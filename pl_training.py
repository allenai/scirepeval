from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
import torch.nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.plugins import DDPPlugin

from datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, TripletDataset, \
    CustomChainDataset
from tasks import TaskFamily, load_tasks
from strategies import BatchingStrategy

pl.seed_everything(42, workers=True)


def init_weights(modules):
    for module in modules:
        module.linear.weight.data.normal_(mean=0.0, std=0.02)
        if module.linear.bias is not None:
            module.linear.bias.data.zero_()


pl_to_split_map = {"fit": "train", "validate": "dev", "test": "test", "predict": "test"}


class PhantasmLight(pl.LightningModule):
    def __init__(self, batch_size: int, task_dict: Dict[str, TaskFamily] = None):
        super().__init__()
        self.task_dict = load_tasks() if not task_dict else task_dict
        self.heads = torch.nn.ModuleDict(
            {t.name: t.head for t in self.task_dict.values() if t.head}
        )
        init_weights(self.heads.values())
        self.multi_train = None
        self.multi_test = None
        self.multi_val = None
        self.encoder = AutoModel.from_pretrained("allenai/specter")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        self.batch_size = batch_size

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-6)
        return optimizer

    def calc_loss(self, train_batch, batch_idx, mode="train"):
        losses = []
        for name, batch in train_batch.items():
            task = self.task_dict[name]
            if task.type == "triplet":
                query, pos, neg = batch[0][0], batch[0][1], batch[0][2]
                query_emb, pos_emb, neg_emb = self(query), self(pos), self(neg)
                curr_loss = task.loss(query_emb.pooler_output, pos_emb.pooler_output, neg_emb.pooler_output)
            else:
                x, y = batch[0], batch[1]
                encoding = self(x)
                logits = self.heads[name](encoding.pooler_output)
                curr_loss = task.loss(logits, y)
                if task.multi_label:
                    curr_loss = torch.mean(curr_loss, dim=1)
            losses.append(curr_loss)
        loss = torch.mean(torch.cat(losses))
        loss_key = "{}_loss".format(mode)
        self.log(loss_key, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {loss_key if mode != "train" else "loss": loss}

    def training_step(self, train_batch, batch_idx):
        return self.calc_loss(train_batch, batch_idx)

    def validation_step(self, train_batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.calc_loss(train_batch, batch_idx, mode="val")

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.trainer.data_parallel:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
        results = {"avg_val_loss": avg_loss}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        return results

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)
        self.log('avg_val_loss', ret["avg_val_loss"], on_epoch=True, prog_bar=True, sync_dist=True)

    def load_data(self, split) -> CustomChainDataset:
        dataset_list = []
        for t_name, task in self.task_dict.items():
            if task.type == "classification":
                if task.multi_label:
                    dataset_list.append(
                        MultiLabelClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                        tokenizer=self.tokenizer,
                                                        fields=["title", "abstract"],
                                                        label_field=task.labels_field,
                                                        labels=task.labels))
                else:
                    dataset_list.append(ClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                              tokenizer=self.tokenizer,
                                                              sample_size=400000 if split == "train" else 80000,
                                                              fields=["title", "abstract"],
                                                              label_field=task.labels_field,
                                                              labels=task.labels))
            else:
                dataset_list.append(TripletDataset(task_name=t_name, json_file=task.data_files[split],
                                                   sample_size=400000 if split == "train" else 80000,
                                                   tokenizer=self.tokenizer, fields=["title", "abstract"]))
        multi_dataset = CustomChainDataset(dataset_list, batch_size=self.batch_size,
                                           device_rank=self.trainer.global_rank, num_devices=self.trainer.world_size,
                                           batching_strategy=BatchingStrategy.SEQUENTIAL)
        if split == "train":
            self.multi_train = multi_dataset
        elif split == "dev":
            self.multi_val = multi_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data("train")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.multi_train, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=3)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.load_data("dev")
        return DataLoader(self.multi_val, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=3)


if __name__ == '__main__':
    log_dir = "./lightning_logs/full_run/"
    logger = TensorBoardLogger(
        save_dir="./lightning_logs/full_run/",
        version=0,
        name='pl-logs'
    )

    # second part of the path shouldn't be f-string
    filepath = f'{log_dir}/version_{logger.version}/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        dirpath=filepath,
        filename='ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=1,
        verbose=True,
        monitor='avg_val_loss',  # monitors metrics logged by self.log.
        mode='min'
    )

    model = PhantasmLight(batch_size=16)
    trainer = pl.Trainer(logger=logger,
                         gpus=[0, 1, 2],
                         strategy=DDPPlugin(find_unused_parameters=True),
                         enable_checkpointing=True,
                         callbacks=[checkpoint_callback],
                         val_check_interval=10000,
                         num_sanity_val_steps=3,
                         max_epochs=2)
    trainer.fit(model)