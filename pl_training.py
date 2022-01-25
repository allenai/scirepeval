from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
import torch.nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, TripletDataset, \
    CustomChainDataset
from tasks import TaskFamily, load_tasks


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
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
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
                print(self.heads[name].linear.weight)
                x, y = batch[0], batch[1]
                encoding = self(x)
                logits = self.heads[name](encoding.pooler_output)
                curr_loss = task.loss(logits, y)
                if task.multi_label:
                    curr_loss = torch.mean(curr_loss, dim=1)
            losses.append(curr_loss)
        loss = torch.mean(torch.cat(losses))
        loss_key = "{}_loss".format(mode)
        self.log(loss_key, loss)
        return {loss_key: loss}

    def training_step(self, train_batch, batch_idx):
        return self.calc_loss(train_batch, batch_idx)

    def validation_step(self, train_batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.calc_loss(train_batch, batch_idx, mode="val")

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
        results = {"avg_val_loss": avg_loss}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        return results

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)
        self.log('avg_val_loss', ret["avg_val_loss"], on_epoch=True, prog_bar=True)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_list = []
        split = pl_to_split_map.get(stage, "train")
        for t_name, task in self.task_dict.items():
            if task.type == "classification":
                if task.multi_label:
                    dataset_list.append(
                        MultiLabelClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                        tokenizer=self.tokenizer,
                                                        fields=["title", "abstract"],
                                                        label_field=task.labels_field,
                                                        labels=task.labels, sample_size=100))
                else:
                    dataset_list.append(ClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                              tokenizer=self.tokenizer,
                                                              fields=["title", "abstract"],
                                                              label_field=task.labels_field,
                                                              labels=task.labels, sample_size=100))
            else:
                dataset_list.append(TripletDataset(task_name=t_name, json_file=task.data_files[split],
                                                   tokenizer=self.tokenizer, fields=["title", "abstract"],
                                                   sample_size=100))
        if split == "train":
            self.multi_train = CustomChainDataset(dataset_list, batch_size=self.batch_size)
        elif split == "dev":
            self.multi_val = CustomChainDataset(dataset_list, batch_size=self.batch_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.multi_train, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=1)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.multi_val, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=1)


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)
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
        mode='min',
        prefix=''
    )

    model = PhantasmLight(batch_size=32)
    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback],
                         val_check_interval=1.0,
                         fast_dev_run=2,
                         num_sanity_val_steps=3)
    trainer.fit(model)
