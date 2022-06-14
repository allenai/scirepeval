from typing import Dict, Optional, Any

import pytorch_lightning as pl
import torch
import torch.nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel

from datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, IRDataset, \
    CustomChainDataset, TripletDataset
from schedulers import InverseSquareRootSchedule, InverseSquareRootScheduleConfig
from strategies import BatchingStrategy
from tasks import TaskFamily, load_tasks
from bert_pals import BertPalsEncoder
from adapter_fusion import AdapterFactory

pl.seed_everything(42, workers=True)


def init_weights(modules):
    for module in modules:
        module.linear.weight.data.normal_(mean=0.0, std=0.02)
        if module.linear.bias is not None:
            module.linear.bias.data.zero_()


pl_to_split_map = {"fit": "train", "validate": "dev", "test": "test", "predict": "test"}


class PhantasmLight(pl.LightningModule):
    def __init__(self, batch_size: int, lr, tokenizer: str, model: str, warmup_steps: int, log_dir: str,
                 use_ctrl_tokens=False,
                 task_dict: Dict[str, TaskFamily] = None,
                 pals_cfg: str = None, adapter_type: str = None):
        super().__init__()
        self.task_dict = load_tasks() if not task_dict else task_dict
        print(self.task_dict.keys())
        self.heads = torch.nn.ModuleDict(
            {t.name: t.head for t in self.task_dict.values() if t.head}
        )
        init_weights(self.heads.values())
        self.warmup_steps = warmup_steps
        self.multi_train = None
        self.multi_test = None
        self.multi_val = None
        self.pals = pals_cfg is not None
        self.adapters = adapter_type is not None
        self.use_ctrl_tokens = use_ctrl_tokens
        # task_ids = [task for task in self.task_dict]
        spl_ctrl_tokens = set()
        for t in self.task_dict.values():
            if type(t.ctrl_token) == str:
                spl_ctrl_tokens.add(t.ctrl_token)
            else:
                spl_ctrl_tokens.update(t.ctrl_token.values())
        spl_ctrl_tokens = sorted(list(spl_ctrl_tokens))
        task_ids = spl_ctrl_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if self.adapters:
            adapters_dir = f'{log_dir}/model/adapters/'
            self.encoder = AdapterFactory.get_adapter(model, task_ids,
                                                      adapter_type == "fusion", adapters_dir)
        else:
            self.encoder = AutoModel.from_pretrained(model)
            if self.pals:
                self.encoder = BertPalsEncoder(f"bert_pals_config/{pals_cfg}", task_ids, self.encoder)
        if self.use_ctrl_tokens:
            print("Using Control Tokens", spl_ctrl_tokens)
            special_tokens_dict = {'additional_special_tokens': spl_ctrl_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            self.encoder.resize_token_embeddings(len(self.tokenizer))
            # if num_added_toks:
            #     with torch.no_grad():
            #         self.encoder.embeddings.word_embeddings.weight[-num_added_toks:,
            #         :] += self.encoder.embeddings.word_embeddings.weight[self.tokenizer.cls_token_id, :]
        self.batch_size = batch_size
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x, token_idx=0, task_id=None):
        if not self.pals:
            embedding = self.encoder(x) if not self.adapters else self.encoder(x, task_id)
            return embedding.last_hidden_state[:, token_idx, :]
        else:
            embedding = self.encoder(x, task_id)
            return embedding[:, token_idx, :]

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.lr, eps=1e-8
        )

        self.opt = optimizer
        scheduler_config = InverseSquareRootScheduleConfig(warmup_updates=self.warmup_steps, warmup_init_lr=self.lr,
                                                           lr=5e-5)
        scheduler = InverseSquareRootSchedule(scheduler_config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1}
        }

    def calc_loss(self, train_batch, batch_idx):
        losses, loss_per_task = [], dict()
        scl = torch.tensor(0.0)
        for name, batch in train_batch.items():
            # print(name, batch[0].shape if name not in ("s2and", "search", "specter") else batch[0][0].shape, self.global_rank)
            task = self.task_dict[name]
            idx = 0 if not self.use_ctrl_tokens else 1
            task_id = task.ctrl_token
            if task.type != "classification":
                query, pos, neg = batch[0][0], batch[0][1], batch[0][2]
                query_ctrl = cand_ctrl = task_id
                if type(task_id) == dict:
                    query_ctrl = task_id["query"]
                    cand_ctrl = task_id["candidates"]
                query_emb, pos_emb, neg_emb = self(query, idx, query_ctrl), self(pos, idx, cand_ctrl), self(neg, idx,
                                                                                                            cand_ctrl)
                curr_loss = task.loss(query_emb, pos_emb, neg_emb)
            else:
                x, y = batch[0], batch[1]
                encoding = self(x, idx, task_id)
                logits = self.heads[name](encoding)
                curr_loss = task.loss(logits, y)
                if task.multi_label:
                    curr_loss = torch.mean(curr_loss, dim=1)
                elif task.contrastive_loss:
                    scl = task.contrastive_loss(encoding, y, self.heads[name].num_labels)
                    curr_loss = 0.1 * curr_loss + 0.9 * scl

            loss_per_task[name] = torch.mean(curr_loss)
            losses.append(curr_loss)
        loss = torch.mean(torch.cat(losses))
        return loss, loss_per_task

    def training_step(self, train_batch, batch_idx):

        loss, loss_per_task = self.calc_loss(train_batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, train_batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss, loss_per_task = self.calc_loss(train_batch, batch_idx)
        for task, task_loss in loss_per_task.items():
            self.log(f"val_loss_{task}", task_loss, on_step=True, on_epoch=True, prog_bar=False,
                     batch_size=self.batch_size, rank_zero_only=True)
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("avg_val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return {"val_loss": loss}

    # def validation_epoch_end(self, outputs: list) -> dict:
    #     # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     # print(avg_loss)
    #     # avg_loss = torch.mean(self.all_gather(avg_loss))
    #     # self.log("avg_val_loss", avg_loss, on_epoch=True, prog_bar=True, rank_zero_only=True, logger=True, sync_dist=True)
    #     for task in self.task_counts:
    #         self.task_counts[task] = torch.tensor(0)

    def load_data(self, split) -> CustomChainDataset:
        dataset_list = []
        for t_name, task in self.task_dict.items():
            op_token = task.ctrl_token if self.use_ctrl_tokens else None
            if task.type == "classification":
                if task.multi_label:
                    dataset_list.append(
                        MultiLabelClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                        tokenizer=self.tokenizer, ctrl_token=op_token,
                                                        fields=task.input_fields,
                                                        label_field=task.labels_field,
                                                        labels=task.labels,
                                                        sample_size=600000 if split == "train" else 80000))
                else:
                    dataset_list.append(ClassificationDataset(task_name=t_name, json_file=task.data_files[split],
                                                              tokenizer=self.tokenizer, ctrl_token=op_token,
                                                              fields=task.input_fields,
                                                              label_field=task.labels_field,
                                                              labels=task.labels,
                                                              sample_size=600000 if split == "train" else 80000))
            elif task.type == "ir":
                dataset_list.append(
                    IRDataset(task_name=t_name, json_file=task.data_files[split], ctrl_token=op_token,
                              tokenizer=self.tokenizer, fields=task.input_fields,
                              sample_size=600000 if split == "train" else 80000))
            else:
                dataset_list.append(
                    TripletDataset(task_name=t_name, json_file=task.data_files[split], ctrl_token=op_token,
                                   tokenizer=self.tokenizer, fields=task.input_fields,
                                   sample_size=600000 if split == "train" else 80000))
        multi_dataset = CustomChainDataset(dataset_list, batch_size=self.batch_size,
                                           device_rank=self.trainer.global_rank, num_devices=self.trainer.world_size,
                                           batching_strategy=BatchingStrategy.MIXED_PROPORTIONAL)
        if split == "train":
            self.multi_train = multi_dataset
        elif split == "dev":
            self.multi_val = multi_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.load_data("train")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.multi_train, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=4,
                          pin_memory=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        self.load_data("dev")
        return DataLoader(self.multi_val, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=2)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            logger = self.logger
            log_dir = f'{logger.save_dir}/{logger.name}/{logger.version}/checkpoints'
            self.tokenizer.save_pretrained(f'{log_dir}/tokenizer/')
            self.tokenizer.save_vocabulary(f'{log_dir}/tokenizer/')
            if self.pals:
                torch.save(self.encoder.state_dict(),
                           f'{log_dir}/model/pytorch_model.bin')
                self.encoder.bert_config.save_pretrained(f'{log_dir}/model/')
            else:
                self.encoder.save_pretrained(f'{log_dir}/model/')
        except:
            print("Exception encountered while saving, try agin from checkpoint")


if __name__ == '__main__':
    log_dir = "./lightning_logs/"
    logger = TensorBoardLogger(
        save_dir=log_dir,
        version='linkbert_lg',
        name='full_run',
    )

    # second part of the path shouldn't be f-string
    filepath = f'{log_dir}/{logger.name}/{logger.version}/checkpoints/'
    checkpoint_callback = ModelCheckpoint(
        dirpath=filepath,
        filename='ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=4,
        verbose=True,
        monitor='avg_val_loss',  # monitors metrics logged by self.log.
        mode='min'
    )

    model = PhantasmLight(batch_size=8, lr=5e-6,
                          tokenizer="michiyasunaga/BioLinkBERT-large",
                          model="michiyasunaga/BioLinkBERT-large",
                          warmup_steps=600,
                          use_ctrl_tokens=True, pals_cfg=None, adapter_type=None, log_dir=filepath)

    hparams = {"gpus": [0, 1, 2, 3], "val_check_interval": 1.0, "num_sanity_val_steps": 4, "max_epochs": 4,
               "accumulate_grad_batches": 16}

    trainer = pl.Trainer(logger=logger,
                         strategy="ddp",
                         enable_checkpointing=True,
                         callbacks=[checkpoint_callback],
                         precision=16,
                         **hparams)
    logger.log_hyperparams(hparams)
    trainer.fit(model, ckpt_path=f"{filepath}/ep-epoch=2_avg_val_loss-avg_val_loss=0.305.ckpt")
