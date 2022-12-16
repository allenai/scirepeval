import json
import sys

# setting path
sys.path.append('../')

import argparse
from typing import Dict, Optional, Any
import datasets
import pytorch_lightning as pl
import torch
import torch.nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.distributed import sync_ddp_if_available
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch.distributed import ReduceOp
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoConfig

from adapter_fusion import AdapterFactory
from bert_pals import BertPalsEncoder
from mtl_datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, IRDataset, \
    CustomChainDataset, TripletDataset, RegressionDataset
from schedulers import InverseSquareRootSchedule, InverseSquareRootScheduleConfig
from strategies import BatchingStrategy
from tasks import TaskFamily, load_tasks

pl.seed_everything(42, workers=True)


def init_weights(modules):
    for module in modules:
        module.linear.weight.data.normal_(mean=0.0, std=0.02)
        if module.linear.bias is not None:
            module.linear.bias.data.zero_()


pl_to_split_map = {"fit": "train", "validate": "dev", "test": "test", "predict": "test"}


class SciRepTrain(pl.LightningModule):
    def __init__(self, batch_size: int, init_lr: float, peak_lr: float, tokenizer: str, model: str, warmup_steps: int,
                 log_dir: str,
                 use_ctrl_tokens=False,
                 task_dict: Dict[str, TaskFamily] = None,
                 pals_cfg: str = None, adapter_type: str = None, max_len: int = 512, load_adapters_as=None):
        super().__init__()
        self.task_dict = load_tasks() if not task_dict else task_dict
        print(self.task_dict.keys())
        self.heads = torch.nn.ModuleDict(
            {t.name: t.head for t in self.task_dict.values() if t.head}
        )
        self.init_loss = None
        self.task_idx = {t: i for i, t in enumerate(self.task_dict)}
        self.loss_wt = torch.ones(len(self.task_dict)).float()
        init_weights(self.heads.values())
        self.warmup_steps = warmup_steps
        self.multi_train = None
        self.multi_test = None
        self.multi_val = None
        self.pals = pals_cfg is not None
        self.adapters = adapter_type is not None
        self.use_ctrl_tokens = use_ctrl_tokens
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
            adapters_dir = f'{log_dir}/model/adapters/' if not load_adapters_as else load_adapters_as
            try:
                adapters_dir = json.loads(adapters_dir)
            except:
                pass
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
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.max_len = max_len
        self.save_hyperparameters(ignore=["task_dict"])

    def forward(self, input_ids, attention_mask=None, token_idx=0, task_id=None):
        if not self.pals:
            embedding = self.encoder(input_ids, attention_mask=attention_mask) if not self.adapters else self.encoder(
                input_ids,
                attention_mask=attention_mask,
                task_id=task_id)
            return embedding.last_hidden_state[:, token_idx, :]
        else:
            embedding = self.encoder(input_ids, attention_mask=attention_mask, task_id=task_id)
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
            optimizer_grouped_parameters, lr=self.init_lr, eps=1e-8
        )

        self.opt = optimizer
        if self.pals or self.adapters:
            scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, 77500)
        else:
            scheduler_config = InverseSquareRootScheduleConfig(warmup_updates=self.warmup_steps,
                                                               warmup_init_lr=self.init_lr,
                                                               lr=self.peak_lr)
            scheduler = InverseSquareRootSchedule(scheduler_config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1}
        }

    def calc_loss(self, train_batch, batch_idx):
        losses, loss_per_task = [], torch.zeros(len(self.task_dict)).cuda()
        scl = torch.tensor(0.0)
        for name, batch in train_batch.items():
            task = self.task_dict[name]
            idx = 0 if not self.use_ctrl_tokens else 1
            task_id = task.ctrl_token
            if task.type not in set(["classification", "regression"]):
                query, pos, neg = batch[0][0], batch[0][1], batch[0][2]
                query_ctrl = cand_ctrl = task_id
                if type(task_id) == dict:
                    query_ctrl = task_id["query"]
                    cand_ctrl = task_id["candidates"]
                query_emb, pos_emb, neg_emb = self(query['input_ids'], query['attention_mask'], idx, query_ctrl), self(
                    pos['input_ids'], pos['attention_mask'], idx, cand_ctrl), self(neg['input_ids'],
                                                                                   neg['attention_mask'], idx,
                                                                                   cand_ctrl)
                margin = neg["margin"] if "margin" in neg else None

                curr_loss = task.loss(query_emb, pos_emb, neg_emb, margin)
            else:
                x, y = batch[0], batch[1]
                encoding = self(x['input_ids'], x['attention_mask'], idx, task_id)
                logits = self.heads[name](encoding)
                if task.type == "regression":
                    logits = logits.squeeze()
                curr_loss = task.loss(logits, y)
                if task.multi_label:
                    curr_loss = torch.mean(curr_loss, dim=1)
                elif task.contrastive_loss:
                    scl = task.contrastive_loss(encoding, y, self.heads[name].num_labels)
                    curr_loss = 0.1 * curr_loss + 0.9 * scl
            loss_per_task[self.task_idx[name]] = torch.mean(curr_loss)
        return loss_per_task

    def training_step(self, train_batch, batch_idx):
        loss_per_task = self.calc_loss(train_batch, batch_idx)
        loss = torch.sum(loss_per_task)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log("lr", self.lr_schedulers().get_last_lr()[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, train_batch, batch_idx) -> Optional[STEP_OUTPUT]:
        loss_per_task = self.calc_loss(train_batch, batch_idx)
        # loss_per_task = torch.mul(self.loss_wt.cuda(), loss_per_task)
        loss = torch.sum(loss_per_task)
        dist_loss_per_task = loss_per_task.clone().data
        dist_loss_per_task = sync_ddp_if_available(dist_loss_per_task, reduce_op=ReduceOp.SUM)
        for task in self.task_dict:
            self.log(f"val_loss_{task}", dist_loss_per_task[self.task_idx[task]], on_step=True, on_epoch=True,
                     prog_bar=False,
                     batch_size=self.batch_size, rank_zero_only=True)
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("avg_val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return {"val_loss": loss}

    def load_data(self, split) -> CustomChainDataset:
        hf_split = "validation" if split == "dev" else "train"
        dataset_list = []
        task_dataset_map = {"classification": ClassificationDataset, "regression": RegressionDataset, "ir": IRDataset}
        for t_name, task in self.task_dict.items():
            data_file = {hf_split: task.data_files[split]} if task.data_files else None
            dataset_name = (task.dataset, hf_split)
            data_src = data_file if data_file else dataset_name
            op_token = task.ctrl_token if self.use_ctrl_tokens else None
            if type(data_src) == dict:
                data = datasets.load_dataset("json", data_files=data_src, streaming=True)[
                    next(iter(data_src.keys()))]
            else:
                data = datasets.load_dataset(**data_src[0], split=data_src[1], streaming=True)
            kwargs = {"data": data, "ctrl_token": op_token, "max_len": self.max_len, "task_name": t_name,
                      "tokenizer": self.tokenizer, "fields": task.input_fields,
                      "sample_size": task.sample_size[split] if type(task.sample_size) == dict else task.sample_size}

            if task.type == "classification":
                kwargs.update({"label_field": task.labels_field, "labels": task.labels})
            elif task.type == "regression":
                kwargs.update({"label_field": task.labels_field})
            if task.multi_label:
                dataset_list.append(MultiLabelClassificationDataset(**kwargs))
            else:
                dataset_list.append(task_dataset_map.get(task.type, TripletDataset)(**kwargs))
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
            self.encoder.save_pretrained(f'{log_dir}/model')
        except:
            print("Exception encountered while saving, try agin from checkpoint")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-config', help='path to the task config file', default="sample_data/tasks_config.json")
    parser.add_argument('model', help='HuggingFace model to be used')
    parser.add_argument('--tokenizer', help='HuggingFace tokenizer to be used (same as model name if not supplied)',
                        default=None)
    parser.add_argument('--output', help='dir to save checkpoints and finetuned model', default="./lightning_logs/")
    parser.add_argument('version', help='experiment version')
    parser.add_argument('--pals-config', default=None, help='path to config file for PALS architecture')
    parser.add_argument('--adapter-type', default=None, help='type of adapter architecture (single/fusion)')
    parser.add_argument('--adapters-chkpt', default=None,
                        help='Adapters to be loaded either from a directory path or a dictionary of pretrained huggingface adapters with id')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--peak-lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--warmup', type=int, default=700, help='number of warmup steps')
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--grad-accum', type=int, default=8, help='grad accumulation steps')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--gpu', type=int, default=None, help='number of gpus')
    parser.add_argument('--max-len', type=int, default=512, help='max sequence length')
    parser.add_argument('--val-check_interval', type=float, default=1.0, help='validation loop interval')
    parser.add_argument('--checkpoint', default=None, help='resume from checkpoint path')

    args = parser.parse_args()
    mconfig = AutoConfig.from_pretrained(args.model)
    tasks_dict = load_tasks(args.tasks_config, mconfig.hidden_size)
    log_dir = args.output
    logger = TensorBoardLogger(
        save_dir=log_dir,
        version=args.version,
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

    model = SciRepTrain(batch_size=args.batch_size, init_lr=args.lr,
                        peak_lr=args.peak_lr,
                        tokenizer=args.tokenizer if args.tokenizer else args.model,
                        model=args.model,
                        warmup_steps=args.warmup,
                        use_ctrl_tokens=args.ctrl_tokens, task_dict=tasks_dict, pals_cfg=args.pals_config,
                        adapter_type=args.adapter_type, log_dir=filepath, max_len=args.max_len,
                        load_adapters_as=args.adapters_chkpt)

    hparams = {"gpus": args.gpu, "val_check_interval": args.val_check_interval, "num_sanity_val_steps": 4,
               "max_epochs": args.epochs,
               "accumulate_grad_batches": args.grad_accum, "resume_from_checkpoint": args.checkpoint}

    trainer = pl.Trainer(logger=logger,
                         strategy="ddp" if hparams["gpus"] else None,
                         enable_checkpointing=True,
                         callbacks=[checkpoint_callback],
                         precision=16,
                         **hparams)
    logger.log_hyperparams(hparams)
    logger.log_hyperparams({"tasks": {k: str(v) for k, v in tasks_dict.items()}})
    trainer.fit(model)
