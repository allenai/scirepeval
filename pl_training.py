from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch.nn
from torch.utils.data import DataLoader, ChainDataset
from transformers import AutoTokenizer, AutoModel

from datasets import ClassificationDataset, multi_collate, MultiLabelClassificationDataset, TripletDataset, \
    CustomChainDataset
from tasks import TaskFamily, load_tasks


def init_weights(modules):
    for module in modules:
        module.linear.weight.data.normal_(mean=0.0, std=0.02)
        if module.linear.bias is not None:
            module.linear.bias.data.zero_()


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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
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
        self.log('train_loss', loss)
        return loss

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_list = []
        for t_name, task in self.task_dict.items():
            if task.type == "classification":
                if task.multi_label:
                    dataset_list.append(MultiLabelClassificationDataset(task_name=t_name, json_file=task.data_file,
                                                                        tokenizer=self.tokenizer,
                                                                        fields=["title", "abstract"],
                                                                        label_field=task.labels_field,
                                                                        labels=task.labels, sample_size=100))
                else:
                    dataset_list.append(ClassificationDataset(task_name=t_name, json_file=task.data_file,
                                                              tokenizer=self.tokenizer,
                                                              fields=["title", "abstract"],
                                                              label_field=task.labels_field,
                                                              labels=task.labels, sample_size=100))
            else:
                dataset_list.append(TripletDataset(task_name=t_name, json_file=task.data_file,
                                                   tokenizer=self.tokenizer, fields=["title", "abstract"],
                                                   sample_size=100))
        if stage in (None, "fit"):
            self.multi_train = CustomChainDataset(dataset_list)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(self.multi_train, batch_size=self.batch_size, collate_fn=multi_collate, num_workers=2)
        return dataloader


if __name__ == '__main__':
    model = PhantasmLight(batch_size=32)
    trainer = pl.Trainer()
    trainer.fit(model)
