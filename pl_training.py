from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from datasets import ClassificationDataset
from tasks import TaskFamily, Classification


class PhantasmLight(pl.LightningModule):
    def __init__(self, batch_size: int, task_dict: Dict[str, TaskFamily] = None):
        super().__init__()
        self.task_dict = dict() if not task_dict else task_dict
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
        for name, loader in train_batch.items():
            task = self.task_dict[name]
            x, y = loader[0], loader[1]
            encoding = self(x)
            if task.head:
                logits = task.head(encoding.pooler_output)
                loss = task.loss(logits, y)
                self.log('train_loss', loss)
        return loss

    def prepare_data(self) -> None:
        def load_mesh_task() -> None:
            with open("sample_data/mesh_descriptors.txt", "r") as f:
                labels = f.readlines()
            labels = {l.strip(): i for i, l in enumerate(labels)}
            dataset = ClassificationDataset(json_file="sample_data/mesh_small.json", tokenizer=self.tokenizer,
                                            fields=["title", "abstract"], label_field="descriptor", labels=labels)
            self.task_dict["mesh"] = Classification(name="mesh", ctrl_token=None, num_labels=len(labels),
                                                    loss=CrossEntropyLoss(), dataset=dataset)

        load_mesh_task()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloaders = dict()
        for name, task in self.task_dict.items():
            dataloaders[name] = DataLoader(task.dataset, batch_size=self.batch_size)
        return dataloaders


if __name__ == '__main__':
    model = PhantasmLight(batch_size=32)
    trainer = pl.Trainer()
    trainer.fit(model)
