from typing import Union, Dict

import numpy as np

from evaluation.embeddings_generator import EmbeddingsGenerator
from abc import ABC, abstractmethod
from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset, IRDataset
import logging
import datasets
import os
from enum import Enum

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    def __init__(self, meta_dataset: Union[str, tuple], dataset_class, model: Model, batch_size: int, fields: list):
        dataset = dataset_class(meta_dataset, model.tokenizer.sep_token, batch_size, model.task_id, fields)
        self.embeddings_generator = EmbeddingsGenerator(dataset, model)

    def generate_embeddings(self, save_path: str = None):
        return self.embeddings_generator.generate_embeddings(save_path)

    @abstractmethod
    def evaluate(self, embeddings: Union[str, Dict[str, np.ndarray]]):
        pass


class SupervisedTask(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class SupervisedEvaluator(Evaluator):
    def __init__(self, task: SupervisedTask, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple],
                 model: Model, metrics: tuple = None, batch_size: int = 16, fields: list = None):
        super(SupervisedEvaluator, self).__init__(meta_dataset, SimpleDataset, model, batch_size, fields)
        self.test_dataset = test_dataset
        self.metrics = metrics
        self.task = task

    def evaluate(self, embeddings):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("csv", data_files={"train": f"{self.test_dataset}/train.csv",
                                                                     "test": f"{self.test_dataset}/test.csv"})
        else:
            split_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1])

        if os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)
        if self.task == SupervisedTask.CLASSIFICATION:
            self.classify(split_dataset, embeddings)
        elif self.task == SupervisedTask.REGRESSION:
            self.regression(split_dataset, embeddings)

    def classify(self, data, embeddings):
        pass

    def regression(self, data, embeddings):
        pass


class IREvaluator(Evaluator):
    def __init__(self, meta_dataset: Union[str, tuple], test_dataset: str, model: Model, metrics, batch_size: int = 16,
                 fields: list = None):
        super(IREvaluator, self).__init__(meta_dataset, IRDataset, model, batch_size, fields)
        self.test_dataset = test_dataset
        self.metrics = metrics

    def evaluate(self, embeddings):
        pass
