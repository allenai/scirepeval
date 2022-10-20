from typing import Union, Dict, Tuple

import numpy as np
from lightning.classification import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from evaluation.embeddings_generator import EmbeddingsGenerator
from abc import ABC, abstractmethod
from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset, IRDataset
import logging
import datasets
import os
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator(ABC):
    def __init__(self, meta_dataset: Union[str, tuple], dataset_class, model: Model, batch_size: int, fields: list):
        dataset = dataset_class(meta_dataset, model.tokenizer.sep_token, batch_size, model.task_id, fields)
        self.embeddings_generator = EmbeddingsGenerator(dataset, model)

    def generate_embeddings(self, save_path: str = None):
        logger.info("Generating embeddings... this might take a while")
        return self.embeddings_generator.generate_embeddings(save_path)

    @abstractmethod
    def evaluate(self, embeddings: Union[str, Dict[str, np.ndarray]], kwargs):
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

    def evaluate(self, embeddings, kwargs=None):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if type(self.test_dataset) == str and os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("csv", data_files={"train": f"{self.test_dataset}/train.csv",
                                                                     "test": f"{self.test_dataset}/test.csv"})
        else:
            split_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1])
        logger.info(f"Loaded {len(split_dataset['train'])} training and {len(split_dataset['test'])} test documents")
        if type(embeddings) == str and os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)
        if self.task == SupervisedTask.CLASSIFICATION:
            self.classify(split_dataset, embeddings)
        elif self.task == SupervisedTask.REGRESSION:
            self.regression(split_dataset, embeddings)

    def read_dataset(self, data: datasets.DatasetDict, embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train, test = data["train"], data["test"]
        x_train, x_test = np.array([embeddings[paper["paper_id"]] for paper in train]), np.array(
            [embeddings[paper["paper_id"]] for paper in test])
        y_train, y_test = np.array([paper["label"] for paper in train]), np.array([paper["label"] for paper in test])
        return x_train, x_test, y_train, y_test

    def classify(self, data, embeddings, cv=3, kwargs=None):
        x_train, x_test, y_train, y_test = self.read_dataset(data, embeddings)
        estimator = LinearSVC(loss="squared_hinge", random_state=42)
        Cs = np.logspace(-4, 2, 7)
        if cv:
            svm = GridSearchCV(estimator=estimator, cv=cv, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
        else:
            svm = estimator
        svm.fit(x_train, y_train)
        preds = svm.predict(x_test)
        for m in self.metrics:
            print(np.round(100 * f1_score(y_test, preds, average=m), 2))

    def regression(self, data, embeddings, kwargs=None):
        pass


class IREvaluator(Evaluator):
    def __init__(self, meta_dataset: Union[str, tuple], test_dataset: str, model: Model, metrics, batch_size: int = 16,
                 fields: list = None):
        super(IREvaluator, self).__init__(meta_dataset, IRDataset, model, batch_size, fields)
        self.test_dataset = test_dataset
        self.metrics = metrics

    def evaluate(self, embeddings):
        pass
