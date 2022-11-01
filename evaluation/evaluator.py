from typing import Union, Dict, Tuple

import numpy as np
from lightning.classification import LinearSVC
from lightning.regression import LinearSVR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from scipy.stats import kendalltau, pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from evaluation.embeddings_generator import EmbeddingsGenerator
from abc import ABC, abstractmethod
from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset, IRDataset
import logging
import datasets
import os
from enum import Enum
from sklearn.metrics.pairwise import euclidean_distances
import pytrec_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RANDOM_STATE = 42


class Evaluator:
    def __init__(self, name: str, meta_dataset: Union[str, tuple], dataset_class, model: Model, batch_size: int,
                 fields: list, key: str = None, process_fn=None):
        dataset = dataset_class(meta_dataset, model.tokenizer.sep_token, batch_size, model.task_id, fields, key,
                                process_fn)
        self.embeddings_generator = EmbeddingsGenerator(dataset, model)
        self.name = name

    def generate_embeddings(self, save_path: str = None):
        logger.info("Generating embeddings... this might take a while")
        return self.embeddings_generator.generate_embeddings(save_path)

    @abstractmethod
    def evaluate(self, embeddings: Union[str, Dict[str, np.ndarray]], **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def calc_metrics(self, test, preds) -> Dict[str, float]:
        pass

    def print_results(self, results: Dict[str, float]):
        if results:
            print("*****************************************************")
            print(f"                 {self.name}")
            print("*****************************************************")
            for k, v in results.items():
                print(f"                 {k}: {v}")
            print("*****************************************************")


class SupervisedTask(Enum):
    CLASSIFICATION = 1
    MULTILABEL_CLASSIFICATION = 2
    REGRESSION = 3


SUPSERVISED_TASK_METRICS = {
    SupervisedTask.CLASSIFICATION: {"f1": f1_score, "accuracy": accuracy_score, "precision": precision_score,
                                    "recall": recall_score},
    SupervisedTask.REGRESSION: {"mse": mean_squared_error, "r2": r2_score, "pearsonr": pearsonr,
                                "kendalltau": kendalltau}
}


class SupervisedEvaluator(Evaluator):
    def __init__(self, name: str, task: SupervisedTask, meta_dataset: Union[str, tuple],
                 test_dataset: Union[str, tuple],
                 model: Model, metrics: tuple, batch_size: int = 16, fields: list = None):
        super(SupervisedEvaluator, self).__init__(name, meta_dataset, SimpleDataset, model, batch_size, fields)
        self.test_dataset = test_dataset
        self.metrics = metrics
        self.task = task

    def evaluate(self, embeddings, **kwargs):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if type(self.test_dataset) == str and os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("csv", data_files={"train": f"{self.test_dataset}/train.csv",
                                                                     "test": f"{self.test_dataset}/test.csv"})
        else:
            split_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1])
        logger.info(f"Loaded {len(split_dataset['train'])} training and {len(split_dataset['test'])} test documents")
        if type(embeddings) == str and os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)
        x_train, x_test, y_train, y_test = self.read_dataset(split_dataset, embeddings)
        eval_fn = self.regression if self.task == SupervisedTask.REGRESSION else self.classify
        preds = eval_fn(x_train, x_test, y_train)
        results = self.calc_metrics(y_test, preds)
        self.print_results(results)
        return results

    @staticmethod
    def read_dataset(data: datasets.DatasetDict, embeddings: Dict[str, np.ndarray]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train, test = data["train"], data["test"]
        x_train, x_test = np.array(
            [embeddings[str(paper["paper_id"])] for paper in train if str(paper["paper_id"]) in embeddings]), np.array(
            [embeddings[str(paper["paper_id"])] for paper in test if str(paper["paper_id"]) in embeddings])
        y_train, y_test = np.array(
            [paper["label"] for paper in train if str(paper["paper_id"]) in embeddings]), np.array(
            [paper["label"] for paper in test if str(paper["paper_id"]) in embeddings])
        return x_train, x_test, y_train, y_test

    def classify(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, cv: int = 3,
                 n_jobs: int = 5):

        Cs = np.logspace(-4, 2, 7)
        if self.task == SupervisedTask.MULTILABEL_CLASSIFICATION:
            estimator = LinearSVC(max_iter=10000)
            svm = GridSearchCV(estimator=estimator, cv=cv, param_grid={'C': Cs}, n_jobs=5)
            svm = OneVsRestClassifier(svm, n_jobs=4)
        else:
            estimator = LinearSVC(loss="squared_hinge", random_state=RANDOM_STATE)
            if cv:
                svm = GridSearchCV(estimator=estimator, cv=cv, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
            else:
                svm = estimator
        svm.fit(x_train, y_train)
        preds = svm.predict(x_test)
        return preds

    def regression(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, cv: int = 3,
                   n_jobs: int = 5):
        svm = LinearSVR(random_state=RANDOM_STATE)
        Cs = np.logspace(-4, 2, 7)
        svm = GridSearchCV(estimator=svm, cv=cv, param_grid={'C': Cs}, verbose=1, n_jobs=n_jobs)
        svm.fit(x_train, y_train)
        preds = svm.predict(x_test)
        return preds

    def calc_metrics(self, test, preds):
        results = dict()
        if self.task == SupervisedTask.REGRESSION:
            for m in self.metrics:
                if m in SUPSERVISED_TASK_METRICS[self.task]:
                    result = tuple(SUPSERVISED_TASK_METRICS[self.task][m](test, preds))[0]
                    if m != "mse":
                        result = np.round(100 * result, 2)
                    results[m] = result
                else:
                    logger.warning(
                        f"Metric {m} not found...skipping, try one of {SUPSERVISED_TASK_METRICS[self.task].keys()}")
        else:
            for m in self.metrics:
                split_m = m.split("_")
                if split_m[0] in SUPSERVISED_TASK_METRICS[self.task]:
                    if len(split_m) > 1:
                        result = SUPSERVISED_TASK_METRICS[self.task][split_m[0]](test, preds, average=split_m[1])
                    else:
                        result = SUPSERVISED_TASK_METRICS[self.task][split_m[0]](test, preds)
                    results[m] = np.round(100 * result, 2)
                else:
                    logger.warning(
                        f"Metric {m} not found...skipping, try one of {SUPSERVISED_TASK_METRICS[self.task].keys()}")
        return results


class IREvaluator(Evaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple], model: Model,
                 metrics: tuple, dataset_class=IRDataset, batch_size: int = 16, fields: list = None):
        super(IREvaluator, self).__init__(name, meta_dataset, dataset_class, model, batch_size, fields)
        self.test_dataset = test_dataset
        self.metrics = metrics

    def get_qc_pairs(self, dataset):
        pairs = dict()
        for row in dataset:
            if row["query_id"] not in pairs:
                pairs[row["query_id"]] = dict()
            pairs[row["query_id"]][row["cand_id"]] = row["score"]
        return pairs

    def calc_metrics(self, qrels, run):
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(self.metrics))
        results = evaluator.evaluate(run)

        metric_values = {}
        for measure in sorted(self.metrics):
            res = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()]
            )
            metric_values[measure] = np.round(100 * res, 2)
        return metric_values

    def evaluate(self, embeddings, **kwargs):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if type(self.test_dataset) == str and os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("json", data_files={"test": f"{self.test_dataset}/test_qrel.jsonl"})
        else:
            split_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1])
        logger.info(f"Loaded {len(split_dataset['test'])} test query-candidate pairs")
        if type(embeddings) == str and os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)

        qrels = self.get_qc_pairs(split_dataset["test"])
        preds = self.retrieval(embeddings, qrels)
        results = self.calc_metrics(qrels, preds)
        self.print_results(results)
        return results

    def retrieval(self, embeddings, qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        run = dict()
        for qid in qrels:
            if qid in embeddings:
                query = np.array([embeddings[qid]])
                cids = [cid for cid in qrels[qid] if cid in embeddings]
                cands = np.array([embeddings[cid] for cid in qrels[qid] if cid in embeddings])
                scores = euclidean_distances(cands, query).flatten()
                run[qid] = dict()
                for i, cid in enumerate(cids):
                    run[qid][cid] = -scores[i]
        return run
