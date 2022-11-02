import math
from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedKFold

import evaluation.evaluator
from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask
from tqdm import tqdm


class FewShotEvaluator(SupervisedEvaluator):
    def __init__(self, name: str, task: SupervisedTask, meta_dataset: Union[str, tuple],
                 test_dataset: Union[str, tuple], sample_size: int, num_iterations: int,
                 model: Model, metrics: tuple = None, batch_size: int = 16, fields: list = None):
        super(FewShotEvaluator, self).__init__(name, task, meta_dataset, test_dataset, model, metrics, batch_size,
                                               fields)
        self.sample_size = sample_size
        self.num_iterations = num_iterations

    def classify(self, x, x_test, y, cv=3, n_jobs=1):
        stage_preds = []
        if self.task == SupervisedTask.MULTILABEL_CLASSIFICATION:
            for k in tqdm(range(self.num_iterations)):
                idx_set = set()
                np.random.seed(evaluation.evaluator.RANDOM_STATE + k)
                for yi in range(y.shape[1]):
                    idx_set.update(
                        np.random.choice(np.where(y[:, yi] == 1)[0], self.sample_size, replace=False).tolist())
                req_idx = list(idx_set)
                x_train, y_train = x[req_idx], y[req_idx]
                preds = super().classify(x_train, x_test, y_train)
                stage_preds.append(preds)
            np.random.seed(evaluation.evaluator.RANDOM_STATE)
        else:
            skf = StratifiedKFold(n_splits=math.ceil(x.shape[0] / self.sample_size))
            count = 0
            for _, train in tqdm(skf.split(x, y), total=self.num_iterations):
                x_train, y_train = x[train], y[train]
                res = super().classify(x_train, x_test, y_train, cv=0)
                stage_preds.append(res)
                count += 1
                if count == self.num_iterations:
                    break
        return stage_preds

    def calc_metrics(self, test, preds_list):
        stage_results = dict()
        for preds in preds_list:
            res = super().calc_metrics(test, preds)
            for k, v in res.items():
                if k not in stage_results:
                    stage_results[k] = []
                stage_results[k].append(v)

        results = {k: np.mean(v) for k, v in stage_results.items()}
        return results
