import math
from typing import Union

import numpy as np
from sklearn.model_selection import StratifiedKFold

from evaluation.encoders import Model
from evaluation.evaluator import SupervisedEvaluator, SupervisedTask


class FewShotEvaluator(SupervisedEvaluator):
    def __init__(self, name: str, task: SupervisedTask, meta_dataset: Union[str, tuple],
                 test_dataset: Union[str, tuple], sample_size: int, num_iterations: int,
                 model: Model, metrics: tuple = None, batch_size: int = 16, fields: list = None):
        super(FewShotEvaluator, self).__init__(name, task, meta_dataset, test_dataset, model, metrics, batch_size,
                                               fields)
        self.sample_size = sample_size
        self.num_iterations = num_iterations

    def classify(self, x, x_test, y, y_test, multi_label=False, cv=3, n_jobs=1):
        if self.task == SupervisedTask.MULTILABEL_CLASSIFICATION:
            pass
        else:
            stage_results = dict()
            skf = StratifiedKFold(n_splits=math.ceil(x.shape[0] / self.sample_size))
            count = 0
            for _, train in skf.split(x, y):
                x_train, y_train = x[train], y[train]
                res = super().classify(x_train, x_test, y_train, y_test, cv=None)
                for k, v in res.items():
                    if k not in stage_results:
                        stage_results[k] = []
                    stage_results[k].append(v)
                count += 1
                if count == self.num_iterations:
                    break
            results = {k: np.mean(v) for k, v in stage_results.items()}
            self.print_results(results)