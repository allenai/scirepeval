from typing import Union

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

    def classify(self, x_train, x_test, y_train, y_test, multi_label=False, cv=3, n_jobs=1):
