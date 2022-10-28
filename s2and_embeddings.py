from typing import Union, Dict

from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import Evaluator
import numpy as np
import pickle


class S2ANDEvaluator(Evaluator):

    def __init__(self, meta_dataset: Union[str, tuple], model: Model, batch_size: int = 16):
        super(S2ANDEvaluator, self).__init__("S2AND", meta_dataset, SimpleDataset, model, batch_size, [], "paper_id")

    def generate_embeddings(self, save_path: str = None):
        if not save_path:
            raise ValueError("No save path specified")
        results = super().embeddings_generator.generate_embeddings(save_path)
        paper_ids, embs = np.array([str(k) for k in results]), np.array(
            [results[k] for k in results])
        pickle.dump((embs, paper_ids), open(save_path, "wb"))

    def evaluate(self, embeddings: Union[str, Dict[str, np.ndarray]], **kwargs) -> Dict[str, float]:
        raise NotImplementedError("S2AND evaluation is not implemented")

    def calc_metrics(self, test, preds) -> Dict[str, float]:
        raise NotImplementedError("S2AND evaluation is not implemented")
