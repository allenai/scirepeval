import pickle

import numpy as np

from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import Evaluator


class S2ANDEvaluator:

    def __init__(self, data_dir: str, model: Model, batch_size: int = 16):
        blocks = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
        self.data_dir = data_dir
        self.evaluators = [
            Evaluator(block, f"{data_dir}/{block}/{block}_papers.json", SimpleDataset, model, batch_size, [],
                      "paper_id") for block in blocks]

    def generate_embeddings(self, suffix: str = ""):
        for evaluator in self.evaluators:
            results = evaluator.generate_embeddings()
            paper_ids, embs = np.array([str(k) for k in results]), np.array(
                [results[k] for k in results])
            pickle.dump((embs, paper_ids),
                        open(f"{self.data_dir}/{evaluator.name}/{evaluator.name}_{suffix}.pkl", "wb"))
