from typing import Union, Dict

import logging
import os
import datasets
import numpy as np
from tqdm import tqdm

from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import IREvaluator
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ReviewerMatchingEvaluator(IREvaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple],
                 reviewer_metadata: Union[str, tuple], model: Model,
                 metrics: tuple, batch_size: int = 16, fields: list = None):
        super(ReviewerMatchingEvaluator, self).__init__(name, meta_dataset, test_dataset, model, metrics, SimpleDataset,
                                                        batch_size, fields, )
        self.reviewer_metadata = reviewer_metadata

    def retrieval(self, embeddings, qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        logger.info("Loading reviewer metadata...")
        if type(self.reviewer_metadata) == str and os.path.isdir(self.test_dataset):
            reviewer_dataset = datasets.load_dataset("json", data_files={
                "metadata": f"{self.test_dataset}/reviewer_metadata.jsonl"})
        else:
            reviewer_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1], split="metadata")
        logger.info(f"Loaded {len(reviewer_dataset)} reviewer metadata")
        reviewer_papers = {d["r_id"]: d["papers"] for d in reviewer_dataset}

        run = dict()
        for qid in tqdm(qrels):
            query = np.array([embeddings[qid]])
            cand_papers = {cid: np.array([embeddings[pid] for pid in reviewer_papers[cid]]) for cid in qrels[qid] if
                           cid in reviewer_papers}
            scores = {cid: cosine_similarity(cand_papers[cid], query).flatten() for cid in cand_papers}
            sorted_scores = {cid: sorted(scores[cid], reverse=True) for cid in scores}
            run[qid] = {cid: np.mean(sorted_scores[cid][:3]) for cid in sorted_scores}
        return run
