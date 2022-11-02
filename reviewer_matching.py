from typing import Union, Dict

import logging
import os
import datasets
import numpy as np
from tqdm import tqdm

from evaluation.embeddings_generator import EmbeddingsGenerator
from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import IREvaluator
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ReviewerMatchingEvaluator(IREvaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple],
                 reviewer_metadata: Union[str, tuple], model: Model,
                 metrics: tuple = ("P_5", "P_10"), batch_size: int = 16, fields: list = None):
        super(ReviewerMatchingEvaluator, self).__init__(name, meta_dataset, test_dataset, model, metrics, SimpleDataset,
                                                        batch_size, fields, )
        self.reviewer_metadata = reviewer_metadata

    def evaluate(self, embeddings, **kwargs):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        if type(self.test_dataset) == str and os.path.isdir(self.test_dataset):
            split_dataset = datasets.load_dataset("json",
                                                  data_files={"test_hard": f"{self.test_dataset}/test_hard_qrel.jsonl",
                                                              "test_soft": f"{self.test_dataset}/test_soft_qrel.jsonl"})
        else:
            split_dataset = datasets.load_dataset(self.test_dataset[0], self.test_dataset[1])
        logger.info(f"Loaded {len(split_dataset['test_hard'])} test query-candidate pairs for hard and soft tests")
        if type(embeddings) == str and os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)

        qrels_hard = self.get_qc_pairs(split_dataset["test_hard"])
        qrels_soft = self.get_qc_pairs(split_dataset["test_soft"])
        preds = self.retrieval(embeddings, qrels_hard)
        results = {f"hard_{k}": v for k, v in self.calc_metrics(qrels_hard, preds).items()}
        results.update({f"soft_{k}": v for k, v in self.calc_metrics(qrels_soft, preds).items()})
        self.print_results(results)
        return results

    def retrieval(self, embeddings, qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        logger.info("Loading reviewer metadata...")
        if type(self.reviewer_metadata) == str and os.path.isdir(self.reviewer_metadata):
            reviewer_dataset = datasets.load_dataset("json", data_files={
                "metadata": f"{self.reviewer_metadata}/reviewer_metadata.jsonl"})["metadata"]
        else:
            reviewer_dataset = datasets.load_dataset(self.reviewer_metadata[0], self.reviewer_metadata[1],
                                                     split="metadata")
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
