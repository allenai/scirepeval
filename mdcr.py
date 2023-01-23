import json
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

logger = logging.getLogger(__name__)


class MDCREvaluator(IREvaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple], model: Model,
                 metrics: tuple = None, batch_size: int = 16, fields: list = None, key="paper_id"):
        super(MDCREvaluator, self).__init__(name, meta_dataset, test_dataset, model, metrics, SimpleDataset,
                                            batch_size, fields, key)

    def get_qc_pairs(self, dataset):
        qrpairs = dict()
        for fos_dict in dataset["test"]:
            for fos in fos_dict:
                for query in fos_dict[fos]:
                    qrpairs[query] = dict()
                    for model in fos_dict[fos][query]:
                        cands = fos_dict[fos][query][model]
                        qrpairs[query].update({v: 1 if model == "true" else 0 for v in cands})
        return qrpairs

    def evaluate(self, embeddings, **kwargs):
        logger.info(f"Loading test dataset from {self.test_dataset}")
        split_dataset = datasets.load_dataset("json",
                                              data_files={"test": self.test_dataset})
        logger.info(f"Loaded {len(split_dataset['test'])} test query-candidate pairs")
        if type(embeddings) == str and os.path.isfile(embeddings):
            embeddings = EmbeddingsGenerator.load_embeddings_from_jsonl(embeddings)

        qrels_hard = self.get_qc_pairs(split_dataset["test"])
        preds = self.retrieval(embeddings, qrels_hard)
        results = dict()
        for q, cscores in tqdm(preds):
            for c in cscores:
                results[f"{q}_{c}"] = cscores[c]
        json.dump(results, open("scirepeval_mdcr.json", "w"))
        return dict()