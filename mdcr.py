import json
from typing import Union, Dict

import logging
import os
import datasets
import numpy as np
from tqdm import tqdm
import argparse
from evaluation.gpt3_encoder import GPT3Model

from evaluation.embeddings_generator import EmbeddingsGenerator
from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import IREvaluator
from evaluation.instructor import InstructorModel

logger = logging.getLogger(__name__)


class MDCREvaluator(IREvaluator):
    def __init__(self, name: str, meta_dataset: Union[str, tuple], test_dataset: Union[str, tuple], model: Model,
                 metrics: tuple = None, batch_size: int = 16, fields: list = None, key="paper_id"):
        super(MDCREvaluator, self).__init__(name, meta_dataset, test_dataset, model, metrics, SimpleDataset,
                                            batch_size, fields, key)

    def get_qc_pairs(self, dataset):
        qrpairs = dict()
        for fos_dict in dataset:
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
        for q, cscores in tqdm(preds.items()):
            for c in cscores:
                results[f"{q}_{c}"] = cscores[c]
        json.dump(results, open("scirepeval_mdcr.json", "w"))
        return dict()

import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtype', help='Model variant to be used (default, pals, adapters, fusion)', default="default")
    parser.add_argument('--model', '-m', help='HuggingFace model to be used')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--adapters-dir', help='path to the adapter checkpoints', default=None)
    parser.add_argument('--adapters-chkpt', help='hf adapter names keyed on tasks', default=None, type=json.loads)
    parser.add_argument('--fusion-dir', help='path to the fusion checkpoints', default=None)
    parser.add_argument('--instructor', action='store_true', default=False, help='use an instructor model for eval')
    parser.add_argument('--gpt3-model', help='Name of embedding model in case of using openai api', default=None)

    args = parser.parse_args()
    adapters_load_from = args.adapters_dir if args.adapters_dir else args.adapters_chkpt
    if args.gpt3_model:
        model = GPT3Model(embed_model=args.gpt3_model)
    elif args.instructor:
        model = InstructorModel(args.model)
        model.task_id = "[PRX]"
    else:
        model = Model(variant=args.mtype, base_checkpoint=args.model, adapters_load_from=adapters_load_from,
                      fusion_load_from=args.fusion_dir, use_ctrl_codes=args.ctrl_tokens,
                      task_id="[PRX]", all_tasks=["[CLF]", "[PRX]", "[RGN]", "[QRY]"])
    evaluator = MDCREvaluator("mdcr", "../mdcr/mdcr_test_data.jsonl", "../mdcr/mdcr_test.json", model, batch_size=32)
    embeddings = evaluator.generate_embeddings(save_path="mdcr_embeddings.json")
    evaluator.evaluate(embeddings)
