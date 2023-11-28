import pickle

import numpy as np

from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import Evaluator
import argparse
from tqdm import tqdm
from evaluation.instructor import InstructorModel

import json


def read_data(file_path):
    task_data = json.load(open(file_path, "r"))
    task_data = list(task_data.values())
    return task_data


class S2ANDEvaluator:

    def __init__(self, data_dir: str, model: Model, batch_size: int = 16):
        blocks = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
        self.data_dir = data_dir
        self.evaluators = [
            Evaluator(block, f"{data_dir}/{block}/{block}_papers.json", SimpleDataset, model, batch_size, [],
                      "paper_id", process_fn=read_data) for block in blocks]

    def generate_embeddings(self, suffix: str):
        for evaluator in tqdm(self.evaluators):
            print(evaluator.name)
            results = evaluator.generate_embeddings()
            paper_ids, embs = np.array([str(k) for k in results]), np.array(
                [results[k] for k in results])
            pickle.dump((embs, paper_ids),
                        open(f"{self.data_dir}/{evaluator.name}/{evaluator.name}_{suffix}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtype', help='Model variant to be used (default, pals, adapters, fusion)', default="default")
    parser.add_argument('--model', '-m', help='HuggingFace model to be used')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--adapters-dir', help='path to the adapter checkpoints', default=None)
    parser.add_argument('--adapters-chkpt', help='hf adapter names keyed on tasks', default=None, type=json.loads)
    parser.add_argument('--fusion-dir', help='path to the fusion checkpoints', default=None)
    parser.add_argument("--data-dir", help="path to the data directory")
    parser.add_argument("--suffix", help="suffix for output embedding files")
    parser.add_argument('--instructor', action='store_true', default=False, help='use an instructor model for eval')

    args = parser.parse_args()
    adapters_load_from = args.adapters_dir if args.adapters_dir else args.adapters_chkpt
    if args.instructor:
        model = InstructorModel(args.model)
        model.task_id = "[PRX]"
    else:
        model = Model(variant=args.mtype, base_checkpoint=args.model, adapters_load_from=adapters_load_from,
                      fusion_load_from=args.fusion_dir, use_ctrl_codes=args.ctrl_tokens,
                      task_id="[PRX]", all_tasks=["[CLF]", "[PRX]", "[RGN]", "[QRY]"])
    evaluator = S2ANDEvaluator(args.data_dir, model)
    evaluator.generate_embeddings(args.suffix)
