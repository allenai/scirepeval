import pickle

import numpy as np

from evaluation.encoders import Model
from evaluation.eval_datasets import SimpleDataset
from evaluation.evaluator import Evaluator
import argparse
from tqdm import tqdm


class S2ANDEvaluator:

    def __init__(self, data_dir: str, model: Model, batch_size: int = 16):
        blocks = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
        self.data_dir = data_dir
        self.evaluators = [
            Evaluator(block, f"{data_dir}/{block}/{block}_papers.json", SimpleDataset, model, batch_size, [],
                      "paper_id") for block in blocks]

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
    parser.add_argument('--model', '--m', help='HuggingFace model to be used')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--adapters-dir', help='path to the adapter checkpoints', default=None)
    parser.add_argument("--data-dir", help="path to the output dir")
    parser.add_argument("--suffix", help="suffix for output embedding files")

    args = parser.parse_args()

    model = Model(variant=args.mtype, base_checkpoint=args.model, adapters_load_from=args.adapters_dir,
                  use_ctrl_codes=args.ctrl_tokens,
                  task_id="[PRX]", all_tasks=["[CLF]", "[PRX]", "[RGN]", "[QRY]"])
    evaluator = S2ANDEvaluator(args.data_dir, model)
    evaluator.generate_embeddings(args.suffix)
