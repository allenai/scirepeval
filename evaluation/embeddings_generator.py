from typing import Dict, List, Union

from evaluation.encoders import Model
from tqdm import tqdm
import numpy as np
import json
import pathlib
import logging

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    def __init__(self, datasets, models: Union[Model, List[Model]]):
        self.datasets = datasets
        self.models = models

    def generate_embeddings(self, save_path: str = None) -> Dict[str, np.ndarray]:
        results = dict()
        try:
            for dataset, model in zip(self.datasets, self.models):
                for batch, batch_ids in tqdm(dataset.batches(), total=len(dataset) // dataset.batch_size):
                    emb = model(batch, batch_ids)
                    for paper_id, embedding in zip(batch_ids, emb.unbind()):
                        if type(paper_id) == tuple:
                            paper_id = paper_id[0]
                        if paper_id not in results:
                            results[paper_id] = embedding.detach().cpu().numpy()
                        else:
                            results[paper_id] += embedding.detach().cpu().numpy()
                    del batch
                    del emb
            results = {k: v/len(self.models) for k, v in results.items()}
        except Exception as e:
            print(e)
        finally:
            if save_path:
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as fout:
                    for k, v in results.items():
                        fout.write(json.dumps({"doc_id": k, "embedding": v.tolist()}) + '\n')
        logger.info(f"Generated {len(results)} embeddings")
        return results

    @staticmethod
    def load_embeddings_from_jsonl(embeddings_path: str) -> Dict[str, np.ndarray]:
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc=f'reading embeddings from {embeddings_path}'):
                line_json = json.loads(line)
                embeddings[line_json['doc_id']] = np.array(line_json['embedding'], dtype=np.float16)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings
