from typing import Dict

from encoders import Model
from tqdm import tqdm
import numpy as np
import json
import pathlib


class EmbeddingsGenerator:
    def __init__(self, dataset, model: Model):
        self.dataset = dataset
        self.model = model

    def generate_embeddings(self, save_path: str = None) -> Dict[str, np.ndarray]:
        results = dict()
        try:
            for batch, batch_ids in tqdm(self.dataset.batches(), total=len(self.dataset) // self.dataset.batch_size):
                emb = self.model(batch, batch_ids)
                for paper_id, embedding in zip(batch_ids, emb.unbind()):
                    if type(paper_id) == tuple:
                        paper_id = paper_id[0]
                    results[paper_id] = embedding.detach().cpu().numpy().tolist()
                del batch
                del emb
        except Exception as e:
            print(e)
        finally:
            if save_path:
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as fout:
                    for k, v in results.items():
                        fout.write(json.dumps({"doc_id": k, "embedding": v}) + '\n')
        return results

    @staticmethod
    def load_embeddings_from_jsonl(embeddings_path: str) -> Dict[str, np.ndarray]:
        embeddings = {}
        with open(embeddings_path, 'r') as f:
            for line in tqdm(f, desc=f'reading embeddings from {embeddings_path}'):
                line_json = json.loads(line)
                embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
        return embeddings
