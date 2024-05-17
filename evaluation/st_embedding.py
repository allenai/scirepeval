import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class Sentence_Transformer_Model:
    
    def __init__(self, embed_model: str):
        self.encoder = SentenceTransformer(embed_model)
        self.encoder.eval()
        self.task_id = None
        self.retrieval_tasks = {"[PRX]", "[SRCH]"}
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.tokenizer.sep_token = self.tokenizer.eos_token
       
    def __call__(self, batch, batch_ids=None):
        embeddings = self.encoder.encode(batch)
        return embeddings