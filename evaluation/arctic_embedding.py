import torch
from transformers import AutoModel, AutoTokenizer

instr_format = "Represent this sentence for searching relevant passages: "


class ArcticModel:
    def __init__(self, embed_model: str):
        self.encoder = AutoModel.from_pretrained(embed_model, add_pooling_layer=False)
        self.encoder.eval()
        self.task_id = None
        self.retrieval_tasks = {"[PRX]", "[SRCH]"}
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.tokenizer.sep_token = self.tokenizer.eos_token

    def __call__(self, batch, batch_ids=None):
        if type(self.task_id) == dict or self.task_id not in self.retrieval_tasks:
            batch = [f"{instr_format if [b[1]] == 'q' else ''}{batch[i]}" for i, b in enumerate(batch_ids)]
        inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            batch_embed = self.encoder(**inputs).last_hidden_state[:, 0]
            batch_embed =  torch.nn.functional.normalize(batch_embed, p=2, dim=1)
        return batch_embed
