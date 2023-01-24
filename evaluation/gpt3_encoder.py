import os
import openai
import torch
from transformers import GPT2TokenizerFast


class GPT3Model:
    def __init__(self, embed_model: str):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.embed_model = embed_model
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def __call__(self, batch, batch_ids=None):
        batch_embed = []
        for iptext in batch:
            try:
                response = openai.Embedding.create(
                    input=iptext,
                    model=self.embed_model
                )
                embeddings = response['data'][0]['embedding']
                batch_embed.append(embeddings)
            except:
                response = openai.Embedding.create(
                    input=" ".join(iptext.split(" ")[:450]),
                    model=self.embed_model
                )
                embeddings = response['data'][0]['embedding']
                batch_embed.append(embeddings)
        return torch.tensor(batch_embed)
