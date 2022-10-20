from typing import Dict, Union, List

from transformers import AutoModel, AutoTokenizer
import os
from bert_pals import BertPalsEncoder
from adapter_fusion import AdapterEncoder, AdapterFusion
import torch


class EncoderFactory:
    def __init__(self, base_checkpoint: str = None, adapters_load_from: Union[str, Dict] = None,
                 task_id: Union[str, Dict] = None, all_tasks: list = None):
        self.base_checkpoint = f"{base_checkpoint}/model" if os.path.isdir(base_checkpoint) else base_checkpoint
        self.all_tasks = all_tasks
        self.task_id = task_id
        self.adapters_load_from = f"{adapters_load_from}/model/adapters/" if adapters_load_from and os.path.isdir(
            adapters_load_from) else adapters_load_from

    def get_encoder(self, variant: str):
        if variant == "default":
            return AutoModel.from_pretrained(self.base_checkpoint)
        elif variant == "pals":
            # needs all task names and a local checkpoint path
            return BertPalsEncoder(config=f"{self.base_checkpoint}/config.json", task_ids=self.all_tasks,
                                   checkpoint=f"{self.base_checkpoint}/pytorch_model.bin")
        elif variant == "adapters":
            # needs a base model checkpoint and the adapters to be loaded from local path or dict of (task_id,
            # adapter) from adapters hub
            reqd_tasks = [self.task_id] if type(self.task_id) == str else list(self.task_id.values())
            return AdapterEncoder(self.base_checkpoint, reqd_tasks, load_as=self.adapters_load_from)
        elif variant == "fusion":
            # needs a base model and list of adapters/local adapter checkpoint paths to be fused
            return AdapterFusion(self.base_checkpoint, self.all_tasks, load_adapters_as=self.adapters_load_from,
                                 inference=True)
        else:
            raise ValueError("Unknown encoder type: {}".format(variant))


class Model:
    def __init__(self, variant: str = "default", base_checkpoint: str = None,
                 adapters_load_from: Union[str, Dict] = None,
                 use_ctrl_codes: bool = False, task_id: Union[str, Dict] = None,
                 all_tasks: list = None, hidden_dim: int = 768, max_len: int = 512):
        self.variant = variant
        self.encoder = EncoderFactory(base_checkpoint, adapters_load_from, task_id, all_tasks).get_encoder(variant)
        self.encoder.to('cuda')
        tokenizer_checkpoint = f"{base_checkpoint}/tokenizer" if os.path.isdir(base_checkpoint) else base_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.reqd_token_idx = 0 if not use_ctrl_codes else 1
        self.task_id = task_id
        self.hidden_dim = hidden_dim
        self.max_length = max_len

    def __call__(self, batch, batch_ids):
        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", max_length=self.max_length)
        input_ids.to('cuda')
        if self.variant == "default":
            output = self.encoder(**input_ids)
        elif type(self.task_id) != dict:
            output = self.encoder(task_id=self.task_id, **input_ids)
        else:
            x = input_ids["input_ids"]
            output = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to("cuda")
            q_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "q"])
            c_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "c"])

            if not q_idx.shape[0]:
                output = self.encoder(task_id=self.task_id["candidates"], **input_ids)
            else:
                for i, v in enumerate(sorted(self.task_id.values())):
                    curr_input_idx = q_idx if v == "[QRY]" else c_idx
                    curr_input = x[curr_input_idx]
                    curr_output = self.encoder(task_id=v, input_ids=curr_input,
                                               attention_mask=input_ids["attention_mask"][curr_input_idx])
                    try:
                        output[curr_input_idx] = curr_output  # adapters
                    except:
                        output[curr_input_idx] = curr_output.last_hidden_state  # pals
        try:
            return output.last_hidden_state[:, self.reqd_token_idx, :]  # cls token
        except:
            return output[:, self.reqd_token_idx, :]  # cls token
