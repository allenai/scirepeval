from typing import Dict, Union, List

from transformers import AutoModel, AutoTokenizer
import os
from bert_pals import BertPalsEncoder, BertPalConfig, BertModel
from adapter_fusion import AdapterEncoder, AdapterFusion
import torch
import logging

logger = logging.getLogger(__name__)


class EncoderFactory:
    def __init__(self, base_checkpoint: str = None, adapters_load_from: Union[str, Dict] = None,
                 fusion_load_from: str = None, all_tasks: list = None):
        self.base_checkpoint = f"{base_checkpoint}/model" if os.path.isdir(base_checkpoint) else base_checkpoint
        self.all_tasks = all_tasks
        self.adapters_load_from = f"{adapters_load_from}/model/adapters" if (type(
            adapters_load_from) == str and os.path.isdir(
            adapters_load_from)) else adapters_load_from
        self.fusion_load_from = f"{fusion_load_from}/model"

    def get_encoder(self, variant: str):
        if variant == "default":
            return AutoModel.from_pretrained(self.base_checkpoint)
        elif variant == "pals":
            # needs all task names and a local checkpoint path
            if os.path.isdir(self.base_checkpoint):
                return BertPalsEncoder(config=f"{self.base_checkpoint}/config.json", task_ids=self.all_tasks,
                                       checkpoint=f"{self.base_checkpoint}/pytorch_model.bin")
            else:
                pals_config = BertPalConfig.from_pretrained(self.base_checkpoint)
                pals_model = BertModel.from_pretrained(self.base_checkpoint)
                return BertPalsEncoder(config=pals_config, task_ids=self.all_tasks,
                                       checkpoint=pals_model)
        elif variant == "adapters":
            # needs a base model checkpoint and the adapters to be loaded from local path or dict of (task_id,
            # adapter) from adapters hub
            return AdapterEncoder(self.base_checkpoint, self.all_tasks, load_as=self.adapters_load_from)
        elif variant == "fusion":
            # needs a base model and list of adapters/local adapter checkpoint paths to be fused
            return AdapterFusion(self.base_checkpoint, self.all_tasks, load_adapters_as=self.adapters_load_from,
                                 fusion_dir=self.fusion_load_from, inference=True)
        else:
            raise ValueError("Unknown encoder type: {}".format(variant))


class Model:
    def __init__(self, variant: str = "default", base_checkpoint: str = None,
                 adapters_load_from: Union[str, Dict] = None, fusion_load_from: str = None,
                 use_ctrl_codes: bool = False, task_id: Union[str, Dict] = None,
                 all_tasks: list = None, hidden_dim: int = 768, max_len: int = 512, use_fp16=False):
        self.variant = variant
        self.encoder = EncoderFactory(base_checkpoint, adapters_load_from, fusion_load_from, all_tasks).get_encoder(
            variant)
        if torch.cuda.is_available():
            self.encoder.to('cuda')
        self.encoder.eval()
        tokenizer_checkpoint = f"{base_checkpoint}/tokenizer" if os.path.isdir(base_checkpoint) else base_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.use_ctrl_codes = use_ctrl_codes
        self.reqd_token_idx = 0 if not use_ctrl_codes else 1
        self._task_id = task_id
        if self._task_id:
            if use_ctrl_codes:
                logger.info(f"Control code used: {self._task_id}")
            elif variant != "default":
                logger.info(f"Task id used: {self._task_id}")

        self.hidden_dim = hidden_dim
        self.max_length = max_len
        self.use_fp16 = use_fp16

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        if self.use_ctrl_codes:
            logger.info(f"Control code used: {value}")
        elif self.variant != "default":
            logger.info(f"Task id used: {value}")
        self._task_id = value

    def __call__(self, batch, batch_ids=None):
        def append_ctrl_code(batch, batch_ids):
            if type(self._task_id) == dict:
                batch = [f"{self.task_id['query']} {text}" if bid[1] == "q" else f"{self.task_id['candidates']} {text}"
                         for text, bid in zip(batch, batch_ids)]
            else:
                batch = [f"{self.task_id} {text}" for text in batch]
            return batch

        batch = [batch] if type(batch) == str else batch
        batch_ids = [] if not batch_ids else batch_ids
        if self.use_ctrl_codes:
            batch = append_ctrl_code(batch, batch_ids)
        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=self.max_length)
        if torch.cuda.is_available():
            input_ids.to('cuda')
        if self.variant == "default":
            output = self.encoder(**input_ids)
        elif type(self._task_id) != dict:
            output = self.encoder(task_id=self._task_id, **input_ids)
        else:
            x = input_ids["input_ids"]
            output = torch.zeros(x.shape[0], x.shape[1], self.hidden_dim).to("cuda")
            q_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "q"])
            c_idx = torch.tensor([i for i, b in enumerate(batch_ids) if b[1] == "c"])

            if not q_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["candidates"], **input_ids)
            elif not c_idx.shape[0]:
                output = self.encoder(task_id=self._task_id["query"], **input_ids)
            else:
                for i, v in enumerate(sorted(self._task_id.values())):
                    curr_input_idx = q_idx if v == "[QRY]" else c_idx
                    curr_input = x[curr_input_idx]
                    curr_output = self.encoder(task_id=v, input_ids=curr_input,
                                               attention_mask=input_ids["attention_mask"][curr_input_idx])
                    try:
                        output[curr_input_idx] = curr_output  # adapters
                    except:
                        output[curr_input_idx] = curr_output.last_hidden_state  # pals
        try:
            embedding = output.last_hidden_state[:, self.reqd_token_idx, :]  # cls token
        except:
            embedding = output[:, self.reqd_token_idx, :]  # cls token
        return embedding.half() if self.use_fp16 else embedding
