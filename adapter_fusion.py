from typing import List, Optional
from transformers.adapters import PfeifferConfig
from transformers.adapters import AutoAdapterModel
from transformers.adapters.composition import Fuse
from abc import ABC, abstractmethod
import torch
import os


class AdapterFactory:
    @staticmethod
    def get_adapter(checkpoint_name: str, task_ids: List[str], fuse_adapters: bool, adapters_dir: str = None):
        if not fuse_adapters:
            return AdapterEncoder(checkpoint_name, task_ids)
        else:
            return AdapterFusion(checkpoint_name, task_ids, adapters_dir)


class AbstractAdapter(torch.nn.Module, ABC):
    def __init__(self, checkpoint_name):
        super(AbstractAdapter, self).__init__()
        self.model = AutoAdapterModel.from_pretrained(checkpoint_name)  # checkpoint

    @abstractmethod
    def save_pretrained(self, save_path: str):
        self.model.save_all_adapters(save_path)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        return self.model.resize_token_embeddings(new_num_tokens)


class AdapterEncoder(AbstractAdapter):
    def __init__(self, checkpoint_name, task_ids: List[str]):
        super(AdapterEncoder, self).__init__(checkpoint_name)
        # Add a new adapter
        for t_id in task_ids:
            self.model.add_adapter(t_id, config="pfeiffer")
        self.model.train_adapter(adapter_setup=task_ids, train_embeddings=True)

    def forward(self, x, task_id):
        self.model.base_model.set_active_adapters(task_id)
        return self.model(x)

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_all_adapters(save_path)


class AdapterFusion(AbstractAdapter):
    def __init__(self, checkpoint_name, task_ids: List[str], adapters_dir: str):
        super(AdapterFusion, self).__init__(checkpoint_name)
        # Add a new adapter
        for t_id in task_ids:
            self.model.load_adapter(f"{adapters_dir}/{t_id}/", load_as=t_id)
        self.fusion_mods_dict = dict()
        for t_id in task_ids:
            task_fuse = Fuse(task_ids)
            self.fusion_mods_dict[t_id] = task_fuse
            self.model.add_adapter_fusion(Fuse(task_ids))
            self.model.train_adapter_fusion(task_fuse)
        self.model.get_input_embeddings().train()
        # self.model.train_adapter(adapter_setup=task_ids, train_embeddings=True)

    def forward(self, x, task_id):
        self.model.base_model.set_active_adapters(self.fusion_mods_dict[task_id])
        return self.model(x)

    def save_pretrained(self, save_path: str):
        for t_id in self.fusion_mods_dict:
            self.model.save_adapter_fusion(f'{save_path}/{t_id}_fusion/')
