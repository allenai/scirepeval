from typing import List, Optional, Union, Dict
from transformers.adapters import PfeifferConfig
from transformers.adapters import AutoAdapterModel
from transformers.adapters.composition import Fuse
from abc import ABC, abstractmethod
import torch
import os


class AdapterFactory:
    @staticmethod
    def get_adapter(checkpoint_name: str, task_ids: List[str], fuse_adapters: bool,
                    adapters_dir: Union[str, Dict] = None):
        print(task_ids)
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
    def __init__(self, checkpoint_name, task_ids: List[str], load_as=None):
        super(AdapterEncoder, self).__init__(checkpoint_name)
        # Add a new adapter
        for t_id in task_ids:
            if not load_as:
                self.model.add_adapter(t_id, config="pfeiffer")
            else:
                # load_as can str for a local path or dict to be loaded from adapters hub
                if type(load_as) == str:
                    self.model.load_adapter(f"{load_as}/{t_id}/", load_as=t_id)
                else:
                    self.model.load_adapter(load_as[t_id], load_as=t_id)
        self.model.train_adapter(adapter_setup=task_ids, train_embeddings=False)

    def forward(self, input_ids, attention_mask, task_id):
        self.model.base_model.set_active_adapters(task_id)
        return self.model(input_ids, attention_mask=attention_mask)

    def save_pretrained(self, save_path: str, adapter_names: List[str] = None):
        # self.model.save_pretrained(save_path)
        save_path = f'{save_path}/adapters/'
        os.makedirs(save_path, exist_ok=True)
        if not adapter_names:
            self.model.save_all_adapters(save_path)
        else:
            for a_name in adapter_names:
                self.model.save_adapter(f"{save_path}/{a_name}/", a_name)


class AdapterFusion(AbstractAdapter):
    def __init__(self, checkpoint_name, task_ids: List[str], load_adapters_as: Union[str, dict], fusion_dir: str = None,
                 inference=False):
        super(AdapterFusion, self).__init__(checkpoint_name)
        # Add a new adapter
        # load_adapters_as can str for a local path with adapters and fusion dirs or dict to be loaded from adapters hub,
        # the adapters hub version of single adapters should have the suffix _fusion
        if not fusion_dir:
            fusion_dir = load_adapters_as.replace("/adapters/", "") if inference and type(
                load_adapters_as) == str else None
        load_adapters_as = load_adapters_as.replace("fusion", "adapters") if type(
            load_adapters_as) == str else load_adapters_as
        for t_id in task_ids:
            if type(load_adapters_as) == str and os.path.isdir(load_adapters_as):
                self.model.load_adapter(f"{load_adapters_as}/{t_id}/", load_as=t_id)
            else:
                self.model.load_adapter(load_adapters_as[t_id], load_as=t_id)
        self.fusion_mods_dict = dict()
        for i, t_id in enumerate(task_ids):
            task_fuse = Fuse(*([t_id] + task_ids[:i] + task_ids[i + 1:]))
            self.fusion_mods_dict[t_id] = task_fuse
            if not inference:
                self.model.add_adapter_fusion(task_fuse)
            else:
                if fusion_dir:
                    self.model.load_adapter_fusion(f"{fusion_dir}/{t_id}_fusion/")
                else:
                    self.model.load_adapter_fusion(f"{load_adapters_as[t_id]}_fusion")
        self.model.train_adapter_fusion(list(self.fusion_mods_dict.values()))
        print(self.model.active_adapters)
        # self.model.get_input_embeddings().train()
        # self.model.train_adapter(adapter_setup=task_ids, train_embeddings=True)

    def forward(self, input_ids, attention_mask, task_id):
        self.model.base_model.set_active_adapters(self.fusion_mods_dict[task_id])
        return self.model(input_ids, attention_mask=attention_mask)

    def save_pretrained(self, save_path: str):
        # self.model.save_pretrained(save_path)
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        for t_id, t_fuse in self.fusion_mods_dict.items():
            self.model.save_adapter_fusion(f'{save_path}/{t_id}_fusion/', t_fuse)
