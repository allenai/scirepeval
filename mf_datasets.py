import decimal
from typing import Iterator, Tuple, List, Dict, Union, Any, Iterable
import torch
from torch.utils.data import IterableDataset, DataLoader, ChainDataset, get_worker_info
from torch.utils.data.dataset import T_co, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer
import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from abc import ABC, abstractmethod
import itertools
from torch.utils.data._utils.collate import default_collate
from collections import defaultdict
from strategies import BatchingStrategy
import random


class AbstractMultiTaskDataset(ABC, IterableDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 sample_size, ctrl_token: str, max_len: int):
        self.task_name = task_name
        self.data_src = data_src
        self.tokenizer = tokenizer
        self.fields = fields
        self.sample_size = sample_size
        self.ctrl_token = ctrl_token
        self.max_len = max_len
        self._effective_sample_size = sample_size

    def sub_sample(self, json_parse: Iterator[Dict]) -> Iterator:
        curr_len = 0
        try:
            for _ in range(self.effective_sample_size):
                curr_len += 1
                yield next(json_parse)
        except StopIteration:
            print(
                f"Reqd sample size {self.effective_sample_size} greater than {self.task_name} dataset size {curr_len}, using complete dataset")

    @abstractmethod
    def preprocess(self, line: Dict[str, str]) -> Union[
        Tuple[str, BatchEncoding, torch.Tensor], List[Tuple[str, List[BatchEncoding]]]]:
        pass

    def postprocess_iter(self, curr_iter):
        return curr_iter

    @property
    def effective_sample_size(self):
        return self._effective_sample_size

    @effective_sample_size.setter
    def effective_sample_size(self, val):
        self._effective_sample_size = val

    def __iter__(self) -> Iterator[T_co]:
        # data is assumed to be a json file
        # try:
        #     file_iter = open(self.data_src, "rb")
        #     json_parse = ijson.items(file_iter, 'item')
        #     peek = next(json_parse)
        #     json_parse = itertools.chain([peek], json_parse)
        # except:
        #     file_iter = open(self.data_src, "rb")
        #     json_parse = ijson.items(file_iter, '', multiple_values=True)
        if type(self.data_src) == dict:
            json_parse = datasets.load_dataset("json", data_files=self.data_src, streaming=True)["train"]
        else:
            json_parse = datasets.load_dataset(self.data_src[0], self.task_name, split=self.data_src[1], streaming=True)
        if self.sample_size == -1:
            map_itr = map(self.preprocess, json_parse)
        else:
            map_itr = map(self.preprocess, self.sub_sample(json_parse))
        return self.postprocess_iter(map_itr)

    def tokenized_input(self, input_data: Union[Dict[str, str], str], ctrl_token_key: str = None) -> BatchEncoding:
        text = []
        if type(input_data) == dict:
            for field in self.fields:
                if input_data[field]:
                    if type(input_data[field]) == decimal.Decimal:
                        input_data[field] = str(int(input_data[field]))
                    text.append(input_data[field])
            text = (f" {self.tokenizer.sep_token} ".join(text)).strip()
        else:
            text = input_data
        if self.ctrl_token:
            ctrl_token = self.ctrl_token if not ctrl_token_key else self.ctrl_token[ctrl_token_key]
            text = ctrl_token + " " + text
        input_ids = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt",
                                   max_length=self.max_len)
        # if self.ctrl_token:
        #     input_ids["input_ids"] = input_ids["input_ids"][:,1:]
        #     input_ids["attention_mask"] = input_ids["attention_mask"][:,1:]
        return {"input_ids": input_ids["input_ids"].flatten(), "attention_mask": input_ids["attention_mask"].flatten()}


class ClassificationDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 label_field: str, labels: Dict[str, int], sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, data_src, tokenizer, fields, sample_size, ctrl_token, max_len)
        self.labels = labels
        self.label_field = label_field

    def label_transform(self, label_raw: str) -> Union[int, np.ndarray]:
        return self.labels[label_raw]

    def preprocess(self, line: Dict[str, str]) -> Tuple[str, BatchEncoding, int]:
        # Splits the line into text and label and applies preprocessing to the text
        label = line[self.label_field]
        input_ids = self.tokenized_input(line)
        return self.task_name, input_ids, self.label_transform(label)

    def sub_sample(self, json_parse: Iterator[Dict]) -> Iterator:
        # json_itr_list = itertools.tee(json_parse, 2)
        # json_parse = json_itr_list[0]
        X, y = zip(*[(d, self.labels[d[self.label_field]]) for d in json_parse])
        X, y = np.array(X), np.array(y)
        if X.shape[0] < self.effective_sample_size:
            print(
                f"Reqd sample size {self.effective_sample_size} greater than {self.task_name} dataset size {X.shape[0]}, using complete dataset")
            X_sub = X
        else:
            X_sub, _, _, _ = train_test_split(X, y, train_size=self.effective_sample_size, random_state=42, stratify=y)
        for d in X_sub:
            yield d


class MultiLabelClassificationDataset(ClassificationDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 label_field: str, labels: Dict[str, int], sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, data_src, tokenizer, fields, label_field, labels, sample_size, ctrl_token, max_len)
        self.labels = dict(sorted(labels.items()))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(self.labels.keys())])

    def label_transform(self, label_raw: List[str]) -> Union[int, np.ndarray]:
        return self.mlb.transform([label_raw]).flatten().astype(float)

    def sub_sample(self, json_parse: Iterator[Dict]) -> Iterator:
        X, y = zip(*[(d, tuple(d[self.label_field])) for d in json_parse])
        X, y = np.array(X), self.mlb.transform(y)
        if X.shape[0] < self.effective_sample_size:
            print(
                f"Reqd sample size {self.effective_sample_size} greater than {self.task_name} dataset size {X.shape[0]}, using complete dataset")
            X_sub = X
        else:
            sub_sample_ratio = self.effective_sample_size / X.shape[0]
            stratifier = IterativeStratification(n_splits=2, order=1,
                                                 sample_distribution_per_fold=[sub_sample_ratio,
                                                                               1 - sub_sample_ratio, ])
            _, indices = next(stratifier.split(X, y))
            X_sub = X[indices]
        for d in X_sub:
            yield d


class IRDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, data_src, tokenizer, fields, sample_size, ctrl_token, max_len)
        self.effective_sample_size //= 5

    def preprocess(self, line: Dict[str, str]) -> List[Tuple[str, List[BatchEncoding]]]:
        # Splits the line into text and label and applies preprocessing to the text
        query, candidates = line["query"], line["candidates"]
        pos_candidates, neg_candidates = [c for c in candidates if c["score"]], [c for c in candidates if
                                                                                 not c["score"]]
        num_trips = min(5, len(neg_candidates))
        new_pos_candidates = pos_candidates.copy()
        if pos_candidates:
            pos_candidates = itertools.cycle(pos_candidates)
            while len(new_pos_candidates) < num_trips:
                new_pos_candidates.append(next(pos_candidates))
        query_ctrl_key, cand_ctrl_key = None, None
        if type(self.ctrl_token) == dict:
            query_ctrl_key = "query"
            cand_ctrl_key = "candidates"
        tokenized_query = self.tokenized_input(query, query_ctrl_key)

        for pos in new_pos_candidates[:num_trips]:
            neg = neg_candidates.pop()
            tokenized_pos = self.tokenized_input(pos, cand_ctrl_key)
            tokenized_neg = self.tokenized_input(neg, cand_ctrl_key)
            yield (self.task_name, [tokenized_query, tokenized_pos, tokenized_neg])

    def postprocess_iter(self, curr_iter):
        # chained_iter = itertools.chain(*curr_iter)
        batched_list = []
        try:
            while True:
                while len(batched_list) < 1000:
                    batched_list += next(curr_iter)
                random.shuffle(batched_list)
                for x in batched_list:
                    yield x
                batched_list.clear()
        except StopIteration:
            random.shuffle(batched_list)
            for x in batched_list:
                yield x


class TripletDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, data_src, tokenizer, fields, sample_size, ctrl_token, max_len)

    def preprocess(self, line: Dict[str, str]) -> Union[
        Tuple[str, BatchEncoding, torch.Tensor], List[Tuple[str, List[BatchEncoding]]]]:
        triplet = []
        for key in ("query", "pos", "neg"):
            triplet.append(self.tokenized_input(line[key]))
        return self.task_name, triplet


class CustomChainDataset(ChainDataset):
    def __init__(self, datasets: Iterable[Dataset], batch_size, device_rank=0, num_devices=1,
                 batching_strategy=BatchingStrategy.SEQUENTIAL):
        super().__init__(datasets)
        self.batch_size = batch_size
        self.batching = batching_strategy
        self.device_rank = device_rank
        self.num_devices = num_devices
        self.effective_batch_size = batch_size * num_devices

    def iter_slice(self, curr_iter, worker_info):
        curr_batch, idx = dict(), 0
        try:
            while True:
                for _ in range(self.effective_batch_size):
                    curr_batch[idx] = next(curr_iter)
                    idx += 1
                for i, x in curr_batch.items():
                    if (i // self.batch_size) % self.num_devices == self.device_rank:
                        if (i // self.effective_batch_size) % worker_info.num_workers == worker_info.id:
                            yield x
                curr_batch.clear()
        except StopIteration:
            curr_batch.clear()

    def __iter__(self):
        batch_itr = self.batching.value.get_batch_iter(self.datasets, self.effective_batch_size)
        worker_info = get_worker_info()
        if worker_info:
            batch_itr = self.iter_slice(batch_itr, worker_info)

        return batch_itr


class RegressionDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, data_src: Union[Dict[str,str],Tuple[str, str]], tokenizer: PreTrainedTokenizer, fields: List[str],
                 label_field: str, sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, data_src, tokenizer, fields, sample_size, ctrl_token, max_len)
        self.label_field = label_field

    def preprocess(self, line: Dict[str, str]) -> Tuple[str, Dict[str, BatchEncoding], Union[int, float]]:
        # Splits the line into text and label and applies preprocessing to the text
        label = np.float32(line[self.label_field])
        input_ids = self.tokenized_input(line)
        return self.task_name, input_ids, label


def multi_collate(batch: List[Any]) -> Dict[str, List[Any]]:
    task_sub_batch = defaultdict(list)
    for b in batch:
        task_sub_batch[b[0]].append(b[1:])
    return {task: default_collate(sub_batch) for task, sub_batch in task_sub_batch.items()}


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    tokenizer.add_special_tokens({'additional_special_tokens': ["[CLF]"]})
    with open("sample_data/mesh_descriptors.txt", "r") as f:
        labels = f.readlines()
    labels = {l.strip(): i for i, l in enumerate(labels)}
    cls_dataset = ClassificationDataset(task_name="mesh", data_src="../../scidocs/data/mesh_plus/train.json",
                                        tokenizer=tokenizer,
                                        fields=["title", "abstract"],
                                        label_field="descriptor", labels=labels, sample_size=400000)
    trip_dataset = IRDataset(task_name="s2and", data_src="sample_data/s2and_small.json",
                             tokenizer=tokenizer,
                             fields=["title", "abstract"], sample_size=400000)
    specter_dataset = TripletDataset(task_name="specter", data_src="../../scidocs/data/specter_triplets/train.json",
                                     tokenizer=tokenizer,
                                     fields=["title", "abstract"], sample_size=400000)
    search_dataset = IRDataset(task_name="search", data_src="sample_data/search_small.jsonl",
                               tokenizer=tokenizer,
                               fields=["title", "abstract", "venue", "year"], sample_size=100)
    with open("sample_data/fos_labels.txt", "r") as f:
        mlc_labels = f.readlines()
    mlc_labels = {l.strip(): i for i, l in enumerate(mlc_labels)}

    ml_cls_dataset = MultiLabelClassificationDataset(task_name="fos", data_src="sample_data/fos_small.json",
                                                     tokenizer=tokenizer,
                                                     fields=["title", "abstract"],
                                                     label_field="labels_text", labels=mlc_labels, sample_size=100,
                                                     ctrl_token="[CLF]")

    batch_size = 16
    multi_dataset = CustomChainDataset([ml_cls_dataset], batch_size=batch_size,
                                       batching_strategy=BatchingStrategy.MIXED_PROPORTIONAL)
    dataloader = DataLoader(multi_dataset, batch_size=batch_size, collate_fn=multi_collate, num_workers=4)
    for i, data in enumerate(dataloader):
        print(i)
        for task, batch in data.items():
            d = batch[-1][-1] if task in ("s2and", "specter", "search") else batch[-1]
            print(task, d.shape[0])
            print(batch)
