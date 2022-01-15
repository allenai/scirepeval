from typing import Iterator, Tuple, List, Dict, Union

from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer
import ijson
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from abc import ABC, abstractmethod
import itertools


class AbstractMultiTaskDataset(ABC, IterableDataset):
    def __init__(self, task_name: str, json_file: str, tokenizer: PreTrainedTokenizer, fields: List[str],
                 sample_size, ctrl_token: str, max_len: int):
        self.task_name = task_name
        self.data_file = json_file
        self.tokenizer = tokenizer
        self.fields = fields
        self.sample_size = sample_size
        self.ctrl_token = ctrl_token
        self.max_len = max_len

    @abstractmethod
    def sub_sample(self, json_parse: List[Dict]) -> Iterator:
        pass

    @abstractmethod
    def preprocess(self, line: Dict[str, str]) -> Union[
        Tuple[str, BatchEncoding, int], List[Tuple[str, List[BatchEncoding]]]]:
        pass

    def __iter__(self) -> Iterator[T_co]:
        # data is assumed to be a json file
        file_iter = open(self.data_file, "rb")
        json_parse = ijson.items(file_iter, 'item')
        if self.sample_size == -1:
            map_itr = map(self.preprocess, json_parse)
        else:
            map_itr = map(self.preprocess, self.sub_sample(list(json_parse)))
        return map_itr

    def tokenized_input(self, input_map: Dict[str, str]) -> BatchEncoding:
        text = "" if not self.ctrl_token else self.ctrl_token
        for field in self.fields:
            text += " " + input_map[field]
        input_ids = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt",
                                   max_length=self.max_len)
        return input_ids


class ClassificationDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, json_file: str, tokenizer: PreTrainedTokenizer, fields: List[str],
                 label_field: str, labels: Dict[str, int], sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, json_file, tokenizer, fields, sample_size, ctrl_token, max_len)
        self.labels = labels
        self.label_field = label_field

    def preprocess(self, line: Dict[str, str]) -> Tuple[str, BatchEncoding, int]:
        # Splits the line into text and label and applies preprocessing to the text
        label = line[self.label_field]
        input_ids = self.tokenized_input(line)
        return self.task_name, input_ids["input_ids"].flatten(), self.labels[label]

    def sub_sample(self, json_parse: List[Dict]) -> Iterator:
        X_ids = np.array([d["corpus_id"] for d in json_parse])
        y = np.array([labels[d[self.label_field]] for d in json_parse])
        ids, _, _, _ = train_test_split(X_ids, y, train_size=self.sample_size, random_state=42,
                                        stratify=y)
        X = [d for d in json_parse if d["corpus_id"] in ids]
        return X


class MultiLabelClassificationDataset(ClassificationDataset):
    def __init__(self, task_name: str, json_file: str, tokenizer: PreTrainedTokenizer, fields: List[str],
                 label_field: str, labels: Dict[str, int], sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, json_file, tokenizer, fields, label_field, labels, sample_size, ctrl_token, max_len)
        self.labels = dict(sorted(labels.items()))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([list(self.labels.values())])

    def preprocess(self, line: Dict[str, str]) -> Tuple[str, BatchEncoding, int]:
        task, X, y = super().preprocess(line)
        return task, X, self.mlb.transform(y)

    def sub_sample(self, json_parse: List[Dict]) -> Iterator:
        X_ids = np.array([d["corpus_id"] for d in json_parse])
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform([tuple(d[self.label_field]) for d in json_parse])
        sub_sample_ratio = self.sample_size / len(json_parse)
        stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[sub_sample_ratio,
                                                                                                1 - sub_sample_ratio, ])
        indices, _ = next(stratifier.split(X_ids, y))
        ids = X_ids[indices]
        X = [d for d in json_parse if d["corpus_id"] in ids]
        return X


class TripletDataset(AbstractMultiTaskDataset):
    def __init__(self, task_name: str, json_file: str, tokenizer: PreTrainedTokenizer, fields: List[str],
                 sample_size=-1, ctrl_token: str = None, max_len: int = 512):
        super().__init__(task_name, json_file, tokenizer, fields, sample_size, ctrl_token, max_len)

    def preprocess(self, line: Dict[str, str]) -> List[Tuple[str, List[BatchEncoding]]]:
        # Splits the line into text and label and applies preprocessing to the text
        query, candidates = line["query"], line["candidates"]
        pos_candidates, neg_candidates = {c for c in candidates if c["score"]}, {c for c in candidates if
                                                                                 not c["score"]}
        tokenized_input_list = []
        tokenized_query = self.tokenized_input(query)
        for pos in pos_candidates:
            neg = None
            if neg_candidates:
                neg = neg_candidates.pop()
            if neg:
                tokenized_pos = self.tokenized_input(pos)
                tokenized_neg = self.tokenized_input(neg)
                tokenized_input_list.append((self.task_name, [tokenized_query, tokenized_pos, tokenized_neg]))
        return tokenized_input_list

    def sub_sample(self, json_parse: List[Dict]) -> Iterator:
        return json_parse[:self.sample_size//10]

    def __iter__(self):
        return itertools.chain(*super().__iter__())


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    with open("sample_data/mesh_descriptors.txt", "r") as f:
        labels = f.readlines()
    labels = {l.strip(): i for i, l in enumerate(labels)}
    dataset = ClassificationDataset(task_name="mesh", json_file="sample_data/mesh_small.json", tokenizer=tokenizer,
                                    fields=["title", "abstract"],
                                    label_field="descriptor", labels=labels, sample_size=100)
    dataloader = DataLoader(dataset, batch_size=32)
    for task, X, y in dataloader:
        print(len(X))
        print(len(y))
