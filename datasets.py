from typing import Iterator, Tuple, List, Dict

from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer, BatchEncoding, AutoTokenizer
import ijson


class ClassificationDataset(IterableDataset):
    def __init__(self, json_file: str, tokenizer: PreTrainedTokenizer, fields: List[str], label_field: str,
                 labels: Dict[str, int], ctrl_token: str = None, max_len: int = 512):
        self.data_file = json_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fields = fields
        self.labels = labels
        self.label_field = label_field
        self.ctrl_token = ctrl_token

    def preprocess(self, line: Dict[str, str]) -> Tuple[BatchEncoding, int]:
        # Splits the line into text and label and applies preprocessing to the text
        text = "" if not self.ctrl_token else self.ctrl_token
        for field in self.fields:
            text += " " + line[field]
        label = line[self.label_field]
        input_ids = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt",
                                   max_length=self.max_len)
        return input_ids["input_ids"].flatten(), self.labels[label]

    def __iter__(self) -> Iterator[T_co]:
        # data is assumed to be a json file
        file_iter = open(self.data_file, "rb")
        json_parse = ijson.items(file_iter, 'item')
        map_itr = map(self.preprocess, json_parse)
        return map_itr


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
    with open("sample_data/mesh_descriptors.txt", "r") as f:
        labels = f.readlines()
    labels = {l.strip(): i for i, l in enumerate(labels)}
    dataset = ClassificationDataset(json_file="sample_data/mesh_small.json", tokenizer=tokenizer, fields=["title", "abstract"],
                                    label_field="descriptor", labels=labels)
    dataloader = DataLoader(dataset, batch_size=32)
    for X, y in dataloader:
        print(len(X))
        print(len(y))
