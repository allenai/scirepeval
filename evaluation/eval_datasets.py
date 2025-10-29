import logging
import os
from typing import Union, List

import datasets

logger = logging.getLogger(__name__)


class SimpleDataset:

    def __init__(self, data_path: Union[str, tuple], sep_token: str, batch_size=32,
                 fields: List = None, key: str = None, processing_fn=None):
        self.batch_size = batch_size
        self.sep_token = sep_token
        if not fields:
            fields = ["title", "abstract"]
        self.fields = fields
        logger.info(f"Loading test metadata from {data_path}")
        if not processing_fn:
            if type(data_path) == str and os.path.isfile(data_path):
                self.data = datasets.load_dataset("json", data_files={"test": data_path})["test"]
            else:
                self.data = datasets.load_dataset(data_path[0], data_path[1], split="evaluation", trust_remote_code=False)
        else:
            self.data = processing_fn(data_path)
        logger.info(f"Loaded {len(self.data)} documents")
        self.seen_ids = set()
        self.key = key

    def __len__(self):
        return len(self.data)

    def batches(self):
        return self.process_batches(self.data)

    def process_batches(self, data: Union[datasets.Dataset, List]):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        key = "doc_id" if not self.key else self.key
        for d in data:
            if key in d and d[key] not in self.seen_ids:
                bid = d[key]
                self.seen_ids.add(bid)
                text = []
                for field in self.fields:
                    if d.get(field):
                        text.append(str(d[field]))
                text = (f" {self.sep_token} ".join(text)).strip()
                if (i) % batch_size != 0 or i == 0:
                    batch_ids.append(bid)
                    batch.append(text)
                else:
                    yield batch, batch_ids
                    batch_ids = [bid]
                    batch = [text]
                i += 1
        if len(batch) > 0:
            yield batch, batch_ids


class IRDataset(SimpleDataset):
    def __init__(self, data_path, sep_token, batch_size=32, fields=None, key=None, processing_fn=None):
        super().__init__(data_path, sep_token, batch_size, fields, key, processing_fn)
        self.queries, self.candidates = [], []
        for d in self.data:
            if type(d["query"]) == str:
                self.queries.append({"title": d["query"], "doc_id": d["doc_id"]})
            else:
                self.queries.append(d["query"])
            self.candidates += (d["candidates"])

    def __len__(self):
        return len(self.queries) + len(self.candidates)

    def batches(self):
        query_gen = self.process_batches(self.queries)
        cand_gen = self.process_batches(self.candidates)
        for q, q_ids in query_gen:
            q_ids = [(v, "q") for v in q_ids]
            yield q, q_ids
        for c, c_ids in cand_gen:
            c_ids = [(v, "c") for v in c_ids]
            yield c, c_ids
