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


class ParquetBinaryDataset(IRDataset):
    """Dataset loader for flat Parquet files with binary (positive/negative) labels.

    Each row has: task_id (query ID), query (query text), example_id (candidate ID),
    quote_text (candidate text), label ("positive" or "negative").

    Rows are grouped by task_id to reconstruct per-query candidate lists.
    """

    def __init__(self, data_path: str, sep_token: str, batch_size=32, fields=None, key=None, processing_fn=None, max_samples: int = None):
        # Skip SimpleDataset.__init__ — we build queries/candidates directly from Parquet rows
        self.batch_size = batch_size
        self.sep_token = sep_token
        self.seen_ids = set()
        self.key = key
        self.fields = ["query", "quote"]

        logger.info(f"Loading Parquet binary dataset from {data_path}")
        import s3fs, pandas as pd
        fs = s3fs.S3FileSystem(anon=False)
        files = fs.glob(data_path.replace("s3://", ""))
        if not files:
            # Fall back to flat layout: <dir>/*_val.parquet (or any .parquet in parent dir)
            parent = data_path.replace("s3://", "").rsplit("/split=", 1)[0]
            files = fs.glob(f"{parent}/*_val.parquet") or fs.glob(f"{parent}/*.parquet")
            if files:
                logger.info(f"No files at {data_path}, falling back to flat layout at {parent}/")
            else:
                raise FileNotFoundError(f"No Parquet files found at {data_path} or flat layout under {parent}/")
        raw = datasets.Dataset.from_pandas(pd.concat([pd.read_parquet(fs.open(f)) for f in files], ignore_index=True))
        logger.info(f"Loaded {len(raw)} rows")

        # Group rows by task_id to build per-query candidate lists
        seen_queries = {}  # task_id -> query text (first seen)
        seen_candidates = set()  # example_ids already added

        self.queries = []
        self.candidates = []
        self.qrels = {}

        for row in raw:
            task_id = str(row["task_id"])
            example_id = str(row["example_id"])

            if task_id not in seen_queries:
                if max_samples is not None and len(seen_queries) >= max_samples:
                    continue
                seen_queries[task_id] = row.get("canonical_query", row["query"])
                self.queries.append({"query": row.get("canonical_query", row["query"]), "doc_id": task_id})
                self.qrels[task_id] = {}

            if task_id in seen_queries:
                self.qrels[task_id][example_id] = 1 if row["label"] == "positive" else 0
                if example_id not in seen_candidates:
                    seen_candidates.add(example_id)
                    self.candidates.append({"quote": row["quote_text"], "doc_id": example_id})

        logger.info(f"Built {len(self.queries)} queries and {len(self.candidates)} candidates")

    def __len__(self):
        return len(self.queries) + len(self.candidates)
