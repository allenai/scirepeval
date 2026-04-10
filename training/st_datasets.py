import itertools
import random
import warnings

import datasets

from tasks import TaskFamily


def _load_split(task: TaskFamily, split: str) -> datasets.Dataset:
    hf_split = "validation" if split == "dev" else "train"
    if task.data_files:
        return datasets.load_dataset("json", data_files={hf_split: task.data_files[split]})[hf_split]
    return datasets.load_dataset(**task.dataset, split=hf_split)


def _make_text_fn(task: TaskFamily):
    sep = "\n\n"
    fields = task.input_fields

    def _text(doc):
        if isinstance(doc, dict):
            parts = [str(doc[f]) for f in fields if doc.get(f)]
        else:
            parts = [doc]
        return sep.join(parts)

    return _text


def build_st_dataset(
    task: TaskFamily,
    split: str,
    num_negatives: int = 1,
    num_positives: int = 2,
    queries_per_dataset: int | None = 25000,
) -> datasets.Dataset:
    """Load a triplet/IR task dataset and return an HF Dataset with anchor/positive/negative_1/.../negative_K columns.

    Sampling is query-first for consistency across all K values:
      1. Sample min(queries_per_dataset, n_queries) unique queries.
      2. For each query, sample min(num_positives, n_pos) positives.
      3. For each positive, emit one row with the same K negatives drawn from the query's negative pool.
    """
    data = _load_split(task, split)
    _text = _make_text_fn(task)

    def _neg_cols(neg_texts: list[str]) -> dict:
        if len(neg_texts) == 1:
            return {"negative": neg_texts[0]}
        return {f"negative_{i+1}": t for i, t in enumerate(neg_texts)}

    groups: list[dict] = []
    if task.type == "triplet":
        query_map: dict[str, dict] = {}
        for ex in data:
            q = _text(ex["query"])
            if q not in query_map:
                query_map[q] = {"query": q, "positives": [], "negatives": []}
            query_map[q]["positives"].append(_text(ex["pos"]))
            query_map[q]["negatives"].append(_text(ex["neg"]))
        groups = list(query_map.values())
    else:
        for ex in data:
            candidates = ex["candidates"]
            pos_texts = [_text(c) for c in candidates if c["score"]]
            neg_texts = [_text(c) for c in candidates if not c["score"]]
            if not pos_texts or not neg_texts:
                continue
            groups.append({"query": _text(ex["query"]), "positives": pos_texts, "negatives": neg_texts})

    n_queries = len(groups)
    if queries_per_dataset is None or queries_per_dataset >= n_queries:
        if queries_per_dataset is not None and queries_per_dataset > n_queries:
            warnings.warn(
                f"queries_per_dataset={queries_per_dataset} exceeds available queries ({n_queries}) "
                f"for task '{task.name}' split='{split}'. Using all {n_queries} queries.",
                stacklevel=2,
            )
        sampled = groups
    else:
        sampled = random.sample(groups, queries_per_dataset)

    rows = []
    for g in sampled:
        pos_pool = g["positives"]
        neg_pool = g["negatives"]
        n_pos = min(num_positives, len(pos_pool))
        chosen_pos = random.sample(pos_pool, n_pos)
        chosen_neg = random.sample(neg_pool, num_negatives) if len(neg_pool) >= num_negatives else random.choices(neg_pool, k=num_negatives)
        for pos_text in chosen_pos:
            row = {"anchor": g["query"], "positive": pos_text}
            row.update(_neg_cols(chosen_neg))
            rows.append(row)

    return datasets.Dataset.from_list(rows)


def build_triplet_eval_dataset(task: TaskFamily, max_samples: int | None = None) -> datasets.Dataset:
    """Load a triplet task's dev split as flat (anchor, positive, negative) rows.

    Preserves the original per-row pairings — no random re-pairing.
    """
    data = _load_split(task, "dev")
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    _text = _make_text_fn(task)
    return datasets.Dataset.from_dict({
        "anchor":   [_text(ex["query"]) for ex in data],
        "positive": [_text(ex["pos"])   for ex in data],
        "negative": [_text(ex["neg"])   for ex in data],
    })


def build_ir_infonce_eval_dataset(
    task: TaskFamily, max_samples: int | None = None, max_negs_per_query: int = 5
) -> datasets.Dataset:
    """Flatten an IR task's dev split into (anchor, positive, negative) triplets, mirroring PL's IRDataset.preprocess.

    For each query: take up to max_negs_per_query negatives, cycle positives to match, yield one row per neg.
    max_samples caps the number of output rows (not queries).
    """
    data = _load_split(task, "dev")
    _text = _make_text_fn(task)

    anchors, positives, negatives = [], [], []
    for ex in data:
        candidates = ex["candidates"]
        pos_texts = [_text(c) for c in candidates if c["score"]]
        neg_texts  = [_text(c) for c in candidates if not c["score"]]
        if not pos_texts or not neg_texts:
            continue
        num_trips = min(max_negs_per_query, len(neg_texts))
        cyc_pos = list(itertools.islice(itertools.cycle(pos_texts), num_trips))
        for pos, neg in zip(cyc_pos, neg_texts[:num_trips]):
            anchors.append(_text(ex["query"]))
            positives.append(pos)
            negatives.append(neg)
            if max_samples is not None and len(anchors) >= max_samples:
                return datasets.Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})

    return datasets.Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


def build_ir_eval_data(
    task: TaskFamily, max_samples: int | None = None
) -> tuple[dict, dict, dict]:
    """Load an IR task's dev split and return (queries, corpus, relevant_docs) for InformationRetrievalEvaluator."""
    data = _load_split(task, "dev")
    _text = _make_text_fn(task)

    queries, corpus, relevant_docs = {}, {}, {}
    text_to_did: dict[str, str] = {}
    for ex in data:
        candidates = ex["candidates"]
        pos_texts = [_text(c) for c in candidates if c["score"]]
        neg_texts = [_text(c) for c in candidates if not c["score"]]
        if not pos_texts or not neg_texts:
            continue
        qid = str(len(queries))
        queries[qid] = _text(ex["query"])
        relevant_docs[qid] = set()
        for text in pos_texts + neg_texts:
            if text not in text_to_did:
                text_to_did[text] = str(len(corpus))
                corpus[text_to_did[text]] = text
        for text in pos_texts:
            relevant_docs[qid].add(text_to_did[text])
        if max_samples is not None and len(queries) >= max_samples:
            break

    return queries, corpus, relevant_docs
