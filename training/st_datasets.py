import itertools
import pickle
import random
import warnings

import datasets
import pickle5

from tasks import TaskFamily

# Cache for easy_ids to avoid recomputing on every dataset load
_easy_ids_cache: dict[tuple[str, str], set[str]] = {}  # (task_name, split) -> set of IDs


def _load_pkl(path: str):
    """Load a pickle file from a local path or s3:// URI using pickle5 for better streaming."""
    if path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        with fs.open(path, "rb") as f:
            return pickle5.load(f)
    with open(path, "rb") as f:
        return pickle5.load(f)


def _get_citation_negatives(entry: dict, neg_type: str) -> list:
    """Return the appropriate negative candidate list from a train_mined entry."""
    if neg_type == "hard":
        return entry.get("hard_negatives", [])
    if neg_type == "easy":
        return entry.get("easy_negatives", [])
    # "all": combine hard + easy
    return entry.get("hard_negatives", []) + entry.get("easy_negatives", [])


def _paper_text(paper_texts: dict, corpus_id: int) -> str | None:
    """Look up a paper by corpus_id and return 'title\n\nabstract', or None if missing."""
    doc = paper_texts.get(corpus_id)
    if not doc:
        return None
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    return "\n\n".join(parts) if parts else None


def build_citation_dataset(
    task: TaskFamily,
    split: str,
    num_negatives: int = 1,
    num_positives: int = 1,
    target_rows: int | None = 25000,
) -> datasets.Dataset:
    """Load the citation triplet pkl and return anchor/positive/negative rows.

    Sampling is index-based to avoid pre-resolving all 557K anchors upfront:
    randomly pick an anchor index, resolve its texts on demand, pick 1 pos and 1 neg.
    Duplicate (anchor, pos, neg) triples are retried. split/num_negatives/num_positives
    are accepted for API compatibility but ignored.
    """
    paper_texts = _load_pkl(task.paper_texts_path)
    train_data = _load_pkl(task.pkl_path)
    anchor_ids = list(train_data.keys())

    n_rows = len(anchor_ids) if target_rows is None else target_rows
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    max_attempts = n_rows * 10
    attempts = 0
    while len(rows) < n_rows and attempts < max_attempts:
        attempts += 1
        entry = train_data[random.choice(anchor_ids)]
        anchor_text = _paper_text(paper_texts, entry["anchor_id"])
        if not anchor_text:
            continue
        pos_candidates = entry.get("positives", [])
        neg_candidates = _get_citation_negatives(entry, task.neg_type)
        if not pos_candidates or not neg_candidates:
            continue
        pos_text = _paper_text(paper_texts, random.choice(pos_candidates)["corpus_id"])
        neg_text = _paper_text(paper_texts, random.choice(neg_candidates)["corpus_id"])
        if not pos_text or not neg_text:
            continue
        key = (anchor_text, pos_text, neg_text)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"anchor": anchor_text, "positive": pos_text, "negative": neg_text})

    if len(rows) < n_rows:
        warnings.warn(
            f"Citation dataset: only produced {len(rows)} unique rows (target {n_rows}). "
            f"Candidate pool may be too small for the requested size.",
            stacklevel=2,
        )

    return datasets.Dataset.from_list(rows)


def build_citation_eval_dataset(
    task: TaskFamily,
    max_samples: int | None = None,
) -> datasets.Dataset:
    """Load eval_test_in.pkl and return flat (anchor, positive, negative) rows for InfoNCE eval.

    eval_test_in maps anchor corpus_id -> {positives: [{corpus_id,...}], negatives: [{corpus_id,...}], ...}.
    Takes the first positive and first negative per anchor.
    """
    if not task.eval_pkl_path:
        raise ValueError(f"eval_pkl_path must be set on task '{task.name}' for citation eval")

    paper_texts = _load_pkl(task.paper_texts_path)
    eval_data = _load_pkl(task.eval_pkl_path)

    anchors, positives, negatives = [], [], []
    for anchor_id, entry in eval_data.items():
        anchor_text = _paper_text(paper_texts, anchor_id)
        if not anchor_text:
            continue

        pos_list = entry.get("positives", [])
        neg_list = entry.get("negatives", [])
        if not pos_list or not neg_list:
            continue

        pos_text = _paper_text(paper_texts, pos_list[0]["corpus_id"])
        neg_text = _paper_text(paper_texts, neg_list[0]["corpus_id"])
        if not pos_text or not neg_text:
            continue

        anchors.append(anchor_text)
        positives.append(pos_text)
        negatives.append(neg_text)

        if max_samples is not None and len(anchors) >= max_samples:
            break

    return datasets.Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


def _load_split(task: TaskFamily, split: str) -> datasets.Dataset:
    hf_split = "validation" if split == "dev" else "train"
    if task.data_files:
        return datasets.load_dataset("json", data_files={hf_split: task.data_files[split]})[hf_split]
    return datasets.load_dataset(**task.dataset, split=hf_split)


def _load_easy_ids(task: TaskFamily, split: str) -> set[str]:
    """Load the set of easy example/record IDs from the difficulty sidecar for this split.

    Uses PyArrow directly to read only the two needed columns, which is much faster than
    going through HF datasets when there are hundreds of sidecar partition files.
    Results are cached in memory to avoid recomputing on every dataset load.
    """
    import pyarrow.dataset as pad
    import s3fs

    # Check cache first
    cache_key = (task.name, split)
    if cache_key in _easy_ids_cache:
        return _easy_ids_cache[cache_key]

    id_col = "example_id" if task.name == "quote" else "record_id"
    fs = s3fs.S3FileSystem()
    easy_ids: set[str] = set()
    for source in task.sources:
        path = f"{task.s3_prefix}/difficulty_tags/{source}/split={split}".lstrip("s3://")
        try:
            ds = pad.dataset(path, filesystem=fs, format="parquet", partitioning=None)
            table = ds.to_table(columns=[id_col, "difficulty_level"], filter=pad.field("difficulty_level") == "easy")
            easy_ids.update(table[id_col].to_pylist())
        except FileNotFoundError:
            warnings.warn(
                f"No difficulty sidecar found for task='{task.name}' source='{source}' split='{split}'. "
                f"Skipping difficulty filter for this source.",
                stacklevel=3,
            )

    # Store in cache for future calls
    _easy_ids_cache[cache_key] = easy_ids
    return easy_ids


def _list_s3_parquet_files(s3_prefix: str, source: str, split: str) -> list[str]:
    """Return sorted list of s3:// parquet file URLs for a given source/split."""
    import s3fs
    fs = s3fs.S3FileSystem()
    path = f"{s3_prefix.lstrip('s3://')}/{source}/split={split}"
    return sorted(f"s3://{p}" for p in fs.glob(f"{path}/*.parquet"))


def _load_s3_parquet_split(
    task: TaskFamily,
    split: str,
    max_rows: int | None = None,
    easy_ids: set[str] | None = None,
) -> datasets.Dataset:
    """Load S3 parquet for a split, optionally filtered to easy rows, up to max_rows.

    Reads only as many partition files as needed (estimating ~10K rows/file, 3x overread
    when easy-filtering to account for dropped rows). Files are NOT shuffled so that
    anchor-localised ordering within partitions is preserved for the quote ±N index scan.

    Args:
        max_rows: Stop reading after accumulating this many (filtered) rows. None = all partitions.
        easy_ids: Set of IDs to keep. None = no difficulty filtering.
    """
    id_col = "example_id" if task.name == "quote" else "record_id"

    all_files: list[str] = []
    for source in task.sources:
        all_files.extend(_list_s3_parquet_files(task.s3_prefix, source, split))

    if max_rows is not None:
        rows_per_file = 10_000
        filter_factor = 3 if easy_ids is not None else 1
        n_files = max(1, (max_rows * filter_factor + rows_per_file - 1) // rows_per_file)
        files_to_read = all_files[:n_files]
    else:
        files_to_read = all_files

    stream = datasets.load_dataset(
        "parquet", data_files={"data": files_to_read}, split="data", streaming=True
    )

    if easy_ids is None and max_rows is None:
        return datasets.Dataset.from_list(list(stream))

    rows: list[dict] = []
    for row in stream:
        if easy_ids is not None and row.get(id_col) not in easy_ids:
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break

    return datasets.Dataset.from_list(rows)


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


def build_s3_dataset(
    task: TaskFamily,
    split: str,
    num_negatives: int = 1,
    num_positives: int = 1,
    target_rows: int | None = 25000,
    max_rows: int | None = 500_000,
    easy_only: bool = True,
) -> datasets.Dataset:
    """Load a new-format S3 parquet task and return anchor/positive/negative columns.

    For quote: streams parquet directly, pairs rows on-the-fly by scanning ±50 indices
    for same-anchor opposite-label counterparts. Avoids double sampling.

    For paper/statement/etc: loads all rows, then samples target_rows indices directly.
    """
    setting = task.name

    if setting == "quote":
        # Quote: stream parquet and pair on-the-fly, skip the separate _load_s3_parquet_split call
        # Load without difficulty filtering first to ensure we have pos/neg pairs; filter paired results later
        id_col = "example_id"

        all_files: list[str] = []
        for source in task.sources:
            all_files.extend(_list_s3_parquet_files(task.s3_prefix, source, split))

        if max_rows is not None:
            rows_per_file = 10_000
            # Don't use filter_factor since we're not pre-filtering
            n_files = max(1, (max_rows + rows_per_file - 1) // rows_per_file)
            files_to_read = all_files[:n_files]
        else:
            files_to_read = all_files

        stream = datasets.load_dataset(
            "parquet", data_files={"data": files_to_read}, split="data", streaming=True
        )

        # Load all rows without difficulty filtering (to ensure pos/neg balance for pairing)
        data_list: list[dict] = []
        row_count = 0
        for row in stream:
            data_list.append(row)
            row_count += 1
            if max_rows is not None and row_count >= max_rows:
                break


        if not data_list:
            raise ValueError(f"No quote rows loaded for task '{task.name}'")

        # Build index: (canonical_query, label) -> list of (idx, quote_text)
        query_label_index: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for idx, row in enumerate(data_list):
            anchor = row.get("canonical_query") or ""
            text = row.get("quote_text") or ""
            label = row.get("label")
            if anchor and text and label in ("positive", "negative"):
                key = (anchor, label)
                if key not in query_label_index:
                    query_label_index[key] = []
                query_label_index[key].append((idx, text))

        # Pair on-the-fly: for each (query, positive), find negatives from query's negative pool
        n_rows = len(data_list) if target_rows is None else target_rows
        seen: set[tuple[str, str, str]] = set()
        rows: list[dict] = []
        queries_with_pairs = 0
        pairs_skipped_duplicate = 0
        no_neg_counterpart = 0

        for (query, label), candidates in query_label_index.items():
            if label != "positive":
                continue
            opposite_label = "negative"
            if (query, opposite_label) not in query_label_index:
                no_neg_counterpart += 1
                continue  # No negatives for this query
            queries_with_pairs += 1
            neg_candidates = query_label_index[(query, opposite_label)]
            for _, pos_text in candidates:
                sampled_negs = random.sample(neg_candidates, min(1, len(neg_candidates)))
                for _, neg_text in sampled_negs:
                    key = (query, pos_text, neg_text)
                    if key in seen:
                        pairs_skipped_duplicate += 1
                        continue
                    seen.add(key)
                    rows.append({"anchor": query, "positive": pos_text, "negative": neg_text})
                    if len(rows) >= n_rows:
                        break
                if len(rows) >= n_rows:
                    break
            if len(rows) >= n_rows:
                break


        # Apply easy_ids filtering post-pairing if needed
        if easy_only:
            easy_ids = _load_easy_ids(task, split)
            if easy_ids and len(easy_ids) > 0:
                rows = [r for r in rows if any(row.get("example_id") in easy_ids for row in data_list
                        if row.get("canonical_query") == r["anchor"] and
                           (row.get("quote_text") == r["positive"] or row.get("quote_text") == r["negative"]))]

        if len(rows) < n_rows:
            warnings.warn(
                f"Quote dataset: only produced {len(rows)} unique rows (target {n_rows}).",
                stacklevel=2,
            )
        return datasets.Dataset.from_list(rows)

    # paper / statement / section_quote / paper_statement — already triplet rows
    easy_ids = _load_easy_ids(task, split) if easy_only else None
    if easy_ids is not None and len(easy_ids) == 0:
        warnings.warn(f"No easy IDs found for task='{task.name}' split='{split}'; disabling difficulty filter.")
        easy_ids = None
    data = _load_s3_parquet_split(task, split, max_rows=max_rows, easy_ids=easy_ids)
    n_data = len(data)

    if not n_data:
        raise ValueError(f"No rows loaded for task '{task.name}'")

    n_rows = n_data if target_rows is None else target_rows
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    max_attempts = n_rows * 10
    attempts = 0

    if setting == "paper":
        def _extract(row):
            q = row.get("canonical_query") or ""
            ta = row.get("text_a") or ""
            tb = row.get("text_b") or ""
            return (q, ta, tb) if q and ta and tb else None
    else:
        def _extract(row):
            q = row.get("canonical_query") or ""
            ta = row.get("text_a") or ""
            tc = row.get("text_c") or ""
            return (q, ta, tc) if q and ta and tc else None

    while len(rows) < n_rows and attempts < max_attempts:
        attempts += 1
        triplet = _extract(data[random.randrange(n_data)])
        if triplet is None or triplet in seen:
            continue
        seen.add(triplet)
        rows.append({"anchor": triplet[0], "positive": triplet[1], "negative": triplet[2]})

    if len(rows) < n_rows:
        warnings.warn(
            f"{setting} dataset: only produced {len(rows)} unique rows (target {n_rows}).",
            stacklevel=2,
        )

    return datasets.Dataset.from_list(rows)


def build_s3_triplet_eval_dataset(
    task: TaskFamily,
    max_samples: int | None = None,
    max_rows: int | None = 100_000,
    easy_only: bool = True,
) -> datasets.Dataset:
    """Load dev split of a new-format S3 parquet task as flat triplet rows for eval.

    Groups by canonical_query, collects all positives and negatives, then pairs by index
    to min(num_pos, num_neg). Each query generates min(num_pos, num_neg) triplet rows.
    max_samples caps the total number of output rows.
    """
    easy_ids = _load_easy_ids(task, "dev") if easy_only else None
    if easy_ids is not None and len(easy_ids) == 0:
        warnings.warn(f"No easy IDs found for task='{task.name}' split='dev'; disabling difficulty filter.")
        easy_ids = None
    data = _load_s3_parquet_split(task, "dev", max_rows=max_rows, easy_ids=easy_ids)
    setting = task.name

    query_map: dict[str, dict] = {}

    if setting == "quote":
        for ex in data:
            q = ex.get("canonical_query") or ""
            if q not in query_map:
                query_map[q] = {"positives": [], "negatives": []}
            text = ex.get("quote_text") or ""
            if not text:
                continue
            if ex.get("label") == "positive":
                query_map[q]["positives"].append(text)
            else:
                query_map[q]["negatives"].append(text)
    elif setting == "paper":
        for ex in data:
            q = ex.get("canonical_query") or ""
            if q not in query_map:
                query_map[q] = {"positives": [], "negatives": []}
            ta, tb = ex.get("text_a") or "", ex.get("text_b") or ""
            if ta:
                query_map[q]["positives"].append(ta)
            if tb:
                query_map[q]["negatives"].append(tb)
    else:
        # statement, section_quote, paper_statement
        for ex in data:
            q = ex.get("canonical_query") or ""
            if q not in query_map:
                query_map[q] = {"positives": [], "negatives": []}
            ta = ex.get("text_a") or ""
            tc = ex.get("text_c") or ""
            if ta:
                query_map[q]["positives"].append(ta)
            if tc:
                query_map[q]["negatives"].append(tc)

    anchors, positives, negatives = [], [], []
    for q, g in query_map.items():
        if not g["positives"] or not g["negatives"]:
            continue
        # Pair all positives with negatives by index up to min count
        n_pairs = min(len(g["positives"]), len(g["negatives"]))
        for i in range(n_pairs):
            anchors.append(q)
            positives.append(g["positives"][i])
            negatives.append(g["negatives"][i])
            if max_samples is not None and len(anchors) >= max_samples:
                return datasets.Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})

    return datasets.Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})


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
