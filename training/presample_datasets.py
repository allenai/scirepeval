"""
Pre-sample and save triplet datasets to disk for faster training.

Supports four formats:
- pkl: citation triplets
- s3_parquet: quote/paper/statement datasets
- single-parquet: search task with query/candidates structure
- legacy HF: standard triplet tasks

Saves all as Arrow datasets with {anchor, positive, negative} columns.
"""
import argparse
import random
import warnings
from collections import defaultdict
import datasets
from datasets import Dataset
from transformers import AutoConfig

# Try importing from current package, fallback to parent
try:
    from tasks import load_tasks
except ImportError:
    import sys
    sys.path.insert(0, '..')
    from tasks import load_tasks


def _load_pkl(path: str):
    """Load a pickle file from a local path or s3:// URI using pickle5."""
    import pickle5
    import os

    if path.startswith("s3://"):
        # Check if local cached version exists
        cache_dir = "/tmp/scirepeval_cache"
        filename = os.path.basename(path)
        local_cache_path = os.path.join(cache_dir, filename)

        if os.path.exists(local_cache_path):
            print(f"    Loading from local cache: {local_cache_path}")
            with open(local_cache_path, "rb") as f:
                return pickle5.load(f)

        # Fall back to S3
        import s3fs
        print(f"    Loading from S3: {path}")
        # Set connection timeouts for s3fs
        fs = s3fs.S3FileSystem(
            anon=False,
            config_kwargs={'connect_timeout': 30, 'read_timeout': 60},
        )
        try:
            with fs.open(path, "rb") as f:
                return pickle5.load(f)
        except Exception as e:
            print(f"    S3 load failed: {e}")
            raise

    # Local path
    with open(path, "rb") as f:
        return pickle5.load(f)


def _paper_text(paper_texts: dict, corpus_id: int) -> str | None:
    """Look up a paper by corpus_id and return 'title\n\nabstract'."""
    doc = paper_texts.get(corpus_id)
    if not doc:
        return None
    parts = []
    if doc.get("title"):
        parts.append(doc["title"])
    if doc.get("abstract"):
        parts.append(doc["abstract"])
    return "\n\n".join(parts) if parts else None


def _get_citation_negatives(entry: dict, neg_type: str) -> list:
    """Return the appropriate negative candidate list."""
    if neg_type == "hard":
        return entry.get("hard_negatives", [])
    if neg_type == "easy":
        return entry.get("easy_negatives", [])
    return entry.get("hard_negatives", []) + entry.get("easy_negatives", [])


def _list_s3_parquet_files(s3_prefix: str, source: str, split: str) -> list[str]:
    """Return sorted list of s3:// parquet file URLs for a given source/split."""
    import s3fs
    fs = s3fs.S3FileSystem(
        anon=False,
        config_kwargs={'connect_timeout': 30, 'read_timeout': 60},
    )
    path = f"{s3_prefix.lstrip('s3://')}/{source}/split={split}"
    print(f"    Listing S3 files: s3://{path}/*.parquet")
    files = sorted(f"s3://{p}" for p in fs.glob(f"{path}/*.parquet"))
    print(f"    Found {len(files)} files")
    return files


def _load_easy_ids(task, split: str) -> set[str]:
    """Load the set of easy example/record IDs from the difficulty sidecar."""
    import pyarrow.dataset as pad
    import s3fs

    id_col = "example_id" if task.name == "quote" else "record_id"
    fs = s3fs.S3FileSystem(
        anon=False,
        config_kwargs={'connect_timeout': 30, 'read_timeout': 60},
    )
    easy_ids: set[str] = set()
    for source in task.sources:
        path = f"{task.s3_prefix}/difficulty_tags/{source}/split={split}".lstrip("s3://")
        try:
            print(f"    Loading easy IDs from: {path}")
            ds = pad.dataset(path, filesystem=fs, format="parquet", partitioning=None)
            table = ds.to_table(columns=[id_col, "difficulty_level"], filter=pad.field("difficulty_level") == "easy")
            easy_ids.update(table[id_col].to_pylist())
        except FileNotFoundError:
            warnings.warn(
                f"No difficulty sidecar found for task='{task.name}' source='{source}' split='{split}'.",
                stacklevel=3,
            )
    return easy_ids


def _load_s3_parquet_split(task, split: str, max_rows: int | None = None, easy_ids: set[str] | None = None) -> datasets.Dataset:
    """Load S3 parquet for a split, optionally filtered to easy rows."""
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


def presample_pkl(task, target_rows: int, citation_anchors: int) -> datasets.Dataset:
    """Pre-sample citation pkl dataset using efficient sampling."""
    print(f"\nProcessing PKL task: {task.name}")
    print(f"  Loading paper_texts...")
    paper_texts = _load_pkl(task.paper_texts_path)
    print(f"  Loaded {len(paper_texts)} papers")

    # Instead of loading entire pkl, open it and sample keys
    print(f"  Opening pkl (will sample {citation_anchors} anchors)...")
    train_data = _load_pkl(task.pkl_path)
    anchor_ids = list(train_data.keys())
    print(f"  Found {len(anchor_ids)} total anchors")

    # Sample citation_anchors random anchor indices
    sampled_anchors = random.sample(anchor_ids, min(citation_anchors, len(anchor_ids)))
    print(f"  Sampled {len(sampled_anchors)} anchors, generating triplets...")

    rows: list[dict] = []
    skipped = 0
    anchor_not_in_texts = 0
    for i, anchor_id in enumerate(sampled_anchors):
        if (i + 1) % max(1, len(sampled_anchors) // 10) == 0:
            print(f"    {i+1}/{len(sampled_anchors)} anchors processed, {len(rows)} triplets generated")

        entry = train_data[anchor_id]
        anchor_corpus_id = entry.get("anchor_id")
        anchor_text = _paper_text(paper_texts, anchor_corpus_id)
        if not anchor_text:
            anchor_not_in_texts += 1
            skipped += 1
            continue

        pos_candidates = entry.get("positives", [])
        neg_candidates = _get_citation_negatives(entry, task.neg_type)
        if not pos_candidates or not neg_candidates:
            skipped += 1
            continue

        # Debug: check first successful anchor
        if len(rows) == 0:
            print(f"    DEBUG: First successful anchor")
            print(f"      anchor_corpus_id: {anchor_corpus_id}")
            print(f"      positives type: {type(pos_candidates)}, len: {len(pos_candidates)}")
            print(f"      negatives type: {type(neg_candidates)}, len: {len(neg_candidates)}")
            if pos_candidates:
                print(f"      pos[0]: {pos_candidates[0]}")
            if neg_candidates:
                print(f"      neg[0]: {neg_candidates[0]}")

        # Pair pos[i] ↔ neg[i] up to min count
        n_pairs = min(len(pos_candidates), len(neg_candidates))
        for j in range(n_pairs):
            pos_id = pos_candidates[j] if isinstance(pos_candidates[j], int) else pos_candidates[j].get("corpus_id")
            neg_id = neg_candidates[j] if isinstance(neg_candidates[j], int) else neg_candidates[j].get("corpus_id")

            if not pos_id or not neg_id:
                continue

            pos_text = _paper_text(paper_texts, pos_id)
            neg_text = _paper_text(paper_texts, neg_id)

            # Use fallback text if paper texts not found (corpus_ids from other sources)
            if not pos_text:
                pos_text = f"[Paper {pos_id}]"
            if not neg_text:
                neg_text = f"[Paper {neg_id}]"

            rows.append({"anchor": anchor_text, "positive": pos_text, "negative": neg_text})

    # Shuffle and truncate
    random.shuffle(rows)
    rows = rows[:target_rows]
    print(f"  Generated {len(rows)} triplets (skipped {skipped} anchors, {anchor_not_in_texts} missing from paper_texts)")
    return datasets.Dataset.from_list(rows)


def presample_quote(task, target_rows: int, max_rows: int | None) -> datasets.Dataset:
    """Pre-sample S3 parquet quote dataset."""
    print(f"\nProcessing S3 quote task: {task.name}")

    # Try to load easy IDs for negatives
    try:
        easy_neg_ids = _load_easy_ids(task, "train")
        print(f"  Loaded {len(easy_neg_ids)} easy negative IDs")
        filter_to_easy_negs = len(easy_neg_ids) > 0
    except Exception as e:
        print(f"  Warning: Could not load easy negative IDs ({e}), will use all negatives")
        easy_neg_ids = None
        filter_to_easy_negs = False

    # Stream all parquet files and build negatives pool + positives index
    all_files: list[str] = []
    for source in task.sources:
        all_files.extend(_list_s3_parquet_files(task.s3_prefix, source, "train"))
    print(f"  Found {len(all_files)} parquet files")

    if max_rows is not None:
        rows_per_file = 10_000
        n_files = max(1, (max_rows + rows_per_file - 1) // rows_per_file)
        files_to_read = random.sample(all_files, k=n_files)
    else:
        files_to_read = all_files

    easy_negatives_sample = Dataset.from_list(list(datasets.load_dataset(
        "parquet", data_files={"data": files_to_read}, split="data", streaming=True
    ).filter(lambda x: x.get('label')=='negative' and x.get('example_id') in easy_neg_ids and random.random() < 0.3)))
    if len(easy_negatives_sample) < target_rows:
        print(f" Warning: Only {len(easy_negatives_sample)} found")
    easy_negatives_sample = easy_negatives_sample.shuffle().select(range(target_rows))
    sampled_anchors = set(easy_negatives_sample['canonical_query'])

    positives_stream = datasets.load_dataset(
        "parquet", data_files={"data": files_to_read}, split="data", streaming=True
    ).shuffle(seed=42, buffer_size=1000).filter(lambda x: x.get('label')=='positive' and x.get('canonical_query') in sampled_anchors)

    positives_index = defaultdict(list)

    for row in positives_stream:
        positives_index[row.get('canonical_query')].append({'example_id': row.get('example_id'), 'positive': row.get('quote_text')})
    
    triplets = []

    for row in easy_negatives_sample:
        query = row.get('canonical_query')
        positives = positives_index.get(query)
        if not positives:
            continue
        positive = random.choice(positives)
        triplet = {'anchor': query, 'positive': positive.get('quote_text'), 'negative': row.get('quote_text'),'neg_example_id': row.get('example_id'), 'pos_example_id': positive.get('example_id')}
        triplets.append(triplet)

    print(f"  Loaded {len(triplets)} triplets")
    return datasets.Dataset.from_list(triplets)


def presample_s3_parquet(task, target_rows: int, max_rows: int | None) -> datasets.Dataset:
    """Pre-sample S3 parquet paper/statement/etc dataset (already triplet format)."""
    print(f"\nProcessing S3 parquet task: {task.name}")
    # For statement task, don't filter by easy_ids - use all data
    # For other tasks like paper, we can try to load easy_ids for filtering
    use_easy_ids_filter = task.name != "statement"

    easy_ids = None
    if use_easy_ids_filter:
        try:
            easy_ids = _load_easy_ids(task, "train")
            print(f"  Loaded {len(easy_ids)} easy IDs")
            if not easy_ids:
                print(f"  Warning: No easy IDs found, will use all data")
                easy_ids = None
        except Exception as e:
            print(f"  Warning: Could not load easy IDs ({e}), will use all data")
            easy_ids = None
    else:
        print(f"  Skipping easy_ids filter for {task.name}, will use all data")

    data = _load_s3_parquet_split(task, "train", max_rows=max_rows, easy_ids=easy_ids)
    n_data = len(data)
    print(f"  Loaded {n_data} rows from parquet")

    if not n_data:
        raise ValueError(f"No rows loaded for task '{task.name}'")

    # Determine extraction function based on task
    if task.name == "paper":
        def _extract(row):
            q = row.get("canonical_query") or ""
            ta = row.get("text_a") or ""
            tb = row.get("text_b") or ""
            return (q, ta, tb) if q and ta and tb else None
    else:
        # statement, section_quote, paper_statement use text_c instead of text_b
        def _extract(row):
            q = row.get("canonical_query") or ""
            ta = row.get("text_a") or ""
            tc = row.get("text_c") or ""
            return (q, ta, tc) if q and ta and tc else None

    # Random sample with dedup
    rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    max_attempts = target_rows * 10
    attempts = 0

    while len(rows) < target_rows and attempts < max_attempts:
        attempts += 1
        triplet = _extract(data[random.randrange(n_data)])
        if triplet is None or triplet in seen:
            continue
        seen.add(triplet)
        rows.append({"anchor": triplet[0], "positive": triplet[1], "negative": triplet[2]})

    random.shuffle(rows)
    print(f"  Generated {len(rows)} triplets")
    if len(rows) < target_rows:
        warnings.warn(f"{task.name} dataset: only produced {len(rows)} unique rows (target {target_rows}).")
    return datasets.Dataset.from_list(rows)


def presample_search(task, target_rows: int) -> datasets.Dataset:
    """Pre-sample single-parquet search dataset (query -> candidates with scores)."""
    print(f"\nProcessing single-parquet search task: {task.name}")

    # Load parquet
    if task.data_files:
        data = datasets.load_dataset("parquet", data_files=task.data_files["train"], split="train")
    else:
        raise ValueError(f"Search task '{task.name}' must have data_files set")

    print(f"  Loaded {len(data)} queries")

    rows: list[dict] = []
    for ex in data:
        query_obj = ex.get("query")
        # Query can be a dict or string
        if isinstance(query_obj, dict):
            query_text = query_obj.get("text") or ""
        else:
            query_text = query_obj or ""

        if not query_text:
            continue

        candidates = ex.get("candidates", [])
        pos_texts = []
        neg_texts = []

        for c in candidates:
            # Candidate can be dict or string
            if isinstance(c, dict):
                ctext = c.get("title") or c.get("text") or c.get("content") or ""
                score = c.get("score")
            else:
                ctext = str(c)
                score = None

            if not ctext:
                continue

            # Score of 1 = positive, score of 0 or missing = negative
            if score == 1:
                pos_texts.append(ctext)
            else:
                neg_texts.append(ctext)

        if not pos_texts or not neg_texts:
            continue

        # Pair pos[i] ↔ neg[i] up to min count
        n_pairs = min(len(pos_texts), len(neg_texts))
        for i in range(n_pairs):
            rows.append({"anchor": query_text, "positive": pos_texts[i], "negative": neg_texts[i]})

    random.shuffle(rows)
    rows = rows[:target_rows]
    print(f"  Generated {len(rows)} triplets")
    return datasets.Dataset.from_list(rows)


def presample_legacy_triplet(task, target_rows: int) -> datasets.Dataset:
    """Pre-sample legacy HF triplet dataset (query/pos/neg structure)."""
    print(f"\nProcessing legacy triplet task: {task.name}")

    # Load split
    hf_split = "train"
    if task.data_files:
        data = datasets.load_dataset("json", data_files={hf_split: task.data_files["train"]})[hf_split]
    else:
        data = datasets.load_dataset(**task.dataset, split=hf_split)

    print(f"  Loaded {len(data)} examples")

    rows: list[dict] = []
    for ex in data:
        query = ex.get("query") or ""
        pos = ex.get("pos") or ""
        neg = ex.get("neg") or ""

        if query and pos and neg:
            rows.append({"anchor": query, "positive": pos, "negative": neg})

    random.shuffle(rows)
    rows = rows[:target_rows]
    print(f"  Generated {len(rows)} triplets")
    return datasets.Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="Pre-sample and save triplet datasets to disk")
    parser.add_argument("--tasks-config", required=True, help="Path to tasks config JSON")
    parser.add_argument("--output", default="./presampled/", help="Output directory for saved datasets")
    parser.add_argument("--target-rows", type=int, default=25000, help="Target triplets per dataset")
    args = parser.parse_args()

    # Load tasks (need a dummy model config for hidden_size)
    class DummyConfig:
        hidden_size = 768
    mconfig = DummyConfig()
    tasks_dict = load_tasks(args.tasks_config, mconfig.hidden_size)

    # Filter to triplet tasks
    triplet_tasks = {n: t for n, t in tasks_dict.items() if t.type == "triplet"}
    if not triplet_tasks:
        print("No triplet tasks found in config")
        return

    # Derive sampling parameters from target_rows
    # Citation: assume avg 5 pos/neg pairs per anchor, oversample 1.5x to account for variance
    citation_anchors = int(1.5 * args.target_rows / 5)
    # S3 parquet: stream 10x target_rows to have good pool before sampling
    max_rows = 10 * args.target_rows

    for name, task in triplet_tasks.items():
        try:
            print(f"\n{'='*60}")
            # Dispatch to appropriate sampler
            if task.dataset_format == "pkl":
                print(f"[{name}] Starting PKL pre-sampling...")
                dataset = presample_pkl(task, args.target_rows, citation_anchors)
            elif task.dataset_format == "s3_parquet":
                if task.name == "quote":
                    print(f"[{name}] Starting S3 quote pre-sampling...")
                    dataset = presample_quote(task, args.target_rows, max_rows)
                else:
                    print(f"[{name}] Starting S3 {task.name} pre-sampling...")
                    dataset = presample_s3_parquet(task, args.target_rows, max_rows)
            elif task.data_files and "train.parquet" in str(task.data_files):
                # Single-parquet search
                print(f"[{name}] Starting single-parquet search pre-sampling...")
                dataset = presample_search(task, args.target_rows)
            else:
                # Legacy HF dataset
                print(f"[{name}] Starting legacy HF triplet pre-sampling...")
                dataset = presample_legacy_triplet(task, args.target_rows)

            # Save dataset
            out_path = f"{args.output}/{name}"
            print(f"[{name}] Saving {len(dataset)} rows to {out_path}...")
            dataset.save_to_disk(out_path)
            print(f"[{name}] ✓ Done!")

        except KeyboardInterrupt:
            print(f"\n[{name}] Interrupted by user")
            raise
        except Exception as e:
            print(f"\n[{name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"[{name}] Skipping task and continuing...")


if __name__ == "__main__":
    main()
