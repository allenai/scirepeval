import argparse
import json
from typing import List, Union
import importlib.metadata

from evaluation.encoders import Model
from evaluation.evaluator import IREvaluator, SupervisedEvaluator, SupervisedTask, ParquetBinaryIREvaluator
from evaluation.few_shot_evaluator import FewShotEvaluator
from evaluation.gpt3_encoder import GPT3Model
import gc
import torch
import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_training_dir = os.path.join(_script_dir, 'training')
sys.path.insert(0, _training_dir)
import schedulers

# Import appropriate instructor model based on transformers version
def _get_transformers_version():
    """Get installed transformers version."""
    try:
        version = importlib.metadata.version("transformers")
        return tuple(int(x) for x in version.split('.')[:2])  # Major.minor only
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return (0, 0)

# Dynamically import the appropriate model class
_transformers_version = _get_transformers_version()
if _transformers_version >= (4, 51):
    # Newer transformers: can use new models
    from evaluation.instructor_new import SentenceTransformerModel, GritLMModel, Voyage4Model, load_prompts_from_file
    InstructorModel = None
    NEW_MODELS_AVAILABLE = True
else:
    # Older transformers: only legacy INSTRUCTOR
    from evaluation.instructor import InstructorModel
    NEW_MODELS_AVAILABLE = False
    SentenceTransformerModel = None
    GritLMModel = None
    Voyage4Model = None

from reviewer_matching import ReviewerMatchingEvaluator
from evaluation.eval_datasets import SimpleDataset, IRDataset


def _load_s3_parquet_ir(s3_glob: str):
    """Load S3 parquet files into an HF Dataset with the IRDataset row schema.

    Each parquet row must have: doc_id, query (dict with corpus_id), candidates
    (list of dicts with corpus_id and score).
    Returns (dataset, qrels) where qrels is {query_corpus_id: {cand_corpus_id: score}}.
    """
    import s3fs
    import pandas as pd
    import datasets as hf_datasets

    fs = s3fs.S3FileSystem(anon=False)
    files = fs.glob(s3_glob.replace("s3://", ""))
    if not files:
        raise FileNotFoundError(f"No Parquet files found at {s3_glob}")
    df = pd.concat([pd.read_parquet(fs.open(f)) for f in files], ignore_index=True)
    raw = hf_datasets.Dataset.from_pandas(df)

    # Normalize to IRDataset schema: promote id fields to top-level doc_id
    # Two query shapes:
    #   - search: query.id + query.text
    #   - others: query.corpus_id + query.title + query.abstract
    normalized = []
    qrels = {}
    for row in raw:
        q = row["query"]
        if "id" in q:
            qid = str(q["id"])
            query = {"doc_id": qid, "title": q["text"]}
        else:
            qid = str(q["corpus_id"])
            query = {"doc_id": qid, "title": q.get("title", ""), "abstract": q.get("abstract", "")}
        candidates = [{**c, "doc_id": str(c["corpus_id"])} for c in row["candidates"]]
        normalized.append({"doc_id": qid, "query": query, "candidates": candidates})
        qrels[qid] = {str(c["corpus_id"]): c["score"] for c in row["candidates"]}

    return hf_datasets.Dataset.from_list(normalized), qrels
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK_IDS = {"classification": "[CLF]", "regression": "[RGN]", "proximity": "[PRX]",
            "adhoc_search": {"query": "[QRY]", "candidates": "[PRX]"},
            "binary_retrieval": {"query": "[QRY]", "candidates": "[PRX]"}}

import pytorch_lightning as pl

pl.seed_everything(42, workers=True)


class SciRepEval:

    def __init__(self, tasks_config: str = "scirepeval_tasks.jsonl", task_list: List[str] = None,
                 task_formats: List[str] = None, batch_size: int = 32, embedding_save_path = None, excluded_tasks: List[str] = None, task_specific_prompts: bool = False, max_samples: int = None):
        tasks_dict = dict()
        task_by_formats = dict()
        with open(tasks_config, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                tasks_dict[d["name"]] = d
                if d["type"] not in task_by_formats:
                    task_by_formats[d["type"]] = []
                task_by_formats[d["type"]].append(d["name"])
        if not task_list and not task_formats:
            self.tasks = tasks_dict
        elif task_list:
            self.tasks = {k: tasks_dict[k] for k in task_list}
        elif task_formats:
            self.tasks = dict()
            for task_format in task_formats:
                if task_format not in excluded_tasks:
                    self.tasks.update({k: tasks_dict[k] for k in task_by_formats[task_format]})
        self.batch_size = batch_size
        self.embedding_save_path = embedding_save_path
        self.task_specifc_prompts = task_specific_prompts
        self.max_samples = max_samples

    def evaluate(self, model: Union[Model, List[Model]], output: str):
        final_results = dict()
        if type(model) != list:
            model = [model]
        for task_name, task in self.tasks.items():
            try:
                for m in model:
                    m.task_id = TASK_IDS[task["type"]]
                    if self.task_specifc_prompts:
                        m.task_name = task_name
                kwargs = dict()
                task_data = task["data"]
                use_s3 = (task_data.get("source")
                          and task["type"] in {"proximity", "adhoc_search"})
                if task["type"] != "binary_retrieval" and not use_s3:
                    if not task_data.get("meta"):
                        raise ValueError(f"Task {task_name} has no test metadata")
                    if task_data.get("meta"):
                        metadata = task_data["meta"]
                        kwargs["meta_dataset"] = metadata if type(metadata) != dict else (metadata["name"], metadata["config"])

                    if not task_data.get("test"):
                        if type(metadata) == dict:
                            kwargs["test_dataset"] = (metadata["name"], metadata["config"])
                        else:
                            raise ValueError(f"Task {task_name} has no test data")
                    if task_data.get("test"):
                        testdata = task_data["test"]
                        kwargs["test_dataset"] = testdata if type(testdata) != dict else (testdata["name"], testdata["config"])

                kwargs["metrics"] = tuple(task["metrics"])

                kwargs["batch_size"] = task["batch_size"] if "batch_size" in task else self.batch_size

                if "fields" in task:
                    kwargs["fields"] = task["fields"]
                save_path, load_path = None, None
                if "embeddings" in task:
                    save_path = task["embeddings"].get("save")
                    load_path = task["embeddings"].get("load")
                    if self.embedding_save_path and save_path:
                        save_path = os.path.join(self.embedding_save_path, save_path)
                    if self.embedding_save_path and load_path:
                        load_path = os.path.join(self.embedding_save_path, load_path)
                few_shot_evaluators = []
                if task["type"] in {"classification", "regression"}:
                    subtype = SupervisedTask.CLASSIFICATION if task[
                                                                   "type"] == "classification" else SupervisedTask.REGRESSION
                    if task.get("multi_label"):
                        subtype = SupervisedTask.MULTILABEL_CLASSIFICATION
                    evaluator = SupervisedEvaluator(task_name, subtype, model=model,
                                                    **kwargs)
                    if task.get("few_shot"):
                        for run in task["few_shot"]:
                            few_shot_evaluators.append(
                                FewShotEvaluator(f"{task_name} {run['sample_size']} shot", subtype, model=model,
                                                 sample_size=run["sample_size"], num_iterations=run["iterations"],
                                                 **kwargs))
                else:
                    if task["type"] == "binary_retrieval":
                        source = task_data.get("source")
                        if not source:
                            raise ValueError(f"Task {task_name} requires a 'source' S3 glob in the task config")
                        evaluator = ParquetBinaryIREvaluator(
                            task_name,
                            meta_dataset=source,
                            test_dataset=source,
                            model=model,
                            metrics=kwargs["metrics"],
                            batch_size=kwargs["batch_size"],
                            fields=kwargs.get("fields"),
                            max_samples=self.max_samples,
                        )
                    elif task_name == "Paper-Reviewer Matching":
                        if not task_data.get("reviewers") and not task_data.get("hf_reviewers"):
                            raise ValueError(f"Task {task_name} has no reviewer metadata locally or hf_metadata")
                        if task_data.get("reviewers"):
                            reviewers = task_data["reviewers"]
                            kwargs["reviewer_metadata"] = reviewers if type(reviewers) != dict else (
                                reviewers["name"], reviewers["config"])
                        evaluator = ReviewerMatchingEvaluator(task_name, model=model, **kwargs)
                    else:
                        data_class = SimpleDataset if task_data.get("simple_format") else IRDataset
                        if use_s3:
                            source = task_data["source"]
                            s3_dataset, s3_qrels = _load_s3_parquet_ir(source)
                            kwargs["processing_fn"] = lambda _: s3_dataset
                            kwargs["meta_dataset"] = source
                            kwargs["prebuilt_qrels"] = s3_qrels
                        evaluator = IREvaluator(task_name, model=model, dataset_class=data_class, **kwargs)
                embeddings = evaluator.generate_embeddings(save_path) if not load_path else load_path
                results = evaluator.evaluate(embeddings)
                if not few_shot_evaluators:
                    final_results[task_name] = results
                else:
                    final_results[task_name] = dict()
                    final_results[task_name]["complete"] = results
                    final_results[task_name]["few_shot"] = []

                for few_shot in few_shot_evaluators:
                    final_results[task_name]["few_shot"].append(
                        {"sample_size": few_shot.sample_size, "results": few_shot.evaluate(embeddings)})
                with open(output, "w") as f:
                    json.dump(final_results, f, indent=4)
            except Exception as e:
                print(f"Error evaluating task {task_name}: {str(e)}")
                final_results[task_name] = {"error": str(e)}
                with open(output, "w") as f:
                    json.dump(final_results, f, indent=4)
            finally:
                # Clear CUDA cache between tasks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-config', help='path to the task config file', default="scirepeval_tasks.jsonl")
    parser.add_argument('--task-list', help='List of tasks to run. Task formats is not used if this is specified.', default=None, nargs="+", type=str)
    parser.add_argument('--excluded-tasks', help='List of tasks to exclude.', default=None, nargs="+", type=str)
    parser.add_argument('--task-formats', help='Types of tasks to run', nargs='+', type=str, default=None)
    parser.add_argument('--mtype', help='Model variant to be used (default, pals, adapters, fusion)', default="default")
    parser.add_argument('--gpt3-model', help='Name of embedding model in case of using openai api', default=None)
    parser.add_argument('--model', '-m', help='HuggingFace model to be used')
    parser.add_argument('--checkpoint', help='Path to PyTorch checkpoint file', default=None)
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--adapters-dir', help='path to the adapter checkpoints', default=None)
    parser.add_argument('--fusion-dir', help='path to the fusion checkpoints', default=None)
    parser.add_argument('--adapters-chkpt', help='hf adapter names keyed on tasks', default=None, type=json.loads)
    parser.add_argument('--pooling-mode',  default="cls", help='Pooling mode to get embeddings from encoder, must be one of "cls" or "mean"')
    parser.add_argument('--output', help="path to the output file", default="scirepeval_results.json")
    parser.add_argument('--fp16', action='store_true', default=False, help='use floating point 16 precision')
    parser.add_argument('--instructor', action='store_true', default=False, help='use an instructor model for eval')
    parser.add_argument('--model-type', help='Instructor model type to use. Only valid if instructor is True')
    parser.add_argument('--prompt-file', type=str, help='JSON-formatted file containing multiple instruction prompts')
    parser.add_argument('--prompt-name', type=str, help='Name of prompt within prompt file to use.', default="blank")
    parser.add_argument('--embeddings-save-path', type=str, default=None, help='Path to parent directory where embeddings will be saved. If specified, config paths are treated as relative to this path.')
    parser.add_argument('--task-specific-prompts', action='store_true')
    # Voyage4 API-specific arguments
    parser.add_argument('--voyage-api', action='store_true', default=False, help='Use Voyage API for docs (local nano for queries). Requires VOYAGE_API_KEY env var')
    parser.add_argument('--truncate-dim', type=int, default=None, help='Embedding truncation dimension for instructor models that support it (e.g. qwen3)')

    parser.add_argument('--max-samples', type=int, default=None, help='Cap number of queries for binary_retrieval tasks (useful for quick tests).')
    parser.add_argument('--trust-remote-code', action='store_true', default=False, help='Pass trust_remote_code=True to SentenceTransformer')
    parser.add_argument('--prompt-name-map', type=json.loads, default=None, help='JSON mapping of task IDs to model preset prompt names, e.g. \'{"[CLF]": "Classification", "[SRCH]": {"q": "Retrieval-query", "c": "Retrieval-document"}}\'')

    parser.add_argument('--max-seq-length', type=int, default=512, help='Override encoder max_seq_length')

    args = parser.parse_args()
    adapters_load_from = args.adapters_dir if args.adapters_dir else args.adapters_chkpt
    if args.gpt3_model:
        model = GPT3Model(embed_model=args.gpt3_model)
    elif args.instructor:
        if args.model_type == "instr":
            model = InstructorModel(args.model)
        elif args.model_type == "voyage4":
            model = Voyage4Model(
                args.model,
                ckpt_path=args.checkpoint,
                use_api=args.voyage_api,
                truncate_dim=args.truncate_dim or 1024,
            )
        elif args.model_type == "gritlm":
            if not args.prompt_file or not os.path.exists(args.prompt_file):
                raise ValueError("GritLM requires a prompt file.")
            task_prompts = load_prompts_from_file(args.prompt_file, args.prompt_name)
            model = GritLMModel(args.model, task_prompts, ckpt_path=args.checkpoint)
        else:
            if args.prompt_name_map:
                task_prompts = None
            elif args.prompt_file and os.path.exists(args.prompt_file):
                task_prompts = load_prompts_from_file(args.prompt_file, args.prompt_name)
            else:
                raise ValueError("Provide either --prompt-file or --prompt-name-map.")
            model = SentenceTransformerModel(
                args.model,
                task_prompts=task_prompts,
                prompt_name_map=args.prompt_name_map,
                ckpt_path=args.checkpoint,
                truncate_dim=args.truncate_dim,
                trust_remote_code=args.trust_remote_code,
                max_seq_length=args.max_seq_length,
            )
    else:
        model = Model(
            variant=args.mtype,
            base_checkpoint=args.model,
            adapters_load_from=adapters_load_from,
            fusion_load_from=args.fusion_dir,
            use_ctrl_codes=args.ctrl_tokens,
            task_id="",
            all_tasks=["[CLF]", "[PRX]", "[QRY]", "[RGN]"],
            pooling_mode=args.pooling_mode,
            use_fp16=args.fp16
        )
    evaluator = SciRepEval(tasks_config=args.tasks_config, batch_size=args.batch_size, embedding_save_path=args.embeddings_save_path, excluded_tasks=args.excluded_tasks, task_formats=args.task_formats, task_list=args.task_list, task_specific_prompts=args.task_specific_prompts, max_samples=args.max_samples)
    evaluator.evaluate(model, args.output)
