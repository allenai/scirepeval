import argparse
import json
from typing import List

from evaluation.encoders import Model
from evaluation.evaluator import IREvaluator, SupervisedEvaluator, SupervisedTask
from evaluation.few_shot_evaluator import FewShotEvaluator
from reviewer_matching import ReviewerMatchingEvaluator
from evaluation.eval_datasets import SimpleDataset, IRDataset

TASK_IDS = {"classification": "[CLF]", "regression": "[RGN]", "proximity": "[PRX]",
            "adhoc_search": {"query": "[QRY]", "candidates": "[PRX]"}}


class SciRepEval:

    def __init__(self, tasks_config: str = "scirepeval_tasks.jsonl", task_list: List[str] = None,
                 task_formats: List[str] = None):
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
                self.tasks.update({k: tasks_dict[k] for k in task_by_formats[task_format]})

    def evaluate(self, model: Model):
        final_results = dict()
        for task_name, task in self.tasks.items():
            model.task_id = TASK_IDS[task["type"]]
            kwargs = dict()
            task_data = task["data"]
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
            if "batch_size" in task:
                kwargs["batch_size"] = task["batch_size"]
            if "fields" in task:
                kwargs["fields"] = task["fields"]
            save_path, load_path = None, None
            if "embeddings" in task:
                save_path = task["embeddings"].get("save")
                load_path = task["embeddings"].get("load")
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
                if task_name == "Paper-Reviewer Matching":
                    if not task_data.get("reviewers") and not task_data.get("hf_reviewers"):
                        raise ValueError(f"Task {task_name} has no reviewer metadata locally or hf_metadata")
                    if task_data.get("reviewers"):
                        reviewers = task_data["reviewers"]
                        kwargs["reviewer_metadata"] = reviewers if type(reviewers) != dict else (
                            reviewers["name"], reviewers["config"])
                    evaluator = ReviewerMatchingEvaluator(task_name, model=model, **kwargs)
                else:
                    data_class = SimpleDataset if task_data.get("simple_format") else IRDataset
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-confg', help='path to the task config file', default="scirepeval_tasks.jsonl")
    parser.add_argument('--mtype', help='Model variant to be used (default, pals, adapters, fusion)', default="default")
    parser.add_argument('--model', '--m', help='HuggingFace model to be used')
    parser.add_argument('--ctrl-tokens', action='store_true', default=False, help='use control codes for tasks')
    parser.add_argument('--adapters-dir', help='path to the adapter checkpoints', default=None)
    parser.add_argument("--output", help="path to the output file", default="scirepeval_results.json")

    args = parser.parse_args()

    model = Model(variant=args.mtype, base_checkpoint=args.model, adapters_load_from=args.adapters_dir,
                  use_ctrl_codes=args.ctrl_tokens,
                  task_id="", all_tasks=["[CLF]", "[PRX]", "[RGN]", "[QRY]"])
    evaluator = SciRepEval(tasks_config=args.tasks_confg)
    results = evaluator.evaluate(model)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
