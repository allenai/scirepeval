"""
Minimal SentenceTransformer training script for scirepeval IR tasks.
Replaces the PL-based SciRepTrain for the case of triplet/IR tasks with optional instruction prompts.
"""
import sys
sys.path.append('../')

import argparse
from datasets import DatasetDict
from transformers import AutoConfig
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, models
from sentence_transformers.evaluation import SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.losses import CachedGISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers  # type: ignore[import]

from tasks import load_tasks
from training.infonce_evaluator import InfoNCEEvaluator
from st_datasets import build_st_dataset, build_triplet_eval_dataset, build_ir_infonce_eval_dataset, build_ir_eval_data


def build_loss(
    model: SentenceTransformer,
    temperature: float,
    mini_batch_size: int,
    guide_model: SentenceTransformer,
    contrast_anchors: bool,
    contrast_positives: bool,
) -> CachedGISTEmbedLoss:
    return CachedGISTEmbedLoss(
        model=model,
        guide=guide_model,
        temperature=temperature,
        mini_batch_size=mini_batch_size,
        contrast_anchors=contrast_anchors,
        contrast_positives=contrast_positives,
    )


def build_prompts(ir_tasks: dict, num_negatives: int = 1) -> dict | None:
    """Build nested {task_name: {col_name: prompt}} dict for SentenceTransformerTrainingArguments."""
    training_prompts = {}
    for name, task in ir_tasks.items():
        if not task.instr_prompt:
            continue
        if isinstance(task.instr_prompt, dict):
            # asymmetric task (e.g. search): different prompts for query vs candidates
            cand_prompt = task.instr_prompt.get("candidates", "")
            neg_cols = {"negative": cand_prompt} if num_negatives == 1 else {f"negative_{i+1}": cand_prompt for i in range(num_negatives)}
            training_prompts[name] = {"anchor": task.instr_prompt["query"], "positive": cand_prompt, **neg_cols}
        else:
            # symmetric task: apply prompt to anchor only (candidates get no prompt, matches legacy behaviour)
            training_prompts[name] = {"anchor": task.instr_prompt}
    return training_prompts if training_prompts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace model or ST model name")
    parser.add_argument("--tasks-config", default="sample_data/tasks_config.json")
    parser.add_argument("--output", default="./st_output/")
    parser.add_argument("--guide-model", default="allenai/specter2", help="Guide model for CachedGISTEmbedLoss")
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for CachedGISTEmbedLoss")
    parser.add_argument("--no-contrast-anchors", action="store_true", default=False, help="Disable anchor-anchor contrastive signal in CachedGISTEmbedLoss")
    parser.add_argument("--no-contrast-positives", action="store_true", default=False, help="Disable positive-positive contrastive signal in CachedGISTEmbedLoss")
    parser.add_argument("--num-negatives", type=int, default=1, help="Hard negatives per sample (K); use negative_1..negative_K columns")
    parser.add_argument("--num-positives", type=int, default=2, help="Positives per query to expand into samples (P)")
    parser.add_argument("--queries-per-dataset", type=int, default=25000, help="Unique queries to sample per dataset; warns if > dataset size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=32, help="Mini-batch size for CachedGISTEmbedLoss embedding computation")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.03, help="Warmup proportion (< 1) or absolute steps (>= 1)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--val-check-interval", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--checkpoint-n-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1, help="Cap total training steps; -1 means no limit")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Truncate each eval dataset to N samples for quick runs")
    parser.add_argument("--use-cosine-schedule", action="store_true", default=False)
    parser.add_argument('--guide-model-pooling', type=str, choices=['cls', 'lasttoken', 'max', 'mean'], default="cls")
    parser.add_argument('--model-pooling', type=str, choices=['cls', 'lasttoken', 'max', 'mean'], default="lasttoken")
    args = parser.parse_args()

    mconfig = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tasks_dict = load_tasks(args.tasks_config, mconfig.hidden_size)

    ir_tasks = {n: t for n, t in tasks_dict.items() if t.type in ("ir", "triplet")}
    if not ir_tasks:
        raise ValueError("No IR/triplet tasks found in tasks config")

    encoder = models.Transformer(args.guide_model, model_args={"trust_remote_code":True})
    pooling = models.Pooling(encoder.get_word_embedding_dimension(), pooling_mode=args.guide_model_pooling)
    guide_model = SentenceTransformer(modules=[encoder, pooling], trust_remote_code=True)
    guide_model.max_seq_length = args.max_len

    encoder = models.Transformer(args.model, model_args={"trust_remote_code": True})
    pooling = models.Pooling(encoder.get_word_embedding_dimension(), pooling_mode=args.model_pooling)
    model = SentenceTransformer(modules=[encoder, pooling], trust_remote_code=True)
    model.max_seq_length = args.max_len

    train_datasets, infonce_evaluators, ir_evaluators, losses = {}, [], [], {}
    for name, task in ir_tasks.items():
        train_datasets[name] = build_st_dataset(task, "train", args.num_negatives, args.num_positives, args.queries_per_dataset)
        if task.type == "triplet":
            eval_ds = build_triplet_eval_dataset(task, max_samples=args.max_eval_samples)
            infonce_evaluators.append(InfoNCEEvaluator(eval_ds=eval_ds, name=name, temperature=args.temperature))
        else:
            eval_ds = build_ir_infonce_eval_dataset(task, max_samples=args.max_eval_samples)
            infonce_evaluators.append(InfoNCEEvaluator(eval_ds=eval_ds, name=name, temperature=args.temperature))
            queries, corpus, relevant_docs = build_ir_eval_data(task, max_samples=args.max_eval_samples)
            ir_evaluators.append(InformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=relevant_docs, name=f"{name}_ir", show_progress_bar=False))
        losses[name] = build_loss(model, args.temperature, args.mini_batch_size, guide_model, not args.no_contrast_anchors, not args.no_contrast_positives)
    if infonce_evaluators and ir_evaluators:
        # infonce_seq score = mean InfoNCE loss (lower=better); used as primary metric
        # IR evaluators follow; outer sequential score is not used for model selection
        infonce_seq = SequentialEvaluator(infonce_evaluators, main_score_function=lambda scores: sum(scores) / len(scores))
        evaluator = SequentialEvaluator([infonce_seq] + ir_evaluators, main_score_function=lambda scores: scores[0])
        best_metric, greater_is_better = "eval_sequential_score", False
    elif infonce_evaluators:
        evaluator = SequentialEvaluator(infonce_evaluators, main_score_function=lambda scores: sum(scores) / len(scores))
        best_metric, greater_is_better = "eval_sequential_score", False
    elif ir_evaluators:
        evaluator = SequentialEvaluator(ir_evaluators, main_score_function=lambda scores: sum(scores) / len(scores))
        best_metric, greater_is_better = "eval_sequential_score", True
    else:
        raise ValueError("No evaluators built — check tasks config")


    lr_scheduler = "cosine" if args.use_cosine_schedule else "linear"
    warmup = args.warmup if args.warmup < 1 else int(args.warmup)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=warmup if isinstance(warmup, float) else 0.0,
        warmup_steps=warmup if isinstance(warmup, int) else 0,
        lr_scheduler_type=lr_scheduler,
        max_steps=args.max_steps,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.val_check_interval,
        save_strategy="steps",
        save_steps=args.checkpoint_n_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        logging_steps=10,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        prompts=build_prompts(ir_tasks, args.num_negatives),
        push_to_hub=False,
        dataloader_drop_last=True
    )
    model.model_card_data.widget = []
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=DatasetDict(train_datasets),
        evaluator=evaluator,
        loss=losses,
    )
    trainer.train()
    model.save_pretrained(f"{args.output}/final_model")


if __name__ == "__main__":
    main()
