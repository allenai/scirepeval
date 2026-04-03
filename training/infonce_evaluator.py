import datasets
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator


class InfoNCEEvaluator(SentenceEvaluator):
    """
    Evaluates InfoNCE loss using only the per-sample hard negatives (no in-batch negatives).

    For each sample i, the "batch" is [positive, negative_1, ..., negative_K].
    Loss = cross_entropy(cosine_sim(q, [p, n1, ..., nK]) / temperature, label=0).

    This matches the training objective (CachedGISTEmbedLoss) in geometry (cosine)
    and objective (InfoNCE), without the in-batch negatives that make no sense at eval time.
    """

    def __init__(
        self,
        eval_ds,
        name: str = "",
        batch_size: int = 64,
        temperature: float = 0.01,
    ):
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature

        self.anchors = eval_ds["anchor"]
        self.positives = eval_ds["positive"]
        if "negative" in eval_ds.column_names:
            self.neg_cols = [eval_ds["negative"]]
        else:
            k = 1
            self.neg_cols = []
            while f"negative_{k}" in eval_ds.column_names:
                self.neg_cols.append(eval_ds[f"negative_{k}"])
                k += 1

    @classmethod
    def from_triplet_task(cls, task, max_samples: int | None = None, **kwargs) -> "InfoNCEEvaluator":
        """Load a triplet task's dev split directly, preserving original (q, pos, neg) pairings.

        Bypasses build_st_dataset to avoid random re-pairing at eval time.
        Always K=1 (one neg per row, as the dataset defines).
        """
        sep = "\n\n"
        fields = task.input_fields

        def _text(doc):
            if isinstance(doc, dict):
                parts = [str(doc[f]) for f in fields if doc.get(f)]
            else:
                parts = [doc]
            return sep.join(parts)

        if task.data_files:
            data = datasets.load_dataset("json", data_files={"validation": task.data_files["dev"]})["validation"]
        else:
            data = datasets.load_dataset(**task.dataset, split="validation")

        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))

        eval_ds = datasets.Dataset.from_dict({
            "anchor":   [_text(ex["query"]) for ex in data],
            "positive": [_text(ex["pos"])   for ex in data],
            "negative": [_text(ex["neg"])   for ex in data],
        })
        return cls(eval_ds=eval_ds, **kwargs)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        encode = lambda texts: torch.tensor(
            model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
        )
        q_emb = encode(self.anchors)   # (N, D)
        p_emb = encode(self.positives)  # (N, D)
        neg_embs = [encode(negs) for negs in self.neg_cols]  # K x (N, D)

        # logits shape: (N, 1+K)
        sims = [F.cosine_similarity(q_emb, p_emb).unsqueeze(1)] + [
            F.cosine_similarity(q_emb, n_emb).unsqueeze(1) for n_emb in neg_embs
        ]
        logits = torch.cat(sims, dim=1) / self.temperature  # (N, 1+K)
        labels = torch.zeros(len(q_emb), dtype=torch.long)  # positive is always index 0
        loss = F.cross_entropy(logits, labels).item()

        print(f"[{self.name}] infonce_loss={loss:.4f} (epoch={epoch}, steps={steps})")
        self.primary_metric = f"{self.name}_infonce_loss"
        return {self.primary_metric: loss}
