"""
InfoNCE loss using only explicit hard negatives — no in-batch negatives.

For each sample i the "batch" is [positive, negative_1, ..., negative_K].
Loss = cross_entropy(cosine_sim(q, [p, n1, ..., nK]) / temperature, label=0).

Isolates the effect of the InfoNCE loss formulation from in-batch negatives,
allowing comparison against CachedGISTEmbedLoss (InfoNCE + in-batch) and
TripletLoss (margin-based, no in-batch).
"""
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer


def hard_negative_infonce(
    q: torch.Tensor,
    pos: torch.Tensor,
    negs: list[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    """InfoNCE over explicit hard negatives only. All inputs must be L2-normalized.

    Args:
        q:    (N, D) query embeddings
        pos:  (N, D) positive embeddings
        negs: K x (N, D) negative embeddings
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    sim_pos = (q * pos).sum(dim=1, keepdim=True)                        # (N, 1)
    sim_negs = [(q * n).sum(dim=1, keepdim=True) for n in negs]         # K x (N, 1)
    logits = torch.cat([sim_pos] + sim_negs, dim=1) / temperature       # (N, 1+K)
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  # positive is index 0
    return F.cross_entropy(logits, labels)


class HardNegativeInfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature: float = 0.05):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, sentence_features: list, labels=None) -> torch.Tensor:  # noqa: ARG002
        # sentence_features: [anchor_batch, positive_batch, neg_1_batch, ..., neg_K_batch]
        embeddings = [F.normalize(self.model(sf)["sentence_embedding"], p=2, dim=1) for sf in sentence_features]
        return hard_negative_infonce(embeddings[0], embeddings[1], embeddings[2:], self.temperature)

    def get_config_dict(self):
        return {"temperature": self.temperature}
