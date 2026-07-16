"""Evaluation metrics for the sign-translation system.

* ``retrieval_recall_at_k`` - quality of the shared manifold: can motion find its
  paired language (and vice-versa) among distractors?
* ``mean_per_joint_position_error`` (MPJPE) - motion reconstruction/generation
  error, the standard 3D-pose metric.
* ``top1_accuracy`` - isolated-sign classification accuracy.
* ``word_error_rate`` - continuous-recognition gloss error (re-exported).
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..models.recognition import word_error_rate  # re-export


def retrieval_recall_at_k(similarity: torch.Tensor,
                          ks: Sequence[int] = (1, 5)) -> Dict[int, float]:
    """Recall@k for cross-modal retrieval.

    Args:
        similarity: ``(N, N)`` matrix; ``similarity[i, j]`` is the score between
            query ``i`` and candidate ``j``. The correct match for query ``i`` is
            candidate ``i`` (diagonal).
        ks: cut-offs to evaluate.

    Returns:
        ``{k: recall@k}`` averaged over queries.
    """
    if similarity.dim() != 2 or similarity.size(0) != similarity.size(1):
        raise ValueError("similarity must be a square (N, N) matrix")
    n = similarity.size(0)
    # Rank of each column within its row (descending score).
    ranking = similarity.argsort(dim=1, descending=True)
    targets = torch.arange(n, device=similarity.device).unsqueeze(1)
    # Position of the correct candidate in each row's ranking.
    hit_positions = (ranking == targets).float().argmax(dim=1)  # (N,)
    out: Dict[int, float] = {}
    for k in ks:
        out[k] = float((hit_positions < k).float().mean())
    return out


def mean_per_joint_position_error(pred: torch.Tensor,
                                  target: torch.Tensor) -> float:
    """MPJPE: mean Euclidean joint error over channels->frames->joints.

    Inputs are ``(N, C, T, V)``; the L2 norm is taken over the channel axis,
    then averaged over all joints, frames, and samples.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape")
    per_joint = torch.linalg.vector_norm(pred - target, dim=1)  # (N, T, V)
    return float(per_joint.mean())


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct top-1 predictions."""
    if logits.dim() != 2:
        raise ValueError("logits must be (N, num_classes)")
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean())
