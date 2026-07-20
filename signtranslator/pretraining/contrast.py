"""Cross-modal contrast + retrieval (Doc-11 §3).

The document's L_NCE is exactly the symmetric InfoNCE already in
`models/alignment.py`; it is REUSED here, not reimplemented. This module adds
retrieval recall@k and the explicit-negatives InfoNCE (wav2vec-2 form: one positive
+ K distractors) that the linguistic hard negatives (stage 11d) plug into.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from ..models.alignment import info_nce_loss   # symmetric InfoNCE (reused)

__all__ = ["info_nce_loss", "l2_normalize", "recall_at_k", "retrieval_recall",
           "info_nce_against_negatives"]


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def recall_at_k(similarity: torch.Tensor, k: int) -> float:
    """Fraction of rows whose diagonal (correct) index is within the top-k.

    ``similarity`` (N, N); row i's correct match is column i.
    """
    n = similarity.shape[0]
    if similarity.shape != (n, n):
        raise ValueError("similarity must be square (N, N)")
    if not 1 <= k <= n:
        raise ValueError("k out of range")
    topk = similarity.topk(k, dim=1).indices                 # (N, k)
    correct = torch.arange(n, device=similarity.device).unsqueeze(1)
    return float((topk == correct).any(dim=1).float().mean())


def retrieval_recall(z_a: torch.Tensor, z_b: torch.Tensor,
                     ks: Sequence[int] = (1, 5)) -> dict:
    """Symmetric retrieval recall@k between paired embeddings (cosine)."""
    za, zb = l2_normalize(z_a), l2_normalize(z_b)
    sim = za @ zb.t()
    out = {}
    for k in ks:
        out[f"a2b_recall@{k}"] = recall_at_k(sim, k)
        out[f"b2a_recall@{k}"] = recall_at_k(sim.t(), k)
    return out


def info_nce_against_negatives(anchors: torch.Tensor, positives: torch.Tensor,
                               negatives: torch.Tensor,
                               temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE with one positive per anchor + an explicit shared negative pool.

    loss_i = −log exp(a_i·p_i/τ) / ( exp(a_i·p_i/τ) + Σ_j exp(a_i·n_j/τ) ).
    ``anchors``/``positives`` (N, d) paired; ``negatives`` (M, d). Embeddings are
    L2-normalised internally. Perfect anchor==positive with orthogonal negatives
    drives the loss to 0 as τ→0.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if anchors.shape != positives.shape:
        raise ValueError("anchors and positives must share shape")
    a = l2_normalize(anchors)
    p = l2_normalize(positives)
    n = l2_normalize(negatives)
    pos = (a * p).sum(-1, keepdim=True) / temperature        # (N, 1)
    neg = a @ n.t() / temperature                            # (N, M)
    logits = torch.cat([pos, neg], dim=1)                    # (N, 1+M)
    target = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, target)
