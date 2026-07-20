"""Temporal/part consistency + augmentation guard (Doc-11 §5).

Order/boundary prediction, multi-view alignment (RGB/2D/3D/hands of the same clip
are positives), and a handedness/direction-preserving augmentation guard that
REFUSES a horizontal flip unless the linguistic direction (handedness, loci,
agreement direction) is transformed with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .contrast import info_nce_loss, l2_normalize, recall_at_k


# ---------------------------------------------------------------------------
# temporal order
# ---------------------------------------------------------------------------
def recover_order_from_timestamps(timestamps: Sequence[float]) -> np.ndarray:
    """The unique ascending order of segments (argsort of distinct timestamps)."""
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(np.unique(ts)) != len(ts):
        raise ValueError("timestamps must be distinct for a unique order")
    return np.argsort(ts, kind="stable")


def pairwise_precedence_accuracy(pred_order: Sequence[int],
                                 true_order: Sequence[int]) -> float:
    """Fraction of ordered pairs whose relative order matches the truth."""
    pred_rank = {v: i for i, v in enumerate(pred_order)}
    true_rank = {v: i for i, v in enumerate(true_order)}
    items = list(true_rank)
    total = correct = 0
    for a_i in range(len(items)):
        for b_i in range(a_i + 1, len(items)):
            a, b = items[a_i], items[b_i]
            total += 1
            if (pred_rank[a] < pred_rank[b]) == (true_rank[a] < true_rank[b]):
                correct += 1
    return correct / total if total else 1.0


# ---------------------------------------------------------------------------
# multi-view alignment
# ---------------------------------------------------------------------------
def align_views(view_a: torch.Tensor, view_b: torch.Tensor,
                temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE aligning two views of the same clips (reuse)."""
    loss, _ = info_nce_loss(l2_normalize(view_a), l2_normalize(view_b),
                            temperature=temperature)
    return loss


def view_retrieval_recall1(view_a: torch.Tensor, view_b: torch.Tensor) -> float:
    za, zb = l2_normalize(view_a), l2_normalize(view_b)
    return recall_at_k(za @ zb.t(), 1)


# ---------------------------------------------------------------------------
# handedness / direction-preserving augmentation guard (innovation)
# ---------------------------------------------------------------------------
class AugmentationError(RuntimeError):
    """Raised when an augmentation would corrupt linguistic direction."""


@dataclass(frozen=True)
class LinguisticDirection:
    """Direction-bearing labels a horizontal flip must transform."""

    dominant_hand: int            # 0 = left, 1 = right
    loci: Tuple[float, ...]       # signed x-positions of spatial loci
    agreement_sign: int           # +1 / -1 directional verb orientation

    def flipped(self) -> "LinguisticDirection":
        return LinguisticDirection(
            dominant_hand=1 - self.dominant_hand,
            loci=tuple(-x for x in self.loci),
            agreement_sign=-self.agreement_sign,
        )


def augment_appearance(x: np.ndarray, scale: float = 1.0,
                       translate: Tuple[float, float] = (0.0, 0.0),
                       noise_std: float = 0.0, seed: int = 0) -> np.ndarray:
    """Content-PRESERVING augmentation (scale, translate, additive noise).

    ``x`` is (T, J, C) with channel 0 = x-coord, 1 = y-coord. None of these change
    handedness or direction, so no relabelling is required.
    """
    rng = np.random.default_rng(seed)
    out = x.astype(np.float64) * scale
    out[..., 0] += translate[0]
    if out.shape[-1] > 1:
        out[..., 1] += translate[1]
    if noise_std > 0:
        out = out + rng.normal(0.0, noise_std, out.shape)
    return out


def horizontal_flip(x: np.ndarray,
                    direction: Optional[LinguisticDirection]
                    ) -> Tuple[np.ndarray, LinguisticDirection]:
    """Horizontally flip motion AND transform its linguistic direction.

    Refuses (raises) if ``direction`` is None: a flip that does not transform
    handedness/loci/agreement changes meaning. When a direction is supplied, the
    x-coordinate is negated and the direction is flipped consistently.
    """
    if direction is None:
        raise AugmentationError(
            "horizontal flip requires a LinguisticDirection to relabel; "
            "flipping without transforming handedness/loci/direction changes meaning")
    out = x.astype(np.float64).copy()
    out[..., 0] = -out[..., 0]                       # mirror x-coordinate
    return out, direction.flipped()
