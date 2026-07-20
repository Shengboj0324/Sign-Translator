"""Auxiliary heads and annotation-masked losses (docs/HAND_GRAPH.md §8).

Heads: handshape classification, palm orientation (6D rotation regression, scored
by the Doc-04 geodesic loss), selected fingers (multilabel), contact, symmetry.

Every loss is **annotation-masked**: samples/entries without a label are excluded
(weight 0), never assigned a fabricated target -- the Doc-03/Doc-04 provenance
discipline. A batch with no labels for a task contributes exactly 0 to that task
(and never a NaN), so it cannot skew training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pose.rotations import rotation_6d_to_matrix, geodesic_distance


# ---------------------------------------------------------------------------
# masked loss primitives
# ---------------------------------------------------------------------------
def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """CE over labelled samples only. ``logits`` (N,C), ``targets`` (N,) long,
    ``mask`` (N,) in [0,1]. Unlabelled targets are clamped for the gather and then
    zeroed by the mask, so an invalid placeholder (e.g. -1) never errors or leaks.
    """
    C = logits.shape[-1]
    safe_targets = targets.clamp(0, C - 1)
    per = F.cross_entropy(logits, safe_targets, reduction="none")   # (N,)
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """Multilabel BCE over labelled entries. ``logits``/``targets`` (N,K);
    ``mask`` (N,) or (N,K) in [0,1] (independent Bernoulli per label)."""
    per = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype),
                                             reduction="none")       # (N,K)
    m = mask.unsqueeze(-1) if mask.dim() == 1 else mask
    m = m.to(logits.dtype)
    return (per * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
class HandshapeHead(nn.Module):
    """Per-hand handshape classification."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)

    def loss(self, h, targets, mask):
        return masked_cross_entropy(self(h), targets, mask)


class SelectedFingersHead(nn.Module):
    """Which of the 5 fingers are selected/extended (multilabel)."""

    def __init__(self, dim: int, num_fingers: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_fingers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)

    def loss(self, h, targets, mask):
        return masked_bce_with_logits(self(h), targets, mask)


class PalmOrientationHead(nn.Module):
    """Regress palm orientation as a 6D rotation; scored by the geodesic loss."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, 6)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return rotation_6d_to_matrix(self.fc(h))

    def loss(self, h, target_R, mask):
        geo = geodesic_distance(self(h), target_R)                  # (N,)
        return (geo * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# contact + symmetry losses
# ---------------------------------------------------------------------------
def contact_loss(p_contact: torch.Tensor, labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None, eps: float = 1e-6
                 ) -> torch.Tensor:
    """BCE of predicted contact probabilities vs hard labels, over labelled pairs."""
    p = p_contact.clamp(eps, 1 - eps)
    per = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p))
    if mask is None:
        return per.mean()
    return (per * mask).sum() / mask.sum().clamp_min(1.0)


def mirror_points(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Reflect points across the plane perpendicular to ``axis`` (negate that coord)."""
    x2 = x.clone()
    x2[..., axis] = -x2[..., axis]
    return x2


def symmetry_loss(left_xyz: torch.Tensor, right_xyz: torch.Tensor,
                  mask: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Deviation of a two-handed sign from left/right mirror symmetry.

    For a symmetric sign, the left hand equals the mirror of the right hand:
    ``left ≈ mirror(right)``. ``left_xyz``/``right_xyz`` (N, L, 3); ``mask`` (N,)
    selects symmetric-sign samples only.
    """
    diff = left_xyz - mirror_points(right_xyz, axis)                # (N, L, 3)
    per = diff.pow(2).sum(-1).mean(-1)                              # (N,)
    return (per * mask).sum() / mask.sum().clamp_min(1.0)
