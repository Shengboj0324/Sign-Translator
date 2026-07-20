"""Hand-specific evaluation metrics (docs/HAND_GRAPH.md §9).

The document requires reporting *fingertip error in hand scale*, joint geodesic
error, handshape accuracy, contact F1, collision rate, left/right consistency, and
mirror/handedness behaviour -- because a small fingertip error can change meaning
and body-only MPJPE hides it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..pose.fitting import self_collision_penalty
from .hetero_graph import WRIST, MIDDLE_MCP, FINGERTIPS


def hand_scale(hand_xyz: torch.Tensor) -> torch.Tensor:
    """Wrist -> middle-MCP distance, the intrinsic hand size. (..., 21, 3) -> (...,)."""
    return torch.linalg.norm(hand_xyz[..., MIDDLE_MCP, :] - hand_xyz[..., WRIST, :],
                             dim=-1)


def fingertip_error_in_hand_scale(pred: torch.Tensor, gt: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
    """Mean fingertip position error divided by the (ground-truth) hand size.

    Scale-invariant: scaling ``pred`` and ``gt`` by the same factor leaves it
    unchanged. ``pred``/``gt`` (..., 21, 3).
    """
    tips = torch.tensor(FINGERTIPS, device=pred.device)
    err = (pred.index_select(-2, tips) - gt.index_select(-2, tips)).norm(dim=-1)  # (...,5)
    scale = hand_scale(gt).clamp_min(eps)                    # (...,)
    return (err.mean(-1) / scale).mean()


def handshape_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Top-1 accuracy over labelled samples. ``logits`` (N,C), ``targets`` (N,)."""
    pred = logits.argmax(-1)
    correct = (pred == targets).to(torch.float64)
    if mask is None:
        return correct.mean()
    m = mask.to(torch.float64)
    return (correct * m).sum() / m.sum().clamp_min(1.0)


@dataclass
class ContactPRF:
    precision: float
    recall: float
    f1: float


def contact_prf(pred: torch.Tensor, true: torch.Tensor) -> ContactPRF:
    """Precision / recall / F1 of predicted vs true contact indicators (bool)."""
    pred = pred.bool(); true = true.bool()
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return ContactPRF(precision, recall, f1)


def collision_rate(joints: torch.Tensor, radii: torch.Tensor,
                   adjacency: Optional[torch.Tensor] = None,
                   tol: float = 1e-9) -> float:
    """Fraction of frames with any self-penetration (Doc-04 sphere proxy > 0).

    ``joints`` (T, J, 3), ``radii`` (J,).
    """
    pen = self_collision_penalty(joints, radii, adjacency)   # (T,)
    return float((pen > tol).to(torch.float64).mean())


def mirror_hand(hand_xyz: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Reflect a hand across the plane perpendicular to ``axis`` (negate a coord)."""
    out = hand_xyz.clone()
    out[..., axis] = -out[..., axis]
    return out


def left_right_consistency(left_xyz: torch.Tensor, right_xyz: torch.Tensor,
                           axis: int = 0) -> torch.Tensor:
    """Mean symmetry error ‖left − mirror(right)‖ in hand scale (0 = symmetric).

    Normalised by the left hand's scale so it is comparable across signers.
    """
    diff = (left_xyz - mirror_hand(right_xyz, axis)).norm(dim=-1)   # (..., 21)
    scale = hand_scale(left_xyz).clamp_min(1e-8)
    return (diff.mean(-1) / scale).mean()
