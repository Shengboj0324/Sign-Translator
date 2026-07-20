"""Non-manual loss suite (docs/FACIAL_NMM.md §4).

    L = λ_bce L_NMM + λ_b L_boundary + λ_s L_scope + λ_v L_smooth.

* ``L_NMM`` — the Doc-03 masked multi-label BCE (reused).
* ``L_boundary`` — supervise marker onset/offset frames (a BCE on a boundary target
  derived from the label's time difference); zero iff onsets/offsets match.
* ``L_scope`` — the non-manual scope must CONTAIN the manual event it marks
  (Doc-03 ``scope_containment_loss``); zero iff contained.
* ``L_smooth = Σ_t ‖p_{t+1}−p_t‖₁`` — penalises flicker; zero iff constant in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..grammar.nonmanual import multilabel_scope_bce, scope_containment_loss


def nmm_bce(logits: torch.Tensor, targets: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L_NMM: masked multi-label BCE (Doc-03)."""
    return multilabel_scope_bce(logits, targets, mask)


def boundary_targets(labels: torch.Tensor) -> torch.Tensor:
    """Onset/offset frames of each channel's active runs. ``labels`` (N, T, K) 0/1.

    A frame is a boundary iff it is ACTIVE and its run starts there (onset: previous
    frame inactive) or ends there (offset: next frame inactive). Returns (N, T, K).
    """
    a = labels > 0.5
    prev = torch.zeros_like(a); prev[:, 1:] = a[:, :-1]     # active at t-1 (False at t=0)
    nxt = torch.zeros_like(a); nxt[:, :-1] = a[:, 1:]       # active at t+1 (False at t=T-1)
    onset = a & ~prev
    offset = a & ~nxt
    return (onset | offset).to(labels.dtype)


def boundary_loss(boundary_logits: torch.Tensor, labels: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """BCE of predicted boundaries vs the onset/offset targets from ``labels``."""
    tgt = boundary_targets(labels)
    return multilabel_scope_bce(boundary_logits, tgt, mask)


def scope_loss(marker_start: torch.Tensor, marker_end: torch.Tensor,
               unit_start: torch.Tensor, unit_end: torch.Tensor) -> torch.Tensor:
    """L_scope: the non-manual scope must contain the manual event (Doc-03)."""
    return scope_containment_loss(marker_start, marker_end, unit_start, unit_end)


def temporal_smoothness(probs: torch.Tensor) -> torch.Tensor:
    """L_smooth = mean_t ‖p_{t+1} − p_t‖₁ along the time axis. 0 iff constant."""
    if probs.shape[1] < 2:
        return probs.new_zeros(())
    return (probs[:, 1:] - probs[:, :-1]).abs().mean()


@dataclass
class NMMWeights:
    bce: float = 1.0
    boundary: float = 0.5
    scope: float = 0.5
    smooth: float = 0.1


def total_nmm_loss(logits: torch.Tensor, targets: torch.Tensor,
                   boundary_logits: torch.Tensor,
                   probs: torch.Tensor,
                   weights: NMMWeights = NMMWeights(),
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Assemble the BCE + boundary + smoothness terms (scope added by the caller)."""
    return (weights.bce * nmm_bce(logits, targets, mask)
            + weights.boundary * boundary_loss(boundary_logits, targets, mask)
            + weights.smooth * temporal_smoothness(probs))
