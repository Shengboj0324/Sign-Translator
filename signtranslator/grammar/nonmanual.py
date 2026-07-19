"""Non-manual scope as multilabel interval prediction.

The document is explicit: non-manual scope is **multilabel interval prediction,
not punctuation appended after generation.** A non-manual marker (negation,
wh-question, topic, conditional, role shift) is active over a *time interval* and
several markers can be active **simultaneously** with each other and with the
manual stream.

Two consequences drive the design:

1. **Multilabel, not multiclass.** At a manual position ``t`` the target is a
   vector of ``M`` independent Bernoullis ``y_{m,t} in {0,1}`` -- a softmax over
   markers would forbid co-occurrence, which is linguistically wrong (a clause
   can be both negated and a question). Training uses masked BCE per (marker,
   position).

2. **Co-temporal with the manual stream.** A marker's predicted span must
   *contain* the manual events it scopes (Allen ``during``/``contains``), which
   couples the non-manual head to the temporal losses of :mod:`temporal`. This is
   what makes the scope part of the timed graph rather than post-hoc punctuation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal import contains_loss, DEFAULT_EPS


def multilabel_scope_bce(logits: torch.Tensor, targets: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Masked BCE over ``(N, T, M)`` marker activations.

    ``targets`` are 0/1; ``mask`` is ``(N, T)`` with 1 for real positions. Each
    (marker, position) is an independent Bernoulli -- co-occurring markers are
    representable, which a softmax cross-entropy could not do.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must share shape (N, T, M)")
    per = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype),
                                             reduction="none")
    if mask is not None:
        m = mask.to(per.dtype).unsqueeze(-1)             # (N, T, 1)
        return (per * m).sum() / m.sum().clamp_min(1.0) / logits.shape[-1]
    return per.mean()


@dataclass
class MarkerSpan:
    """One non-manual marker active over ``[start, end)`` covering manual units."""

    marker: int
    start: float
    end: float
    covered_units: Tuple[int, ...] = ()


def spans_from_activations(probs: torch.Tensor, unit_starts: Sequence[float],
                           unit_ends: Sequence[float],
                           threshold: float = 0.5) -> List[MarkerSpan]:
    """Decode ``(T, M)`` per-position marker probabilities into marker spans.

    Contiguous runs of active positions for a marker become one span whose time
    interval is the union of the covered manual units' intervals. This is the
    inverse of the multilabel head: activations -> timed, labelled scopes.
    """
    if probs.dim() != 2:
        raise ValueError("probs must be (T, M)")
    T, M = probs.shape
    spans: List[MarkerSpan] = []
    active = probs >= threshold
    for m in range(M):
        t = 0
        while t < T:
            if not bool(active[t, m]):
                t += 1
                continue
            run_start = t
            while t < T and bool(active[t, m]):
                t += 1
            units = tuple(range(run_start, t))
            spans.append(MarkerSpan(
                marker=m,
                start=float(min(unit_starts[u] for u in units)),
                end=float(max(unit_ends[u] for u in units)),
                covered_units=units))
    return spans


def scope_containment_loss(marker_start: torch.Tensor, marker_end: torch.Tensor,
                           unit_start: torch.Tensor, unit_end: torch.Tensor,
                           eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Penalty for a marker span failing to contain a scoped manual unit.

    Zero exactly when the marker interval contains the unit interval
    (Allen ``contains``), so a trained scope is co-temporal with -- not merely
    adjacent to -- the manual events it marks.
    """
    return contains_loss(marker_start, marker_end, unit_start, unit_end, eps)


class NonmanualScopeHead(nn.Module):
    """Predict multilabel marker activations and per-marker span endpoints.

    From per-position node states it emits ``(N, T, M)`` marker logits. Marker
    span endpoints for the containment coupling are derived from the manual
    units the run covers (via :func:`spans_from_activations`), keeping the head
    aligned to the manual timeline.
    """

    def __init__(self, d_model: int, num_markers: int) -> None:
        super().__init__()
        self.marker_head = nn.Linear(d_model, num_markers)

    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        if node_states.dim() != 3:
            raise ValueError("node_states must be (N, T, d)")
        return self.marker_head(node_states)             # (N, T, M) logits

    def loss(self, node_states: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return multilabel_scope_bce(self.forward(node_states), targets, mask)
