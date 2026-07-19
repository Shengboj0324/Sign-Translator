"""Allen's interval algebra and its differentiable constraint losses.

Two intervals ``X = [xs, xe)`` and ``Y = [ys, ye)`` stand in exactly one of the
13 Allen (1983) relations. This module provides:

* crisp predicates for each relation (with an ``eps`` tolerance for the boundary
  equalities), and
* **differentiable hinge losses** that are exactly zero when a relation holds
  (up to a margin ``eps > 0`` on strict inequalities) and grow linearly with the
  violation, so gradient descent on interval endpoints can *enforce* a relation.

The precedence loss is the document's own:

    L_prec(i,j) = max(0, xe - ys + eps)   == 0  iff  xe <= ys - eps  (X before Y)

Every loss is proved (tests) to be non-negative, zero iff its relation holds,
finite, sub-differentiable, and to drive its relation to satisfaction under a
few gradient steps.
"""

from __future__ import annotations

from enum import Enum
from typing import Tuple

import torch

DEFAULT_EPS = 1e-3


class AllenRelation(Enum):
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"          # X during Y  (Y contains X)
    CONTAINS = "contains"      # X contains Y
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


Interval = Tuple[float, float]


def _s(iv):
    return iv[0]


def _e(iv):
    return iv[1]


# ---------------------------------------------------------------------------
# Crisp classification -- jointly exhaustive, mutually exclusive
# ---------------------------------------------------------------------------
def classify_relation(x: Interval, y: Interval, eps: float = DEFAULT_EPS
                      ) -> AllenRelation:
    """Return the single Allen relation between ``x`` and ``y``.

    Boundary equalities use a tolerance ``eps``; strict orderings use it as a
    separating margin so the 13 cases stay mutually exclusive.
    """
    xs, xe = float(_s(x)), float(_e(x))
    ys, ye = float(_s(y)), float(_e(y))

    def eq(a, b):
        return abs(a - b) <= eps

    def lt(a, b):
        return a < b - eps

    if eq(xs, ys) and eq(xe, ye):
        return AllenRelation.EQUALS
    if lt(xe, ys):
        return AllenRelation.BEFORE
    if lt(ye, xs):
        return AllenRelation.AFTER
    if eq(xe, ys):
        return AllenRelation.MEETS
    if eq(ye, xs):
        return AllenRelation.MET_BY
    if eq(xs, ys):                    # shared start
        return AllenRelation.STARTS if lt(xe, ye) else AllenRelation.STARTED_BY
    if eq(xe, ye):                    # shared end
        return AllenRelation.FINISHES if lt(ys, xs) else AllenRelation.FINISHED_BY
    if lt(ys, xs) and lt(xe, ye):
        return AllenRelation.DURING
    if lt(xs, ys) and lt(ye, xe):
        return AllenRelation.CONTAINS
    if lt(xs, ys) and lt(ys, xe) and lt(xe, ye):
        return AllenRelation.OVERLAPS
    if lt(ys, xs) and lt(xs, ye) and lt(ye, xe):
        return AllenRelation.OVERLAPPED_BY
    # Fallback for near-degenerate configs within the tolerance band.
    return AllenRelation.EQUALS


def intervals_intersect(x: Interval, y: Interval) -> bool:
    return _s(x) < _e(y) and _s(y) < _e(x)


# ---------------------------------------------------------------------------
# Differentiable losses.  Each takes (xs, xe, ys, ye) tensors and returns a
# non-negative scalar tensor that is 0 exactly when the relation holds.
# ---------------------------------------------------------------------------
def _hinge(z: torch.Tensor) -> torch.Tensor:
    return torch.clamp(z, min=0.0)


def validity_loss(ts: torch.Tensor, te: torch.Tensor,
                  eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff ``ts <= te - eps`` (start strictly before end)."""
    return _hinge(ts - te + eps)


def precedence_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Document's L_prec: 0 iff ``xe <= ys - eps`` (X strictly before Y)."""
    return _hinge(xe - ys + eps)


def meets_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff ``xe == ys`` (X meets Y)."""
    return torch.abs(xe - ys)


def contains_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff X contains Y: ``xs < ys - eps`` and ``ye < xe - eps``.

    This is the SCOPE constraint: a non-manual X must *contain* the manual span Y.
    """
    return _hinge(xs - ys + eps) + _hinge(ye - xe + eps)


def during_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff X during Y (Y contains X). Symmetric to :func:`contains_loss`."""
    return contains_loss(ys, ye, xs, xe, eps)


def overlap_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff the intervals intersect: not-before AND not-after.

    Penalises the two ways they can be disjoint: X entirely before Y
    (``xe <= ys``) or X entirely after Y (``ye <= xs``).
    """
    return _hinge(xs - ye + eps) + _hinge(ys - xe + eps)


def equals_loss(xs, xe, ys, ye, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """0 iff both endpoints coincide."""
    return torch.abs(xs - ys) + torch.abs(xe - ye)


# ---------------------------------------------------------------------------
# Edge-type -> loss dispatch (used by the SIR temporal objective)
# ---------------------------------------------------------------------------
_EDGE_LOSS = {
    "precedence": precedence_loss,
    "overlap": overlap_loss,
    "scope": contains_loss,        # non-manual source contains manual target
}


def edge_temporal_loss(edge_type: str, xs, xe, ys, ye,
                       eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Loss for the temporal relation an edge type encodes (0 if not temporal)."""
    fn = _EDGE_LOSS.get(edge_type)
    if fn is None:
        return torch.zeros((), dtype=torch.as_tensor(xs).dtype)
    return fn(xs, xe, ys, ye, eps)


def sir_temporal_loss(starts: torch.Tensor, ends: torch.Tensor,
                      edges: "list", eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Total temporal loss over an SIR: per-node validity + per-edge relations.

    ``starts``/``ends`` are ``(N,)`` endpoint tensors indexed by event id;
    ``edges`` is a list of ``(source, target, edge_type_str)`` triples.
    """
    loss = validity_loss(starts, ends, eps).sum()
    for src, tgt, etype in edges:
        loss = loss + edge_temporal_loss(
            etype, starts[src], ends[src], starts[tgt], ends[tgt], eps)
    return loss
