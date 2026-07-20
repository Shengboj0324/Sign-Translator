"""Non-manual channels, scoped intervals, and the scope algebra (docs/FACIAL_NMM.md §1-2).

Non-manual events are concurrent grammatical channels (brows, eye aperture / gaze,
head / torso, cheeks, mouth), each a scoped interval
``n_k = (channel, marker, value, t_s, t_e, confidence)``. The concurrent channels'
scopes must **nest**, never partially cross; the scope algebra (built on the Doc-03
Allen relations) checks that and forms the nesting forest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

from ..grammar.temporal import AllenRelation, classify_relation


class Channel(IntEnum):
    BROW = 0
    EYE_APERTURE = 1
    GAZE = 2
    HEAD = 3
    TORSO = 4
    CHEEK = 5
    MOUTH = 6


class Marker(IntEnum):
    """Grammatical non-manual markers (not cosmetic emotion)."""

    YN_Q = 0        # yes/no question   (brow raise)
    WH_Q = 1        # wh question       (brow furrow)
    NEG = 2         # negation          (head shake)
    TOPIC = 3       # topicalisation    (brow raise + head tilt)
    COND = 4        # conditional       (held brow raise + head tilt)
    AFFECT = 5      # affective (non-grammatical) -- kept separate for disentanglement


# For each grammatical marker, the concurrent channel activations (direction × base
# intensity). The event's ``value`` further scales these (§7 articulation).
MARKER_CHANNELS: Dict[Marker, Dict[Channel, float]] = {
    Marker.YN_Q:  {Channel.BROW: +1.0, Channel.HEAD: +0.3},
    Marker.WH_Q:  {Channel.BROW: -1.0, Channel.HEAD: +0.2},
    Marker.NEG:   {Channel.HEAD: +1.0},
    Marker.TOPIC: {Channel.BROW: +1.0, Channel.HEAD: +0.5},
    Marker.COND:  {Channel.BROW: +1.0, Channel.HEAD: +0.5, Channel.EYE_APERTURE: +0.3},
    Marker.AFFECT: {Channel.CHEEK: +1.0, Channel.MOUTH: +0.5},
}

GRAMMATICAL_MARKERS = frozenset({Marker.YN_Q, Marker.WH_Q, Marker.NEG,
                                 Marker.TOPIC, Marker.COND})


@dataclass(frozen=True)
class NonmanualEvent:
    channel: Channel
    marker: Marker
    value: float
    t_s: float
    t_e: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (self.t_s < self.t_e):
            raise ValueError("t_s must be < t_e")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError("value must be in [0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    @property
    def duration(self) -> float:
        return self.t_e - self.t_s

    @property
    def is_grammatical(self) -> bool:
        return self.marker in GRAMMATICAL_MARKERS


# ---------------------------------------------------------------------------
# scope algebra
# ---------------------------------------------------------------------------
_CROSSING = frozenset({AllenRelation.OVERLAPS, AllenRelation.OVERLAPPED_BY})
# x is (strictly or equally) inside y:
_INSIDE = frozenset({AllenRelation.DURING, AllenRelation.STARTS,
                     AllenRelation.FINISHES})


def scope_relation(a: NonmanualEvent, b: NonmanualEvent,
                   eps: float = 1e-6) -> AllenRelation:
    """The Allen relation between two events' scopes."""
    return classify_relation((a.t_s, a.t_e), (b.t_s, b.t_e), eps)


def is_properly_nested(events: List[NonmanualEvent], eps: float = 1e-6) -> bool:
    """True iff no two scopes partially cross (every overlapping pair is nested).

    Concurrent non-manual channels may nest (a WH-question scope containing a topic
    scope) but must not partially overlap -- that would be an ill-formed grammatical
    structure.
    """
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if scope_relation(events[i], events[j], eps) in _CROSSING:
                return False
    return True


def nesting_parents(events: List[NonmanualEvent],
                    eps: float = 1e-6) -> List[Optional[int]]:
    """Parent index of each event = its smallest strict container (else ``None``).

    Forms the nesting forest of the concurrent scopes.
    """
    parents: List[Optional[int]] = [None] * len(events)
    for i, ev in enumerate(events):
        best, best_dur = None, float("inf")
        for j, other in enumerate(events):
            if i == j:
                continue
            if scope_relation(ev, other, eps) in _INSIDE and other.duration < best_dur:
                best, best_dur = j, other.duration
        parents[i] = best
    return parents
