"""Bridge: doc-02 typed SignPlan -> doc-03 SIR temporal graph -> gloss.

The planning layer (doc 02) emits a **typed** ``SignPlan``: an ordered list of
manual lexeme units, fingerspelling indices, non-manual scope spans over unit
ranges, per-referent loci, and a semantic frame. The grammar layer (doc 03)
consumes a **temporal graph** (SIR). This module is the faithful, information-
preserving map between them.

Design discipline (kept deliberately strict):

* We encode *only* what the plan actually specifies. The plan gives an ordered
  manual stream and non-manual spans over unit *indices*; it does NOT give a
  per-unit referent assignment, so we do not invent one. A ``unit_referents``
  argument may be supplied by a caller that genuinely has that mapping (e.g. the
  planner's own bookkeeping); absent it, manual events carry no referent/locus.
* Non-manual spans become NONMANUAL events with SCOPE edges to exactly the manual
  units they cover (``manual_units[start:end+1]`` -- the schema's inclusive
  range), so the multi-channel structure survives the round trip.
* Fingerspelled units become FINGERSPELL events, not (hallucinated) lexical signs.

Because the map is faithful, the manual gloss projection of the resulting SIR
must equal the plan's manual-unit order -- a property we test, not assert.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .sir import EventKind, EdgeType, SIREvent, SIREdge, SIRGraph


def plan_to_sir(plan, vocab=None, unit_time: float = 1.0,
                unit_referents: Optional[Dict[int, int]] = None) -> SIRGraph:
    """Map a doc-02 ``SignPlan`` to a doc-03 ``SIRGraph``.

    ``unit_referents`` optionally maps a manual-unit index -> referent id; when
    given, and the referent has a locus in ``plan.loci``, the corresponding
    manual event is annotated with that referent and locus (and a LOCUS edge is
    added). Without it, no referent/locus is fabricated.
    """
    unit_referents = unit_referents or {}
    events: List[SIREvent] = []
    edges: List[SIREdge] = []
    fingerspelled = set(plan.fingerspelling)

    # --- manual stream: one event per unit, in plan order (time = index).
    unit_event_id: Dict[int, int] = {}       # unit index -> event id
    eid = 0
    for idx, lex in enumerate(plan.manual_units):
        kind = EventKind.FINGERSPELL if idx in fingerspelled else EventKind.MANUAL
        ref = unit_referents.get(idx)
        locus = plan.loci.get(ref) if ref is not None else None
        events.append(SIREvent(id=eid, kind=kind, label=lex,
                               t_start=idx * unit_time,
                               t_end=(idx + 1) * unit_time,
                               referent=ref, locus=locus))
        unit_event_id[idx] = eid
        eid += 1

    # precedence chain over the manual stream
    manual_ids = [unit_event_id[i] for i in range(len(plan.manual_units))]
    for a, b in zip(manual_ids, manual_ids[1:]):
        edges.append(SIREdge(a, b, EdgeType.PRECEDENCE))

    # --- non-manual spans: NONMANUAL event + SCOPE edges to covered units.
    n_units = len(plan.manual_units)
    for span in plan.nonmanual:
        lo = max(0, span.start)
        hi = min(n_units - 1, span.end)          # inclusive per schema
        if hi < lo:
            continue                             # empty / out-of-range span
        s_time = lo * unit_time
        e_time = (hi + 1) * unit_time
        nm_id = eid
        events.append(SIREvent(id=nm_id, kind=EventKind.NONMANUAL,
                               label=span.marker, t_start=s_time, t_end=e_time))
        eid += 1
        for u in range(lo, hi + 1):
            edges.append(SIREdge(nm_id, unit_event_id[u], EdgeType.SCOPE))

    # --- LOCUS edges between co-referent manual units (persistent spatial index)
    if unit_referents:
        by_ref: Dict[int, List[int]] = {}
        for idx, r in unit_referents.items():
            if idx < n_units:
                by_ref.setdefault(r, []).append(unit_event_id[idx])
        for r, ids in by_ref.items():
            ids.sort()
            for a, b in zip(ids, ids[1:]):
                edges.append(SIREdge(a, b, EdgeType.COREF))

    return SIRGraph(events=events, edges=edges)


def plan_manual_units(plan) -> List[int]:
    """The plan's manual lexeme stream (the gloss the SIR projection must match)."""
    return list(plan.manual_units)
