"""The Structured Intermediate Representation: a temporal graph of sign events.

Nodes are manual or non-manual *events*, each with a half-open time interval
``[t_start, t_end)`` (``t_start < t_end``). Edges are typed: precedence, overlap,
scope, co-reference, and spatial locus. A deterministic manual-label projection
can be derived from the manual sub-graph (docs/GRAMMAR_SIR.md §2), but that
projection is not thereby an authentic gloss annotation.

The graph is validated structurally rule-by-rule; its manual-label projection is
a topological sort of the manual events' precedence DAG. Neither structural
validity nor this projection establishes authentic ASL annotation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast


SIR_SCHEMA_VERSION = "1.0.0"


class EventKind(Enum):
    MANUAL = "manual"          # a lexical sign
    CLASSIFIER = "classifier"  # a depicting / classifier construction (manual)
    FINGERSPELL = "fingerspell"  # a fingerspelled item (manual)
    NONMANUAL = "nonmanual"    # a facial / body marker

    @property
    def is_manual(self) -> bool:
        return self is not EventKind.NONMANUAL


class EdgeType(Enum):
    PRECEDENCE = "precedence"  # source finishes no later than target starts
    OVERLAP = "overlap"        # intervals intersect
    SCOPE = "scope"            # non-manual source contains manual target (during)
    COREF = "coref"            # source and target share a referent
    LOCUS = "locus"            # target is placed at a spatial locus


@dataclass
class SIREvent:
    """One node: an event with a time interval and typed content."""

    id: int
    kind: EventKind
    label: int                 # lexeme id (manual) or marker id (non-manual)
    t_start: float
    t_end: float
    referent: Optional[int] = None   # discourse referent this event involves
    locus: Optional[int] = None      # spatial locus, if placed

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def overlaps_time(self, other: "SIREvent") -> bool:
        return self.t_start < other.t_end and other.t_start < self.t_end


@dataclass(frozen=True)
class SIREdge:
    source: int                # event id
    target: int                # event id
    type: EdgeType


@dataclass
class SIRGraph:
    """A temporal graph ``G = (V, E)`` of sign events."""

    events: List[SIREvent] = field(default_factory=list)
    edges: List[SIREdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id: Dict[int, SIREvent] = {e.id: e for e in self.events}

    def rebuild_index(self) -> None:
        self._by_id = {e.id: e for e in self.events}

    def event(self, event_id: int) -> SIREvent:
        return self._by_id[event_id]

    def has_event(self, event_id: int) -> bool:
        return event_id in self._by_id

    def manual_events(self) -> List[SIREvent]:
        return [e for e in self.events if e.kind.is_manual]

    def nonmanual_events(self) -> List[SIREvent]:
        return [e for e in self.events if not e.kind.is_manual]

    def edges_of(self, edge_type: EdgeType) -> List[SIREdge]:
        return [e for e in self.edges if e.type is edge_type]


# ---------------------------------------------------------------------------
# Topological ordering / manual-label projection
# ---------------------------------------------------------------------------
def _precedence_adjacency(graph: SIRGraph, node_ids: Set[int]
                          ) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {n: [] for n in node_ids}
    for e in graph.edges_of(EdgeType.PRECEDENCE):
        if e.source in node_ids and e.target in node_ids:
            adj[e.source].append(e.target)
    return adj


def _has_cycle(node_ids: Set[int], adj: Dict[int, List[int]]) -> bool:
    """Kahn's algorithm: a DAG has a full topological order; a cycle does not."""
    indeg = {n: 0 for n in node_ids}
    for u in node_ids:
        for v in adj[u]:
            indeg[v] += 1
    queue = deque(n for n in node_ids if indeg[n] == 0)
    seen = 0
    while queue:
        u = queue.popleft()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen != len(node_ids)


def is_topological_order(order: Sequence[int], graph: SIRGraph) -> bool:
    """Whether ``order`` respects every precedence edge among its nodes."""
    position = {n: i for i, n in enumerate(order)}
    for e in graph.edges_of(EdgeType.PRECEDENCE):
        if e.source in position and e.target in position:
            if position[e.source] >= position[e.target]:
                return False
    return True


def manual_label_projection(graph: SIRGraph) -> List[int]:
    """Project the SIR to one deterministic manual-label sequence.

    The manual events are topologically sorted by their precedence edges; ties
    are broken by start time then id. The result is a sequence of integer
    lexeme identifiers, not evidence that an authentic gloss annotation exists.
    A linear projection also necessarily omits concurrent and non-manual content.
    """
    violations = validate_sir(graph)
    if violations:
        raise ValueError(f"cannot project invalid SIR: {violations}")
    manual = graph.manual_events()
    event_by_id = {event.id: event for event in manual}
    ids = {e.id for e in manual}
    adj = _precedence_adjacency(graph, ids)
    if _has_cycle(ids, adj):
        raise ValueError("precedence edges among manual events contain a cycle")

    indeg = {n: 0 for n in ids}
    for u in ids:
        for v in adj[u]:
            indeg[v] += 1
    start_of = {e.id: (e.t_start, e.id) for e in manual}
    # ready set ordered by (start time, id) for a deterministic linearisation
    ready = sorted((n for n in ids if indeg[n] == 0), key=lambda n: start_of[n])
    order: List[int] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for v in sorted(adj[n], key=lambda x: start_of[x]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
        ready.sort(key=lambda x: start_of[x])
    return [event_by_id[n].label for n in order]


def gloss_projection(graph: SIRGraph) -> List[int]:
    """Backward-compatible name for :func:`manual_label_projection`.

    The return value must not be stored or represented as authentic gloss unless
    an independently governed human annotation explicitly establishes that
    interpretation.
    """
    return manual_label_projection(graph)


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------
def validate_sir(graph: SIRGraph, num_loci: Optional[int] = None,
                 lexicon: Optional[object] = None) -> List[str]:
    """Return the list of violated structural rules (empty == valid).

    Rules mirror docs/GRAMMAR_SIR.md §2.1.
    """
    violations: List[str] = []
    if not isinstance(graph, SIRGraph):
        return ["invalid_graph_type"]
    if not isinstance(graph.events, list) or not isinstance(graph.edges, list):
        return ["invalid_graph_collections"]
    if num_loci is not None and (
            isinstance(num_loci, bool) or not isinstance(num_loci, int)
            or num_loci < 1):
        violations.append("invalid_num_loci")

    def exact_nonnegative_int(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def finite_number(value: object) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)))

    valid_events: Dict[int, SIREvent] = {}
    for event in graph.events:
        if not isinstance(event, SIREvent):
            violations.append("invalid_event_type")
            continue
        if not exact_nonnegative_int(event.id):
            violations.append("invalid_event_id")
        elif event.id not in valid_events:
            valid_events[event.id] = event
        if not isinstance(event.kind, EventKind):
            violations.append("invalid_event_kind")
        if not exact_nonnegative_int(event.label):
            violations.append("invalid_event_label")
        if not finite_number(event.t_start) or not finite_number(event.t_end) \
                or not float(event.t_start) < float(event.t_end):
            violations.append("invalid_interval")
        if event.referent is not None and not exact_nonnegative_int(event.referent):
            violations.append("invalid_referent")
        if event.locus is not None and not exact_nonnegative_int(event.locus):
            violations.append("invalid_locus")

    ids = {e.id for e in graph.events
           if isinstance(e, SIREvent) and exact_nonnegative_int(e.id)}
    if len(ids) != len(graph.events):
        violations.append("duplicate_event_id")

    # 3. edges reference existing nodes
    for edge in graph.edges:
        if not isinstance(edge, SIREdge):
            violations.append("invalid_edge_type")
            continue
        if not exact_nonnegative_int(edge.source) \
                or not exact_nonnegative_int(edge.target):
            violations.append("invalid_edge_endpoint")
        if not isinstance(edge.type, EdgeType):
            violations.append("invalid_edge_relation")
        if edge.source not in ids or edge.target not in ids:
            violations.append("edge_references_missing_node")
        if edge.source == edge.target:
            violations.append("self_edge")

    valid_edge_keys = [
        (edge.source, edge.target, edge.type)
        for edge in graph.edges
        if isinstance(edge, SIREdge) and isinstance(edge.type, EdgeType)
        and exact_nonnegative_int(edge.source) and exact_nonnegative_int(edge.target)
    ]
    if len(valid_edge_keys) != len(set(valid_edge_keys)):
        violations.append("duplicate_edge")

    # 2. precedence is acyclic (over ALL events, not just manual)
    all_ids = set(ids)
    adj: Dict[int, List[int]] = {n: [] for n in all_ids}
    for edge in graph.edges:
        if isinstance(edge, SIREdge) and edge.type is EdgeType.PRECEDENCE \
                and edge.source in all_ids and edge.target in all_ids:
            adj[edge.source].append(edge.target)
    if _has_cycle(all_ids, adj):
        violations.append("precedence_cycle")

    # Edge types are temporal claims, not decorative relation labels.
    for edge in graph.edges:
        if not isinstance(edge, SIREdge) or not isinstance(edge.type, EdgeType) \
                or edge.source not in valid_events or edge.target not in valid_events:
            continue
        source, target = valid_events[edge.source], valid_events[edge.target]
        if not all(finite_number(value) for value in (
                source.t_start, source.t_end, target.t_start, target.t_end)):
            continue
        if edge.type is EdgeType.PRECEDENCE and source.t_end > target.t_start:
            violations.append("precedence_time_contradiction")
        elif edge.type is EdgeType.OVERLAP and not source.overlaps_time(target):
            violations.append("overlap_time_contradiction")
        elif edge.type is EdgeType.SCOPE and not (
                source.t_start <= target.t_start and source.t_end >= target.t_end):
            violations.append("scope_time_contradiction")

    # 4. scope edges: non-manual source, manual target
    for edge in graph.edges:
        if isinstance(edge, SIREdge) and edge.type is EdgeType.SCOPE \
                and edge.source in valid_events and edge.target in valid_events:
            source, target = valid_events[edge.source], valid_events[edge.target]
            if isinstance(source.kind, EventKind) and source.kind.is_manual:
                violations.append("scope_source_not_nonmanual")
            if isinstance(target.kind, EventKind) and not target.kind.is_manual:
                violations.append("scope_target_not_manual")

    # 5. coref events share a referent
    for edge in graph.edges:
        if isinstance(edge, SIREdge) and edge.type is EdgeType.COREF \
                and edge.source in valid_events and edge.target in valid_events:
            a, b = valid_events[edge.source], valid_events[edge.target]
            if a.referent is None or b.referent is None or a.referent != b.referent:
                violations.append("coref_referent_mismatch")

    # 6. loci in range and distinct per referent
    if isinstance(num_loci, int) and not isinstance(num_loci, bool) and num_loci >= 1:
        placed: Dict[int, int] = {}          # locus -> referent
        for e in graph.events:
            if isinstance(e, SIREvent) and exact_nonnegative_int(e.locus):
                locus = cast(int, e.locus)
                if not 0 <= locus < num_loci:
                    violations.append("locus_out_of_range")
                ref = cast(int, e.referent) \
                    if exact_nonnegative_int(e.referent) else None
                if locus in placed and ref is not None and placed[locus] != ref:
                    violations.append("locus_collision")
                elif ref is not None:
                    placed[locus] = ref

    # 7. hallucination rule (manual events in the lexicon or fingerspelled)
    if lexicon is not None:
        for e in graph.events:
            if not isinstance(e, SIREvent) or not isinstance(e.kind, EventKind) \
                    or not e.kind.is_manual or not exact_nonnegative_int(e.label):
                continue
            if e.kind is EventKind.FINGERSPELL:
                continue
            in_lex = (lexicon.contains(e.label) if hasattr(lexicon, "contains")
                      else e.label in cast(Any, lexicon))
            if not in_lex:
                violations.append("hallucinated_manual_event")

    seen, uniq = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq


def sir_to_dict(graph: SIRGraph) -> Dict[str, Any]:
    """Return a canonical, order-independent JSON representation of a valid SIR.

    Serialization fails closed: malformed or temporally contradictory graphs are
    never assigned a content identity.
    """
    violations = validate_sir(graph)
    if violations:
        raise ValueError(f"cannot serialize invalid SIR: {violations}")
    events = sorted(graph.events, key=lambda event: event.id)
    edges = sorted(graph.edges, key=lambda edge: (
        edge.source, edge.target, edge.type.value))
    return {
        "schema_version": SIR_SCHEMA_VERSION,
        "events": [{
            "id": event.id,
            "kind": event.kind.value,
            "label": event.label,
            "t_start": float(event.t_start),
            "t_end": float(event.t_end),
            "referent": event.referent,
            "locus": event.locus,
        } for event in events],
        "edges": [{
            "source": edge.source,
            "target": edge.target,
            "type": edge.type.value,
        } for edge in edges],
    }


def sir_from_dict(value: Mapping[str, Any]) -> SIRGraph:
    """Parse the exact canonical SIR schema and reject unknown or lossy fields."""
    if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "events", "edges"}:
        raise ValueError("SIR fields must be exactly schema_version, events, and edges")
    if value["schema_version"] != SIR_SCHEMA_VERSION:
        raise ValueError("unsupported SIR schema version")
    if not isinstance(value["events"], list) or not isinstance(value["edges"], list):
        raise ValueError("SIR events and edges must be lists")
    events: List[SIREvent] = []
    for item in value["events"]:
        required = {"id", "kind", "label", "t_start", "t_end", "referent", "locus"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(f"SIR event fields must be exactly {sorted(required)}")
        try:
            kind = EventKind(item["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown SIR event kind") from error
        events.append(SIREvent(
            id=item["id"], kind=kind, label=item["label"],
            t_start=item["t_start"], t_end=item["t_end"],
            referent=item["referent"], locus=item["locus"],
        ))
    edges: List[SIREdge] = []
    for item in value["edges"]:
        required = {"source", "target", "type"}
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(f"SIR edge fields must be exactly {sorted(required)}")
        try:
            edge_type = EdgeType(item["type"])
        except (TypeError, ValueError) as error:
            raise ValueError("unknown SIR edge type") from error
        edges.append(SIREdge(source=item["source"], target=item["target"],
                             type=edge_type))
    graph = SIRGraph(events=events, edges=edges)
    violations = validate_sir(graph)
    if violations:
        raise ValueError(f"invalid SIR payload: {violations}")
    return graph


def sir_sha256(graph: SIRGraph) -> str:
    """SHA-256 identity of a valid SIR graph, independent of list ordering."""
    payload = json.dumps(
        sir_to_dict(graph), ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
