"""The Structured Intermediate Representation: a temporal graph of sign events.

Nodes are manual or non-manual *events*, each with a half-open time interval
``[t_start, t_end)`` (``t_start < t_end``). Edges are typed: precedence, overlap,
scope, co-reference, and spatial locus. A gloss is one topological projection of
the manual sub-graph (docs/GRAMMAR_SIR.md §2).

The graph is validated structurally rule-by-rule; the gloss projection is a
topological sort of the manual events' precedence DAG.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple


class EventKind(Enum):
    MANUAL = "manual"          # a lexical sign
    CLASSIFIER = "classifier"  # a depicting / classifier construction (manual)
    FINGERSPELL = "fingerspell"  # a fingerspelled item (manual)
    NONMANUAL = "nonmanual"    # a facial / body marker

    @property
    def is_manual(self) -> bool:
        return self is not EventKind.NONMANUAL


class EdgeType(Enum):
    PRECEDENCE = "precedence"  # source strictly before target (Allen before)
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
# Topological ordering / gloss projection
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


def gloss_projection(graph: SIRGraph) -> List[int]:
    """Project the SIR to one gloss token sequence (lexeme labels).

    The manual events are topologically sorted by their precedence edges; ties
    are broken by start time then id, so the projection is deterministic. This
    is *one* observed gloss -- the graph carries strictly more (timing, overlap,
    non-manual scope) than any single linearisation.
    """
    manual = graph.manual_events()
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
    return [graph.event(n).label for n in order]


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------
def validate_sir(graph: SIRGraph, num_loci: Optional[int] = None,
                 lexicon: Optional[object] = None) -> List[str]:
    """Return the list of violated structural rules (empty == valid).

    Rules mirror docs/GRAMMAR_SIR.md §2.1.
    """
    violations: List[str] = []
    ids = {e.id for e in graph.events}
    if len(ids) != len(graph.events):
        violations.append("duplicate_event_id")

    # 1. interval validity
    for e in graph.events:
        if not (e.t_start < e.t_end):
            violations.append("invalid_interval")

    # 3. edges reference existing nodes
    for edge in graph.edges:
        if edge.source not in ids or edge.target not in ids:
            violations.append("edge_references_missing_node")

    # 2. precedence is acyclic (over ALL events, not just manual)
    all_ids = set(ids)
    adj = {n: [] for n in all_ids}
    for edge in graph.edges_of(EdgeType.PRECEDENCE):
        if edge.source in all_ids and edge.target in all_ids:
            adj[edge.source].append(edge.target)
    if _has_cycle(all_ids, adj):
        violations.append("precedence_cycle")

    # 4. scope edges: non-manual source, manual target
    for edge in graph.edges_of(EdgeType.SCOPE):
        if edge.source in graph._by_id and edge.target in graph._by_id:
            if graph.event(edge.source).kind.is_manual:
                violations.append("scope_source_not_nonmanual")
            if not graph.event(edge.target).kind.is_manual:
                violations.append("scope_target_not_manual")

    # 5. coref events share a referent
    for edge in graph.edges_of(EdgeType.COREF):
        if edge.source in graph._by_id and edge.target in graph._by_id:
            a, b = graph.event(edge.source), graph.event(edge.target)
            if a.referent is None or b.referent is None or a.referent != b.referent:
                violations.append("coref_referent_mismatch")

    # 6. loci in range and distinct per referent
    if num_loci is not None:
        placed: Dict[int, int] = {}          # locus -> referent
        for e in graph.events:
            if e.locus is not None:
                if not 0 <= e.locus < num_loci:
                    violations.append("locus_out_of_range")
                ref = e.referent
                if e.locus in placed and ref is not None and placed[e.locus] != ref:
                    violations.append("locus_collision")
                elif ref is not None:
                    placed[e.locus] = ref

    # 7. hallucination rule (manual events in the lexicon or fingerspelled)
    if lexicon is not None:
        for e in graph.manual_events():
            if e.kind is EventKind.FINGERSPELL:
                continue
            in_lex = (lexicon.contains(e.label) if hasattr(lexicon, "contains")
                      else e.label in lexicon)
            if not in_lex:
                violations.append("hallucinated_manual_event")

    seen, uniq = set(), []
    for v in violations:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq
