"""Synthetic temporal-graph fixtures for implementation-level perturbation tests.

The historical specification names perturbation categories such as negation,
question type, topicalization, aspect, reference, and role shift. Here they are
only fixture fields used to exercise typed graph behavior.

The builder is a deterministic pure function, so flipping one input field and
diffing the outputs reveals exactly which implementation fields changed. It is
not a linguistic oracle, a reference ASL grammar, or human annotation evidence.
Its hard-coded mappings are usable only for software invariants until qualified
ASL reviewers validate or replace them under a governed convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .sir import EventKind, EdgeType, SIREvent, SIREdge, SIRGraph
from .nonmanual import MarkerSpan

# Fixture marker ids; qualified-ASL meaning has not been established.
NM_NEG = 0
NM_WH = 1
NM_YN = 2
NM_TOPIC = 3
NM_COND = 4
NM_ROLESHIFT = 5


class Aspect(Enum):
    NONE = "none"
    CONTINUATIVE = "continuative"   # reduplicated / lengthened
    ITERATIVE = "iterative"


class QuestionType(Enum):
    NONE = "none"
    YESNO = "yesno"
    WH = "wh"


@dataclass(frozen=True)
class GrammarFeatures:
    """Synthetic input fields for deterministic graph construction."""

    predicate: int
    subject: Optional[int] = None      # referent id
    object: Optional[int] = None       # referent id
    negated: bool = False
    question: QuestionType = QuestionType.NONE
    topicalized: bool = False          # object fronted as topic
    conditional: bool = False
    aspect: Aspect = Aspect.NONE
    plural_subject: bool = False
    role_shift: bool = False           # constructed action / reported speech

    def referents(self) -> List[int]:
        out: List[int] = []
        for r in (self.subject, self.object):
            if r is not None and r not in out:
                out.append(r)
        return out


#: Implementation-local perturbation contract. This table is not evidence that
#: the listed changes are linguistically complete or correct for ASL.
LICENSED: Dict[str, Set[str]] = {
    "predicate": {"manual_labels"},
    "subject": {"referents", "loci", "manual_labels", "edges"},
    "object": {"referents", "loci", "manual_labels", "edges"},
    "negated": {"nonmanual"},
    "question": {"nonmanual"},
    "topicalized": {"order", "nonmanual"},       # fronting reorders + topic marker
    "conditional": {"nonmanual"},
    "aspect": {"durations"},                     # fixture duration perturbation
    "plural_subject": {"manual_labels"},         # fixture label perturbation
    "role_shift": {"nonmanual", "loci"},         # fixture marker/locus perturbation
}

# Arbitrary fixture identifiers; these are not entries in a validated ASL lexicon.
_LEX_BASE = 10           # predicate signs start here
_LEX_REF = 20            # referent-naming signs
_LEX_PLURAL_INFLECT = 100  # offset selecting the plural-inflected lexeme


class ControllableASLBuilder:
    """Historical API name for a deterministic synthetic SIR fixture builder."""

    def __init__(self, num_loci: int = 7, unit_dur: float = 1.0,
                 gap: float = 0.0) -> None:
        self.num_loci = num_loci
        self.unit_dur = unit_dur
        self.gap = gap

    def build(self, feats: GrammarFeatures,
              locus_assignment: Optional[Dict[int, int]] = None) -> SIRGraph:
        """Realise features as an SIR. ``locus_assignment`` pins referent loci
        (for discourse persistence); otherwise loci are assigned deterministically.
        """
        refs = feats.referents()
        if locus_assignment is None:
            loci = {r: (i % self.num_loci) for i, r in enumerate(refs)}
        else:
            loci = dict(locus_assignment)

        # --- declared fixture order; this is not a validated ASL ordering rule
        order: List[Tuple[str, int]] = []      # (role, referent-or-None)
        if feats.topicalized and feats.object is not None:
            order.append(("object", feats.object))
            if feats.subject is not None:
                order.append(("subject", feats.subject))
        else:
            if feats.subject is not None:
                order.append(("subject", feats.subject))
            if feats.object is not None:
                order.append(("object", feats.object))
        order.append(("predicate", None))

        events: List[SIREvent] = []
        cursor = 0.0
        eid = 0
        pred_event_id = None
        for role, ref in order:
            dur = self.unit_dur
            if role == "predicate" and feats.aspect is Aspect.CONTINUATIVE:
                dur = self.unit_dur * 2.0          # lengthened
            if role == "predicate":
                label = _LEX_BASE + feats.predicate
            else:
                label = _LEX_REF + (ref if ref is not None else 0)
                # A distinct fixture label exercises a label-only perturbation;
                # it does not claim a valid ASL plural realization.
                if (role == "subject" and feats.plural_subject
                        and ref is not None):
                    label += _LEX_PLURAL_INFLECT
            ev = SIREvent(id=eid, kind=EventKind.MANUAL, label=label,
                          t_start=cursor, t_end=cursor + dur,
                          referent=ref, locus=loci.get(ref) if ref is not None else None)
            events.append(ev)
            if role == "predicate":
                pred_event_id = eid
            cursor += dur + self.gap
            eid += 1

        # --- precedence edges follow the linear manual order
        manual_ids = [e.id for e in events]
        edges: List[SIREdge] = [SIREdge(manual_ids[i], manual_ids[i + 1],
                                        EdgeType.PRECEDENCE)
                                for i in range(len(manual_ids) - 1)]

        span_start = events[0].t_start
        span_end = events[-1].t_end

        # --- non-manual markers (scoping the whole clause unless noted)
        def add_nm(marker: int, s: float, e: float, target_id: int) -> None:
            nonlocal eid
            events.append(SIREvent(id=eid, kind=EventKind.NONMANUAL, label=marker,
                                   t_start=s, t_end=e))
            edges.append(SIREdge(eid, target_id, EdgeType.SCOPE))
            eid += 1

        pred = pred_event_id
        if feats.negated:
            add_nm(NM_NEG, events[pred].t_start - 0.01, span_end + 0.01, pred)
        if feats.question is QuestionType.WH:
            add_nm(NM_WH, span_start - 0.01, span_end + 0.01, pred)
        elif feats.question is QuestionType.YESNO:
            add_nm(NM_YN, span_start - 0.01, span_end + 0.01, pred)
        if feats.topicalized and feats.object is not None:
            add_nm(NM_TOPIC, events[0].t_start - 0.01, events[0].t_end + 0.01,
                   manual_ids[0])
        if feats.conditional:
            add_nm(NM_COND, span_start - 0.01, span_end + 0.01, pred)
        if feats.role_shift:
            add_nm(NM_ROLESHIFT, span_start - 0.01, span_end + 0.01, pred)

        g = SIRGraph(events=events, edges=edges)
        return g


# ---------------------------------------------------------------------------
# Minimal-pair diffing
# ---------------------------------------------------------------------------
def _sir_fields(graph: SIRGraph) -> Dict[str, object]:
    """Extract comparable fields for minimal-pair diffing."""
    manual = graph.manual_events()
    return {
        "manual_labels": tuple(sorted(e.label for e in manual)),
        # Sequencing signature: the time-order of referent SLOTS (predicate = -1),
        # invariant to which lexeme fills a slot. Topicalization changes this
        # (object fronts); relabeling a sign (e.g. plural inflection) does not.
        "order": tuple(e.referent if e.referent is not None else -1
                       for e in sorted(manual, key=lambda x: x.t_start)),
        "referents": tuple(sorted(e.referent for e in manual
                                  if e.referent is not None)),
        "loci": tuple(sorted((e.referent, e.locus) for e in manual
                             if e.locus is not None and e.referent is not None)),
        "nonmanual": tuple(sorted(e.label for e in graph.nonmanual_events())),
        "durations": tuple(round(e.duration, 3)
                           for e in sorted(manual, key=lambda x: x.t_start)),
        # Structural (manual precedence) edges ONLY. Scope/coref/locus edges are
        # consequences of markers and referents and are captured by the
        # "nonmanual"/"loci" fields; folding them in here would conflate a
        # non-manual marker's scope edge with manual sentence structure.
        "edges": tuple(sorted((e.source, e.target)
                              for e in graph.edges
                              if e.type is EdgeType.PRECEDENCE)),
    }


def changed_sir_fields(a: SIRGraph, b: SIRGraph) -> Set[str]:
    fa, fb = _sir_fields(a), _sir_fields(b)
    return {k for k in fa if fa[k] != fb[k]}


@dataclass
class MinimalPairResult:
    feature: str
    changed: Set[str]
    licensed: Set[str]

    @property
    def unlicensed_changes(self) -> Set[str]:
        return self.changed - self.licensed

    @property
    def is_licensed(self) -> bool:
        return not self.unlicensed_changes


def minimal_pair(builder: ControllableASLBuilder, base: GrammarFeatures,
                 feature: str, new_value) -> MinimalPairResult:
    """Flip one feature; report whether only its licensed SIR fields changed."""
    if feature not in LICENSED:
        raise ValueError(f"unknown feature {feature}")
    other = replace(base, **{feature: new_value})
    changed = changed_sir_fields(builder.build(base), builder.build(other))
    return MinimalPairResult(feature=feature, changed=changed,
                             licensed=LICENSED[feature])


# ---------------------------------------------------------------------------
# Spatial-locus persistence across discourse
# ---------------------------------------------------------------------------
def build_discourse(builder: ControllableASLBuilder,
                    sentences: Sequence[GrammarFeatures]) -> List[SIRGraph]:
    """Exercise persistent referent-locus bookkeeping across fixture graphs.

    Loci are assigned once, in order of first mention, and reused. This verifies
    implementation state persistence, not linguistic adequacy.
    """
    assignment: Dict[int, int] = {}
    graphs: List[SIRGraph] = []
    for feats in sentences:
        for r in feats.referents():
            if r not in assignment:
                assignment[r] = len(assignment) % builder.num_loci
        graphs.append(builder.build(feats, locus_assignment=assignment))
    return graphs


def locus_of_referent(graph: SIRGraph, referent: int) -> Optional[int]:
    for e in graph.events:
        if e.referent == referent and e.locus is not None:
            return e.locus
    return None


# ---------------------------------------------------------------------------
# OOV / name coverage via fingerspelling
# ---------------------------------------------------------------------------
def realise_with_fingerspelling(labels: Sequence[int], lexicon,
                                start: float = 0.0, unit: float = 1.0
                                ) -> SIRGraph:
    """Build a fixture sequence, marking out-of-lexicon items as FINGERSPELL.

    This tests the explicit OOV branch. It does not prove that arbitrary integer
    labels have a valid fingerspelled realization.
    """
    events: List[SIREvent] = []
    edges: List[SIREdge] = []
    cursor = start
    for i, lab in enumerate(labels):
        in_lex = lexicon.contains(lab) if hasattr(lexicon, "contains") else (lab in lexicon)
        kind = EventKind.MANUAL if in_lex else EventKind.FINGERSPELL
        events.append(SIREvent(id=i, kind=kind, label=lab,
                               t_start=cursor, t_end=cursor + unit))
        if i > 0:
            edges.append(SIREdge(i - 1, i, EdgeType.PRECEDENCE))
        cursor += unit
    return SIRGraph(events=events, edges=edges)
