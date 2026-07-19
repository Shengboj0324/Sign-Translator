"""Controllable ASL grammar builder for the required minimal-pair tests.

The document requires minimal pairs for negation, yes/no vs WH questions,
topicalization, conditionals, aspect, plural reference, and role shift; plus
spatial-locus persistence across multi-sentence discourse and OOV coverage via
fingerspelling.

As in the doc-02 counterfactual work, the *reference* builder is a deterministic,
rule-based oracle: a pure function from grammatical features to an SIR. Because it
is a function, flipping exactly one feature and diffing the SIRs reveals precisely
which SIR fields that feature licenses to change -- the minimal-pair property.
A learned model can never establish this exactly; a controllable oracle can.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .sir import EventKind, EdgeType, SIREvent, SIREdge, SIRGraph
from .nonmanual import MarkerSpan

# Non-manual marker ids (ASL grammatical markers).
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
    """Grammatical content to be realised as ASL, independent of surface form."""

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


#: for each feature, the SIR fields it is licensed to change (a minimal pair
#: flipping this feature must change nothing outside this set).
LICENSED: Dict[str, Set[str]] = {
    "predicate": {"manual_labels"},
    "subject": {"referents", "loci", "manual_labels", "edges"},
    "object": {"referents", "loci", "manual_labels", "edges"},
    "negated": {"nonmanual"},
    "question": {"nonmanual"},
    "topicalized": {"order", "nonmanual"},       # fronting reorders + topic marker
    "conditional": {"nonmanual"},
    "aspect": {"durations"},                     # aspect changes timing, not order
    "plural_subject": {"manual_labels"},         # plural morphology on the sign
    "role_shift": {"nonmanual", "loci"},         # body shift + spatial referencing
}

# Lexeme id conventions for the toy ASL lexicon.
_LEX_BASE = 10           # predicate signs start here
_LEX_REF = 20            # referent-naming signs
_LEX_PLURAL_INFLECT = 100  # offset selecting the plural-inflected lexeme


class ControllableASLBuilder:
    """Deterministic GrammarFeatures -> SIRGraph."""

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

        # --- manual event order (ASL topic-comment; object fronts if topicalized)
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
                # Plural = morphological inflection of the subject sign
                # (reduplication selects a plural-marked lexeme). Modelled as a
                # distinct lexeme id, so it changes the LABEL only -- no extra
                # event, no reorder, no timing change.
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
    """Realise a multi-sentence discourse with PERSISTENT referent loci.

    A referent introduced in sentence 1 keeps its spatial locus in later
    sentences -- a core ASL discourse property. Loci are assigned once, in order
    of first mention, and reused.
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
    """Realise a token sequence, fingerspelling any out-of-lexicon item.

    Every token becomes a manual event; a token absent from the lexicon is a
    FINGERSPELL event rather than a (hallucinated) lexical sign, so names and OOV
    terms are always covered.
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
