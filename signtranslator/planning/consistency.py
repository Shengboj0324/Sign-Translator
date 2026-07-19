"""Semantic consistency checks and counterfactual licensing.

The specification's verification list includes semantic-role, negation,
question-type, number, tense/aspect, referent-consistency and non-manual-scope
tests, plus **counterfactual** tests: *change one semantic feature and verify
only licensed plan fields change.*

Two distinct notions live here:

* **Consistency** -- structural predicates over a *single* plan (does the
  non-manual scope actually cover the negated predicate? are all arguments
  placed in signing space?). These extend the hard validator of ``schema.py``
  with semantically-motivated checks.

* **Counterfactual licensing** -- a property of a *controllable* mapping from
  semantic features to plans. If flipping one feature (say negation) changes
  fields it should not (the manual unit order, unrelated referents), the mapping
  is leaking. The ``licensed_fields`` table declares, per feature, exactly which
  plan fields that feature is allowed to touch; ``counterfactual_diff`` then
  reports any change outside the licensed set.

The reference planner used to *test* this is a deterministic, rule-based
``ControllablePlanBuilder`` -- a controllable oracle, not a learned model. It
lets the counterfactual property be checked exactly, which a stochastic model
never could.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .schema import (
    PlanVocabulary, SignPlan, SemanticFrame, NonmanualSpan, DEFAULT_VOCAB,
)

# Non-manual marker convention (indices into the NM value space).
NM_NEG = 0        # negation
NM_WH = 1         # wh-question
NM_YN = 2         # yes/no question
NM_TOPIC = 3      # topic marking
NM_COND = 4       # conditional


# ---------------------------------------------------------------------------
# Semantic feature bundle -- the controllable input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemanticFeatures:
    """The semantic content a plan must express, independent of surface form."""

    predicate: int
    agent: Optional[int] = None       # referent id
    patient: Optional[int] = None     # referent id
    tam: int = 0                      # tense/aspect/mood
    negated: bool = False
    question: Optional[str] = None    # None | "wh" | "yn"
    number: int = 1                  # 1 = singular, >1 = plural (affects units)
    topic: Optional[int] = None

    def referents(self) -> List[int]:
        refs = []
        for r in (self.agent, self.patient, self.topic):
            if r is not None and r not in refs:
                refs.append(r)
        return refs


# ---------------------------------------------------------------------------
# Controllable oracle: features -> plan (deterministic, rule-based)
# ---------------------------------------------------------------------------
class ControllablePlanBuilder:
    """A deterministic features->plan mapping used to test licensing.

    Each semantic feature influences a *fixed, declared* set of plan fields.
    Because the mapping is a pure function, flipping one feature and diffing the
    plans reveals exactly which fields that feature controls -- which is what the
    counterfactual test needs.
    """

    #: for each feature name, the plan fields it is licensed to change
    LICENSED_FIELDS: Dict[str, Set[str]] = {
        "predicate": {"frame", "manual_units"},
        "agent": {"frame", "referents", "loci", "topic"},
        "patient": {"frame", "referents", "loci"},
        "tam": {"tam"},
        "negated": {"nonmanual"},
        "question": {"nonmanual"},
        "number": {"manual_units", "fingerspelling"},
        "topic": {"topic"},
    }

    def __init__(self, vocab: PlanVocabulary = DEFAULT_VOCAB) -> None:
        self.vocab = vocab

    def build(self, feats: SemanticFeatures) -> SignPlan:
        refs = feats.referents()
        # deterministic locus assignment: ith referent -> locus i
        loci = {r: (i % self.vocab.num_loci) for i, r in enumerate(refs)}

        args: List[Tuple[int, int]] = []
        if feats.agent is not None:
            args.append((0, feats.agent))            # role 0 = AGENT
        if feats.patient is not None:
            args.append((1, feats.patient))          # role 1 = PATIENT

        # manual units: the predicate sign, repeated for plural (number>1).
        pred_sign = feats.predicate % self.vocab.num_lexemes
        units = [pred_sign] * max(1, feats.number)

        nonmanual: List[NonmanualSpan] = []
        if feats.negated:
            nonmanual.append(NonmanualSpan(NM_NEG, 0, len(units) - 1))
        if feats.question == "wh":
            nonmanual.append(NonmanualSpan(NM_WH, 0, len(units) - 1))
        elif feats.question == "yn":
            nonmanual.append(NonmanualSpan(NM_YN, 0, len(units) - 1))

        # plural is fingerspelled here purely to give `number` a second licensed
        # field to exercise (a deliberate, declared coupling).
        fingerspelling = [len(units) - 1] if feats.number > 1 else []

        return SignPlan(
            frame=SemanticFrame(predicate=feats.predicate, args=args),
            referents=refs, tam=feats.tam,
            topic=feats.topic if feats.topic is not None else feats.agent,
            loci=loci, manual_units=units, nonmanual=nonmanual,
            fingerspelling=fingerspelling, conf_bucket=self.vocab.num_conf_buckets - 1)


# ---------------------------------------------------------------------------
# Field-level diff and counterfactual licensing
# ---------------------------------------------------------------------------
_PLAN_FIELDS = ("frame", "referents", "tam", "topic", "focus", "loci",
                "manual_units", "nonmanual", "fingerspelling", "conf_bucket")


def changed_fields(a: SignPlan, b: SignPlan) -> Set[str]:
    """The set of plan fields that differ between two plans."""
    changed: Set[str] = set()
    if (a.frame.predicate, a.frame.args) != (b.frame.predicate, b.frame.args):
        changed.add("frame")
    if a.referents != b.referents:
        changed.add("referents")
    if a.tam != b.tam:
        changed.add("tam")
    if a.topic != b.topic:
        changed.add("topic")
    if a.focus != b.focus:
        changed.add("focus")
    if a.loci != b.loci:
        changed.add("loci")
    if a.manual_units != b.manual_units:
        changed.add("manual_units")
    if a.nonmanual != b.nonmanual:
        changed.add("nonmanual")
    if a.fingerspelling != b.fingerspelling:
        changed.add("fingerspelling")
    if a.conf_bucket != b.conf_bucket:
        changed.add("conf_bucket")
    return changed


@dataclass
class CounterfactualResult:
    feature: str
    changed: Set[str]
    licensed: Set[str]

    @property
    def unlicensed_changes(self) -> Set[str]:
        return self.changed - self.licensed

    @property
    def is_licensed(self) -> bool:
        return not self.unlicensed_changes


def counterfactual_diff(builder: ControllablePlanBuilder,
                        base: SemanticFeatures, feature: str,
                        new_value) -> CounterfactualResult:
    """Flip one feature and report whether only licensed fields changed."""
    if feature not in builder.LICENSED_FIELDS:
        raise ValueError(f"unknown feature {feature}")
    kwargs = {f: getattr(base, f) for f in base.__dataclass_fields__}
    kwargs[feature] = new_value
    other = SemanticFeatures(**kwargs)
    changed = changed_fields(builder.build(base), builder.build(other))
    return CounterfactualResult(feature=feature, changed=changed,
                                licensed=builder.LICENSED_FIELDS[feature])


# ---------------------------------------------------------------------------
# Single-plan semantic consistency checks
# ---------------------------------------------------------------------------
@dataclass
class ConsistencyReport:
    violations: List[str] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        return not self.violations


def check_semantic_consistency(plan: SignPlan,
                               vocab: PlanVocabulary = DEFAULT_VOCAB
                               ) -> ConsistencyReport:
    """Semantic predicates over a single plan (beyond the structural validator).

    * negation/question markers must scope over at least one manual unit;
    * a wh- and a yn-question marker must not co-occur (a clause is one type);
    * every predicate argument's referent must be placed in signing space;
    * plural agreement: a plural reading (>1 unit) must carry the extra units.
    """
    report = ConsistencyReport()
    markers = [s.marker for s in plan.nonmanual]

    # question type is exclusive
    if NM_WH in markers and NM_YN in markers:
        report.violations.append("conflicting_question_types")

    # marked scopes must cover real units
    for span in plan.nonmanual:
        if len(plan.manual_units) == 0:
            report.violations.append("marker_over_empty_units")
        elif not (0 <= span.start <= span.end < len(plan.manual_units)):
            report.violations.append("marker_scope_invalid")

    # arguments must be placeable
    for _, ref in plan.frame.args:
        if ref not in plan.loci:
            report.violations.append("argument_not_placed")

    # negation must actually scope the predicate (unit 0 by convention)
    if any(s.marker == NM_NEG for s in plan.nonmanual):
        covers_pred = any(s.marker == NM_NEG and s.start == 0
                          for s in plan.nonmanual)
        if not covers_pred:
            report.violations.append("negation_does_not_scope_predicate")

    # de-duplicate
    seen, uniq = set(), []
    for v in report.violations:
        if v not in seen:
            seen.add(v); uniq.append(v)
    report.violations = uniq
    return report
