"""Verification of the ASL grammar battery: minimal pairs, locus persistence, OOV.

The document requires minimal-pair tests for the core ASL constructions. The
decisive property is *minimality*: flipping exactly one grammatical feature must
change only the SIR fields that feature licenses, and nothing else. Because the
builder is a pure function of features, we can prove this by construction rather
than hope a model learned it.
"""

import pytest

from signtranslator.grammar.sir import EventKind
from signtranslator.grammar.grammar_tests import (
    Aspect, QuestionType, GrammarFeatures, ControllableASLBuilder,
    LICENSED, minimal_pair, changed_sir_fields, build_discourse,
    locus_of_referent, realise_with_fingerspelling,
    NM_NEG, NM_WH, NM_YN, NM_TOPIC, NM_COND, NM_ROLESHIFT,
)


@pytest.fixture
def builder():
    return ControllableASLBuilder()


def _base():
    # "GIRL LIKE BOY": subject=1, object=2, predicate=0
    return GrammarFeatures(predicate=0, subject=1, object=2)


# ---------------------------------------------------------------------------
# Minimal pairs: one feature flip changes only its licensed fields
# ---------------------------------------------------------------------------
def test_negation_minimal_pair(builder):
    r = minimal_pair(builder, _base(), "negated", True)
    assert r.is_licensed, r.unlicensed_changes
    assert "nonmanual" in r.changed                       # it DID do something


def test_negation_adds_exactly_the_neg_marker(builder):
    g = builder.build(GrammarFeatures(predicate=0, subject=1, object=2,
                                      negated=True))
    nm = {e.label for e in g.nonmanual_events()}
    assert nm == {NM_NEG}


def test_yesno_vs_wh_are_distinct_markers(builder):
    yn = builder.build(GrammarFeatures(predicate=0, subject=1,
                                       question=QuestionType.YESNO))
    wh = builder.build(GrammarFeatures(predicate=0, subject=1,
                                       question=QuestionType.WH))
    assert {e.label for e in yn.nonmanual_events()} == {NM_YN}
    assert {e.label for e in wh.nonmanual_events()} == {NM_WH}
    # question type only touches the non-manual channel
    r = minimal_pair(builder, GrammarFeatures(predicate=0, subject=1,
                                              question=QuestionType.YESNO),
                     "question", QuestionType.WH)
    assert r.changed == {"nonmanual"}


def test_topicalization_reorders_and_marks(builder):
    r = minimal_pair(builder, _base(), "topicalized", True)
    assert r.is_licensed, r.unlicensed_changes
    assert r.changed == {"order", "nonmanual"}
    # object is now first
    g = builder.build(GrammarFeatures(predicate=0, subject=1, object=2,
                                      topicalized=True))
    manual = sorted(g.manual_events(), key=lambda e: e.t_start)
    assert manual[0].referent == 2                        # object fronted
    assert NM_TOPIC in {e.label for e in g.nonmanual_events()}


def test_conditional_minimal_pair(builder):
    r = minimal_pair(builder, _base(), "conditional", True)
    assert r.is_licensed, r.unlicensed_changes
    assert r.changed == {"nonmanual"}


def test_aspect_changes_timing_not_order(builder):
    r = minimal_pair(builder, _base(), "aspect", Aspect.CONTINUATIVE)
    assert r.is_licensed, r.unlicensed_changes
    assert "durations" in r.changed
    assert "order" not in r.changed                       # aspect is not reordering
    # continuative lengthens the predicate
    plain = builder.build(_base())
    cont = builder.build(GrammarFeatures(predicate=0, subject=1, object=2,
                                         aspect=Aspect.CONTINUATIVE))
    pred_plain = max(plain.manual_events(), key=lambda e: e.t_start)
    pred_cont = max(cont.manual_events(), key=lambda e: e.t_start)
    assert pred_cont.duration > pred_plain.duration


def test_plural_adds_morphology_only(builder):
    r = minimal_pair(builder, _base(), "plural_subject", True)
    assert r.is_licensed, r.unlicensed_changes
    assert "manual_labels" in r.changed
    assert "nonmanual" not in r.changed


def test_role_shift_minimal_pair(builder):
    r = minimal_pair(builder, _base(), "role_shift", True)
    assert r.is_licensed, r.unlicensed_changes
    assert NM_ROLESHIFT in {e.label
                            for e in builder.build(
                                GrammarFeatures(predicate=0, subject=1, object=2,
                                                role_shift=True)).nonmanual_events()}


def test_every_feature_flip_is_licensed(builder):
    """Sweep: no feature may leak changes outside its licensed set."""
    base = _base()
    flips = [
        ("negated", True), ("question", QuestionType.WH),
        ("question", QuestionType.YESNO), ("topicalized", True),
        ("conditional", True), ("aspect", Aspect.CONTINUATIVE),
        ("plural_subject", True), ("role_shift", True),
    ]
    for feat, val in flips:
        r = minimal_pair(builder, base, feat, val)
        assert r.is_licensed, f"{feat}->{val} leaked {r.unlicensed_changes}"


def test_a_feature_flip_actually_changes_something(builder):
    """Guard against a vacuous 'licensed' pass: each flip must change >=1 field."""
    base = _base()
    for feat, val in [("negated", True), ("question", QuestionType.WH),
                      ("topicalized", True), ("conditional", True),
                      ("aspect", Aspect.CONTINUATIVE), ("plural_subject", True),
                      ("role_shift", True)]:
        r = minimal_pair(builder, base, feat, val)
        assert r.changed, f"{feat}->{val} changed nothing"


# ---------------------------------------------------------------------------
# Spatial-locus persistence across discourse
# ---------------------------------------------------------------------------
def test_locus_persists_across_sentences(builder):
    """A referent keeps its spatial locus across a multi-sentence discourse."""
    s1 = GrammarFeatures(predicate=0, subject=1, object=2)   # introduce 1 and 2
    s2 = GrammarFeatures(predicate=3, subject=2)             # reuse 2
    s3 = GrammarFeatures(predicate=4, subject=1, object=2)   # reuse both
    graphs = build_discourse(builder, [s1, s2, s3])
    loc1 = locus_of_referent(graphs[0], 1)
    loc2 = locus_of_referent(graphs[0], 2)
    assert loc1 is not None and loc2 is not None and loc1 != loc2
    # same loci reappear later
    assert locus_of_referent(graphs[1], 2) == loc2
    assert locus_of_referent(graphs[2], 1) == loc1
    assert locus_of_referent(graphs[2], 2) == loc2


def test_distinct_referents_get_distinct_loci(builder):
    s = GrammarFeatures(predicate=0, subject=1, object=2)
    g = build_discourse(builder, [s])[0]
    assert locus_of_referent(g, 1) != locus_of_referent(g, 2)


# ---------------------------------------------------------------------------
# OOV coverage via fingerspelling
# ---------------------------------------------------------------------------
class _Lexicon:
    def __init__(self, known):
        self._known = set(known)

    def contains(self, label):
        return label in self._known


def test_oov_tokens_become_fingerspelling():
    lex = _Lexicon({10, 11, 12})
    g = realise_with_fingerspelling([10, 999, 11], lex)   # 999 is OOV (a name)
    kinds = {e.label: e.kind for e in g.events}
    assert kinds[10] == EventKind.MANUAL
    assert kinds[999] == EventKind.FINGERSPELL             # covered, not hallucinated
    assert kinds[11] == EventKind.MANUAL


def test_every_token_is_covered_none_dropped():
    lex = _Lexicon({1})
    labels = [1, 2, 3, 4]
    g = realise_with_fingerspelling(labels, lex)
    assert [e.label for e in g.events] == labels          # nothing silently dropped
    assert sum(1 for e in g.events if e.kind == EventKind.FINGERSPELL) == 3


def test_fingerspelled_sequence_is_temporally_ordered():
    lex = _Lexicon(set())
    g = realise_with_fingerspelling([100, 101, 102], lex)
    starts = [e.t_start for e in sorted(g.events, key=lambda e: e.id)]
    assert starts == sorted(starts)                       # monotone, no overlap gaps
    assert all(e.kind == EventKind.FINGERSPELL for e in g.events)
