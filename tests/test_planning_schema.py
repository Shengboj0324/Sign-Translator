"""Verification of the typed plan schema, serialization, and validator.

Serialize/deserialize is proved to be an exact round-trip, and every structural
rule is triggered by a deliberately-broken plan so a rule that never fires
cannot pass unnoticed.
"""

import random

import pytest

from signtranslator.planning.schema import (
    PlanVocabulary, SignPlan, SemanticFrame, NonmanualSpan,
    serialize_plan, deserialize_plan, validate_plan, DEFAULT_VOCAB,
    DeserializationError, STRUCTURAL, VALUE_KINDS,
)

V = DEFAULT_VOCAB


# ---------------------------------------------------------------------------
# Vocabulary id layout
# ---------------------------------------------------------------------------
def test_token_ids_are_a_contiguous_bijection():
    """Every id in [0, size) decodes to exactly one (kind, value) and back."""
    seen = set()
    for tok in range(V.size):
        kind, value = V.decode(tok)
        assert V.token(kind, value) == tok
        assert tok not in seen
        seen.add(tok)
    assert len(seen) == V.size


def test_structural_and_value_kinds_are_disjoint():
    ids = {}
    for kind in STRUCTURAL:
        ids.setdefault(V.token(kind), []).append(kind)
    for kind in VALUE_KINDS:
        size = {"ROLE": V.num_roles, "REF": V.num_referents, "PRED_V": V.num_predicates,
                "TAM_V": V.num_tam, "LOCUS": V.num_loci, "LEX": V.num_lexemes,
                "NM": V.num_nonmanual, "IDX": V.max_units, "CONF_V": V.num_conf_buckets}[kind]
        for v in range(size):
            ids.setdefault(V.token(kind, v), []).append(f"{kind}[{v}]")
    assert all(len(names) == 1 for names in ids.values()), "token id collision"


def test_token_range_and_kind_validation():
    with pytest.raises(ValueError):
        V.token("BOP", 1)                      # structural takes no value
    with pytest.raises(ValueError):
        V.token("ROLE", V.num_roles)           # out of range
    with pytest.raises(ValueError):
        V.token("NOPE")
    with pytest.raises(ValueError):
        V.decode(-1)
    with pytest.raises(ValueError):
        V.decode(V.size)


def test_confidence_bucket_round_trips():
    for b in range(V.num_conf_buckets):
        assert V.bucket_of_confidence(V.confidence_of_bucket(b)) == b
    assert V.confidence_of_bucket(0) == 0.0
    assert V.confidence_of_bucket(V.num_conf_buckets - 1) == 1.0


def test_vocab_validates_sizes():
    with pytest.raises(ValueError):
        PlanVocabulary(num_predicates=0)
    with pytest.raises(ValueError):
        PlanVocabulary(num_conf_buckets=1)


# ---------------------------------------------------------------------------
# A canonical valid plan
# ---------------------------------------------------------------------------
def _valid_plan():
    return SignPlan(
        frame=SemanticFrame(predicate=2, args=[(0, 1), (1, 3)]),
        referents=[1, 3],
        tam=2,
        topic=1, focus=3,
        loci={1: 2, 3: 4},
        manual_units=[5, 9, 12, 7],
        classifiers=[],
        nonmanual=[NonmanualSpan(marker=0, start=1, end=2)],
        fingerspelling=[3],
        conf_bucket=8,
    )


def test_canonical_plan_is_valid():
    assert validate_plan(_valid_plan()) == []


def _random_valid_plan(rng, vocab=V):
    n_ref = rng.randint(1, min(4, vocab.num_referents))
    refs = rng.sample(range(vocab.num_referents), n_ref)
    loci = dict(zip(refs, rng.sample(range(vocab.num_loci), n_ref)))
    n_units = rng.randint(1, min(6, vocab.max_units))
    units = [rng.randrange(vocab.num_lexemes) for _ in range(n_units)]
    args = [(rng.randrange(vocab.num_roles), rng.choice(refs))
            for _ in range(rng.randint(0, 3))]
    nms = []
    for _ in range(rng.randint(0, 2)):
        i = rng.randrange(n_units)
        j = rng.randrange(i, n_units)
        nms.append(NonmanualSpan(rng.randrange(vocab.num_nonmanual), i, j))
    fs = rng.sample(range(n_units), rng.randint(0, n_units))
    return SignPlan(
        frame=SemanticFrame(rng.randrange(vocab.num_predicates), args),
        referents=refs, tam=rng.randrange(vocab.num_tam),
        topic=rng.choice(refs) if rng.random() < 0.5 else None,
        focus=rng.choice(refs) if rng.random() < 0.5 else None,
        loci=loci, manual_units=units, nonmanual=nms,
        fingerspelling=sorted(fs), conf_bucket=rng.randrange(vocab.num_conf_buckets))


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------
def test_serialize_deserialize_is_identity_on_canonical_plan():
    plan = _valid_plan()
    back = deserialize_plan(serialize_plan(plan))
    assert back.frame.predicate == plan.frame.predicate
    assert back.frame.args == plan.frame.args
    assert back.referents == plan.referents
    assert back.tam == plan.tam
    assert back.topic == plan.topic            # information structure must survive
    assert back.focus == plan.focus
    assert back.classifiers == plan.classifiers
    assert back.loci == plan.loci
    assert back.manual_units == plan.manual_units
    assert back.nonmanual == plan.nonmanual
    assert back.fingerspelling == plan.fingerspelling
    assert back.conf_bucket == plan.conf_bucket


def test_topic_focus_classifiers_round_trip_all_cases():
    """Regression: these three fields were previously dropped by serialization.
    Cover topic-set/focus-none, topic-none/focus-set, and non-empty classifiers."""
    base = dict(frame=SemanticFrame(1, [(0, 1)]), referents=[1, 2],
                manual_units=[3, 4])
    for topic, focus, cls in [(1, None, [7, 9]), (None, 2, []),
                              (1, 2, [5]), (None, None, [])]:
        p = SignPlan(topic=topic, focus=focus, classifiers=cls, **base)
        q = deserialize_plan(serialize_plan(p))
        assert q.topic == topic and q.focus == focus and q.classifiers == cls
        # and re-serialization is an exact fixpoint
        assert serialize_plan(q) == serialize_plan(p)


def test_round_trip_on_many_random_valid_plans():
    rng = random.Random(0)
    for _ in range(500):
        plan = _random_valid_plan(rng)
        tokens = serialize_plan(plan)
        back = deserialize_plan(tokens)
        assert validate_plan(back) == [], f"generator produced invalid plan"
        # re-serializing the deserialized plan gives the identical token stream
        assert serialize_plan(back) == tokens


def test_empty_variable_slots_round_trip():
    """Zero args / refs / units / nms / fs must serialize and parse."""
    plan = SignPlan(frame=SemanticFrame(predicate=0), tam=0, conf_bucket=0)
    tokens = serialize_plan(plan)
    back = deserialize_plan(tokens)
    assert back.frame.args == [] and back.referents == []
    assert back.manual_units == [] and back.nonmanual == []


def test_every_serialized_stream_starts_with_bop_ends_with_eop():
    tokens = serialize_plan(_valid_plan())
    assert V.decode(tokens[0]) == ("BOP", 0)
    assert V.decode(tokens[-1]) == ("EOP", 0)


# ---------------------------------------------------------------------------
# Deserialization rejects malformed skeletons
# ---------------------------------------------------------------------------
def test_deserialize_rejects_missing_bop():
    tokens = serialize_plan(_valid_plan())[1:]        # drop BOP
    with pytest.raises(DeserializationError):
        deserialize_plan(tokens)


def test_deserialize_rejects_truncated_pair():
    """An ARGS role with no following ref is a truncated pair."""
    t = V.token
    tokens = [t("BOP"), t("PRED"), t("PRED_V", 0), t("ARGS"), t("ROLE", 0),
              t("REFS")]                                # ROLE not followed by REF
    with pytest.raises(DeserializationError):
        deserialize_plan(tokens)


def test_deserialize_rejects_trailing_tokens():
    tokens = serialize_plan(_valid_plan()) + [V.token("EOP")]
    with pytest.raises(DeserializationError):
        deserialize_plan(tokens)


def test_deserialize_rejects_wrong_slot_order():
    t = V.token
    tokens = [t("BOP"), t("ARGS")]                     # PRED must come first
    with pytest.raises(DeserializationError):
        deserialize_plan(tokens)


# ---------------------------------------------------------------------------
# Each validation rule fires on a deliberately broken plan
# ---------------------------------------------------------------------------
def test_rule_arg_referent_must_be_declared():
    p = _valid_plan()
    p.frame.args = [(0, 5)]                            # 5 not in referents
    assert "arg_referent_undeclared" in validate_plan(p)


def test_rule_topic_focus_must_be_declared():
    p = _valid_plan(); p.topic = 5
    assert "topic_undeclared" in validate_plan(p)
    p = _valid_plan(); p.focus = 5
    assert "focus_undeclared" in validate_plan(p)


def test_rule_every_referent_needs_a_locus():
    p = _valid_plan(); del p.loci[1]
    assert "referent_without_locus" in validate_plan(p)


def test_rule_loci_must_be_distinct():
    p = _valid_plan(); p.loci = {1: 2, 3: 2}           # both at locus 2
    assert "locus_collision" in validate_plan(p)


def test_rule_locus_for_undeclared_referent():
    p = _valid_plan(); p.loci[4] = 5                   # 4 not declared
    assert "locus_for_undeclared_referent" in validate_plan(p)


def test_rule_nonmanual_scope_within_units():
    p = _valid_plan()
    p.nonmanual = [NonmanualSpan(0, 2, 99)]            # end beyond units
    assert "nonmanual_scope_out_of_bounds" in validate_plan(p)


def test_rule_nonmanual_scope_start_after_end():
    p = _valid_plan()
    p.nonmanual = [NonmanualSpan(0, 3, 1)]             # start > end
    assert "nonmanual_scope_out_of_bounds" in validate_plan(p)


def test_rule_fingerspell_index_in_bounds():
    p = _valid_plan(); p.fingerspelling = [99]
    assert "fingerspell_index_out_of_bounds" in validate_plan(p)


def test_rule_confidence_in_range():
    p = _valid_plan(); p.conf_bucket = 999
    assert "confidence_out_of_range" in validate_plan(p)


def test_rule_lexeme_in_range():
    p = _valid_plan(); p.manual_units = [999]
    assert "lexeme_out_of_range" in validate_plan(p)


def test_rule_predicate_and_tam_ranges():
    p = _valid_plan(); p.frame.predicate = 99
    assert "predicate_out_of_range" in validate_plan(p)
    p = _valid_plan(); p.tam = 99
    assert "tam_out_of_range" in validate_plan(p)


def test_hallucination_rule_requires_a_lexicon():
    """Without a lexicon the hallucination rule is skipped, not silently passed."""
    p = _valid_plan()
    assert "hallucinated_lexical_entry" not in validate_plan(p, lexicon=None)

    class _Lex:
        def __init__(self, entries):
            self.entries = set(entries)
        def contains(self, x):
            return x in self.entries

    # unit 9 present, unit at index 3 (=7) fingerspelled, unit 12 absent+not FS
    lex = _Lex({5, 9, 7})
    viol = validate_plan(p, lexicon=lex)
    assert "hallucinated_lexical_entry" in viol           # unit 12 hallucinated
    # if 12 is added to the lexicon, the rule clears
    lex.entries.add(12)
    assert "hallucinated_lexical_entry" not in validate_plan(p, lexicon=lex)


def test_validation_reports_each_rule_at_most_once():
    p = _valid_plan()
    p.frame.args = [(0, 5), (1, 6)]                     # two undeclared args
    viol = validate_plan(p)
    assert viol.count("arg_referent_undeclared") == 1
