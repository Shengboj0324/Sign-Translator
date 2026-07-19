"""Proofs for the schema automaton and constrained decoding.

The three DFA properties (soundness, liveness, determinism) and the three
masked-distribution properties are checked directly -- soundness by exhaustive
enumeration on a tiny vocabulary, the safety of constrained decoding by many
randomised decodes that are each required to be automaton-accepted.
"""

import math

import pytest
import torch

from signtranslator.planning.schema import (
    PlanVocabulary, deserialize_plan, validate_plan, serialize_plan,
    DeserializationError,
)
from signtranslator.planning.automaton import SchemaAutomaton, S, _TRANSITIONS
from signtranslator.planning.constrained import (
    ConstrainedDecoder, masked_log_softmax, masked_distribution, allowed_mask,
)

# A deliberately tiny vocabulary so exhaustive enumeration is tractable.
TINY = PlanVocabulary(num_predicates=2, num_roles=2, num_referents=2, num_tam=2,
                      num_loci=2, num_lexemes=2, num_nonmanual=2, max_units=2,
                      num_conf_buckets=2)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_automaton_is_deterministic_by_construction():
    SchemaAutomaton()                       # would raise if edges shared a kind
    for state, edges in _TRANSITIONS.items():
        kinds = [k for k, _ in edges]
        assert len(kinds) == len(set(kinds)), state


def test_step_returns_none_for_disallowed_token():
    a = SchemaAutomaton()
    assert a.step(S.START, a.vocab.token("PRED")) is None   # BOP expected first
    assert a.step(S.START, a.vocab.token("BOP")) is S.AFTER_BOP


def test_serialized_plans_are_accepted_by_the_automaton():
    """Everything ``serialize_plan`` produces must be a legal skeleton."""
    from tests.test_planning_schema import _random_valid_plan
    import random
    a = SchemaAutomaton()
    rng = random.Random(1)
    for _ in range(200):
        tokens = serialize_plan(_random_valid_plan(rng))
        assert a.accepts(tokens)


# ---------------------------------------------------------------------------
# Liveness -- no dead ends
# ---------------------------------------------------------------------------
def test_every_non_accepting_reachable_state_has_a_move():
    a = SchemaAutomaton()
    for state in a.reachable_states():
        allowed = a.allowed_tokens(state)
        if a.is_accepting(state):
            assert allowed == set()
        else:
            assert allowed, f"dead end at {state}"


def test_all_states_are_reachable():
    """A state with no path from START would be dead code in the grammar."""
    a = SchemaAutomaton()
    assert a.reachable_states() == set(S)


# ---------------------------------------------------------------------------
# Soundness -- exhaustive on the tiny vocabulary
# ---------------------------------------------------------------------------
def test_every_accepted_string_deserializes_soundly():
    """Exhaustive soundness: within a length bound, every accepted string is a
    well-formed skeleton that deserializes without error.

    NOTE: not every accepted string is a *fixpoint* of serialize.deserialize.
    ``loci`` is a map, so an accepted string with a duplicate referent in the
    LOCI slot (grammatically legal) canonicalizes to a single entry. The correct
    invariant is therefore idempotence: serialize.deserialize applied twice
    equals applying it once.
    """
    a = SchemaAutomaton(TINY)
    count = 0
    # The minimal plan is now 17 tokens (TOPIC/FOCUS/CLS markers added), so the
    # enumeration bound is raised to 19 to admit non-trivial (non-minimal) plans;
    # larger bounds grow super-exponentially without adding qualitatively new
    # cases, so this is the exhaustive-enough bound.
    for tokens in a.enumerate_accepted(max_length=19):
        plan = deserialize_plan(tokens, TINY)              # must not raise
        canonical = serialize_plan(plan, TINY)
        assert a.accepts(canonical)                        # still in the language
        # idempotence: canonicalizing an already-canonical stream is a no-op
        assert serialize_plan(deserialize_plan(canonical, TINY), TINY) == canonical
        count += 1
    assert count > 50, f"enumeration too small to be meaningful ({count})"


def test_canonical_serializations_are_exact_fixpoints():
    """For strings that ARE serializations of a plan, the round-trip is exact."""
    a = SchemaAutomaton(TINY)
    from tests.test_planning_schema import _random_valid_plan
    import random
    rng = random.Random(3)
    for _ in range(200):
        tokens = serialize_plan(_random_valid_plan(rng, TINY), TINY)
        assert a.accepts(tokens)
        assert serialize_plan(deserialize_plan(tokens, TINY), TINY) == tokens


def test_minimal_accepted_plan_has_the_expected_length():
    """The shortest plan is all 14 markers + 3 singleton values (PRED/TAM/CONF)
    with every variable slot and both optional TOPIC/FOCUS referents empty = 17."""
    a = SchemaAutomaton(TINY)
    lengths = [len(t) for t in a.enumerate_accepted(max_length=17)]
    assert min(lengths) == 17


# ---------------------------------------------------------------------------
# Masked distribution
# ---------------------------------------------------------------------------
def test_masked_distribution_is_a_distribution_over_allowed_tokens():
    torch.manual_seed(0)
    logits = torch.randn(10, dtype=torch.float64)
    allowed = {1, 4, 7, 9}
    p = masked_distribution(logits, allowed)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    for i in range(10):
        if i not in allowed:
            assert float(p[i]) == 0.0
        else:
            assert float(p[i]) > 0.0


def test_masked_distribution_preserves_ratios_among_allowed():
    """Constraining removes illegal mass; it never reorders legal preferences."""
    torch.manual_seed(1)
    logits = torch.randn(12, dtype=torch.float64)
    allowed = {2, 3, 5, 8}
    full = torch.softmax(logits, dim=-1)
    masked = masked_distribution(logits, allowed)
    a, b = 2, 8
    assert abs(float(masked[a] / masked[b]) - float(full[a] / full[b])) < 1e-9


def test_masked_distribution_has_no_nan_with_extreme_logits():
    """A naive exp on a masked prob can give 0*inf = NaN; log-space must not."""
    logits = torch.tensor([100.0, -100.0, 50.0, float("-inf"), 0.0],
                          dtype=torch.float64)
    p = masked_distribution(logits, {1, 4})            # exclude the +100 and inf
    assert torch.isfinite(p).all()
    assert float(p[0]) == 0.0 and float(p[3]) == 0.0
    assert abs(float(p.sum()) - 1.0) < 1e-12


def test_single_allowed_token_gets_all_the_mass():
    p = masked_distribution(torch.randn(8), {5})
    assert abs(float(p[5]) - 1.0) < 1e-6


def test_empty_allowed_set_is_rejected():
    with pytest.raises(ValueError):
        masked_distribution(torch.randn(5), set())
    with pytest.raises(ValueError):
        masked_log_softmax(torch.randn(5), set())


def test_masked_log_softmax_requires_1d():
    with pytest.raises(ValueError):
        masked_log_softmax(torch.randn(3, 4), {0})


def test_allowed_mask_marks_exactly_the_allowed_ids():
    mask = allowed_mask({0, 3, 4}, vocab_size=6)
    assert mask.tolist() == [True, False, False, True, True, False]


# ---------------------------------------------------------------------------
# Constrained decoding safety
# ---------------------------------------------------------------------------
def _progressing_logits(vocab, seed):
    """Random logits that bias structural markers by the current length.

    With constant logits, greedy would loop forever on a repeatable slot (a LEX
    token stays argmax). Biasing markers by +len(prefix) guarantees the loop
    eventually advances, so decoding terminates -- while every choice is still
    a legal, automaton-allowed token.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(vocab.size, generator=g, dtype=torch.float64)
    marker_ids = [vocab.token(k) for k in
                  ("BOP", "PRED", "ARGS", "REFS", "TAM", "LOCI", "UNITS",
                   "NMS", "FS", "CONF", "EOP")]

    def fn(prefix):
        logits = base.clone()
        for mid in marker_ids:
            logits[mid] += 0.5 * len(prefix)
        return logits
    return fn


def test_greedy_constrained_decode_always_yields_a_wellformed_skeleton():
    """Constrained decoding guarantees a valid SKELETON, not a consistent plan.

    The automaton enforces the grammar (slot order, value kinds), so the output
    always deserializes. It does NOT enforce cross-slot consistency (a decoded
    arg may reference an undeclared referent), which is the validator's job --
    exactly the skeleton-vs-plan distinction of docs/SEMANTIC_PLANNER.md §3.1.
    So we assert the skeleton is well-formed, not that validate_plan is empty.
    """
    dec = ConstrainedDecoder()
    for seed in range(30):
        tokens = dec.greedy_decode(_progressing_logits(dec.vocab, seed))
        assert dec.automaton.accepts(tokens)
        plan = deserialize_plan(tokens)                # never raises
        # validate_plan may report consistency violations -- that is expected,
        # and is why a separate consistency mechanism exists (Stage 2e).
        assert isinstance(validate_plan(plan), list)


def test_greedy_decode_output_is_a_wellformed_skeleton():
    """Stronger: the decoded token stream deserializes to a parseable plan."""
    dec = ConstrainedDecoder()
    tokens = dec.greedy_decode(_progressing_logits(dec.vocab, 0))
    plan = deserialize_plan(tokens)
    assert serialize_plan(plan) == tokens              # round-trips exactly


def test_sampled_constrained_decode_stays_in_the_language():
    dec = ConstrainedDecoder()
    g = torch.Generator().manual_seed(0)
    for seed in range(30):
        tokens = dec.sample_decode(_progressing_logits(dec.vocab, seed),
                                   generator=g)
        assert dec.automaton.accepts(tokens)
        deserialize_plan(tokens)                       # must parse


def test_decoder_never_selects_a_disallowed_token():
    """The internal asserts must never trip across many random logit streams."""
    dec = ConstrainedDecoder()
    for seed in range(50):
        dec.greedy_decode(_progressing_logits(dec.vocab, seed))   # asserts inside


def test_allowed_at_each_step_matches_a_real_serialization():
    dec = ConstrainedDecoder()
    from tests.test_planning_schema import _valid_plan
    tokens = serialize_plan(_valid_plan())
    steps = dec.allowed_at_each_step(tokens)
    assert len(steps) == len(tokens)
    for tok, allowed in zip(tokens, steps):
        assert tok in allowed                          # each emitted token was legal


def test_allowed_at_each_step_rejects_an_illegal_sequence():
    dec = ConstrainedDecoder()
    with pytest.raises(ValueError):
        dec.allowed_at_each_step([dec.vocab.token("PRED")])    # must start BOP


def test_sample_decode_rejects_nonpositive_temperature():
    dec = ConstrainedDecoder()
    with pytest.raises(ValueError):
        dec.sample_decode(_progressing_logits(dec.vocab, 0), temperature=0.0)


# ---------------------------------------------------------------------------
# Bounded runtime: termination guarantee
# ---------------------------------------------------------------------------
def test_bounded_decoding_terminates_even_with_adversarial_constant_logits():
    """The decisive property: a model that ALWAYS prefers a repeatable token
    would loop forever under the unbounded language. The slot caps make the
    language finite, so bounded decoding must still terminate.

    Here every LEX token is given an overwhelming, constant score, so an
    unbounded greedy decode would emit units without end.
    """
    dec = ConstrainedDecoder()
    v = dec.vocab
    logits = torch.full((v.size,), -10.0, dtype=torch.float64)
    for lx in range(v.num_lexemes):
        logits[v.token("LEX", lx)] = 100.0            # always want another unit

    tokens = dec.greedy_decode(lambda prefix: logits)   # must not raise
    assert dec.automaton.accepts(tokens)
    plan = deserialize_plan(tokens)
    # the units slot is capped at max_units, not infinite
    assert len(plan.manual_units) <= v.max_units


def test_bounded_liveness_advance_marker_always_survives_the_cap():
    """At every decision state, hitting the cap must still leave a legal move."""
    a = SchemaAutomaton()
    from signtranslator.planning.automaton import _REPEAT_DECISION
    for state, (kind, attr) in _REPEAT_DECISION.items():
        capped_counts = {state: getattr(a.vocab, attr)}
        allowed = a.bounded_allowed(state, capped_counts)
        assert allowed, f"{state} dead-ends at its cap"
        # the repeat kind is gone, but a non-repeat (advance) token remains
        assert all(a.vocab.decode(t)[0] != kind for t in allowed)


def test_bounded_run_is_always_a_base_run():
    """Every bounded step is a legal base-DFA step, so soundness transfers."""
    dec = ConstrainedDecoder()
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        logits = torch.randn(dec.vocab.size, generator=g, dtype=torch.float64)
        tokens = dec.greedy_decode(lambda prefix: logits)
        assert dec.automaton.accepts(tokens)          # accepted by the base DFA


def test_max_generated_length_bounds_actual_output():
    dec = ConstrainedDecoder()
    bound = dec.automaton.max_generated_length()
    g = torch.Generator().manual_seed(1)
    for _ in range(10):
        logits = torch.randn(dec.vocab.size, generator=g, dtype=torch.float64)
        tokens = dec.greedy_decode(lambda prefix: logits)
        assert len(tokens) <= bound
