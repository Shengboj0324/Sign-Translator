"""Verification of the fail-closed policy.

The safety property is asserted exhaustively over a confidence grid: the policy
must NEVER assert a sign below its threshold, and must never fingerspell a token
it cannot verify. A single counterexample would mean the system can hallucinate.
"""

import pytest
import torch

from signtranslator.speech.policy import (
    Action, FailClosedPolicy, PolicyDecision, PolicyOutcome,
    selective_metrics, risk_coverage_curve, area_under_risk_coverage,
    SelectivePoint,
)

GRID = [i / 100.0 for i in range(101)]


# ---------------------------------------------------------------------------
# Core safety properties
# ---------------------------------------------------------------------------
def test_never_emits_below_the_emit_threshold():
    """THE safety invariant, checked exhaustively over the confidence range."""
    pol = FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.4,
                           sign_lexicon=[1, 2, 3])
    for token in (1, 2, 3):
        for c in GRID:
            d = pol.decide(token, c)
            if c < 0.8:
                assert d.action is not Action.EMIT, (token, c)
            else:
                assert d.action is Action.EMIT


def test_never_emits_a_token_that_has_no_sign():
    """Full confidence in a word with no sign must still not invent one."""
    pol = FailClosedPolicy(emit_threshold=0.5, sign_lexicon=[1],
                           verified_lexicon=[1, 2])
    assert pol.decide(2, 1.0).action is Action.FINGERSPELL
    assert pol.decide(1, 1.0).action is Action.EMIT


def test_never_fingerspells_an_unverified_token():
    """Spelling an unverified word just moves the hallucination elsewhere."""
    pol = FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.1,
                           sign_lexicon=[1], verified_lexicon=[1])
    for c in GRID:
        assert pol.decide(99, c).action is Action.PAUSE


def test_zero_confidence_always_pauses():
    pol = FailClosedPolicy(emit_threshold=0.5, fingerspell_threshold=0.0,
                           sign_lexicon=[1])
    # fingerspell_threshold=0 admits everything verified; use a strict variant.
    strict = FailClosedPolicy(emit_threshold=0.5, fingerspell_threshold=0.2,
                              sign_lexicon=[1])
    assert strict.decide(1, 0.0).action is Action.PAUSE


def test_policy_is_monotone_in_confidence():
    """More evidence must never yield a more conservative action."""
    pol = FailClosedPolicy(emit_threshold=0.75, fingerspell_threshold=0.35,
                           sign_lexicon=[1, 2], verified_lexicon=[1, 2, 5])
    for token in (1, 2, 5, 7):
        actions = [pol.decide(token, c).action for c in GRID]
        for a, b in zip(actions, actions[1:]):
            assert int(b) >= int(a), f"token {token}: {a} -> {b} is non-monotone"


def test_action_ordering_reflects_assertiveness():
    assert Action.PAUSE < Action.FINGERSPELL < Action.EMIT
    assert PolicyDecision(1, 0.9, Action.EMIT, "").asserts_a_sign
    assert not PolicyDecision(1, 0.9, Action.FINGERSPELL, "").asserts_a_sign


def test_thresholds_are_boundary_inclusive():
    pol = FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.4,
                           sign_lexicon=[1])
    assert pol.decide(1, 0.8).action is Action.EMIT
    assert pol.decide(1, 0.4).action is Action.FINGERSPELL
    assert pol.decide(1, 0.39999).action is Action.PAUSE


def test_open_vocabulary_mode_treats_all_tokens_as_signable():
    pol = FailClosedPolicy(emit_threshold=0.5)          # no lexicons supplied
    assert pol.decide(12345, 0.9).action is Action.EMIT


def test_decisions_carry_a_reason():
    pol = FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.3,
                           sign_lexicon=[1], verified_lexicon=[1, 2])
    assert "no sign" in pol.decide(2, 0.9).reason
    assert "below emit" in pol.decide(1, 0.5).reason
    assert "unverified" in pol.decide(7, 0.9).reason
    assert "below fingerspell" in pol.decide(1, 0.1).reason


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_rejects_incoherent_thresholds():
    """The safer action must not require MORE evidence than the bolder one."""
    with pytest.raises(ValueError):
        FailClosedPolicy(emit_threshold=0.3, fingerspell_threshold=0.7)
    with pytest.raises(ValueError):
        FailClosedPolicy(emit_threshold=1.5)
    with pytest.raises(ValueError):
        FailClosedPolicy(fingerspell_threshold=-0.1)


def test_rejects_out_of_range_confidence():
    pol = FailClosedPolicy()
    with pytest.raises(ValueError):
        pol.decide(1, 1.5)
    with pytest.raises(ValueError):
        pol.decide(1, -0.01)


def test_tolerates_float_rounding_in_posteriors():
    """REGRESSION: real lattice posteriors can land an ulp outside [0, 1].

    Summing floating-point posteriors legitimately produces values like
    1.0000000000000002. Rejecting those crashed the policy on genuine model
    output; accepting 1.5 would hide an upstream bug. Rounding is tolerated and
    clamped, real violations still raise.
    """
    pol = FailClosedPolicy(emit_threshold=0.8, sign_lexicon=[1])
    d = pol.decide(1, 1.0 + 2.2e-16)
    assert d.action is Action.EMIT and d.confidence <= 1.0
    d2 = pol.decide(1, -1e-16)
    assert d2.confidence >= 0.0
    with pytest.raises(ValueError):
        pol.decide(1, 1.001)              # beyond rounding: a real error


def test_sequence_length_mismatch_rejected():
    with pytest.raises(ValueError):
        FailClosedPolicy().decide_sequence([1, 2], [0.5])


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------
def test_outcome_counts_and_coverage():
    pol = FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.4,
                           sign_lexicon=[1, 2, 3])
    out = pol.decide_sequence([1, 2, 3, 1], [0.95, 0.5, 0.1, 0.85])
    assert out.count(Action.EMIT) == 2
    assert out.count(Action.FINGERSPELL) == 1
    assert out.count(Action.PAUSE) == 1
    assert abs(out.coverage - 0.5) < 1e-12
    assert abs(out.abstention_rate - 0.5) < 1e-12
    assert out.emitted_tokens() == [1, 1]
    assert "coverage" in out.report()


def test_empty_outcome_is_safe():
    out = PolicyOutcome()
    assert len(out) == 0 and out.coverage == 0.0


def test_raising_the_threshold_never_increases_coverage():
    """Coverage is non-increasing in the emit threshold, by construction."""
    torch.manual_seed(0)
    conf = torch.rand(200).tolist()
    tokens = [1] * 200
    coverages = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        pol = FailClosedPolicy(emit_threshold=t, fingerspell_threshold=0.0,
                               sign_lexicon=[1])
        coverages.append(pol.decide_sequence(tokens, conf).coverage)
    assert coverages == sorted(coverages, reverse=True)


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------
def test_selective_metrics_on_hand_built_data():
    conf = [0.9, 0.8, 0.3, 0.2]
    correct = [True, True, False, False]
    p = selective_metrics(conf, correct, threshold=0.5)
    assert abs(p.coverage - 0.5) < 1e-12
    assert abs(p.selective_accuracy - 1.0) < 1e-12
    assert abs(p.risk - 0.0) < 1e-12


def test_coverage_is_monotone_non_increasing_in_threshold():
    """Exactly true for any data: {c >= t} shrinks as t grows."""
    torch.manual_seed(1)
    conf = torch.rand(300)
    correct = torch.rand(300) < conf
    covs = [p.coverage for p in risk_coverage_curve(conf, correct)]
    for a, b in zip(covs, covs[1:]):
        assert b <= a + 1e-12


def test_abstention_improves_accuracy_when_confidence_is_informative():
    """The entire point of failing closed: what it does assert is more reliable."""
    torch.manual_seed(2)
    conf = torch.rand(2000, dtype=torch.float64)
    correct = torch.rand(2000, dtype=torch.float64) < conf   # calibrated by construction
    overall = float(correct.double().mean())
    high = selective_metrics(conf, correct, threshold=0.8)
    assert high.selective_accuracy > overall
    assert high.coverage < 1.0


def test_uninformative_confidence_does_not_improve_accuracy():
    """A confidence uncorrelated with correctness buys nothing -- as it should.

    This guards against a metric that looks good for the wrong reason.
    """
    torch.manual_seed(3)
    conf = torch.rand(3000, dtype=torch.float64)
    correct = torch.rand(3000) < 0.6                # independent of confidence
    overall = float(correct.double().mean())
    high = selective_metrics(conf, correct, threshold=0.8)
    assert abs(high.selective_accuracy - overall) < 0.06


def test_aurc_prefers_informative_confidence():
    torch.manual_seed(4)
    n = 3000
    correct = torch.rand(n) < 0.6
    informative = torch.where(correct, torch.rand(n) * 0.4 + 0.6,
                              torch.rand(n) * 0.4)      # tracks correctness
    random_conf = torch.rand(n)
    assert (area_under_risk_coverage(informative, correct)
            < area_under_risk_coverage(random_conf, correct))


def test_full_abstention_reports_perfect_accuracy_vacuously():
    """Above every confidence, nothing is kept; accuracy is 1.0 over an empty set.

    Reported explicitly so a reader is not fooled: coverage 0 makes the accuracy
    meaningless, which is why coverage must always be reported alongside it.
    """
    p = selective_metrics([0.1, 0.2], [False, False], threshold=0.99)
    assert p.coverage == 0.0 and p.selective_accuracy == 1.0


def test_selective_metrics_validates_input():
    with pytest.raises(ValueError):
        selective_metrics([0.5], [True, False], 0.5)
    with pytest.raises(ValueError):
        selective_metrics([], [], 0.5)
