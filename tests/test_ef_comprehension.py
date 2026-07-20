"""Adversarial tests for comprehension + dissociation (Doc-12, stage 12f)."""

import pytest

from signtranslator.eval_framework.comprehension import (
    proposition_prf, comprehension_f1, mean_comprehension_f1,
    proposition_agreement, SystemScores, preference_comprehension_dissociate,
    BlindedTrial,
)


def test_perfect_recovery_is_f1_one():
    intended = {"agent=DOG", "action=CHASE", "patient=CAT"}
    assert comprehension_f1(set(intended), intended) == 1.0


def test_partial_recovery_prf():
    intended = {"a", "b", "c", "d"}
    recovered = {"a", "b", "x"}                         # 2 correct, 1 spurious
    prf = proposition_prf(recovered, intended)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 4)
    assert prf["f1"] == pytest.approx(2 * (2/3) * (1/2) / ((2/3) + (1/2)))


def test_missing_and_hallucinated_propositions_penalised():
    intended = {"neg=TRUE", "q=YESNO"}
    # recovers the wrong polarity -> low overlap.
    recovered = {"neg=FALSE", "q=YESNO"}
    assert comprehension_f1(recovered, intended) < 1.0


def test_mean_comprehension_over_items():
    items = [({"a"}, {"a"}), (set(), {"a"})]            # 1.0 and 0.0
    assert mean_comprehension_f1(items) == pytest.approx(0.5)


def test_proposition_agreement_perfect_and_disagreeing():
    universe = ["p1", "p2", "p3"]
    a = [{"p1", "p2"}, {"p3"}]
    assert proposition_agreement(a, a, universe) == pytest.approx(1.0)
    b = [{"p3"}, {"p1", "p2"}]                          # opposite codings
    assert proposition_agreement(a, b, universe) < 0.5


def test_preference_comprehension_dissociation():
    # System A is preferred but conveys LESS meaning than B.
    a = SystemScores("A", preference=0.8, comprehension_f1=0.6)
    b = SystemScores("B", preference=0.4, comprehension_f1=0.9)
    assert preference_comprehension_dissociate(a, b)
    # aligned ordering does not dissociate.
    c = SystemScores("C", preference=0.9, comprehension_f1=0.95)
    assert not preference_comprehension_dissociate(c, b)


def test_blinded_trial_forbids_text_priming():
    with pytest.raises(ValueError):
        BlindedTrial(show_text_before_signing=True)
    trial = BlindedTrial(rater_language_backgrounds=("Deaf_ASL_native",))
    assert trial.attention_check_pass_rate([1, 0, 1], [1, 0, 1]) == 1.0
    assert trial.attention_check_pass_rate([1, 1, 1], [1, 0, 1]) == pytest.approx(2/3)
