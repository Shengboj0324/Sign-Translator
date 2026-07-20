"""Adversarial tests for the falsifiable contract chain (Doc-12, stage 12a)."""

import pytest

from signtranslator.eval_framework.contracts import (
    Direction, Contract, EvaluationChain,
)

CAV = "transcript accuracy != sign adequacy"


def _c(name, value, threshold, direction, layer="speech", caveat=CAV):
    return Contract(name, layer, value, threshold, direction, caveat)


def test_ge_contract_passes_and_fails():
    assert _c("wer_acc", 0.9, 0.8, Direction.GE).passed
    assert not _c("wer_acc", 0.7, 0.8, Direction.GE).passed


def test_le_contract_passes_and_fails():
    assert _c("wer", 0.1, 0.2, Direction.LE).passed
    assert not _c("wer", 0.3, 0.2, Direction.LE).passed


def test_contract_requires_caveat():
    with pytest.raises(ValueError):
        Contract("x", "speech", 1.0, 0.5, Direction.GE, caveat="")
    with pytest.raises(ValueError):
        Contract("x", "speech", 1.0, 0.5, Direction.GE, caveat="   ")


def test_boundary_is_inclusive():
    assert _c("m", 0.8, 0.8, Direction.GE).passed      # >= is inclusive
    assert _c("m", 0.2, 0.2, Direction.LE).passed


def test_chain_adequate_only_if_all_pass():
    chain = EvaluationChain([
        _c("speech_acc", 0.95, 0.9, Direction.GE, "speech"),
        _c("plan_f1", 0.88, 0.8, Direction.GE, "plan"),
        _c("geo_err", 0.05, 0.1, Direction.LE, "manual"),
    ])
    assert chain.adequate and chain.failures == []


def test_single_passing_metric_is_insufficient():
    # THE PRINCIPLE: a great speech score cannot certify the chain if another
    # layer fails. Adequacy is the conjunction, not any single metric.
    chain = EvaluationChain([
        _c("speech_acc", 0.99, 0.9, Direction.GE, "speech"),   # excellent
        _c("nonmanual_f1", 0.40, 0.8, Direction.GE, "non-manual"),  # fails
    ])
    assert not chain.adequate
    assert [f.layer for f in chain.failures] == ["non-manual"]


def test_empty_chain_is_not_adequate():
    assert not EvaluationChain([]).adequate


def test_adding_a_failing_contract_can_only_lower_adequacy():
    passing = [_c("a", 1.0, 0.5, Direction.GE), _c("b", 1.0, 0.5, Direction.GE)]
    assert EvaluationChain(passing).adequate
    worse = EvaluationChain(passing + [_c("c", 0.0, 0.5, Direction.GE)])
    assert not worse.adequate                          # monotone conjunction
