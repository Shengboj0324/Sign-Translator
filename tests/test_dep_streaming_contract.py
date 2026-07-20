"""Adversarial tests for the streaming contract (Doc-13, stage 13a)."""

import pytest

from signtranslator.deployment.streaming_contract import (
    is_prefix, certify_commit_monotone, StreamingContract, CommitViolationError,
)


def test_is_prefix():
    assert is_prefix([1, 2], [1, 2, 3])
    assert is_prefix([], [1])
    assert not is_prefix([1, 2], [1, 3])
    assert not is_prefix([1, 2, 3], [1, 2])


def test_monotone_history_certified():
    hist = [(), (1,), (1, 2), (1, 2, 3)]
    ok, viol = certify_commit_monotone(hist)
    assert ok and viol is None


def test_non_monotone_history_pinpointed():
    # step 2 rewrites the already-committed token 2 -> 9: a display violation.
    hist = [(1,), (1, 2), (1, 9)]
    ok, viol = certify_commit_monotone(hist)
    assert not ok and viol == 2


def test_commit_must_extend():
    c = StreamingContract()
    c.commit([1])
    c.commit([1, 2])
    assert c.committed == (1, 2)
    with pytest.raises(CommitViolationError):
        c.commit([1, 9])                               # rewrites displayed token 2


def test_suffix_is_freely_revisable():
    c = StreamingContract()
    c.commit([1])
    c.revise_suffix([5, 6])
    c.revise_suffix([7])                               # revising U_t is fine
    assert c.suffix == (7,)
    assert c.certify()[0]


def test_committed_history_certifies_after_valid_run():
    c = StreamingContract()
    for prefix in ([1], [1, 2], [1, 2, 3]):
        c.commit(prefix)
    ok, viol = c.certify()
    assert ok and viol is None


def test_commitment_errors_against_reference():
    c = StreamingContract()
    c.commit([1, 2, 5])                                # 5 is wrong / hallucinated
    assert c.commitment_errors(reference=[1, 2, 3]) == 1
    c2 = StreamingContract()
    c2.commit([1, 2, 3])
    assert c2.commitment_errors(reference=[1, 2, 3]) == 0
