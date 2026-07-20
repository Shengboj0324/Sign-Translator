"""Adversarial tests for governance (Doc-10, stage 10g)."""

import pytest

from signtranslator.data_engineering.schema import ConsentState, Sample
from signtranslator.data_engineering.governance import (
    transition_consent, ConsentError, apply_withdrawal, apply_retention,
    UsagePolicy, gate_action, infer_sensitive_trait, SensitiveInferenceError,
    SENSITIVE_TRAITS,
)


def _sample(sid, signer, retention=None):
    return Sample(
        sample_id=sid, source_id="rec", signer_id_hash=signer,
        target_language="ASL", license="L", consent=ConsentState.GRANTED,
        intended_use="research", smplx_version="1.1", provenance="p",
        split="train", retention_date=retention,
    )


def test_consent_withdrawal_is_terminal():
    s = transition_consent(ConsentState.GRANTED, ConsentState.WITHDRAWN)
    assert s == ConsentState.WITHDRAWN
    with pytest.raises(ConsentError):
        transition_consent(ConsentState.WITHDRAWN, ConsentState.GRANTED)


def test_consent_idempotent_same_state():
    assert transition_consent(ConsentState.GRANTED, ConsentState.GRANTED) \
        == ConsentState.GRANTED


def test_withdrawal_removes_all_records_of_signer():
    corpus = [_sample("a", "g1"), _sample("b", "g1"), _sample("c", "g2")]
    kept = apply_withdrawal(corpus, "g1")
    assert [s.sample_id for s in kept] == ["c"]
    assert all(s.signer_id_hash != "g1" for s in kept)


def test_retention_drops_expired_only():
    corpus = [_sample("old", "g1", retention=10.0),
              _sample("live", "g2", retention=100.0),
              _sample("forever", "g3", retention=None)]
    kept = apply_retention(corpus, now=50.0)
    assert {s.sample_id for s in kept} == {"live", "forever"}


def test_policy_gates_default_closed():
    p = UsagePolicy()
    for action in ("derivative_model", "identity_use", "commercial_use",
                   "redistribution"):
        assert gate_action(p, action) is False


def test_policy_gate_opens_only_named_action():
    p = UsagePolicy(allow_commercial_use=True)
    assert gate_action(p, "commercial_use") is True
    assert gate_action(p, "redistribution") is False


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        gate_action(UsagePolicy(), "sell_biometrics")


def test_sensitive_trait_inference_always_raises():
    s = _sample("a", "g1")
    for trait in SENSITIVE_TRAITS:
        with pytest.raises(SensitiveInferenceError):
            infer_sensitive_trait(s, trait)
