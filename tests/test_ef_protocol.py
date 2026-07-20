"""Adversarial tests for pre-registration + test firewall (Doc-12, stage 12d)."""

import pytest

from signtranslator.eval_framework.protocol import (
    PreRegistration, EvaluationFirewall, ProtocolError, signer_held_out_split,
)
from signtranslator.data_engineering.schema import ConsentState, Sample


def _prereg():
    return PreRegistration.create(
        primary_endpoints=["comprehension_f1", "grammaticality"],
        min_effects={"comprehension_f1": 0.05, "grammaticality": 0.1})


def test_registration_hash_is_deterministic_and_order_independent():
    a = PreRegistration.create(["x", "y"], {"x": 0.1, "y": 0.2})
    b = PreRegistration.create(["y", "x"], {"y": 0.2, "x": 0.1})
    assert a.registration_hash == b.registration_hash


def test_registration_requires_min_effect_for_each_endpoint():
    with pytest.raises(ValueError):
        PreRegistration.create(["x"], {})              # no min effect for x


def test_firewall_blocks_test_selection():
    fw = EvaluationFirewall(_prereg())
    fw.select_hyperparameters("val")                    # allowed
    with pytest.raises(ProtocolError):
        fw.select_hyperparameters("test")               # forbidden


def test_firewall_blocks_unregistered_primary_endpoint():
    fw = EvaluationFirewall(_prereg())
    with pytest.raises(ProtocolError):
        fw.report_primary("bleu")                       # not pre-registered
    fw.report_primary("comprehension_f1")               # registered -> ok


def test_endpoint_confirmed_requires_significant_and_meaningful():
    fw = EvaluationFirewall(_prereg())
    # significant but below the registered min effect (0.05) -> not confirmed.
    assert not fw.endpoint_confirmed("comprehension_f1", effect=0.02, pvalue=0.001)
    # significant and above the min effect -> confirmed.
    assert fw.endpoint_confirmed("comprehension_f1", effect=0.08, pvalue=0.001)


def _sample(sid, signer, source):
    return Sample(sample_id=sid, source_id=source, signer_id_hash=signer,
                  target_language="ASL", license="L", consent=ConsentState.GRANTED,
                  intended_use="research", smplx_version="1.1", provenance="p",
                  split="train")


def test_signer_held_out_split_is_certified_leakage_free():
    samples = [_sample(f"s{k}", f"g{k % 5}", f"rec{k % 3}") for k in range(30)]
    assignment, cert = signer_held_out_split(samples, seed=1)
    assert cert.certified
    # every group lands in exactly one split (no signer spans train/val/test).
    from collections import defaultdict
    groups = defaultdict(set)
    for i, s in enumerate(samples):
        groups[s.group_key].add(assignment[i])
    assert all(len(v) == 1 for v in groups.values())
