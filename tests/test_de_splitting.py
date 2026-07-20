"""Adversarial tests for the leakage-certified grouped split (Doc-10, 10f)."""

from signtranslator.data_engineering.schema import ConsentState, Sample
from signtranslator.data_engineering.splitting import (
    group_samples, grouped_split, certify_no_group_leakage,
    Window, windows_inherit_split, certify_window_split_consistency,
    LeakageCertificate,
)


def _sample(sid, signer, source):
    return Sample(
        sample_id=sid, source_id=source, signer_id_hash=signer,
        target_language="ASL", license="L", consent=ConsentState.GRANTED,
        intended_use="research", smplx_version="1.1", provenance="p", split="train",
    )


def _corpus():
    # 6 signers × 2 sources => 12 groups, several samples each.
    out = []
    k = 0
    for signer in [f"g{i}" for i in range(6)]:
        for source in ["recA", "recB"]:
            for _ in range(3):
                out.append(_sample(f"s{k}", signer, source)); k += 1
    return out


def test_group_key_partitions_by_signer_and_source():
    samples = _corpus()
    groups = group_samples(samples)
    assert len(groups) == 12
    assert all(len(v) == 3 for v in groups.values())


def test_grouped_split_is_leakage_free():
    samples = _corpus()
    assign = grouped_split(samples, (0.6, 0.2, 0.2), seed=3)
    cert = certify_no_group_leakage(samples, assign)
    assert cert.certified and cert.offending_groups == ()


def test_all_samples_assigned_and_ratios_approx():
    samples = _corpus()
    assign = grouped_split(samples, (0.7, 0.15, 0.15), seed=0)
    assert set(assign) == set(range(len(samples)))
    counts = {s: sum(1 for v in assign.values() if v == s)
              for s in ("train", "val", "test")}
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]
    assert sum(counts.values()) == len(samples)


def test_certificate_detects_a_hand_crafted_leak():
    samples = _corpus()
    assign = grouped_split(samples, seed=1)
    # force a leak: move ONE sample of a group to a different split.
    victim_group = samples[0].group_key
    members = [i for i, s in enumerate(samples) if s.group_key == victim_group]
    assign[members[0]] = "train"
    assign[members[1]] = "test"
    cert = certify_no_group_leakage(samples, assign)
    assert not cert.certified and victim_group in cert.offending_groups


def test_windows_inherit_parent_split_no_leak():
    samples = _corpus()
    assign = grouped_split(samples, seed=2)
    # window each sample into overlapping clips AFTER the split.
    windows = [Window(i, t, t + 16) for i in range(len(samples)) for t in (0, 8)]
    splits = windows_inherit_split(windows, assign)
    assert all(sp == assign[w.sample_idx] for w, sp in zip(windows, splits))
    assert certify_window_split_consistency(windows, assign)


def test_invalid_ratios_rejected():
    import pytest
    with pytest.raises(ValueError):
        grouped_split(_corpus(), (0.5, 0.4, 0.3))
