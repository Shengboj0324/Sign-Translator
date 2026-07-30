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
    # Each signer owns two recordings.  The two recordings are one connected
    # component because they share a signer, so there are six indivisible groups.
    out = []
    k = 0
    for signer in [f"g{i}" for i in range(6)]:
        for source in [f"{signer}-recA", f"{signer}-recB"]:
            for _ in range(3):
                out.append(_sample(f"s{k}", signer, source)); k += 1
    return out


def test_components_join_every_recording_from_the_same_signer():
    samples = _corpus()
    groups = group_samples(samples)
    assert len(groups) == 6
    assert all(len(v) == 6 for v in groups.values())


def test_components_close_transitively_over_shared_sources():
    samples = [
        _sample("s0", "alice", "session-a"),
        _sample("s1", "alice", "session-b"),
        _sample("s2", "bob", "session-b"),
        _sample("s3", "carol", "session-c"),
    ]
    groups = list(group_samples(samples).values())
    assert sorted(sorted(group) for group in groups) == [[0, 1, 2], [3]]


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
    victim_signer = samples[0].signer_id_hash
    members = [i for i, s in enumerate(samples) if s.signer_id_hash == victim_signer]
    assign[members[0]] = "train"
    assign[members[1]] = "test"
    cert = certify_no_group_leakage(samples, assign)
    assert not cert.certified
    assert victim_signer in cert.offending_signers


def test_certificate_detects_source_leak_across_different_signers():
    samples = [_sample("s0", "a", "shared"), _sample("s1", "b", "shared")]
    cert = certify_no_group_leakage(samples, {0: "train", 1: "test"})
    assert not cert.certified
    assert cert.offending_sources == ("shared",)


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
