"""Adversarial tests for the gate + provenance chain (Doc-10, stage 10b)."""

from signtranslator.data_engineering.schema import ConsentState
from signtranslator.data_engineering.provenance import (
    content_hash, gate_download, ProvenanceChain, ProvenanceStep,
)

USES = ("research", "education")


def test_gate_allows_only_full_precondition():
    d = gate_download("CC-BY-NC-4.0", ConsentState.GRANTED, "research", USES)
    assert d.allowed and d.reasons == ()


def test_gate_blocks_missing_license():
    d = gate_download("", ConsentState.GRANTED, "research", USES)
    assert not d.allowed and "no_license" in d.reasons


def test_gate_blocks_withdrawn_consent():
    d = gate_download("L", ConsentState.WITHDRAWN, "research", USES)
    assert not d.allowed and "consent_not_granted" in d.reasons


def test_gate_blocks_disallowed_use():
    d = gate_download("L", ConsentState.GRANTED, "surveillance", USES)
    assert not d.allowed and "use_not_permitted" in d.reasons


def test_content_hash_is_order_insensitive_for_dicts():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_changes_on_any_edit():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_chain_root_advances_and_verifies():
    c = ProvenanceChain()
    r0 = c.root
    c.append("download", {"uri": "x", "bytes": 10})
    r1 = c.root
    c.append("triangulate", {"joints": 25})
    r2 = c.root
    assert r0 != r1 != r2 and c.verify()


def test_chain_detects_output_tampering():
    c = ProvenanceChain()
    c.append("download", {"uri": "x"})
    c.append("clean", {"frames": 100})
    # tamper: rewrite a recorded step's output hash.
    c.steps[1] = ProvenanceStep("clean", content_hash({"frames": 999}))
    assert not c.verify()


def test_chain_detects_step_reordering():
    a = ProvenanceChain(); a.append("s1", 1); a.append("s2", 2)
    b = ProvenanceChain(); b.append("s2", 2); b.append("s1", 1)
    # order is part of provenance => different roots.
    assert a.root != b.root


def test_recompute_matches_incremental():
    c = ProvenanceChain()
    for i in range(5):
        c.append(f"step{i}", {"i": i})
    assert ProvenanceChain.recompute_root(c.steps) == c.root
