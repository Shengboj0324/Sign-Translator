"""Adversarial tests for the gate + provenance chain (Doc-10, stage 10b)."""

from signtranslator.data_engineering.schema import (
    AuthorizationBasis, ConsentState, DataAuthorization, PersonalityRightsStatus,
)
from signtranslator.data_engineering.provenance import (
    content_hash, gate_download, ProvenanceChain, ProvenanceStep,
)

USES = ("research", "education")


def _authorization(*, basis=AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT,
                   uses=USES, actions=("download",), rights=PersonalityRightsStatus.VERIFIED,
                   attribution="", limitations=()):
    return DataAuthorization(
        basis=basis, license_identifier="CC-BY-NC-4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        licensor="test licensor", evidence_uri="license-evidence.html",
        evidence_sha256="a" * 64, permitted_uses=tuple(uses),
        permitted_actions=tuple(actions), personality_rights=rights,
        attribution_notice=attribution, limitations=tuple(limitations),
    )


def test_gate_allows_only_full_precondition():
    d = gate_download(_authorization(), ConsentState.GRANTED, "research")
    assert d.allowed and d.reasons == ()


def test_gate_blocks_missing_license_evidence():
    authorization = DataAuthorization(
        **{**_authorization().__dict__, "evidence_sha256": ""})
    d = gate_download(authorization, ConsentState.GRANTED, "research")
    assert not d.allowed and "invalid_authorization_evidence_sha256" in d.reasons


def test_gate_blocks_withdrawn_consent():
    d = gate_download(_authorization(), ConsentState.WITHDRAWN, "research")
    assert not d.allowed and "direct_consent_not_granted" in d.reasons


def test_gate_blocks_disallowed_use():
    d = gate_download(_authorization(), ConsentState.GRANTED, "surveillance")
    assert not d.allowed and "use_not_permitted" in d.reasons


def test_published_noncommercial_license_allows_training_without_fake_consent():
    authorization = _authorization(
        basis=AuthorizationBasis.PUBLISHED_DATASET_LICENSE,
        uses=("non-commercial research",),
        actions=("download", "create_derivatives", "model_training"),
        rights=PersonalityRightsStatus.NOT_VERIFIED,
        attribution="Dataset authors; CC BY-NC 4.0",
        limitations=("No identity, publicity, or privacy rights are asserted.",),
    )
    decision = gate_download(
        authorization, ConsentState.NOT_DIRECTLY_VERIFIED,
        "non-commercial research", ("create_derivatives", "model_training"))
    assert decision.allowed
    commercial = gate_download(
        authorization, ConsentState.NOT_DIRECTLY_VERIFIED,
        "non-commercial research", ("commercial_use",))
    assert not commercial.allowed
    assert "action_not_permitted:commercial_use" in commercial.reasons


def test_published_license_cannot_be_mislabeled_as_direct_consent():
    authorization = _authorization(
        basis=AuthorizationBasis.PUBLISHED_DATASET_LICENSE,
        rights=PersonalityRightsStatus.NOT_VERIFIED,
        attribution="Dataset authors", limitations=("Personality rights unresolved.",))
    decision = gate_download(authorization, ConsentState.GRANTED, "research")
    assert not decision.allowed
    assert "secondary_license_consent_state_mismatch" in decision.reasons


def test_gate_rejects_malformed_authorization_inputs_without_crashing():
    malformed_authorization = gate_download({}, ConsentState.GRANTED, "research")
    assert malformed_authorization.reasons == ("invalid_authorization_type",)
    malformed_actions = gate_download(
        _authorization(), ConsentState.GRANTED, "research", "model_training")
    assert not malformed_actions.allowed
    assert "invalid_requested_actions" in malformed_actions.reasons


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
