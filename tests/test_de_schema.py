"""Adversarial tests for the canonical sample schema (Doc-10, stage 10a)."""

import pytest

from signtranslator.data_engineering.schema import (
    AuthorizationBasis, ConsentState, DataAuthorization, PersonalityRightsStatus,
    Sample, validate_sample, DATASET_MAP, dataset_map_is_complete,
)
from signtranslator.data_engineering.source_portfolio import (
    AccessStatus, CapabilityEvidence, CURRENT_PRE_PHASE_2_DECISION,
    EvidenceLevel, IntendedUse, RequirementBundle, RightsStatus,
    SourceCandidate, assess_source_portfolio, evidence,
)


def _good(**kw):
    authorization = DataAuthorization(
        basis=AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT,
        license_identifier="CC-BY-NC-4.0",
        license_url="https://example.test/license",
        licensor="test participant", evidence_uri="consent.txt",
        evidence_sha256="a" * 64, permitted_uses=("research",),
        permitted_actions=("download",),
        personality_rights=PersonalityRightsStatus.VERIFIED,
    )
    base = dict(
        sample_id="s1", source_id="rec1", signer_id_hash="h_abc",
        target_language="ASL", license="CC-BY-NC-4.0", consent=ConsentState.GRANTED,
        intended_use="research", smplx_version="1.1", provenance="root_deadbeef",
        split="train", authorization=authorization,
    )
    base.update(kw)
    return Sample(**base)


def test_valid_sample_passes():
    assert validate_sample(_good()) == []


@pytest.mark.parametrize("field,val,code", [
    ("sample_id", "", "missing_sample_id"),
    ("signer_id_hash", "", "missing_signer_id_hash"),
    ("license", "", "missing_license"),
    ("intended_use", "", "missing_intended_use"),
    ("provenance", "", "missing_provenance"),
    ("target_language", "", "missing_target_language"),
    ("split", "holdout", "invalid_split"),
])
def test_governance_critical_fields_required(field, val, code):
    assert code in validate_sample(_good(**{field: val}))


def test_explicit_authorization_is_required_and_must_match_license():
    assert "missing_authorization" in validate_sample(_good(authorization=None))
    assert "invalid_authorization_type" in validate_sample(_good(authorization={}))
    mismatched = DataAuthorization(
        **{**_good().authorization.__dict__, "license_identifier": "different"})
    assert "authorization_license_mismatch" in validate_sample(
        _good(authorization=mismatched))


def test_confidence_range_enforced():
    assert "confidence_2d_out_of_range" in validate_sample(_good(confidence_2d=1.5))
    assert "confidence_3d_out_of_range" in validate_sample(_good(confidence_3d=-0.1))
    assert validate_sample(_good(confidence_2d=0.0, confidence_3d=1.0)) == []


def test_group_key_is_signer_and_source():
    s = _good(signer_id_hash="hZ", source_id="recQ")
    assert s.group_key == ("hZ", "recQ")


def test_no_sensitive_trait_field_on_record():
    # §7 non-inference guard begins structurally: the record cannot hold a trait.
    s = _good()
    for banned in ("race", "ethnicity", "gender", "age", "disability", "religion"):
        assert not hasattr(s, banned)


def test_dataset_map_complete_and_non_redistributable():
    assert dataset_map_is_complete()
    assert set(DATASET_MAP) >= {
        "How2Sign", "WLASL", "MS-ASL", "ASLLVD", "PHOENIX14T",
        "SignAvatars", "ASL3DWord",
    }
    # every source is licensed / not redistributable by us (honest scope).
    assert all(not r.redistributable for r in DATASET_MAP.values())


def _source(source_id, capabilities, *, access=AccessStatus.LOCAL_VERIFIED,
            rights=RightsStatus.PERMITTED):
    return SourceCandidate(
        source_id, source_id, "https://example.test/source", access,
        evidence(*capabilities, level=EvidenceLevel.LOCAL_VERIFIED),
        rights, rights, rights, "test-only source",
    )


def test_source_portfolio_forbids_cross_source_capability_stitching():
    requirement = RequirementBundle(
        "coobserved", frozenset({"body_3d", "eye_gaze"}),
        EvidenceLevel.LOCAL_VERIFIED, IntendedUse.RESEARCH_TRAINING,
    )
    split = assess_source_portfolio(
        (_source("body", ("body_3d",)), _source("gaze", ("eye_gaze",))),
        (requirement,),
    )
    assert split.approved is False
    assert split.bundle_decisions[0].satisfying_sources == ()

    coobserved = assess_source_portfolio(
        (_source("complete", ("body_3d", "eye_gaze")),), (requirement,))
    assert coobserved.approved is True
    assert coobserved.bundle_decisions[0].satisfying_sources == ("complete",)


@pytest.mark.parametrize("access,rights,expected", [
    (AccessStatus.REQUEST_REQUIRED, RightsStatus.PERMITTED,
     "not_locally_verified:request_required"),
    (AccessStatus.LOCAL_VERIFIED, RightsStatus.PERMISSION_REQUIRED,
     "rights_not_permitted:permission_required"),
    (AccessStatus.LOCAL_VERIFIED, RightsStatus.UNRESOLVED,
     "rights_not_permitted:unresolved"),
])
def test_source_portfolio_fails_closed_on_access_and_rights(access, rights, expected):
    requirement = RequirementBundle(
        "required", frozenset({"continuous_asl"}),
        EvidenceLevel.LOCAL_VERIFIED, IntendedUse.COMMERCIAL_TRAINING,
    )
    decision = assess_source_portfolio(
        (_source("candidate", ("continuous_asl",), access=access, rights=rights),),
        (requirement,),
    )
    assert decision.approved is False
    assert expected in decision.bundle_decisions[0].source_failures[0][1]


def test_source_portfolio_requires_real_evidence_and_strict_identifiers():
    with pytest.raises(ValueError, match="positive evidence"):
        CapabilityEvidence("body_3d", EvidenceLevel.NONE)
    with pytest.raises(ValueError, match="lowercase ASCII identifier"):
        CapabilityEvidence("body 3D", EvidenceLevel.LOCAL_VERIFIED)
    with pytest.raises(ValueError, match="HTTPS"):
        SourceCandidate(
            "bad", "bad", "http://example.test", AccessStatus.LOCAL_VERIFIED,
            evidence("body_3d"), RightsStatus.PERMITTED, RightsStatus.PERMITTED,
            RightsStatus.PERMITTED, "test-only source",
        )


def test_current_pre_phase_2_gate_remains_closed_with_explicit_failures():
    assert CURRENT_PRE_PHASE_2_DECISION.approved is False
    payload = CURRENT_PRE_PHASE_2_DECISION.to_dict()
    assert payload["cross_source_capability_stitching_allowed"] is False
    assert len(payload["bundle_decisions"]) == 5
    assert all(not item["passed"] for item in payload["bundle_decisions"])
