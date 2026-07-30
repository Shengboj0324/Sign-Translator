"""Adversarial tests for the canonical sample schema (Doc-10, stage 10a)."""

import pytest

from signtranslator.data_engineering.schema import (
    AuthorizationBasis, ConsentState, DataAuthorization, PersonalityRightsStatus,
    Sample, validate_sample, DATASET_MAP, dataset_map_is_complete,
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
