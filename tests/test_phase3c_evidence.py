"""Adversarial regression tests for the Phase 3C evidence boundary."""

import hashlib
import json
from dataclasses import replace

import pytest

from signtranslator.planning.phase3c import (
    CANONICAL_PHASE2_REPRESENTATION,
    PHASE3C_SCHEMA_VERSION,
    EvidenceRole,
    ExternalEvidenceAttestation,
    LexicalMotionEntry,
    LexicalMotionLibrary,
    LexicalMotionReview,
    MotionLookupStatus,
    Phase3CWorkState,
    assess_phase3c_readiness,
    load_external_evidence_attestation,
    load_lexical_motion_library,
)
from signtranslator.planning.supervision import SupervisionBatchCertificate
from signtranslator.pretraining.dependence import (
    DependenceTestConfig,
    HeldOutInterventionScores,
    InterventionAudit,
    evaluate_video_dependence,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _review(index: int = 0, *, lexeme_id: int = 4) -> LexicalMotionReview:
    return LexicalMotionReview(
        creator_pseudonym="creator-1",
        reviewer_pseudonym="reviewer-1",
        creator_qualified_asl=True,
        reviewer_qualified_asl=True,
        reviewer_viewed_motion=True,
        reviewed_lexeme_id=lexeme_id,
        reviewed_asl_convention_sha256=_digest("asl-convention"),
        reviewed_sir_lexicon_sha256=_digest("sir-lexicon"),
        reviewed_phase2_schema_sha256=_digest("phase2-schema"),
        reviewed_motion_manifest_sha256=_digest(f"motion-manifest-{index}"),
        reviewed_motion_payload_sha256=_digest(f"motion-payload-{index}"),
        meaning_review_sha256=_digest("meaning-review"),
        articulation_review_sha256=_digest("articulation-review"),
        creator_qualification_evidence_sha256=_digest("creator-qualification"),
        reviewer_qualification_evidence_sha256=_digest("reviewer-qualification"),
        independence_evidence_sha256=_digest("independence"),
        review_protocol_sha256=_digest("review-protocol"),
        approved_at="2026-09-14T10:00:00-07:00",
    )


def _entry(
    index: int = 0,
    *,
    lexeme_id: int = 4,
    form_id: str | None = None,
    actions: tuple[str, ...] = (
        "commercial_deployment", "research_training"),
) -> LexicalMotionEntry:
    return LexicalMotionEntry(
        schema_version=PHASE3C_SCHEMA_VERSION,
        entry_id=f"entry-{index}",
        lexeme_id=lexeme_id,
        form_id=form_id or f"form-{index}",
        asl_convention_sha256=_digest("asl-convention"),
        sir_lexicon_sha256=_digest("sir-lexicon"),
        representation=CANONICAL_PHASE2_REPRESENTATION,
        phase2_schema_sha256=_digest("phase2-schema"),
        motion_manifest_sha256=_digest(f"motion-manifest-{index}"),
        motion_payload_sha256=_digest(f"motion-payload-{index}"),
        source_recording_id=f"source-{index}",
        signer_id_hash=f"signer-{index}",
        authorization_evidence_sha256=_digest(f"authorization-{index}"),
        permitted_actions=actions,
        review=_review(index, lexeme_id=lexeme_id),
    )


def _library(*entries: LexicalMotionEntry) -> LexicalMotionLibrary:
    items = entries or (_entry(),)
    return LexicalMotionLibrary(
        schema_version=PHASE3C_SCHEMA_VERSION,
        library_id="project-lexical-motion",
        version="1.0.0",
        asl_convention_sha256=_digest("asl-convention"),
        sir_lexicon_sha256=_digest("sir-lexicon"),
        phase2_schema_sha256=_digest("phase2-schema"),
        entries=tuple(items),
    )


def _evidence(role: EvidenceRole, *, artifact: str | None = None,
              index: int = 0) -> ExternalEvidenceAttestation:
    return ExternalEvidenceAttestation(
        schema_version=PHASE3C_SCHEMA_VERSION,
        evidence_id=f"evidence-{role.value}-{index}",
        role=role,
        artifact_sha256=artifact or _digest(f"artifact-{role.value}-{index}"),
        issuer_pseudonym=f"issuer-{index}",
        verifier_pseudonym=f"verifier-{index}",
        verifier_qualification_evidence_sha256=_digest(
            f"verifier-qualification-{index}"),
        verification_protocol_sha256=_digest(f"verification-protocol-{index}"),
        attestation_sha256=_digest(f"attestation-{role.value}-{index}"),
        issued_at="2026-09-14T09:00:00-07:00",
        verified_at="2026-09-14T10:00:00-07:00",
    )


def _video_dependence(*, passes: bool = True):
    n = 32
    audit = InterventionAudit(
        model_sha256=_digest("model"),
        preregistration_sha256=_digest("preregistration"),
        split_certificate_sha256=_digest("split"),
        aligned_input_manifest_sha256=_digest("aligned"),
        blank_video_manifest_sha256=_digest("blank"),
        shuffled_video_manifest_sha256=_digest("shuffled"),
        order_corruption_manifest_sha256=_digest("order"),
        text_only_manifest_sha256=_digest("text-only"),
        training_source_ids=("training-source",),
        training_signer_id_hashes=("training-signer",),
    )
    degraded = (0.5,) * n
    scores = HeldOutInterventionScores(
        sample_ids=tuple(f"sample-{i}" for i in range(n)),
        source_ids=tuple(f"heldout-source-{i}" for i in range(n)),
        signer_id_hashes=tuple(f"heldout-signer-{i}" for i in range(n)),
        shuffle_permutation=tuple((i + 1) % n for i in range(n)),
        aligned=((1.0,) * n if passes else degraded),
        blank_video=degraded,
        shuffled_video=degraded,
        order_corrupted_video=degraded,
        text_only=degraded,
    )
    return evaluate_video_dependence(
        scores,
        audit,
        DependenceTestConfig(
            familywise_alpha=0.05,
            minimum_effective_pairs=30,
            minimum_median_score_drop=0.25,
        ),
    )


def _supervision(*, passes: bool = True) -> SupervisionBatchCertificate:
    return SupervisionBatchCertificate(
        approved_for_research_training=passes,
        annotation_count=1,
        signer_count=1,
        source_recording_count=1,
        violations=() if passes else ("not_authorized",),
    )


def _all_evidence(library: LexicalMotionLibrary):
    return (
        _evidence(
            EvidenceRole.CANONICAL_PHASE2_STATE,
            artifact=library.phase2_schema_sha256,
            index=0,
        ),
        _evidence(EvidenceRole.QUALIFIED_ASL_VALIDATION, index=1),
        _evidence(EvidenceRole.RESEARCH_TRAINING_RIGHTS, index=2),
        _evidence(EvidenceRole.COMMERCIAL_DEPLOYMENT_RIGHTS, index=3),
    )


def test_current_report_is_software_complete_but_empirically_blocked():
    report = assess_phase3c_readiness()
    assert report.preartifact_software_boundary_complete
    assert not report.research_exit_approved
    assert not report.industrial_path_approved
    assert report.blockers == (
        "qualified_text_video_to_sir_supervision_absent",
        "direct_paired_learning_falsification_not_passed",
        "human_approved_lexical_motion_library_absent",
        "canonical_phase2_multichannel_state_absent",
        "independent_qualified_asl_validation_absent",
        "research_training_rights_absent",
        "commercial_training_and_deployment_rights_absent",
    )
    by_id = {item.component_id: item for item in report.work_items}
    assert not by_id["direct_paired_model_training"].software_contract_implemented
    assert by_id["direct_paired_model_training"].state \
        is Phase3CWorkState.EXTERNAL_ARTIFACT_BLOCKED
    assert not by_id["canonical_phase2_motion_export"].software_contract_implemented
    assert by_id["canonical_phase2_motion_export"].state \
        is Phase3CWorkState.PREDECESSOR_PHASE_BLOCKED
    assert all(item.required_artifacts and item.reason for item in report.work_items)


@pytest.mark.parametrize("representation", (
    "openpose_2d_137",
    "cartesian_27_joint",
    "generated_smplx_guess",
    "",
))
def test_noncanonical_or_inferred_motion_representations_are_rejected(
    representation,
):
    with pytest.raises(ValueError, match="canonical Phase-2"):
        replace(_entry(), representation=representation)


@pytest.mark.parametrize("change, message", (
    ({"reviewer_pseudonym": "creator-1"}, "distinct"),
    ({"creator_qualified_asl": False}, "qualified"),
    ({"reviewer_qualified_asl": False}, "qualified"),
    ({"reviewer_viewed_motion": False}, "direct motion review"),
    ({"approved_at": "2026-09-14T10:00:00"}, "timezone"),
    ({"meaning_review_sha256": "not-a-digest"}, "SHA-256"),
    ({"articulation_review_sha256": _digest("meaning-review")}, "roles must be distinct"),
))
def test_lexical_motion_review_fails_closed(change, message):
    with pytest.raises(ValueError, match=message):
        replace(_review(), **change)


def test_library_round_trip_and_hash_are_byte_stable():
    library = _library(_entry(0, lexeme_id=4), _entry(1, lexeme_id=5))
    restored = load_lexical_motion_library(library.canonical_bytes())
    assert restored == library
    assert restored.content_sha256() == library.content_sha256()
    assert restored.canonical_bytes() == library.canonical_bytes()


def test_library_rejects_empty_reordered_duplicate_and_cross_schema_entries():
    with pytest.raises(ValueError, match="bounded typed entries"):
        LexicalMotionLibrary(
            PHASE3C_SCHEMA_VERSION,
            "library",
            "1.0.0",
            _digest("asl-convention"),
            _digest("sir-lexicon"),
            _digest("phase2-schema"),
            (),
        )
    first = _entry(0, lexeme_id=4)
    second = _entry(1, lexeme_id=5)
    with pytest.raises(ValueError, match="canonically ordered"):
        _library(second, first)
    with pytest.raises(ValueError, match="identifiers must be unique"):
        _library(first, replace(second, entry_id=first.entry_id))
    with pytest.raises(ValueError, match="lexeme and form pairs"):
        _library(first, _entry(1, lexeme_id=4, form_id=first.form_id))
    with pytest.raises(ValueError, match="multiple forms"):
        _library(first, replace(
            second,
            motion_payload_sha256=first.motion_payload_sha256,
            review=replace(
                second.review,
                reviewed_motion_payload_sha256=first.motion_payload_sha256,
            ),
        ))
    with pytest.raises(ValueError, match="another governed schema"):
        other_schema = _digest("other-schema")
        _library(first, replace(
            second,
            phase2_schema_sha256=other_schema,
            review=replace(
                second.review,
                reviewed_phase2_schema_sha256=other_schema,
            ),
        ))
    with pytest.raises(ValueError, match="artifact roles must be distinct"):
        replace(
            first,
            authorization_evidence_sha256=first.motion_payload_sha256,
        )


@pytest.mark.parametrize("entry_field, review_field, replacement", (
    ("lexeme_id", "reviewed_lexeme_id", 99),
    ("asl_convention_sha256", "reviewed_asl_convention_sha256", _digest("other-asl")),
    ("sir_lexicon_sha256", "reviewed_sir_lexicon_sha256", _digest("other-lexicon")),
    ("phase2_schema_sha256", "reviewed_phase2_schema_sha256", _digest("other-phase2")),
    ("motion_manifest_sha256", "reviewed_motion_manifest_sha256", _digest("other-manifest")),
    ("motion_payload_sha256", "reviewed_motion_payload_sha256", _digest("other-motion")),
))
def test_review_cannot_be_replayed_onto_another_entry(
    entry_field,
    review_field,
    replacement,
):
    entry = _entry()
    assert getattr(entry.review, review_field) == getattr(entry, entry_field)
    with pytest.raises(ValueError, match="not bound"):
        replace(entry, **{entry_field: replacement})


def test_lookup_never_silently_selects_unknown_ambiguous_or_unauthorized_motion():
    resolved = _library(_entry()).lookup(4, "research_training")
    assert resolved.status is MotionLookupStatus.RESOLVED
    assert resolved.selected_entry_id == "entry-0"
    assert not resolved.abstained

    unknown = _library(_entry()).lookup(999, "research_training")
    assert unknown.status is MotionLookupStatus.UNKNOWN_LEXEME
    assert unknown.selected_entry_id is None and unknown.abstained

    research_only = _entry(actions=("research_training",))
    unauthorized = _library(research_only).lookup(4, "commercial_deployment")
    assert unauthorized.status is MotionLookupStatus.ACTION_NOT_AUTHORIZED
    assert unauthorized.selected_entry_id is None and unauthorized.abstained

    variants = _library(
        _entry(0, lexeme_id=4, form_id="form-a"),
        _entry(1, lexeme_id=4, form_id="form-b"),
    )
    ambiguous = variants.lookup(4, "research_training")
    assert ambiguous.status is MotionLookupStatus.AMBIGUOUS_FORM
    assert ambiguous.selected_entry_id is None and ambiguous.abstained


def test_loader_rejects_noncanonical_duplicate_key_nonfinite_and_oversize_bytes():
    payload = _library(_entry()).canonical_bytes()
    with pytest.raises(ValueError, match="not canonical"):
        load_lexical_motion_library(payload + b" ")
    duplicate = payload.replace(
        b'"library_id":"project-lexical-motion"',
        b'"library_id":"first","library_id":"project-lexical-motion"',
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_lexical_motion_library(duplicate)
    nonfinite = payload.replace(b'"lexeme_id":4', b'"lexeme_id":NaN')
    with pytest.raises(ValueError, match="non-finite"):
        load_lexical_motion_library(nonfinite)
    with pytest.raises(ValueError, match="byte limit"):
        load_lexical_motion_library(payload, max_bytes=len(payload) - 1)
    with pytest.raises(TypeError, match="immutable bytes"):
        load_lexical_motion_library(payload.decode("utf-8"))


def test_external_evidence_requires_independence_chronology_and_exact_schema():
    evidence = _evidence(EvidenceRole.RESEARCH_TRAINING_RIGHTS)
    restored = ExternalEvidenceAttestation.from_dict(evidence.to_dict())
    assert restored == evidence
    assert restored.content_sha256() == evidence.content_sha256()
    assert load_external_evidence_attestation(evidence.canonical_bytes()) == evidence
    with pytest.raises(ValueError, match="distinct"):
        replace(evidence, verifier_pseudonym=evidence.issuer_pseudonym)
    with pytest.raises(ValueError, match="follow issuance"):
        replace(evidence, verified_at=evidence.issued_at)
    with pytest.raises(ValueError, match="roles must be distinct"):
        replace(
            evidence,
            verification_protocol_sha256=evidence.artifact_sha256,
        )
    injected = evidence.to_dict()
    injected["license_assumed"] = True
    with pytest.raises(ValueError, match="fields must be exactly"):
        ExternalEvidenceAttestation.from_dict(injected)
    with pytest.raises(ValueError, match="not canonical"):
        load_external_evidence_attestation(evidence.canonical_bytes() + b"\n")
    duplicate = evidence.canonical_bytes().replace(
        b'"evidence_id":"evidence-research_training_rights-0"',
        b'"evidence_id":"first","evidence_id":"evidence-research_training_rights-0"',
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_external_evidence_attestation(duplicate)


def test_complete_supplied_research_evidence_passes_without_implying_deployment():
    library = _library(_entry(actions=("research_training",)))
    evidence = _all_evidence(library)[:-1]
    report = assess_phase3c_readiness(
        supervision=_supervision(),
        video_dependence=_video_dependence(),
        lexical_motion_library=library,
        external_evidence=evidence,
    )
    assert report.research_exit_approved
    assert not report.industrial_path_approved
    assert report.blockers == (
        "commercial_training_and_deployment_rights_absent",
        "lexical_motion_not_commercially_authorized",
    )


def test_industrial_path_requires_all_evidence_and_per_entry_commercial_scope():
    library = _library(_entry())
    report = assess_phase3c_readiness(
        supervision=_supervision(),
        video_dependence=_video_dependence(),
        lexical_motion_library=library,
        external_evidence=_all_evidence(library),
    )
    assert report.research_exit_approved
    assert report.industrial_path_approved
    assert report.blockers == ()


def test_failed_empirical_certificates_and_schema_mismatch_remain_blockers():
    library = _library(_entry())
    evidence = list(_all_evidence(library))
    evidence[0] = _evidence(
        EvidenceRole.CANONICAL_PHASE2_STATE,
        artifact=_digest("different-phase2-schema"),
    )
    report = assess_phase3c_readiness(
        supervision=_supervision(passes=False),
        video_dependence=_video_dependence(passes=False),
        lexical_motion_library=library,
        external_evidence=evidence,
    )
    assert not report.research_exit_approved
    assert "qualified_text_video_to_sir_supervision_absent" in report.blockers
    assert "direct_paired_learning_falsification_not_passed" in report.blockers
    assert "lexical_motion_phase2_schema_mismatch" in report.blockers


def test_readiness_rejects_duplicate_roles_and_artifact_role_reuse():
    first = _evidence(EvidenceRole.QUALIFIED_ASL_VALIDATION, index=1)
    with pytest.raises(ValueError, match="roles must be unique"):
        assess_phase3c_readiness(external_evidence=(first, first))
    second = _evidence(
        EvidenceRole.RESEARCH_TRAINING_RIGHTS,
        artifact=first.artifact_sha256,
        index=2,
    )
    with pytest.raises(ValueError, match="multiple evidence roles"):
        assess_phase3c_readiness(external_evidence=(first, second))


def test_json_manifest_contains_no_gloss_or_english_label_field():
    manifest = json.loads(_library(_entry()).canonical_bytes())
    encoded_keys = json.dumps(manifest, sort_keys=True)
    assert "gloss" not in encoded_keys.lower()
    assert "english" not in encoded_keys.lower()
    injected = dict(manifest)
    injected["gloss_tokens"] = ["FAKE"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        LexicalMotionLibrary.from_dict(injected)
