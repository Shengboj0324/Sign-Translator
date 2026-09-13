"""Adversarial verification of the governed, blocker-aware Phase 3 boundary."""

import hashlib
import itertools
import math
from dataclasses import replace

import pytest

from signtranslator.data_engineering.schema import (
    AuthorizationBasis,
    ConsentState,
    DataAuthorization,
    PersonalityRightsStatus,
    Sample,
)
from signtranslator.grammar.sir import EventKind, SIREvent, SIRGraph, sir_sha256
from signtranslator.planning.supervision import (
    PHASE3_UNSOLVED_BLOCKERS,
    ArtifactKind,
    GovernedArtifact,
    GovernedSIRAnnotation,
    IndependentReviewAttestation,
    SIRAnnotationOrigin,
    SourceEvidenceBinding,
    certify_supervision_batch,
    current_phase3_readiness,
    validate_annotation_against_sample,
)
from signtranslator.pretraining.dependence import (
    DependenceTestConfig,
    HeldOutInterventionScores,
    InterventionAudit,
    exact_one_sided_sign_pvalue,
    evaluate_video_dependence,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact(kind: ArtifactKind, name: str) -> GovernedArtifact:
    return GovernedArtifact(kind, name, "1.0.0", _digest(name))


def _authorization(*, training: bool = True) -> DataAuthorization:
    actions = ("download", "create_derivatives")
    if training:
        actions += ("model_training",)
    return DataAuthorization(
        basis=AuthorizationBasis.DIRECT_PARTICIPANT_CONSENT,
        license_identifier="PROJECT-CONSENT-1",
        license_url="https://example.invalid/project-consent-1",
        licensor="project-participant",
        evidence_uri="/evidence/consent.json",
        evidence_sha256=_digest("consent"),
        permitted_uses=("research",),
        permitted_actions=actions,
        personality_rights=PersonalityRightsStatus.VERIFIED,
    )


def _sample(index: int, *, signer: str | None = None, split: str = "train",
            training: bool = True) -> Sample:
    return Sample(
        sample_id=f"sample-{index}", source_id=f"source-{index}",
        signer_id_hash=signer or f"signer-{index}", target_language="ASL",
        license="PROJECT-CONSENT-1", consent=ConsentState.GRANTED,
        intended_use="research", smplx_version="unavailable",
        provenance=f"provenance-{index}", split=split,
        authorization=_authorization(training=training),
    )


def _annotation(index: int, sample: Sample, *, convention_name: str = "asl-v1",
                lexicon_name: str = "sir-lexicon-v1") -> GovernedSIRAnnotation:
    convention = _artifact(ArtifactKind.ASL_CONVENTION, convention_name)
    lexicon = _artifact(ArtifactKind.SIR_LEXICON, lexicon_name)
    graph = SIRGraph(events=[
        SIREvent(index, EventKind.MANUAL, index + 10, 0.0, 1.0),
    ])
    annotation_id = f"annotation-{index}"
    video_digest = _digest(f"video-{index}")
    transcript_digest = _digest(f"transcript-{index}")
    review = IndependentReviewAttestation(
        annotator_pseudonym=f"annotator-{index}",
        reviewer_pseudonym=f"reviewer-{index}",
        annotator_qualified_asl=True,
        reviewer_qualified_asl=True,
        annotator_viewed_source_video=True,
        reviewer_viewed_source_video=True,
        annotation_protocol=_artifact(
            ArtifactKind.ANNOTATION_PROTOCOL, "annotation-protocol"),
        review_protocol=_artifact(ArtifactKind.REVIEW_PROTOCOL, "review-protocol"),
        subject_annotation_id=annotation_id,
        reviewed_sir_sha256=sir_sha256(graph),
        reviewed_video_sha256=video_digest,
        reviewed_transcript_sha256=transcript_digest,
        reviewed_convention_sha256=convention.sha256,
        reviewed_lexicon_sha256=lexicon.sha256,
        annotator_qualification_evidence_sha256=_digest("annotator-qualification"),
        reviewer_qualification_evidence_sha256=_digest("reviewer-qualification"),
        independence_evidence_sha256=_digest("reviewer-independence"),
        attestation_sha256=_digest(f"attestation-{index}"),
        approved_at="2026-09-12T11:00:00-07:00",
    )
    source = SourceEvidenceBinding(
        sample_id=sample.sample_id,
        source_recording_id=sample.source_id,
        signer_id_hash=sample.signer_id_hash,
        video_sha256=video_digest,
        transcript_sha256=transcript_digest,
        sample_provenance=sample.provenance,
        authorization_evidence_sha256=sample.authorization.evidence_sha256,
    )
    return GovernedSIRAnnotation.create(
        annotation_id=annotation_id,
        origin=SIRAnnotationOrigin.PROJECT_HUMAN,
        source=source,
        convention=convention,
        lexicon=lexicon,
        lexicon_convention_sha256=convention.sha256,
        graph=graph,
        review=review,
        created_at="2026-09-12T10:00:00-07:00",
    )


def _observed(annotations):
    videos = {item.source.sample_id: item.source.video_sha256 for item in annotations}
    transcripts = {
        item.source.sample_id: item.source.transcript_sha256 for item in annotations}
    return videos, transcripts


def test_governed_annotation_round_trip_is_content_stable():
    sample = _sample(0)
    annotation = _annotation(0, sample)
    restored = GovernedSIRAnnotation.from_manifest(annotation.to_manifest())
    assert restored == annotation
    assert restored.content_sha256() == annotation.content_sha256()


def test_annotation_manifest_rejects_gloss_or_english_injection():
    annotation = _annotation(0, _sample(0))
    manifest = annotation.to_manifest()
    manifest["gloss_tokens"] = ["THIS", "IS", "ENGLISH"]
    with pytest.raises(ValueError, match="fields must be exactly"):
        GovernedSIRAnnotation.from_manifest(manifest)


def test_review_must_be_qualified_video_based_and_independent():
    protocol = _artifact(ArtifactKind.ANNOTATION_PROTOCOL, "annotation-protocol")
    review_protocol = _artifact(ArtifactKind.REVIEW_PROTOCOL, "review-protocol")
    base = dict(
        annotator_pseudonym="same", reviewer_pseudonym="same",
        annotator_qualified_asl=True, reviewer_qualified_asl=True,
        annotator_viewed_source_video=True, reviewer_viewed_source_video=True,
        annotation_protocol=protocol, review_protocol=review_protocol,
        subject_annotation_id="annotation-0",
        reviewed_sir_sha256=_digest("sir"),
        reviewed_video_sha256=_digest("video"),
        reviewed_transcript_sha256=_digest("transcript"),
        reviewed_convention_sha256=_digest("convention"),
        reviewed_lexicon_sha256=_digest("lexicon"),
        annotator_qualification_evidence_sha256=_digest("annotator-qualification"),
        reviewer_qualification_evidence_sha256=_digest("reviewer-qualification"),
        independence_evidence_sha256=_digest("reviewer-independence"),
        attestation_sha256=_digest("attestation"),
        approved_at="2026-09-12T11:00:00Z",
    )
    with pytest.raises(ValueError, match="distinct"):
        IndependentReviewAttestation(**base)
    base["reviewer_pseudonym"] = "reviewer"
    base["reviewer_qualified_asl"] = False
    with pytest.raises(ValueError, match="qualified"):
        IndependentReviewAttestation(**base)


def test_sir_payload_tampering_and_noncanonical_bytes_are_rejected():
    annotation = _annotation(0, _sample(0))
    with pytest.raises(ValueError):
        replace(annotation, sir_payload=annotation.sir_payload + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(annotation, sir_payload_sha256=_digest("wrong"))


def test_lexicon_must_bind_to_the_exact_convention():
    annotation = _annotation(0, _sample(0))
    with pytest.raises(ValueError, match="not bound"):
        replace(annotation, lexicon_convention_sha256=_digest("other-convention"))


@pytest.mark.parametrize("field", [
    "subject_annotation_id", "reviewed_sir_sha256", "reviewed_video_sha256",
    "reviewed_transcript_sha256", "reviewed_convention_sha256",
    "reviewed_lexicon_sha256",
])
def test_review_attestation_cannot_be_replayed_against_other_content(field):
    annotation = _annotation(0, _sample(0))
    replacement = "different-annotation" if field == "subject_annotation_id" \
        else _digest(f"different-{field}")
    replayed_review = replace(annotation.review, **{field: replacement})
    with pytest.raises(ValueError, match="not bound"):
        replace(annotation, review=replayed_review)


def test_sample_binding_and_training_authorization_fail_closed():
    sample = _sample(0, training=False)
    annotation = _annotation(0, sample)
    violations = validate_annotation_against_sample(
        annotation, sample, observed_video_sha256=_digest("different-video"),
        observed_transcript_sha256=annotation.source.transcript_sha256)
    assert "video_sha256_mismatch" in violations
    assert "action_not_permitted:model_training" in violations


def test_sample_must_be_canonically_valid_and_explicitly_asl():
    valid = _sample(0)
    annotation = _annotation(0, valid)
    wrong_language = replace(valid, target_language="BSL")
    violations = validate_annotation_against_sample(
        annotation, wrong_language,
        observed_video_sha256=annotation.source.video_sha256,
        observed_transcript_sha256=annotation.source.transcript_sha256,
    )
    assert "target_language_not_asl" in violations

    mismatched_license = replace(valid, license="DIFFERENT-LICENSE")
    violations = validate_annotation_against_sample(
        annotation, mismatched_license,
        observed_video_sha256=annotation.source.video_sha256,
        observed_transcript_sha256=annotation.source.transcript_sha256,
    )
    assert "sample_schema:authorization_license_mismatch" in violations


def test_supervision_batch_passes_only_uniform_leakage_free_records():
    samples = [_sample(0, split="train"), _sample(1, split="test")]
    annotations = [_annotation(i, sample) for i, sample in enumerate(samples)]
    videos, transcripts = _observed(annotations)
    certificate = certify_supervision_batch(
        annotations, {item.sample_id: item for item in samples}, videos, transcripts)
    assert certificate.approved_for_research_training
    assert certificate.violations == ()


def test_supervision_batch_rejects_mixed_conventions_and_signer_leakage():
    samples = [
        _sample(0, signer="same-signer", split="train"),
        _sample(1, signer="same-signer", split="test"),
    ]
    annotations = [
        _annotation(0, samples[0]),
        _annotation(1, samples[1], convention_name="asl-v2",
                    lexicon_name="sir-lexicon-v2"),
    ]
    videos, transcripts = _observed(annotations)
    certificate = certify_supervision_batch(
        annotations, {item.sample_id: item for item in samples}, videos, transcripts)
    assert not certificate.approved_for_research_training
    assert "mixed_asl_conventions" in certificate.violations
    assert "mixed_sir_lexicons" in certificate.violations
    assert "signer_split_leakage:same-signer" in certificate.violations


def test_current_phase3_readiness_is_explicitly_blocked():
    report = current_phase3_readiness()
    assert not report.research_exit_approved
    assert not report.industrial_path_approved
    assert report.unsolved_blockers == PHASE3_UNSOLVED_BLOCKERS
    assert len(report.unsolved_blockers) == 6


def _brute_sign_tail(wins: int, n: int) -> float:
    outcomes = itertools.product((0, 1), repeat=n)
    return sum(sum(outcome) >= wins for outcome in outcomes) / (2 ** n)


@pytest.mark.parametrize("n", range(1, 9))
def test_sign_test_matches_brute_force_distribution(n):
    for wins in range(n + 1):
        assert exact_one_sided_sign_pvalue(wins, n) == pytest.approx(
            _brute_sign_tail(wins, n), abs=1e-14)


def test_dependence_threshold_must_be_preregistered_and_positive():
    with pytest.raises(ValueError, match="positive and preregistered"):
        DependenceTestConfig(0.01, 30, 0.0)


def _audit(*, training_source: str = "training-source") -> InterventionAudit:
    digest_names = {
        name: _digest(name) for name in (
            "model", "preregistration", "split", "aligned", "blank", "shuffled",
            "order", "text-only",
        )
    }
    return InterventionAudit(
        model_sha256=digest_names["model"],
        preregistration_sha256=digest_names["preregistration"],
        split_certificate_sha256=digest_names["split"],
        aligned_input_manifest_sha256=digest_names["aligned"],
        blank_video_manifest_sha256=digest_names["blank"],
        shuffled_video_manifest_sha256=digest_names["shuffled"],
        order_corruption_manifest_sha256=digest_names["order"],
        text_only_manifest_sha256=digest_names["text-only"],
        training_source_ids=(training_source,),
        training_signer_id_hashes=("training-signer",),
    )


def _scores(*, n: int = 40, no_effect: str | None = None,
            heldout_source_prefix: str = "heldout-source") -> HeldOutInterventionScores:
    aligned = (1.0,) * n
    degraded = (0.5,) * n
    conditions = {
        "blank_video": degraded,
        "shuffled_video": degraded,
        "order_corrupted_video": degraded,
        "text_only": degraded,
    }
    if no_effect is not None:
        conditions[no_effect] = aligned
    return HeldOutInterventionScores(
        sample_ids=tuple(f"heldout-{index}" for index in range(n)),
        source_ids=tuple(f"{heldout_source_prefix}-{index}" for index in range(n)),
        signer_id_hashes=tuple(f"heldout-signer-{index}" for index in range(n)),
        shuffle_permutation=tuple((index + 1) % n for index in range(n)),
        aligned=aligned,
        **conditions,
    )


def test_video_dependence_certificate_requires_all_interventions_and_disjointness():
    config = DependenceTestConfig(
        familywise_alpha=0.01, minimum_effective_pairs=30,
        minimum_median_score_drop=0.25)
    passed = evaluate_video_dependence(_scores(), _audit(), config)
    assert passed.passed
    assert all(result.passed for result in passed.results)
    assert all(result.corrected_alpha == pytest.approx(0.0025)
               for result in passed.results)

    failed_intervention = evaluate_video_dependence(
        _scores(no_effect="text_only"), _audit(), config)
    assert not failed_intervention.passed
    assert not failed_intervention.results[-1].passed

    source_leak = evaluate_video_dependence(
        _scores(heldout_source_prefix="training-source"),
        _audit(training_source="training-source-0"), config)
    assert not source_leak.passed
    assert not source_leak.source_disjoint


def test_intervention_scores_reject_nan_and_invalid_shuffle():
    scores = _scores()
    with pytest.raises(ValueError, match="sample_ids must be a tuple"):
        replace(scores, sample_ids=None)
    with pytest.raises(ValueError, match="derangement"):
        replace(scores, shuffle_permutation=tuple(range(len(scores.sample_ids))))
    bad = list(scores.aligned)
    bad[0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        replace(scores, aligned=tuple(bad))


def test_corpus_scale_sign_test_retains_finite_log_evidence():
    certificate = evaluate_video_dependence(
        _scores(n=2_000), _audit(),
        DependenceTestConfig(0.01, 30, 0.25))
    assert certificate.passed
    assert certificate.results[0].one_sided_sign_pvalue == 0.0
    assert math.isfinite(certificate.results[0].one_sided_sign_log10_pvalue)
    assert certificate.results[0].one_sided_sign_log10_pvalue < -500
