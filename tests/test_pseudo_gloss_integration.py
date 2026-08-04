"""Integration, model-isolation, calibration, artifact, and falsification tests."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
import signtranslator.pseudo_gloss.cli as pseudo_cli_module
import signtranslator.pseudo_gloss.corpus as pseudo_corpus_module
import signtranslator.pseudo_gloss.inference as pseudo_inference_module

from signtranslator.data_engineering import (
    AuthorizationBasis,
    ConsentState,
    DataAuthorization,
    ExtractedSample,
    LandmarkTrack,
    PersonalityRightsStatus,
    Sample,
    promote_reviewed_weak_candidate,
    validate_sample,
)
from signtranslator.pseudo_gloss import (
    AbstentionConfig,
    ActivationCharter,
    AcceptanceFeatures,
    ArtifactBinding,
    CalibrationCertificate,
    CandidateProvenance,
    FusionConfig,
    GlossLexicon,
    HybridPseudoGlossPipeline,
    HumanReferenceCase,
    HumanGlossAnnotation,
    InputSecurityPolicy,
    InterventionSpecification,
    LabelType,
    LogisticAcceptanceCalibrator,
    ModelGovernance,
    NeuralTextProposalModel,
    NoisyLabelObjectiveCertificate,
    OptimizationConfig,
    REQUIRED_FALSIFICATION_TESTS,
    ReviewStatus,
    runtime_environment,
    SourceTokenizer,
    TextProposalConfig,
    TextInitializationEvidence,
    TextTrainingExample,
    VideoCTCEvidenceModel,
    VideoCandidateLatticeTrainingExample,
    VideoEvidenceConfig,
    VideoTrainingExample,
    WeakGlossCandidateRecord,
    WeakSupervisionEvidence,
    apply_video_intervention,
    assign_source_folds,
    assess_activation,
    build_falsification_report,
    candidate_deletion_abstains,
    candidate_lattice_video_loss,
    certify_cross_fit,
    certify_vocabulary_holdout,
    deterministic_source_derangement,
    evaluate_calibration,
    evaluate_human_references,
    fit_text_model,
    fit_video_model,
    paired_source_bootstrap,
    prepare_openpose_features,
    state_dict_sha256,
    source_cluster_calibration_uncertainty,
    token_error_counts,
    training_indices_for_fold,
    load_pipeline_bundle,
    run_corpus_inference,
    verify_candidate_batch,
    verify_bundle,
    validate_training_annotation,
    write_candidate_batch,
    write_bundle,
)
from signtranslator.pseudo_gloss.cli import main as pseudo_gloss_main


DIGEST = "b" * 64


def _dataset_authorization(tmp_path):
    from signtranslator.pseudo_gloss.artifacts import sha256_file

    evidence = tmp_path / "dataset-license-evidence.txt"
    evidence.write_text("unit-test noncommercial research license evidence", encoding="utf-8")
    authorization = DataAuthorization(
        basis=AuthorizationBasis.PUBLISHED_DATASET_LICENSE,
        license_identifier="CC-BY-NC-4.0",
        license_url="https://creativecommons.org/licenses/by-nc/4.0/",
        licensor="unit-test publisher", evidence_uri=str(evidence),
        evidence_sha256=sha256_file(evidence), permitted_uses=("research",),
        permitted_actions=("download", "create_derivatives", "model_training"),
        personality_rights=PersonalityRightsStatus.NOT_VERIFIED,
        attribution_notice="Unit-test dataset citation",
        limitations=("personality rights are not independently verified",),
    )
    path = tmp_path / "dataset-authorization.json"
    path.write_text(json.dumps({
        "schema_version": 1, "intended_use": "research",
        "consent": ConsentState.NOT_DIRECTLY_VERIFIED.name,
        "authorization": authorization.to_manifest(),
    }), encoding="utf-8")
    return path, evidence


def _lexicon():
    return GlossLexicon("asl-unit-v1", "unit-convention-v1",
                        ("UNKNOWN", "HELLO", "BOOK"), DIGEST)


def _tokenizer():
    return SourceTokenizer("english-unit-v1", ("hello", "book", "."), DIGEST)


def _track(frames=8):
    values = np.linspace(0.1, 0.9, 2 * frames * 137, dtype=np.float32).reshape(2, frames, 137)
    confidence = np.full((frames, 137), 0.8, dtype=np.float32)
    validity = np.ones((frames, 137), dtype=np.bool_)
    timestamps = np.arange(frames, dtype=np.float64) / 25.0
    return LandmarkTrack(values, confidence, validity, timestamps)


def _models():
    text = NeuralTextProposalModel(
        _tokenizer(), _lexicon(), TextProposalConfig(
            embedding_dim=8, feedforward_dim=16, layers=1, heads=2,
            beam_size=4, max_candidate_tokens=3, max_source_tokens=16))
    video = VideoCTCEvidenceModel(
        _lexicon(), VideoEvidenceConfig(hidden_channels=4, blocks=1, temporal_kernel=3))
    # A deterministic, deliberately untrained checkpoint for interface tests.
    with torch.no_grad():
        for parameter in text.parameters():
            parameter.zero_()
        text.output.bias.copy_(torch.tensor([-0.3, -4.0, 1.0, 0.5]))
        for parameter in video.parameters():
            parameter.zero_()
    return text, video


def _pipeline(calibrator=None):
    text, video = _models()
    pipeline = HybridPseudoGlossPipeline(
        text, video, FusionConfig(alpha=1.0, beta=0.2,
                                  alignment_entropy_penalty=0.1, candidate_limit=4),
        AbstentionConfig(acceptance_threshold=0.6, minimum_top_posterior=0.0,
                         maximum_dropped_text_mass=1.0),
        InputSecurityPolicy(max_words=16, max_candidate_tokens=3, max_candidates=4),
        calibrator,
    )
    pipeline.freeze_for_inference()
    return pipeline


def _provenance(sample_id="sample-1"):
    from signtranslator.pseudo_gloss.contracts import canonical_json_bytes, sha256_bytes

    return CandidateProvenance(
        source_sample_id=sample_id, source_video_sha256=DIGEST,
        transcript_sha256=DIGEST, visual_feature_sha256=DIGEST,
        generator_model_id="model", model_weight_sha256=DIGEST,
        tokenizer_sha256=DIGEST, prompt_or_template_sha256=DIGEST,
        decoding_config_sha256=DIGEST,
        environment_sha256=sha256_bytes(canonical_json_bytes(runtime_environment())),
        code_revision="revision", random_seed=1,
        created_at="2026-08-04T00:00:00Z",
    )


def _candidate(label_type=LabelType.UNREVIEWED_PSEUDO,
               review_status=ReviewStatus.UNREVIEWED, sample_id="sample-1"):
    human = label_type is not LabelType.UNREVIEWED_PSEUDO
    return WeakGlossCandidateRecord(
        annotation_id=f"ann-{label_type.value}", label_type=label_type,
        review_status=review_status, lexicon_id="asl-unit-v1",
        convention_id="unit-convention-v1", candidate_tokens=("HELLO",),
        candidate_log_score=-0.2, candidate_rank=1, provenance=_provenance(sample_id),
        human_annotator_pseudonym="reviewer" if human else None,
        human_review_protocol="protocol-v1" if human else None,
        review_attestation_sha256=DIGEST if human else None,
        reviewer_qualified_asl=human, source_video_reviewed=human,
        parent_annotation_ids=("machine-parent",)
        if label_type is LabelType.HUMAN_CORRECTED_PSEUDO else (),
    )


def _human_annotation(sample_id="sample-human"):
    return HumanGlossAnnotation(
        annotation_id=f"human-{sample_id}", source_sample_id=sample_id,
        source_video_sha256=DIGEST, label_type=LabelType.PROJECT_HUMAN,
        review_status=ReviewStatus.APPROVED, lexicon_id="asl-unit-v1",
        convention_id="unit-convention-v1", tokens=("HELLO",),
        annotator_pseudonym="reviewer", review_protocol="protocol-v1",
        review_attestation_sha256=DIGEST, reviewer_qualified_asl=True,
        source_video_reviewed=True)


def test_hybrid_pipeline_outputs_separate_machine_records_and_abstains_uncalibrated():
    result = _pipeline().generate(
        transcript="hello book.", track=_track(), source_sample_id="sample-1",
        source_video_sha256=DIGEST, generator_model_id="unit-text+unit-video",
        code_revision="revision", random_seed=3,
        created_at="2026-08-04T00:00:00Z")
    assert result.records
    assert not result.decision.accepted
    assert result.decision.reason == "uncalibrated_acceptance"
    assert result.selected_annotation_id is None
    assert all(record.label_type is LabelType.UNREVIEWED_PSEUDO for record in result.records)
    assert all(record.review_status is ReviewStatus.UNREVIEWED for record in result.records)
    assert result.machine_only and not result.translation_claim


def test_pipeline_is_deterministic_and_video_model_is_transcript_independent(monkeypatch):
    pipeline = _pipeline()
    original_generate = pipeline._generate_deterministic
    deterministic_states = []

    def observe_determinism(**arguments):
        deterministic_states.append(torch.are_deterministic_algorithms_enabled())
        return original_generate(**arguments)

    monkeypatch.setattr(pipeline, "_generate_deterministic", observe_determinism)
    arguments = dict(
        transcript="hello book.", track=_track(), source_sample_id="sample-1",
        source_video_sha256=DIGEST, generator_model_id="unit-text+unit-video",
        code_revision="revision", random_seed=3,
        created_at="2026-08-04T00:00:00Z")
    first = pipeline.generate(**arguments)
    second = pipeline.generate(**arguments)
    assert first == second
    assert deterministic_states == [True, True]
    features = torch.cat((
        torch.from_numpy(_track().values),
        torch.from_numpy(_track().confidence)[None],
        torch.from_numpy(_track().validity_mask.astype(np.float32))[None],
    )).unsqueeze(0)
    with torch.no_grad():
        before = pipeline.video_model(features)
        after = pipeline.video_model(features.clone())
    assert torch.equal(before, after)


def test_blank_video_is_fail_closed_not_zero_filled_evidence():
    pipeline = _pipeline()
    blank = apply_video_intervention(_track(), "blank")
    result = pipeline.generate(
        transcript="hello book.", track=blank, source_sample_id="sample-1",
        source_video_sha256=DIGEST, generator_model_id="unit-hybrid",
        code_revision="revision", random_seed=3,
        created_at="2026-08-04T00:00:00Z")
    assert not result.records and not result.decision.accepted
    assert result.decision.reason == "no_visual_evidence"


def test_wholly_missing_pose_frames_do_not_create_ctc_alignment_slots():
    _, video = _models()
    track = _track()
    track.validity_mask[2:] = False
    track.confidence[2:] = 0
    features, frame_validity = prepare_openpose_features(track)
    target = torch.tensor([1, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="infeasible"):
        video.loss(
            features.unsqueeze(0), target, torch.tensor([2]), torch.tensor([2]),
            frame_validity.unsqueeze(0),
        )
    with pytest.raises(ValueError, match="non-blank lexicon"):
        video.loss(
            features.unsqueeze(0), torch.tensor([0]), torch.tensor([1]),
            torch.tensor([2]), frame_validity.unsqueeze(0))


def _feature(value):
    return AcceptanceFeatures(
        top_posterior=value, posterior_margin=value / 2,
        normalized_video_log_probability=-1 + value,
        normalized_path_entropy=1 - value,
        mean_blank_posterior=1 - value,
        dropped_text_mass=(1 - value) / 2,
    )


def test_calibration_requires_qualified_source_disjoint_reference_and_roundtrips():
    features = [_feature(value) for value in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9)]
    labels = [False, False, False, True, True, True]
    invalid = CalibrationCertificate(DIGEST, "protocol", 6, True, False)
    with pytest.raises(PermissionError, match="qualified"):
        LogisticAcceptanceCalibrator().fit(features, labels, invalid)
    certificate = CalibrationCertificate(DIGEST, "protocol", 6, True, True)
    calibrator = LogisticAcceptanceCalibrator().fit(features, labels, certificate)
    assert calibrator.predict(_feature(0.9)) > calibrator.predict(_feature(0.1))
    restored = LogisticAcceptanceCalibrator.from_state(calibrator.state())
    assert restored.state_sha256() == calibrator.state_sha256()


def test_calibration_metrics_reliability_bins_slices_and_cluster_uncertainty():
    probabilities = [0.1, 0.4, 0.8, 0.9]
    labels = [False, False, True, True]
    sources = ["source-a", "source-a", "source-b", "source-c"]
    evaluation = evaluate_calibration(probabilities, labels, sources, [0.0, 0.5, 1.0])
    assert evaluation.brier_score == pytest.approx(0.055)
    assert evaluation.expected_calibration_error == pytest.approx(0.2)
    assert evaluation.maximum_calibration_error == pytest.approx(0.25)
    assert [item.count for item in evaluation.bins] == [2, 2]
    assert [item.source_id for item in evaluation.source_slices] == [
        "source-a", "source-b", "source-c"]
    uncertainty = source_cluster_calibration_uncertainty(
        probabilities, labels, sources, [0.0, 0.5, 1.0],
        bootstrap_replicates=300, seed=19)
    repeated = source_cluster_calibration_uncertainty(
        probabilities, labels, sources, [0.0, 0.5, 1.0],
        bootstrap_replicates=300, seed=19)
    assert uncertainty == repeated and uncertainty.source_group_count == 3
    assert math.isinf(evaluate_calibration(
        [0.0, 1.0], [True, False], ["a", "b"], [0.0, 0.5, 1.0]).log_loss)


def test_source_cross_fit_has_no_group_leakage_and_is_invariant_to_duplicates():
    source_ids = ["a", "a", "b", "c", "d", "d"]
    assignments = assign_source_folds(source_ids, folds=3, seed=9)
    assert certify_cross_fit(assignments)
    assert assignments[0].fold == assignments[1].fold
    train, held = training_indices_for_fold(assignments, assignments[0].fold)
    assert {source_ids[index] for index in train}.isdisjoint(
        {source_ids[index] for index in held})


def test_training_rejects_unreviewed_pseudo_targets_before_optimization():
    text, _ = _models()
    train = [TextTrainingExample("train-source", "sample-1", "hello", _candidate())]
    validation = [TextTrainingExample("val-source", "sample-1", "hello", _candidate())]
    before = [parameter.detach().clone() for parameter in text.parameters()]
    with pytest.raises(PermissionError, match="unreviewed"):
        fit_text_model(
            text, train, validation, OptimizationConfig(epochs=1),
            TextInitializationEvidence(
                True, "unit-pretrained", "test-only", DIGEST,
                state_dict_sha256(text)))
    assert all(torch.equal(old, new) for old, new in zip(before, text.parameters()))


def test_human_corrected_pseudo_training_requires_crossfit_and_anti_circularity_evidence():
    record = _candidate(LabelType.HUMAN_CORRECTED_PSEUDO, ReviewStatus.APPROVED)
    with pytest.raises(PermissionError, match="weak-supervision evidence"):
        validate_training_annotation(record, source_group_id="held-source")
    evidence = WeakSupervisionEvidence(
        annotation_id=record.annotation_id,
        candidate_model_weight_sha256=record.provenance.model_weight_sha256,
        candidate_text_state_sha256="1" * 64,
        candidate_video_state_sha256="2" * 64,
        generator_training_source_ids=("train-source",),
        generator_held_out_source_ids=("held-source",),
        reference_set_sha256="3" * 64,
        falsification_report_sha256="4" * 64,
        calibrator_state_sha256="5" * 64,
        qualified_asl_reference=True,
        all_required_falsification_tests_passed=True,
        confidence_calibrated=True,
    )
    validate_training_annotation(
        record, source_group_id="held-source", weak_evidence=evidence,
        current_text_state_sha256="6" * 64, current_video_state_sha256="7" * 64)
    with pytest.raises(PermissionError, match="self-train"):
        validate_training_annotation(
            record, source_group_id="held-source", weak_evidence=evidence,
            current_video_state_sha256="2" * 64)
    with pytest.raises(PermissionError, match="cross-fitting"):
        validate_training_annotation(
            record, source_group_id="train-source", weak_evidence=evidence,
            current_video_state_sha256="7" * 64)


def test_multi_candidate_video_objective_is_connected_and_circularity_guarded():
    _, video = _models()
    first = _candidate(sample_id="sample-1")
    second = replace(
        first, annotation_id="ann-second", candidate_tokens=("BOOK",),
        candidate_rank=2)
    example = VideoCandidateLatticeTrainingExample(
        source_id="held-source", sample_id="sample-1", track=_track(),
        candidates=(first, second), candidate_weights=(0.6, 0.4),
        calibrated_confidence=0.8)
    evidence = {
        record.annotation_id: WeakSupervisionEvidence(
            annotation_id=record.annotation_id,
            candidate_model_weight_sha256=record.provenance.model_weight_sha256,
            candidate_text_state_sha256="1" * 64,
            candidate_video_state_sha256="2" * 64,
            generator_training_source_ids=("train-source",),
            generator_held_out_source_ids=("held-source",),
            reference_set_sha256="3" * 64,
            falsification_report_sha256="4" * 64,
            calibrator_state_sha256="5" * 64,
            qualified_asl_reference=True,
            all_required_falsification_tests_passed=True,
            confidence_calibrated=True)
        for record in (first, second)
    }
    objective_certificate = NoisyLabelObjectiveCertificate(
        reference_set_sha256="3" * 64,
        calibration_artifact_sha256="5" * 64,
        falsification_report_sha256="4" * 64,
        qualified_asl_reference=True, source_disjoint=True, cross_fitted=True,
        all_required_falsification_tests_passed=True, confidence_calibrated=True)
    loss = candidate_lattice_video_loss(
        video, [example], evidence, objective_certificate)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in video.parameters())
    circular = dict(evidence)
    circular[first.annotation_id] = replace(
        evidence[first.annotation_id],
        candidate_video_state_sha256=state_dict_sha256(video))
    with pytest.raises(PermissionError, match="cannot optimize"):
        candidate_lattice_video_loss(
            video, [example], circular, objective_certificate)


def test_approved_human_text_training_has_finite_connected_optimization():
    text, _ = _models()
    train_candidate = _human_annotation("sample-train")
    validation_candidate = _human_annotation("sample-val")
    history = fit_text_model(
        text,
        [TextTrainingExample("train-source", "sample-train", "hello", train_candidate)],
        [TextTrainingExample("val-source", "sample-val", "hello", validation_candidate)],
        OptimizationConfig(epochs=2, learning_rate=1e-3, seed=4),
        TextInitializationEvidence(
            True, "unit-pretrained", "test-only", DIGEST,
            state_dict_sha256(text)))
    assert len(history.train_loss) == 2
    assert all(math.isfinite(value) for value in history.train_loss + history.validation_loss)


def test_from_scratch_text_training_is_rejected_and_video_ctc_training_is_finite():
    text, video = _models()
    train_annotation = _human_annotation("sample-train")
    validation_annotation = _human_annotation("sample-val")
    with pytest.raises(PermissionError, match="from-scratch"):
        fit_text_model(
            text,
            [TextTrainingExample("train-source", "sample-train", "hello", train_annotation)],
            [TextTrainingExample("val-source", "sample-val", "hello", validation_annotation)],
            OptimizationConfig(epochs=1),
            TextInitializationEvidence(
                False, "unit-random", "test-only", DIGEST,
                state_dict_sha256(text)))
    with pytest.raises(PermissionError, match="weights do not match"):
        fit_text_model(
            text,
            [TextTrainingExample("train-source", "sample-train", "hello", train_annotation)],
            [TextTrainingExample("val-source", "sample-val", "hello", validation_annotation)],
            OptimizationConfig(epochs=1),
            TextInitializationEvidence(
                True, "unit-pretrained", "test-only", DIGEST, DIGEST))
    history = fit_video_model(
        video,
        [VideoTrainingExample("train-source", "sample-train", _track(), train_annotation)],
        [VideoTrainingExample("val-source", "sample-val", _track(), validation_annotation)],
        OptimizationConfig(epochs=1, learning_rate=1e-3, seed=5))
    assert len(history.train_loss) == 1
    assert math.isfinite(history.train_loss[0]) and math.isfinite(history.validation_loss[0])


def test_candidate_attachment_does_not_silently_become_gloss():
    authorization = DataAuthorization(
        basis=AuthorizationBasis.PUBLISHED_DATASET_LICENSE,
        license_identifier="CC-BY-NC-4.0", license_url="https://example.org/license",
        licensor="publisher", evidence_uri="/evidence", evidence_sha256=DIGEST,
        permitted_uses=("noncommercial research",),
        permitted_actions=("download", "create_derivatives", "model_training"),
        personality_rights=PersonalityRightsStatus.NOT_VERIFIED,
        attribution_notice="citation", limitations=("personality rights unverified",))
    unreviewed = _candidate()
    sample = Sample(
        sample_id="sample-1", source_id="source", signer_id_hash="signer",
        target_language="ASL", license="CC-BY-NC-4.0",
        consent=ConsentState.NOT_DIRECTLY_VERIFIED,
        intended_use="noncommercial research", smplx_version="none",
        provenance=DIGEST, split="train", authorization=authorization,
        weak_gloss_candidates=(unreviewed,))
    assert validate_sample(sample) == []
    extracted = ExtractedSample(
        sample, _track(), (), ("hello",), DIGEST, "openpose", "image_fraction")
    with pytest.raises(PermissionError, match="approved"):
        promote_reviewed_weak_candidate(extracted, unreviewed.annotation_id)


def test_human_corrected_candidate_promotion_preserves_annotation_binding():
    reviewed = _candidate(LabelType.HUMAN_CORRECTED_PSEUDO, ReviewStatus.APPROVED)
    sample = Sample(
        sample_id="sample-1", source_id="source", signer_id_hash="signer",
        target_language="ASL", license="license", consent=ConsentState.GRANTED,
        intended_use="research", smplx_version="none", provenance=DIGEST,
        split="train", weak_gloss_candidates=(reviewed,))
    extracted = ExtractedSample(
        sample, _track(), (), ("hello",), DIGEST, "openpose", "image_fraction")
    promoted = promote_reviewed_weak_candidate(extracted, reviewed.annotation_id)
    assert promoted.gloss_tokens == ("HELLO",)
    assert promoted.gloss_annotation_id == reviewed.annotation_id


def test_bundle_is_content_addressed_chain_verified_and_tamper_evident(tmp_path):
    pipeline = _pipeline()
    result = pipeline.generate(
        transcript="hello book.", track=_track(), source_sample_id="sample-1",
        source_video_sha256=DIGEST, generator_model_id="unit-text+unit-video",
        code_revision="revision", random_seed=3,
        created_at="2026-08-04T00:00:00Z")
    bundle = tmp_path / "bundle"
    governance = ModelGovernance(
        text_model_id="unit-text", text_model_license="test-only",
        text_model_source="local-test", video_model_id="unit-video",
        text_model_artifact_sha256=DIGEST,
        text_model_license_evidence_sha256=DIGEST,
        video_model_license="test-only", video_model_source="local-test",
        video_model_artifact_sha256=DIGEST,
        video_model_license_evidence_sha256=DIGEST,
        training_data_manifest_sha256=DIGEST, dependency_lock_sha256=DIGEST,
        sbom_sha256=DIGEST, intended_use="unit testing")
    manifest_path = write_bundle(
        bundle, pipeline=pipeline, records=(), governance=governance,
        seed=3, code_revision="revision")
    manifest = verify_bundle(bundle)
    assert manifest_path.name == "manifest.json"
    assert manifest["candidate_count"] == 0
    assert not manifest["production_gloss_export_authorized"]
    restored = load_pipeline_bundle(bundle)
    restored_result = restored.generate(
        transcript="hello book.", track=_track(), source_sample_id="sample-1",
        source_video_sha256=DIGEST, generator_model_id="unit-text+unit-video",
        code_revision="revision", random_seed=3,
        created_at="2026-08-04T00:00:00Z")
    assert restored_result == result
    batch = tmp_path / "candidate-batch"
    decision = {
        **asdict(result.decision),
        "selected_annotation_id": result.selected_annotation_id,
        "dropped_text_probability_mass": result.dropped_text_probability_mass,
    }
    from signtranslator.pseudo_gloss.artifacts import sha256_file
    authorization, _ = _dataset_authorization(tmp_path)
    write_candidate_batch(
        batch, records=result.records, decision=decision,
        model_bundle_manifest_sha256=sha256_file(bundle / "manifest.json"),
        dataset_authorization_sha256=sha256_file(authorization),
        source_sample_id="sample-1",
        transcript_sha256=hashlib.sha256(b"hello book.").hexdigest(),
        source_video_sha256=DIGEST, landmark_track_sha256="c" * 64,
        code_revision="revision")
    assert verify_candidate_batch(
        batch, model_bundle=bundle,
        dataset_authorization=authorization)["record_count"] == len(result.records)
    invalid_probability = dict(decision)
    invalid_probability["calibrated_probability"] = math.nan
    with pytest.raises(ValueError, match="calibrated probability"):
        write_candidate_batch(
            tmp_path / "invalid-probability", records=result.records,
            decision=invalid_probability,
            model_bundle_manifest_sha256=sha256_file(bundle / "manifest.json"),
            dataset_authorization_sha256=sha256_file(authorization),
            source_sample_id="sample-1",
            transcript_sha256=hashlib.sha256(b"hello book.").hexdigest(),
            source_video_sha256=DIGEST, landmark_track_sha256="c" * 64,
            code_revision="revision")
    nonexistent_selection = dict(decision)
    nonexistent_selection.update({
        "accepted": True, "calibrated_probability": 0.9,
        "selected_annotation_id": "not-a-recorded-candidate",
    })
    invalid_selection_batch = tmp_path / "invalid-selection"
    write_candidate_batch(
        invalid_selection_batch, records=result.records, decision=nonexistent_selection,
        model_bundle_manifest_sha256=sha256_file(bundle / "manifest.json"),
        dataset_authorization_sha256=sha256_file(authorization),
        source_sample_id="sample-1",
        transcript_sha256=hashlib.sha256(b"hello book.").hexdigest(),
        source_video_sha256=DIGEST, landmark_track_sha256="c" * 64,
        code_revision="revision")
    with pytest.raises(ValueError, match="does not select a recorded"):
        verify_candidate_batch(
            invalid_selection_batch, model_bundle=bundle,
            dataset_authorization=authorization)
    records_path = bundle / "config.json"
    payload = bytearray(records_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    records_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle(bundle)


def test_offline_cli_verifies_and_runs_without_media_copy(tmp_path, monkeypatch):
    pipeline = _pipeline()
    governance = ModelGovernance(
        text_model_id="unit-text", text_model_license="test-only",
        text_model_source="local-test", video_model_id="unit-video",
        text_model_artifact_sha256=DIGEST,
        text_model_license_evidence_sha256=DIGEST,
        video_model_license="test-only", video_model_source="local-test",
        video_model_artifact_sha256=DIGEST,
        video_model_license_evidence_sha256=DIGEST,
        training_data_manifest_sha256=DIGEST, dependency_lock_sha256=DIGEST,
        sbom_sha256=DIGEST, intended_use="unit testing")
    bundle = tmp_path / "model"
    write_bundle(bundle, pipeline=pipeline, records=(), governance=governance,
                 seed=3, code_revision="revision")
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("hello book.", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"test-source-video")
    track_path = tmp_path / "track.npz"
    track = _track()
    np.savez_compressed(track_path, values=track.values, confidence=track.confidence,
                        validity_mask=track.validity_mask, timestamps=track.timestamps)
    output = tmp_path / "inference"
    authorization, license_evidence = _dataset_authorization(tmp_path)
    assert pseudo_gloss_main([
        "verify-model", str(bundle),
    ]) == 0
    assert pseudo_gloss_main([
        "infer-one", "--model-bundle", str(bundle),
        "--transcript-file", str(transcript), "--source-video", str(source_video),
        "--landmark-track", str(track_path),
        "--dataset-authorization", str(authorization), "--sample-id", "sample-cli",
        "--created-at", "2026-08-04T00:00:00Z", "--output", str(output),
    ]) == 0
    assert pseudo_gloss_main([
        "verify-batch", str(output), str(bundle), str(authorization)]) == 0
    assert not any(path.suffix == ".mp4" for path in output.iterdir())
    transcript_link = tmp_path / "transcript-link.txt"
    transcript_link.symlink_to(transcript)
    with pytest.raises(ValueError, match="non-symlink"):
        pseudo_gloss_main([
            "infer-one", "--model-bundle", str(bundle),
            "--transcript-file", str(transcript_link),
            "--source-video", str(source_video), "--landmark-track", str(track_path),
            "--dataset-authorization", str(authorization),
            "--sample-id", "sample-cli", "--created-at", "2026-08-04T00:00:00Z",
            "--output", str(tmp_path / "symlink-output"),
        ])

    class MutatingPipeline:
        def generate(self, **arguments):
            result = pipeline.generate(**arguments)
            transcript.write_text("mutated transcript", encoding="utf-8")
            return result

    monkeypatch.setattr(
        pseudo_inference_module, "load_pipeline_bundle", lambda _bundle: MutatingPipeline())
    with pytest.raises(RuntimeError, match="mutated during inference"):
        pseudo_gloss_main([
            "infer-one", "--model-bundle", str(bundle),
            "--transcript-file", str(transcript), "--source-video", str(source_video),
            "--landmark-track", str(track_path), "--sample-id", "sample-cli",
            "--dataset-authorization", str(authorization),
            "--created-at", "2026-08-04T00:00:00Z",
            "--output", str(tmp_path / "mutated-output"),
        ])
    assert not (tmp_path / "mutated-output").exists()
    license_evidence.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash-mismatched"):
        pseudo_gloss_main([
            "verify-batch", str(output), str(bundle), str(authorization)])


def test_source_group_bootstrap_and_falsification_report_are_deterministic():
    spec = InterventionSpecification(
        "blank_video", minimum_mean_decline=0.1,
        null_hypothesis="blanking does not reduce score",
        effect_measure="paired mean score decline",
        stop_rule="stop if lower bound does not exceed 0.1",
        bootstrap_replicates=500, seed=4)
    baseline = [0.9, 0.8, 0.85, 0.95]
    intervention = [0.1, 0.2, 0.15, 0.1]
    groups = ["a", "a", "b", "c"]
    first = paired_source_bootstrap(baseline, intervention, groups, spec)
    second = paired_source_bootstrap(baseline, intervention, groups, spec)
    assert first == second and first.passed
    shuffled = paired_source_bootstrap(
        baseline, intervention, groups,
        InterventionSpecification(
            "shuffled_video", 0.1, "shuffling does not reduce score",
            "paired mean score decline", "stop if lower bound does not exceed 0.1",
            bootstrap_replicates=500, seed=4))
    order = paired_source_bootstrap(
        baseline, intervention, groups,
        InterventionSpecification(
            "order_corruption", 0.1, "reversal does not reduce score",
            "paired mean score decline", "stop if lower bound does not exceed 0.1",
            bootstrap_replicates=500, seed=4))
    report = build_falsification_report(
        [first, shuffled, order], human_reference_completed=False,
        source_holdout_completed=False)
    assert report.video_reliance_certified
    assert not report.linguistic_validation_certified


def test_vocabulary_holdout_operates_on_declared_families_not_spelling_guess():
    families = {"HELLO": "greeting", "HI": "greeting", "BOOK": "object"}
    assert certify_vocabulary_holdout([("HELLO",)], [("BOOK",)], families)
    assert not certify_vocabulary_holdout([("HELLO",)], [("HI",)], families)


def test_human_reference_evaluation_reports_recall_order_and_edit_decomposition():
    counts = token_error_counts(("A", "B", "C"), ("A", "X", "C", "D"))
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 1)
    cases = [
        HumanReferenceCase(
            "sample-1", "source-a", ("HELLO", "BOOK"),
            (("HELLO", "BOOK"), ("BOOK", "HELLO")), ("HELLO", "BOOK"),
            ("order", "nonmanual"), {"order": True, "nonmanual": False}),
        HumanReferenceCase(
            "sample-2", "source-b", ("BOOK",), (("HELLO",),), None,
            ("fingerspelling",), {"fingerspelling": False}),
    ]
    evaluation = evaluate_human_references(cases)
    assert evaluation.candidate_recall == 0.5
    assert evaluation.coverage == 0.5
    assert evaluation.exact_match_rate_among_accepted == 1.0
    assert evaluation.token_error_rate_among_accepted == 0.0
    assert evaluation.order_error_rate_among_accepted == 0.0
    assert {item.name: item.acceptability_rate for item in evaluation.construction_slices} == {
        "fingerspelling": 0.0, "nonmanual": 0.0, "order": 1.0}


def test_shuffled_video_derangement_and_candidate_deletion_are_fail_closed():
    sources = ["source-a", "source-a", "source-b", "source-c", "source-d"]
    strata = ["short", "short", "short", "long", "long"]
    first = deterministic_source_derangement(sources, strata, seed=17)
    assert first == deterministic_source_derangement(sources, strata, seed=17)
    assert all(sources[index] != sources[donor] and strata[index] == strata[donor]
               for index, donor in enumerate(first))
    with pytest.raises(ValueError, match="fewer than two"):
        deterministic_source_derangement(["only"], ["short"], seed=17)
    assert candidate_deletion_abstains("selected", ["survivor-a", "survivor-b"])
    assert not candidate_deletion_abstains("selected", ["selected", "survivor"])


def test_training_annotation_must_bind_exact_sample_not_only_source_group():
    text, _ = _models()
    annotation = _human_annotation("true-sample")
    before = [parameter.detach().clone() for parameter in text.parameters()]
    with pytest.raises(ValueError, match="does not match"):
        fit_text_model(
            text,
            [TextTrainingExample("recording-a", "wrong-sample", "hello", annotation)],
            [TextTrainingExample("recording-b", "true-sample", "hello", annotation)],
            OptimizationConfig(epochs=1),
            TextInitializationEvidence(
                True, "unit-pretrained", "test-only", DIGEST,
                state_dict_sha256(text)))
    assert all(torch.equal(old, new) for old, new in zip(before, text.parameters()))


def test_activation_gate_requires_and_binds_all_external_evidence(tmp_path, monkeypatch):
    from signtranslator.pseudo_gloss.artifacts import sha256_file
    from signtranslator.pseudo_gloss.contracts import canonical_json_bytes, sha256_bytes

    reference_records = [{
        "annotation_id": f"human-{index}", "source_id": f"source-{index}",
        "tokens": ["HELLO" if index % 2 else "BOOK"],
        "label_type": "project_human", "reviewer_pseudonym": "qualified-reviewer",
    } for index in range(6)]
    dataset_authorization, _ = _dataset_authorization(tmp_path)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "schema_version": 1, "language": "ASL",
        "convention_id": "unit-convention-v1", "records": reference_records,
    }), encoding="utf-8")
    reference_hash = sha256_file(reference)
    calibration_fit_reference = tmp_path / "calibration-fit-reference.json"
    calibration_fit_reference.write_text(json.dumps({
        "schema_version": 1, "language": "ASL",
        "convention_id": "unit-convention-v1", "records": [{
            **record, "annotation_id": f"fit-{index}",
            "source_id": f"fit-source-{index}",
        } for index, record in enumerate(reference_records)],
    }), encoding="utf-8")
    calibration_fit_hash = sha256_file(calibration_fit_reference)
    certificate = CalibrationCertificate(
        calibration_fit_hash, "protocol-v1", 6, True, True)
    features = [_feature(value) for value in (0.1, 0.2, 0.3, 0.7, 0.8, 0.9)]
    calibrator = LogisticAcceptanceCalibrator().fit(
        features, [False, False, False, True, True, True], certificate)
    pipeline = _pipeline(calibrator)

    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(json.dumps({
        "lexicon_id": "asl-unit-v1", "convention_id": "unit-convention-v1",
        "tokens": ["UNKNOWN", "HELLO", "BOOK"], "source_sha256": DIGEST,
    }), encoding="utf-8")
    convention = tmp_path / "convention.json"
    convention.write_text(json.dumps({
        "schema_version": 1, "convention_id": "unit-convention-v1", "language": "ASL",
        "token_rules": "closed uppercase lexicon", "nonmanual_policy": "separate tier",
        "fingerspelling_policy": "explicit", "spatial_policy": "explicit",
    }), encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({
        "schema_version": 1, "qualified_asl_reference": True,
        "independent_from_candidate_generation": True, "source_disjoint": True,
        "reference_set_sha256": reference_hash, "convention_id": "unit-convention-v1",
        "review_protocol": "protocol-v1", "reviewer_pseudonyms": ["qualified-reviewer"],
    }), encoding="utf-8")
    calibration_fit_attestation = tmp_path / "calibration-fit-attestation.json"
    calibration_fit_attestation.write_text(json.dumps({
        "schema_version": 1, "qualified_asl_reference": True,
        "independent_from_candidate_generation": True, "source_disjoint": True,
        "reference_set_sha256": calibration_fit_hash,
        "convention_id": "unit-convention-v1", "review_protocol": "protocol-v1",
        "reviewer_pseudonyms": ["qualified-fit-reviewer"],
    }), encoding="utf-8")
    preregistration = tmp_path / "preregistration.json"
    decoding_config_hash = sha256_bytes(canonical_json_bytes({
        "abstention": asdict(pipeline.abstention),
        "fusion": asdict(pipeline.fusion),
        "security": asdict(pipeline.security),
    }))
    falsification_specifications = {
        name: {
            "minimum_mean_decline": 0.1,
            "null_hypothesis": f"{name} does not reduce the primary score",
            "effect_measure": "paired mean score decline",
            "stop_rule": "stop if lower confidence bound does not exceed 0.1",
            "confidence_level": 0.95, "bootstrap_replicates": 500, "seed": 4,
        } for name in REQUIRED_FALSIFICATION_TESTS
    }
    preregistration.write_text(json.dumps({
        "schema_version": 1, "primary_endpoint": "candidate recall",
        "stop_rules": ["video non-use"],
        "falsification_thresholds": {
            name: 0.1 for name in REQUIRED_FALSIFICATION_TESTS
        },
        "falsification_specifications": falsification_specifications,
        "calibration_bin_edges": [0.0, 0.5, 1.0],
        "thresholds_frozen": True, "test_set_locked": True,
        "analysis_plan": "paired source bootstrap",
        "decoding_config_sha256": decoding_config_hash,
    }), encoding="utf-8")
    text_artifact = tmp_path / "text-checkpoint.bin"
    text_license = tmp_path / "text-license.txt"
    video_artifact = tmp_path / "video-checkpoint.bin"
    video_license = tmp_path / "video-license.txt"
    text_artifact.write_bytes(b"text checkpoint evidence")
    text_license.write_text("test-only license evidence", encoding="utf-8")
    video_artifact.write_bytes(b"video checkpoint evidence")
    video_license.write_text("test-only license evidence", encoding="utf-8")
    label_policy = tmp_path / "label-policy.json"
    label_policy.write_text(json.dumps({
        "schema_version": 1, "policy_id": "policy-v1", "policy_owner": "unit-owner",
        "allowed_training_label_types": [
            "official_human", "project_human", "human_corrected_pseudo"],
        "unreviewed_may_enter_gloss_tokens": False,
        "promotion_function": "promote_reviewed_weak_candidate",
        "source_video_review_required": True,
        "machine_parent_preservation_required": True,
    }), encoding="utf-8")
    review_workflow = tmp_path / "review-workflow.json"
    review_workflow.write_text(json.dumps({
        "schema_version": 1, "protocol_id": "protocol-v1",
        "qualified_asl_required": True, "source_video_visible": True,
        "independent_review_required": True, "reviewer_attestation_required": True,
        "rejection_and_abstention_recorded": True,
    }), encoding="utf-8")
    dependency_lock = tmp_path / "requirements.lock"
    dependency_lock.write_text("torch==2.13.0\n", encoding="utf-8")
    training_manifest = tmp_path / "training-data-manifest.json"
    training_manifest.write_text(json.dumps({
        "schema_version": 1, "source_group_type": "VIDEO_ID",
        "local_text_training_source_ids": ["train-source-text"],
        "local_video_training_source_ids": ["train-source-video"],
        "external_pretraining_dataset_ids": ["unit-pretraining-corpus"],
        "external_pretraining_source_overlap_assessed": True,
        "signer_disjoint_claim": False,
    }), encoding="utf-8")
    sbom = tmp_path / "sbom.json"
    sbom.write_text(json.dumps({
        "bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
        "components": [],
    }), encoding="utf-8")
    model_bundle = tmp_path / "model-bundle"
    governance = ModelGovernance(
        text_model_id="unit-text", text_model_license="test-only",
        text_model_source="local-test", video_model_id="unit-video",
        text_model_artifact_sha256=sha256_file(text_artifact),
        text_model_license_evidence_sha256=sha256_file(text_license),
        video_model_license="test-only", video_model_source="local-test",
        video_model_artifact_sha256=sha256_file(video_artifact),
        video_model_license_evidence_sha256=sha256_file(video_license),
        training_data_manifest_sha256=sha256_file(training_manifest),
        dependency_lock_sha256=sha256_file(dependency_lock),
        sbom_sha256=sha256_file(sbom), intended_use="unit testing")
    write_bundle(model_bundle, pipeline=pipeline, records=(), governance=governance,
                 seed=3, code_revision="revision")
    falsification = tmp_path / "falsification.json"
    falsification.write_text(json.dumps({
        "schema_version": 1,
        "model_manifest_sha256": sha256_file(model_bundle / "manifest.json"),
        "preregistration_sha256": sha256_file(preregistration),
        "human_reference_set_sha256": reference_hash,
        "results": [{
            "name": name, "passed": True, "mean_decline": 0.2,
            "lower_confidence_bound": 0.15, "upper_confidence_bound": 0.25,
            "required_decline": 0.1,
            "sample_count": 6, "source_group_count": 3,
            "specification_sha256": sha256_bytes(canonical_json_bytes({
                "name": name, **falsification_specifications[name],
            })),
        } for name in REQUIRED_FALSIFICATION_TESTS],
        "source_holdout_completed": True, "human_reference_completed": True,
        "untouched_test_set_evaluations": 1, "stop_rules_triggered": [],
    }), encoding="utf-8")
    calibration_evaluation = tmp_path / "calibration-evaluation.json"
    calibration_evaluation.write_text(json.dumps({
        "schema_version": 1, "reference_set_sha256": reference_hash,
        "calibrator_state_sha256": calibrator.state_sha256(), "held_out": True,
        "bin_edges": [0.0, 0.5, 1.0], "count": 6, "source_group_count": 6,
        "brier_score": 0.1, "log_loss": 0.3,
        "expected_calibration_error": 0.1, "maximum_calibration_error": 0.2,
        "confidence_level": 0.95, "bootstrap_replicates": 500,
        "brier_interval": [0.05, 0.2],
        "expected_calibration_error_interval": [0.04, 0.18],
    }), encoding="utf-8")

    charter = ActivationCharter(
        schema_version=1,
        lexicon=ArtifactBinding(str(lexicon), sha256_file(lexicon)),
        annotation_convention=ArtifactBinding(str(convention), sha256_file(convention)),
        dataset_authorization=ArtifactBinding(
            str(dataset_authorization), sha256_file(dataset_authorization)),
        calibration_fit_reference_set=ArtifactBinding(
            str(calibration_fit_reference), calibration_fit_hash),
        calibration_fit_attestation=ArtifactBinding(
            str(calibration_fit_attestation), sha256_file(calibration_fit_attestation)),
        human_reference_set=ArtifactBinding(str(reference), reference_hash),
        qualified_reference_attestation=ArtifactBinding(
            str(attestation), sha256_file(attestation)),
        preregistration=ArtifactBinding(str(preregistration), sha256_file(preregistration)),
        label_provenance_policy=ArtifactBinding(
            str(label_policy), sha256_file(label_policy)),
        review_workflow=ArtifactBinding(
            str(review_workflow), sha256_file(review_workflow)),
        falsification_report=ArtifactBinding(
            str(falsification), sha256_file(falsification)),
        calibration_evaluation=ArtifactBinding(
            str(calibration_evaluation), sha256_file(calibration_evaluation)),
        training_data_manifest=ArtifactBinding(
            str(training_manifest), sha256_file(training_manifest)),
        dependency_lock=ArtifactBinding(
            str(dependency_lock), sha256_file(dependency_lock)),
        sbom=ArtifactBinding(str(sbom), sha256_file(sbom)),
        text_model_artifact=ArtifactBinding(
            str(text_artifact), sha256_file(text_artifact)),
        text_model_license_evidence=ArtifactBinding(
            str(text_license), sha256_file(text_license)),
        video_model_artifact=ArtifactBinding(
            str(video_artifact), sha256_file(video_artifact)),
        video_model_license_evidence=ArtifactBinding(
            str(video_license), sha256_file(video_license)),
        model_bundle=ArtifactBinding(
            str(model_bundle), sha256_file(model_bundle / "manifest.json")),
        signer_mapping=None, signer_generalization_claim=False)
    report = assess_activation(charter)
    assert report.activation_approved
    assert all(check.passed for check in report.checks)
    assert not report.linguistic_validation_approved
    assert not report.production_gloss_export_approved
    activation_charter_path = tmp_path / "activation-charter.json"
    activation_charter_path.write_text(
        json.dumps(asdict(charter)), encoding="utf-8")

    transcript_paths = []
    video_paths = []
    track_paths = []
    for index in range(2):
        transcript_path = tmp_path / f"corpus-transcript-{index}.txt"
        transcript_path.write_text("hello book.", encoding="utf-8")
        video_path = tmp_path / f"corpus-video-{index}.mp4"
        video_path.write_bytes(f"video-{index}".encode("ascii"))
        track_path = tmp_path / f"corpus-track-{index}.npz"
        track = _track()
        np.savez_compressed(
            track_path, values=track.values, confidence=track.confidence,
            validity_mask=track.validity_mask, timestamps=track.timestamps)
        transcript_paths.append(transcript_path)
        video_paths.append(video_path)
        track_paths.append(track_path)
    corpus_manifest = tmp_path / "corpus-inputs.jsonl"
    corpus_manifest.write_text("".join(json.dumps({
        "sample_id": f"corpus-{index}", "source_id": f"corpus-source-{index}",
        "transcript_file": str(transcript_paths[index]),
        "source_video": str(video_paths[index]),
        "landmark_track": str(track_paths[index]),
        "created_at": "2026-08-04T00:00:00Z",
    }) + "\n" for index in (1, 0)), encoding="utf-8")
    corpus_output = tmp_path / "corpus-output"
    original_run_inference = pseudo_corpus_module.run_inference
    calls = 0

    def interrupt_second(context, request, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original_run_inference(context, request, **kwargs)

    monkeypatch.setattr(pseudo_corpus_module, "run_inference", interrupt_second)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_corpus_inference(
            input_manifest=corpus_manifest, output=corpus_output,
            model_bundle=model_bundle, dataset_authorization=dataset_authorization,
            activation_charter=activation_charter_path, resume=False)
    monkeypatch.setattr(pseudo_corpus_module, "run_inference", original_run_inference)
    state_path = run_corpus_inference(
        input_manifest=corpus_manifest, output=corpus_output,
        model_bundle=model_bundle, dataset_authorization=dataset_authorization,
        activation_charter=activation_charter_path, resume=True)
    first_state = state_path.read_bytes()
    state = json.loads(first_state)
    assert state["complete"] is True
    assert set(state["completed"]) == {"corpus-0", "corpus-1"}
    assert {path.name for path in (corpus_output / "samples").iterdir()} \
        == {"corpus-0", "corpus-1"}
    assert run_corpus_inference(
        input_manifest=corpus_manifest, output=corpus_output,
        model_bundle=model_bundle, dataset_authorization=dataset_authorization,
        activation_charter=activation_charter_path, resume=True).read_bytes() == first_state
    assert pseudo_gloss_main([
        "infer-corpus", "--input-manifest", str(corpus_manifest),
        "--output", str(corpus_output), "--model-bundle", str(model_bundle),
        "--dataset-authorization", str(dataset_authorization),
        "--activation-charter", str(activation_charter_path), "--resume",
    ]) == 0

    manifest_original = corpus_manifest.read_text(encoding="utf-8")
    corpus_manifest.write_text(
        manifest_original.replace("corpus-source-0", "changed-source"), encoding="utf-8")
    with pytest.raises(ValueError, match="input_manifest_sha256"):
        run_corpus_inference(
            input_manifest=corpus_manifest, output=corpus_output,
            model_bundle=model_bundle, dataset_authorization=dataset_authorization,
            activation_charter=activation_charter_path, resume=True)
    corpus_manifest.write_text(manifest_original, encoding="utf-8")

    falsification_original = falsification.read_text(encoding="utf-8")
    incomplete_falsification = json.loads(falsification_original)
    incomplete_falsification["results"].pop()
    falsification.write_text(json.dumps(incomplete_falsification), encoding="utf-8")
    incomplete_charter = replace(
        charter, falsification_report=ArtifactBinding(
            str(falsification), sha256_file(falsification)))
    incomplete_report = assess_activation(incomplete_charter)
    assert not incomplete_report.activation_approved
    assert any(check.name == "complete_falsification_suite" and not check.passed
               for check in incomplete_report.checks)
    falsification.write_text(falsification_original, encoding="utf-8")

    training_manifest_original = training_manifest.read_text(encoding="utf-8")
    overlapping_training_manifest = json.loads(training_manifest_original)
    overlapping_training_manifest["local_video_training_source_ids"] = ["source-0"]
    training_manifest.write_text(
        json.dumps(overlapping_training_manifest), encoding="utf-8")
    overlap_charter = replace(
        charter, training_data_manifest=ArtifactBinding(
            str(training_manifest), sha256_file(training_manifest)))
    overlap_report = assess_activation(overlap_charter)
    assert not overlap_report.activation_approved
    assert any(check.name == "training_human_reference_source_disjointness"
               and not check.passed for check in overlap_report.checks)
    training_manifest.write_text(training_manifest_original, encoding="utf-8")

    lexicon.write_text("{}", encoding="utf-8")
    failed = assess_activation(charter)
    assert not failed.activation_approved
    assert any(not check.passed for check in failed.checks)
