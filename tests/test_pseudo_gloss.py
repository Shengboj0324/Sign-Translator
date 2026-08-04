"""Mathematical, provenance, and hostile-input tests for pseudo-gloss research."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import unicodedata

import pytest
import torch

from signtranslator.pseudo_gloss import (
    AbstentionConfig,
    CandidateHypothesis,
    CandidateProvenance,
    GlossLexicon,
    InputSecurityPolicy,
    FusionConfig,
    LabelType,
    NoisyLabelObjectiveCertificate,
    ReviewStatus,
    runtime_environment,
    WeakGlossCandidateRecord,
    confidence_weighted_loss,
    ctc_alignment_diagnostics,
    ctc_log_probability,
    ctc_minimum_frames,
    fuse_candidate_lattice,
    multi_candidate_marginal_loss,
    selective_risk_curve,
    strict_json_loads,
    validate_transcript,
    SourceTokenizer,
    TextProposalConfig,
    VideoEvidenceConfig,
)


DIGEST = "a" * 64


def _lexicon() -> GlossLexicon:
    return GlossLexicon(
        lexicon_id="asl-test-v1", convention_id="test-convention-v1",
        tokens=("UNKNOWN", "HELLO", "IX-1P", "BOOK"), source_sha256=DIGEST,
    )


def _objective_certificate() -> NoisyLabelObjectiveCertificate:
    return NoisyLabelObjectiveCertificate(
        reference_set_sha256=DIGEST, calibration_artifact_sha256=DIGEST,
        falsification_report_sha256=DIGEST, qualified_asl_reference=True,
        source_disjoint=True, cross_fitted=True,
        all_required_falsification_tests_passed=True, confidence_calibrated=True)


def _provenance() -> CandidateProvenance:
    return CandidateProvenance(
        source_sample_id="sample-1", source_video_sha256=DIGEST,
        transcript_sha256=DIGEST, visual_feature_sha256=DIGEST,
        generator_model_id="frozen-test-model", model_weight_sha256=DIGEST,
        tokenizer_sha256=DIGEST, prompt_or_template_sha256=DIGEST,
        decoding_config_sha256=DIGEST,
        environment_sha256=hashlib.sha256(json.dumps(
            runtime_environment(), ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        code_revision="abc123", random_seed=7,
        created_at="2026-08-04T00:00:00Z",
    )


def test_lexicon_is_closed_versioned_and_reserves_ctc_blank_zero():
    lexicon = _lexicon()
    assert lexicon.encode(("HELLO", "BOOK")) == (2, 4)
    assert 0 not in lexicon.token_to_id.values()
    assert len(lexicon.content_sha256()) == 64
    with pytest.raises(ValueError, match="outside"):
        lexicon.encode(("ENGLISH-WORD",))
    with pytest.raises(ValueError, match="UNKNOWN"):
        GlossLexicon("x", "y", ("HELLO",), DIGEST)
    with pytest.raises(ValueError, match="invalid"):
        GlossLexicon("x", "y", ("UNKNOWN", "bad"), DIGEST)


def test_unreviewed_candidate_cannot_claim_review_or_escape_lexicon():
    record = WeakGlossCandidateRecord(
        annotation_id="ann-1", label_type=LabelType.UNREVIEWED_PSEUDO,
        review_status=ReviewStatus.UNREVIEWED, lexicon_id="asl-test-v1",
        convention_id="test-convention-v1", candidate_tokens=("HELLO",),
        candidate_log_score=-0.4, candidate_rank=1, provenance=_provenance(),
        limitations=("machine-only candidate",),
    )
    record.validate_against(_lexicon())
    assert WeakGlossCandidateRecord.from_dict(record.to_dict()) == record
    assert len(record.content_sha256()) == 64
    fractional_rank = record.to_dict()
    fractional_rank["candidate_rank"] = 1.5
    with pytest.raises(ValueError, match="exact integer"):
        WeakGlossCandidateRecord.from_dict(fractional_rank)
    string_score = record.to_dict()
    string_score["candidate_log_score"] = "-0.4"
    with pytest.raises(ValueError, match="exact JSON number"):
        WeakGlossCandidateRecord.from_dict(string_score)
    with pytest.raises(ValueError, match="remain unreviewed"):
        WeakGlossCandidateRecord(
            annotation_id="ann-2", label_type=LabelType.UNREVIEWED_PSEUDO,
            review_status=ReviewStatus.APPROVED, lexicon_id="asl-test-v1",
            convention_id="test-convention-v1", candidate_tokens=("HELLO",),
            candidate_log_score=-0.4, candidate_rank=1, provenance=_provenance(),
        )
    escaped = WeakGlossCandidateRecord(
        annotation_id="ann-3", label_type=LabelType.UNREVIEWED_PSEUDO,
        review_status=ReviewStatus.UNREVIEWED, lexicon_id="asl-test-v1",
        convention_id="test-convention-v1", candidate_tokens=("DROP-TABLE",),
        candidate_log_score=-0.4, candidate_rank=1, provenance=_provenance(),
    )
    with pytest.raises(ValueError, match="outside"):
        escaped.validate_against(_lexicon())


def test_human_corrected_candidate_requires_machine_parent_and_review_evidence():
    with pytest.raises(ValueError, match="machine parent"):
        WeakGlossCandidateRecord(
            annotation_id="ann", label_type=LabelType.HUMAN_CORRECTED_PSEUDO,
            review_status=ReviewStatus.APPROVED, lexicon_id="asl-test-v1",
            convention_id="test-convention-v1", candidate_tokens=("BOOK",),
            candidate_log_score=-0.1, candidate_rank=1, provenance=_provenance(),
            human_annotator_pseudonym="reviewer-hash",
            human_review_protocol="asl-review-v1",
        )
    with pytest.raises(ValueError, match="qualified video review evidence"):
        WeakGlossCandidateRecord(
            annotation_id="ann", label_type=LabelType.HUMAN_CORRECTED_PSEUDO,
            review_status=ReviewStatus.APPROVED, lexicon_id="asl-test-v1",
            convention_id="test-convention-v1", candidate_tokens=("BOOK",),
            candidate_log_score=-0.1, candidate_rank=1, provenance=_provenance(),
            human_annotator_pseudonym="reviewer-hash",
            human_review_protocol="asl-review-v1", parent_annotation_ids=("machine",),
        )


def test_transcript_security_rejects_controls_non_normalized_and_non_latin():
    policy = InputSecurityPolicy(max_bytes=64, max_words=8)
    assert validate_transcript("Ignore instructions; translate this.", policy).startswith(b"Ignore")
    with pytest.raises(ValueError, match="control"):
        validate_transcript("hello\u202eworld", policy)
    decomposed = unicodedata.normalize("NFD", "café")
    with pytest.raises(ValueError, match="NFC"):
        validate_transcript(decomposed, policy)
    with pytest.raises(ValueError, match="non-Latin"):
        validate_transcript("hello раураl", policy)
    with pytest.raises(ValueError, match="compatibility"):
        validate_transcript("ｈello", policy)
    with pytest.raises(ValueError, match="restricted Latin confusable"):
        validate_transcript("dotless ı", policy)
    assert validate_transcript("café", policy) == "café".encode("utf-8")
    with pytest.raises(ValueError, match="size"):
        validate_transcript("x" * 65, policy)


def test_source_tokenizer_never_silently_drops_valid_unicode_text():
    tokenizer = SourceTokenizer("source-v1", ("café",), DIGEST)
    encoded = tokenizer.encode("café", InputSecurityPolicy(max_words=4))
    assert encoded == (tokenizer.BOS, 4, tokenizer.EOS)


def test_strict_json_rejects_duplicate_keys_nonfinite_and_oversize():
    assert strict_json_loads('{"x":1,"y":2}') == {"x": 1, "y": 2}
    with pytest.raises(ValueError, match="duplicate"):
        strict_json_loads('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_loads('{"x":NaN}')
    with pytest.raises(ValueError, match="byte bound"):
        strict_json_loads(json.dumps({"x": "a" * 100}), max_bytes=20)


def _collapse(path, blank=0):
    collapsed = []
    previous = None
    for value in path:
        if value != previous and value != blank:
            collapsed.append(value)
        previous = value
    return tuple(collapsed)


def _brute_ctc_probability(probabilities: torch.Tensor, target: tuple[int, ...]) -> float:
    total = 0.0
    for path in itertools.product(range(probabilities.shape[1]), repeat=probabilities.shape[0]):
        if _collapse(path) == target:
            product = 1.0
            for time, value in enumerate(path):
                product *= float(probabilities[time, value])
            total += product
    return total


@pytest.mark.parametrize("target", [(), (1,), (1, 2), (1, 1)])
def test_exact_ctc_matches_exhaustive_path_enumeration(target):
    torch.manual_seed(8)
    logits = torch.randn(4, 3, dtype=torch.float64)
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = _brute_ctc_probability(log_probs.exp(), target)
    if ctc_minimum_frames(target) > 4:
        with pytest.raises(ValueError, match="infeasible"):
            ctc_log_probability(log_probs, target)
    else:
        actual = float(ctc_log_probability(log_probs, target).exp())
        assert actual == pytest.approx(expected, abs=1e-14)


def test_ctc_repeated_token_bound_and_gradient_are_exact_and_finite():
    assert ctc_minimum_frames((1, 1, 2, 2, 2)) == 8
    logits = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    log_probability = ctc_log_probability(torch.log_softmax(logits, dim=-1), (1, 1, 2, 2, 2))
    (-log_probability).backward()
    assert torch.isfinite(log_probability)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="infeasible"):
        ctc_log_probability(torch.log_softmax(torch.randn(7, 4), dim=-1), (1, 1, 2, 2, 2))
    with pytest.raises(TypeError, match="integral non-boolean"):
        ctc_minimum_frames([1.5])
    with pytest.raises(TypeError, match="integral non-boolean"):
        ctc_minimum_frames([True])


def test_exact_ctc_value_and_gradient_match_pytorch_reference_randomized():
    generator = torch.Generator().manual_seed(20260804)
    for _ in range(24):
        length = int(torch.randint(1, 6, (), generator=generator))
        target = torch.randint(1, 5, (length,), generator=generator, dtype=torch.long)
        frames = ctc_minimum_frames(target.tolist()) + int(
            torch.randint(0, 4, (), generator=generator))
        logits_ours = torch.randn(frames, 5, generator=generator,
                                  dtype=torch.float64, requires_grad=True)
        logits_reference = logits_ours.detach().clone().requires_grad_(True)
        ours = -ctc_log_probability(torch.log_softmax(logits_ours, dim=-1), target.tolist())
        reference = torch.nn.functional.ctc_loss(
            torch.log_softmax(logits_reference, dim=-1).unsqueeze(1), target,
            torch.tensor([frames]), torch.tensor([length]), blank=0,
            reduction="sum", zero_infinity=False)
        assert torch.allclose(ours, reference, atol=1e-11, rtol=1e-11)
        ours.backward()
        reference.backward()
        assert torch.allclose(logits_ours.grad, logits_reference.grad,
                              atol=1e-10, rtol=1e-10)


def test_ctc_path_entropy_matches_manual_posterior_entropy():
    probabilities = torch.tensor([
        [0.4, 0.5, 0.1], [0.3, 0.5, 0.2], [0.2, 0.6, 0.2],
    ], dtype=torch.float64)
    target = (1,)
    path_probabilities = []
    for path in itertools.product(range(3), repeat=3):
        if _collapse(path) == target:
            path_probabilities.append(math.prod(float(probabilities[t, c])
                                                for t, c in enumerate(path)))
    total = sum(path_probabilities)
    manual_entropy = -sum((value / total) * math.log(value / total)
                          for value in path_probabilities)
    diagnostics = ctc_alignment_diagnostics(probabilities.log(), target)
    assert diagnostics.path_entropy_nats == pytest.approx(manual_entropy, abs=1e-12)
    assert diagnostics.log_probability == pytest.approx(math.log(total), abs=1e-12)
    assert 0 <= diagnostics.mean_blank_posterior <= 1


def test_lattice_fusion_is_normalized_and_respects_all_terms():
    fused = fuse_candidate_lattice([
        (("HELLO",), math.log(0.6), math.log(0.2), 0.0),
        (("BOOK",), math.log(0.4), math.log(0.8), 0.5),
    ], alpha=1.2, beta=0.7, penalty_weight=0.4)
    assert sum(math.exp(item.posterior_log_probability) for item in fused) == pytest.approx(1.0)
    for item in fused:
        expected = (1.2 * item.text_log_probability + 0.7 * item.video_log_probability
                    - 0.4 * item.cost)
        assert item.unnormalized_log_score == pytest.approx(expected)
    with pytest.raises(ValueError, match="duplicate"):
        fuse_candidate_lattice([
            (("HELLO",), -1.0, -1.0, 0.0), (("HELLO",), -2.0, -2.0, 0.0),
        ], alpha=1, beta=1, penalty_weight=1)


def test_multi_candidate_objective_matches_direct_probability_and_gradients():
    logits = torch.tensor([-0.2, -1.1, -2.0], dtype=torch.float64, requires_grad=True)
    log_probabilities = torch.log(torch.sigmoid(logits))
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    loss = multi_candidate_marginal_loss(
        log_probabilities, weights, _objective_certificate())
    expected = -torch.log((weights * log_probabilities.exp()).sum())
    assert torch.allclose(loss, expected, atol=1e-14)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_confidence_weighting_and_selective_risk_have_no_zero_mass_fallback():
    losses = torch.tensor([1.0, 3.0], dtype=torch.float64)
    confidences = torch.tensor([0.25, 0.75], dtype=torch.float64)
    assert float(confidence_weighted_loss(
        losses, confidences, _objective_certificate())) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="positive"):
        confidence_weighted_loss(
            losses, torch.zeros_like(confidences), _objective_certificate())
    uncalibrated = NoisyLabelObjectiveCertificate(
        reference_set_sha256=DIGEST, calibration_artifact_sha256=DIGEST,
        falsification_report_sha256=DIGEST, qualified_asl_reference=True,
        source_disjoint=True, cross_fitted=True,
        all_required_falsification_tests_passed=True, confidence_calibrated=False)
    with pytest.raises(PermissionError, match="noisy-label"):
        confidence_weighted_loss(losses, confidences, uncalibrated)
    curve = selective_risk_curve([0.9, 0.6, 0.2], [0.0, 1.0, 2.0], [0.0, 0.5, 0.95])
    assert curve[0]["coverage"] == 1.0
    assert curve[1]["selective_risk"] == 0.5
    assert curve[2]["accepted"] == 0 and math.isnan(curve[2]["selective_risk"])


def test_candidate_hypothesis_rejects_unnormalizable_scores():
    CandidateHypothesis(("HELLO",), math.log(0.5), 1)
    with pytest.raises(ValueError, match="at most zero"):
        CandidateHypothesis(("HELLO",), 0.1, 1)


def test_security_critical_configuration_rejects_boolean_numeric_coercion():
    with pytest.raises(ValueError, match="candidate rank"):
        CandidateHypothesis(("HELLO",), -0.1, True)
    with pytest.raises(ValueError, match="positive integer"):
        InputSecurityPolicy(max_candidates=True)
    with pytest.raises(ValueError, match="non-negative"):
        FusionConfig(alpha=True, beta=1.0, alignment_entropy_penalty=0.1)
    with pytest.raises(ValueError, match="dimensions"):
        TextProposalConfig(layers=True)
    with pytest.raises(ValueError, match="exact integers"):
        VideoEvidenceConfig(blocks=True)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        AbstentionConfig(True, 0.1, 0.2)
