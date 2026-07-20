"""Adversarial tests for the latency budget (Doc-13, stage 13b)."""

import pytest

from signtranslator.deployment.latency_budget import (
    PipelineStage, bottleneck_stage, steady_state_throughput, first_output_latency,
    first_output_latency_budget, latency_percentiles, LatencyClaim,
    latency_claim_is_credible, HardwareLatencyReport,
)


def _pipe():
    return [
        PipelineStage("buffer", 0.02), PipelineStage("asr", 0.05),
        PipelineStage("plan", 0.03), PipelineStage("motion", 0.08),
        PipelineStage("render", 0.016),
    ]


def test_throughput_bounded_by_slowest_stage():
    stages = _pipe()
    assert bottleneck_stage(stages).name == "motion"       # 0.08 s
    assert steady_state_throughput(stages) == pytest.approx(1 / 0.08)


def test_first_output_is_sum_of_stages():
    assert first_output_latency(_pipe()) == pytest.approx(0.02 + 0.05 + 0.03 + 0.08 + 0.016)


def test_first_output_budget_matches_document_formula():
    L = first_output_latency_budget(0.02, 0.05, 0.03, 0.08, 0.016)
    assert L == pytest.approx(0.196)
    # first-output latency (sum) exceeds the per-item throughput period (bottleneck).
    assert L > 1.0 / steady_state_throughput(_pipe())


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        first_output_latency_budget(-0.1, 0.05, 0.03, 0.08, 0.016)


def test_percentiles_ordering():
    lat = [0.10, 0.12, 0.11, 0.13, 0.30, 0.14, 0.12, 0.11, 0.15, 0.12]
    p = latency_percentiles(lat)
    assert p["p50"] <= p["p95"] <= p["p99"]


def test_latency_claim_not_credible_without_disclosure():
    bare = LatencyClaim(target_ms=200)                     # "<200 ms" and nothing else
    assert not latency_claim_is_credible(bare)
    partial = LatencyClaim(200, chunk_size_ms=40, lookahead_ms=80)   # missing 2
    assert not latency_claim_is_credible(partial)


def test_latency_claim_credible_with_full_disclosure():
    full = LatencyClaim(200, chunk_size_ms=40, lookahead_ms=80,
                        allows_sentence_reordering=False, quality_loss=0.01)
    assert latency_claim_is_credible(full)


def test_hardware_report_requires_named_device_and_batch1():
    good = HardwareLatencyReport("JetsonOrin", 1, (0.1, 0.12, 0.3))
    assert good.is_reportable()
    assert good.percentiles()["p95"] >= good.percentiles()["p50"]
    assert not HardwareLatencyReport("", 1, (0.1,)).is_reportable()      # unnamed
    assert not HardwareLatencyReport("GPU", 8, (0.1,)).is_reportable()   # not batch 1
