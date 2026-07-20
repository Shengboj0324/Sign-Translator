"""Doc-13 stage 13h: end-to-end deployment integration + replay + cycle stress."""

import numpy as np
import torch

from signtranslator.deployment import (
    StreamingContract, CommitViolationError,
    PipelineStage, steady_state_throughput, first_output_latency_budget,
    simulate_queue, bounded_queue_latency,
    symmetric_int8_quantize, symmetric_int8_dequantize,
    certify_optimization, online_softmax_attention,
    clarification_gate, TokenCategory, RuntimeAction,
    TelemetrySnapshot, PrivacyRingBuffer,
    StaticShapeExecutor, OptimizationPlan, OptimizationStep,
    SemanticCheckpoint, ReplayHarness,
)
from signtranslator.speech.policy import FailClosedPolicy


def test_full_deployment_pipeline():
    # 1) streaming commit contract: monotone, cannot rewrite displayed signs.
    contract = StreamingContract()
    contract.commit([1]); contract.commit([1, 2])
    try:
        contract.commit([1, 9])
        assert False
    except CommitViolationError:
        pass
    assert contract.certify()[0]

    # 2) latency budget: throughput bounded by slowest stage; first-output = sum.
    stages = [PipelineStage("asr", 0.05), PipelineStage("motion", 0.08)]
    assert steady_state_throughput(stages) == 1 / 0.08
    assert first_output_latency_budget(0.02, 0.05, 0.03, 0.08, 0.016) == 0.196

    # 3) backpressure bounds the backlog under overload.
    sim = simulate_queue(3.0, 1.0, capacity=4.0, steps=100, backpressure=True)
    assert sim.peak_occupancy <= 4.0 + 1e-9
    assert bounded_queue_latency(4.0, 1.0) == 4.0

    # 4) an INT8 transform is gated on numerical equivalence + quality.
    x = torch.randn(128, dtype=torch.float64)
    q, scale = symmetric_int8_quantize(x)
    xhat = symmetric_int8_dequantize(q, scale)
    cert = certify_optimization(x, xhat, 0.9, 0.9, atol=scale)
    assert cert.accepted

    # 5) chunked attention equals standard attention exactly.
    scores = torch.randn(20, dtype=torch.float64)
    values = torch.randn(20, 4, dtype=torch.float64)
    ref = torch.softmax(scores, 0) @ values
    assert torch.allclose(online_softmax_attention(scores, values, 5), ref, atol=1e-12)

    # 6) runtime controls: names clarify, telemetry + privacy behave.
    pol = FailClosedPolicy(emit_threshold=0.8)
    assert clarification_gate(1, 0.9, TokenCategory.NAME, pol) == RuntimeAction.CLARIFY
    assert TelemetrySnapshot(60, 0.5, 0.01, 10).healthy()
    buf = PrivacyRingBuffer(window=50)
    buf.push(np.random.randn(80)); buf.clear()
    assert buf.retained_samples == 0

    # 7) static executor + optimization order.
    ex = StaticShapeExecutor(lambda t: t + 1); ex.capture(torch.zeros(3))
    assert torch.equal(ex.replay(torch.zeros(3)), torch.ones(3))
    plan = OptimizationPlan(); plan.establish_baseline()
    plan.apply(OptimizationStep.DISTILL_QUANTIZE)

    # 8) replay harness: a correct decoder passes all checkpoints, monotone.
    chunks = [(0.0, 1), (0.1, 2), (0.2, 3)]
    checkpoints = [SemanticCheckpoint(0.1, (1,)), SemanticCheckpoint(0.2, (1, 2))]
    harness = ReplayHarness(chunks, checkpoints)
    result = harness.replay(decode=lambda cids: tuple(cids[:-1]) if len(cids) > 1 else ())
    assert result.passed and result.monotone


def test_replay_flags_a_checkpoint_miss():
    chunks = [(0.0, 1), (0.1, 2)]
    checkpoints = [SemanticCheckpoint(0.1, (1, 2, 3))]      # expects more than decoded
    harness = ReplayHarness(chunks, checkpoints)
    result = harness.replay(decode=lambda cids: (cids[0],) if cids else ())
    assert not result.passed and len(result.failures) == 1


def test_cycle_stress_determinism():
    x = torch.randn(64, dtype=torch.float64)
    q1, s1 = symmetric_int8_quantize(x)
    q2, s2 = symmetric_int8_quantize(x)
    assert torch.equal(q1, q2) and s1 == s2
    a = simulate_queue(2.0, 1.0, 5.0, 50, backpressure=True)
    b = simulate_queue(2.0, 1.0, 5.0, 50, backpressure=True)
    assert a.occupancy == b.occupancy
