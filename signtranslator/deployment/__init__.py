"""Real-time deployment and optimization layer (Document 13).

Implements `13_real_time_deployment_optimization.md` (see docs/DEPLOYMENT.md): the
streaming commit contract, latency-budget algebra, the backpressure bounded-queue
theorem, quantization error bounds, a numerically-certified optimization gate with
online-softmax exactness, runtime controls (confidence gate, fallback, telemetry,
privacy), a static-shape guard + optimization-order gate, and a replay harness.
Reuses Doc-01 (revision), Doc-03 (fail-closed policy), and Doc-12 (contract chain).
The sandbox is CPU-only, so this is the deployment MATH/LOGIC, not GPU execution.
"""

from .streaming_contract import (
    is_prefix, certify_commit_monotone, StreamingContract, CommitViolationError,
)
from .latency_budget import (
    PipelineStage, bottleneck_stage, steady_state_throughput, first_output_latency,
    first_output_latency_budget, latency_percentiles, LatencyClaim,
    latency_claim_is_credible, HardwareLatencyReport,
)
from .backpressure import (
    QueueSim, simulate_queue, backlog_growth_per_step, bounded_queue_latency,
    should_apply_backpressure,
)
from .quantization import (
    FP16_MACHINE_EPS, fp16_round_trip, fp16_max_relative_error,
    symmetric_int8_quantize, symmetric_int8_dequantize,
    affine_uint8_quantize, affine_uint8_dequantize, quantization_error,
    calibrate_minmax, calibrate_percentile, range_coverage,
)
from .optimization_gate import (
    max_abs_error, max_rel_error, TransformCertificate, certify_optimization,
    online_softmax, online_softmax_attention,
)
from .runtime_controls import (
    TokenCategory, RuntimeAction, clarification_gate, fallback_render,
    TelemetryThresholds, TelemetrySnapshot, PrivacyRingBuffer,
)
from .static_execution import (
    StaticShapeExecutor, ShapeInstabilityError, OptimizationStep,
    OptimizationPlan, OptimizationOrderError,
)
from .replay import SemanticCheckpoint, ReplayResult, ReplayHarness

__all__ = [
    "is_prefix", "certify_commit_monotone", "StreamingContract",
    "CommitViolationError",
    "PipelineStage", "bottleneck_stage", "steady_state_throughput",
    "first_output_latency", "first_output_latency_budget", "latency_percentiles",
    "LatencyClaim", "latency_claim_is_credible", "HardwareLatencyReport",
    "QueueSim", "simulate_queue", "backlog_growth_per_step",
    "bounded_queue_latency", "should_apply_backpressure",
    "FP16_MACHINE_EPS", "fp16_round_trip", "fp16_max_relative_error",
    "symmetric_int8_quantize", "symmetric_int8_dequantize",
    "affine_uint8_quantize", "affine_uint8_dequantize", "quantization_error",
    "calibrate_minmax", "calibrate_percentile", "range_coverage",
    "max_abs_error", "max_rel_error", "TransformCertificate", "certify_optimization",
    "online_softmax", "online_softmax_attention",
    "TokenCategory", "RuntimeAction", "clarification_gate", "fallback_render",
    "TelemetryThresholds", "TelemetrySnapshot", "PrivacyRingBuffer",
    "StaticShapeExecutor", "ShapeInstabilityError", "OptimizationStep",
    "OptimizationPlan", "OptimizationOrderError",
    "SemanticCheckpoint", "ReplayResult", "ReplayHarness",
]
