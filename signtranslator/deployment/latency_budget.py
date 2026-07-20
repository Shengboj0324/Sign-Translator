"""Latency budget algebra (Doc-13 §2).

For mostly-pipelined stages, steady-state throughput is bounded by the slowest stage
while first-output latency is the dependency-critical path. Report p50/p95/p99 at
batch size 1 on named hardware. A latency CLAIM is not credible without disclosing
chunk size, lookahead, reordering, and quality loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..speech.streaming import percentile


@dataclass(frozen=True)
class PipelineStage:
    name: str
    service_time_s: float

    def __post_init__(self):
        if self.service_time_s < 0:
            raise ValueError("service_time_s must be >= 0")


def bottleneck_stage(stages: Sequence[PipelineStage]) -> PipelineStage:
    """The slowest stage — it caps steady-state throughput."""
    if not stages:
        raise ValueError("no stages")
    return max(stages, key=lambda s: s.service_time_s)


def steady_state_throughput(stages: Sequence[PipelineStage]) -> float:
    """Pipelined throughput = 1 / max_i service_time_i (items/sec)."""
    slowest = bottleneck_stage(stages).service_time_s
    if slowest <= 0:
        raise ValueError("bottleneck service time must be > 0")
    return 1.0 / slowest


def first_output_latency(stages: Sequence[PipelineStage]) -> float:
    """First-output latency = sum of stage times (the serial critical path)."""
    return float(sum(s.service_time_s for s in stages))


def first_output_latency_budget(l_buffer: float, l_asr: float, l_plan: float,
                                l_motion: float, l_render: float) -> float:
    """L_first ≈ L_buffer + L_ASR + L_plan + L_motion + L_render (the document)."""
    parts = [l_buffer, l_asr, l_plan, l_motion, l_render]
    if any(p < 0 for p in parts):
        raise ValueError("latency components must be >= 0")
    return float(sum(parts))


def latency_percentiles(latencies_s: Sequence[float],
                        qs: Sequence[float] = (50.0, 95.0, 99.0)) -> Dict[str, float]:
    """p50/p95/p99 (default) of measured latencies (batch size 1)."""
    vals = list(latencies_s)
    return {f"p{int(q)}": percentile(vals, q) for q in qs}


# ---------------------------------------------------------------------------
# credibility guard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatencyClaim:
    """A latency target with the disclosures that make it credible (or not)."""

    target_ms: float
    chunk_size_ms: Optional[float] = None
    lookahead_ms: Optional[float] = None
    allows_sentence_reordering: Optional[bool] = None
    quality_loss: Optional[float] = None       # e.g. delta in the quality metric


def latency_claim_is_credible(claim: LatencyClaim) -> bool:
    """A '<X ms' claim is credible only if it discloses chunk/lookahead/reordering/quality.

    Encodes the document's skepticism: a universal '<200 ms' target is not credible
    without specifying chunk size, lookahead, sentence reordering, and quality loss.
    """
    return (claim.chunk_size_ms is not None
            and claim.lookahead_ms is not None
            and claim.allows_sentence_reordering is not None
            and claim.quality_loss is not None)


@dataclass(frozen=True)
class HardwareLatencyReport:
    """A latency report is only meaningful with named hardware + percentiles."""

    device: str
    batch_size: int
    latencies_s: Tuple[float, ...]

    def percentiles(self) -> Dict[str, float]:
        return latency_percentiles(self.latencies_s)

    def is_reportable(self) -> bool:
        return bool(self.device) and self.batch_size == 1 and len(self.latencies_s) > 0
