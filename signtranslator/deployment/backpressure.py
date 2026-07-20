"""Backpressure + bounded-queue latency (Doc-13 §3).

A stage is a queue of capacity B served at rate mu with arrivals lambda. Without
backpressure and lambda>mu the backlog (hence latency) grows unbounded; with
backpressure (slow/pause the source when full) the backlog is <= B and queueing
latency is <= B/mu. The document's rule becomes a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QueueSim:
    occupancy: List[float]     # backlog after each step
    total_blocked: float       # items refused by backpressure (0 if disabled)

    @property
    def peak_occupancy(self) -> float:
        return max(self.occupancy) if self.occupancy else 0.0


def simulate_queue(arrival_rate: float, service_rate: float, capacity: float,
                   steps: int, backpressure: bool) -> QueueSim:
    """Fluid discrete-time queue simulation.

    Each step: serve up to ``service_rate``, then admit arrivals. With backpressure
    the backlog is capped at ``capacity`` (excess is blocked/slowed); without it the
    backlog can grow without bound.
    """
    if arrival_rate < 0 or service_rate <= 0 or capacity <= 0 or steps < 1:
        raise ValueError("bad queue parameters")
    o = 0.0
    occ: List[float] = []
    blocked = 0.0
    for _ in range(steps):
        o = max(0.0, o - service_rate)              # serve
        if backpressure:
            admitted = min(arrival_rate, capacity - o)
            blocked += arrival_rate - admitted
            o += admitted
        else:
            o += arrival_rate                        # unbounded admission
        occ.append(o)
    return QueueSim(occ, blocked)


def backlog_growth_per_step(arrival_rate: float, service_rate: float) -> float:
    """Unbounded-regime growth rate max(0, lambda - mu) with no backpressure."""
    return max(0.0, arrival_rate - service_rate)


def bounded_queue_latency(capacity: float, service_rate: float) -> float:
    """Worst-case queueing latency under backpressure: B / mu (bounded)."""
    if service_rate <= 0:
        raise ValueError("service_rate must be > 0")
    return capacity / service_rate


def should_apply_backpressure(occupancy: float, capacity: float,
                              high_watermark: float = 0.9) -> bool:
    """Slow/pause the source when the backlog crosses the high watermark."""
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    return occupancy >= high_watermark * capacity
