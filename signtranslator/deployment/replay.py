"""Replay validation harness (Doc-13 §8).

Replays timestamped audio against expected semantic checkpoints deterministically,
asserting each checkpoint is committed at or before its time and that the committed
prefix stays monotone (nothing displayed is silently changed). Exercises chunk
boundaries, interruptions, corrections, and cold starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .streaming_contract import is_prefix, certify_commit_monotone

Tokens = Tuple[int, ...]
DecodeFn = Callable[[List[int]], Sequence[int]]


@dataclass(frozen=True)
class SemanticCheckpoint:
    time_s: float
    expected: Tokens          # committed prefix expected to be present by time_s


@dataclass
class ReplayResult:
    passed: bool
    monotone: bool
    failures: List[Tuple[SemanticCheckpoint, Tokens]] = field(default_factory=list)


class ReplayHarness:
    """Deterministic replay of timestamped chunks against semantic checkpoints."""

    def __init__(self, chunks: Sequence[Tuple[float, int]],
                 checkpoints: Sequence[SemanticCheckpoint]) -> None:
        # chunks: (arrival_time_s, chunk_id); checkpoints sorted by time.
        self.chunks = sorted(chunks, key=lambda c: c[0])
        self.checkpoints = sorted(checkpoints, key=lambda c: c.time_s)

    def _chunks_until(self, t: float) -> List[int]:
        return [cid for (ct, cid) in self.chunks if ct <= t]

    def replay(self, decode: DecodeFn) -> ReplayResult:
        """Run ``decode`` at each checkpoint time; verify checkpoint + monotonicity."""
        committed_history: List[Tokens] = []
        failures: List[Tuple[SemanticCheckpoint, Tokens]] = []
        for cp in self.checkpoints:
            committed = tuple(decode(self._chunks_until(cp.time_s)))
            committed_history.append(committed)
            if not is_prefix(cp.expected, committed):
                failures.append((cp, committed))
        monotone, _ = certify_commit_monotone(committed_history)
        return ReplayResult(passed=not failures and monotone,
                            monotone=monotone, failures=failures)
