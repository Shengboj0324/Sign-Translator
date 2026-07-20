"""Runtime controls (Doc-13 §6).

Confidence gate + clarification for names/numbers/low-confidence ASR (reuse the
Doc-03 fail-closed policy), a deterministic fallback path, telemetry gauges with
thresholds, and a privacy ring buffer with immediate deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional

import numpy as np

from ..speech.policy import FailClosedPolicy, Action


# ---------------------------------------------------------------------------
# confidence gate + clarification
# ---------------------------------------------------------------------------
class TokenCategory(Enum):
    ORDINARY = "ordinary"
    NAME = "name"
    NUMBER = "number"


class RuntimeAction(IntEnum):
    PAUSE = 0
    CLARIFY = 1
    FINGERSPELL = 2
    EMIT = 3


def clarification_gate(token: int, confidence: float, category: TokenCategory,
                       policy: FailClosedPolicy,
                       clarify_threshold: float = 0.95) -> RuntimeAction:
    """Route high-stakes low-confidence tokens (names/numbers) to CLARIFY.

    Names and numbers are unrecoverable if guessed wrong, so below a strict
    ``clarify_threshold`` the system asks rather than asserts; otherwise it defers
    to the Doc-03 fail-closed policy (PAUSE < FINGERSPELL < EMIT).
    """
    if category in (TokenCategory.NAME, TokenCategory.NUMBER) and confidence < clarify_threshold:
        return RuntimeAction.CLARIFY
    base = policy.decide(token, confidence).action
    return {Action.PAUSE: RuntimeAction.PAUSE,
            Action.FINGERSPELL: RuntimeAction.FINGERSPELL,
            Action.EMIT: RuntimeAction.EMIT}[base]


# ---------------------------------------------------------------------------
# deterministic fallback
# ---------------------------------------------------------------------------
def fallback_render(token: int, verified_phrases: Dict[int, str]) -> str:
    """Deterministic fallback: a verified retrieved phrase, else fingerspelling.

    Side-effect-free and deterministic — the always-available safe path.
    """
    if token in verified_phrases:
        return verified_phrases[token]
    return "fs:" + "-".join(str(token))          # deterministic fingerspelling


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TelemetryThresholds:
    max_thermal_c: float = 85.0
    max_memory_frac: float = 0.90
    max_dropped_frame_rate: float = 0.05
    max_desync_ms: float = 50.0


@dataclass(frozen=True)
class TelemetrySnapshot:
    thermal_c: float
    memory_frac: float
    dropped_frame_rate: float
    desync_ms: float

    def violations(self, thr: TelemetryThresholds = TelemetryThresholds()) -> List[str]:
        v: List[str] = []
        if self.thermal_c > thr.max_thermal_c:
            v.append("thermal")
        if self.memory_frac > thr.max_memory_frac:
            v.append("memory")
        if self.dropped_frame_rate > thr.max_dropped_frame_rate:
            v.append("dropped_frames")
        if self.desync_ms > thr.max_desync_ms:
            v.append("desync")
        return v

    def healthy(self, thr: TelemetryThresholds = TelemetryThresholds()) -> bool:
        return not self.violations(thr)


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------
class PrivacyRingBuffer:
    """On-device audio ring buffer with immediate deletion (privacy mode)."""

    def __init__(self, window: int, privacy_mode: bool = True) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.privacy_mode = privacy_mode
        self._buf = np.zeros(0, dtype=np.float64)

    def push(self, samples: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float64)])
        if self.privacy_mode and self._buf.size > self.window:
            # privacy mode never retains beyond the window: drop the oldest.
            self._buf = self._buf[-self.window:]

    @property
    def retained_samples(self) -> int:
        return int(self._buf.size)

    def clear(self) -> None:
        """Immediately zeroise and drop the buffer (no residual audio)."""
        self._buf[:] = 0.0
        self._buf = np.zeros(0, dtype=np.float64)

    def snapshot(self) -> np.ndarray:
        return self._buf.copy()
