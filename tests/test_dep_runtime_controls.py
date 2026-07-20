"""Adversarial tests for runtime controls (Doc-13, stage 13f)."""

import numpy as np
import pytest

from signtranslator.deployment.runtime_controls import (
    TokenCategory, RuntimeAction, clarification_gate, fallback_render,
    TelemetryThresholds, TelemetrySnapshot, PrivacyRingBuffer,
)
from signtranslator.speech.policy import FailClosedPolicy


def _policy():
    return FailClosedPolicy(emit_threshold=0.8, fingerspell_threshold=0.5)


def test_low_confidence_name_triggers_clarification():
    p = _policy()
    # a name at 0.9 confidence would EMIT ordinarily, but names demand >= 0.95.
    assert clarification_gate(1, 0.9, TokenCategory.NAME, p) == RuntimeAction.CLARIFY
    assert clarification_gate(1, 0.9, TokenCategory.NUMBER, p) == RuntimeAction.CLARIFY


def test_high_confidence_name_emits():
    p = _policy()
    assert clarification_gate(1, 0.99, TokenCategory.NAME, p) == RuntimeAction.EMIT


def test_ordinary_token_defers_to_policy():
    p = _policy()
    assert clarification_gate(1, 0.95, TokenCategory.ORDINARY, p) == RuntimeAction.EMIT
    assert clarification_gate(1, 0.2, TokenCategory.ORDINARY, p) == RuntimeAction.PAUSE


def test_fallback_is_deterministic():
    verified = {5: "HELLO"}
    assert fallback_render(5, verified) == "HELLO"          # verified retrieval
    a = fallback_render(42, verified)
    b = fallback_render(42, verified)
    assert a == b and a.startswith("fs:")                    # deterministic fingerspell


def test_telemetry_healthy_and_violations():
    thr = TelemetryThresholds()
    ok = TelemetrySnapshot(thermal_c=60, memory_frac=0.5,
                           dropped_frame_rate=0.01, desync_ms=10)
    assert ok.healthy(thr) and ok.violations(thr) == []
    hot = TelemetrySnapshot(thermal_c=95, memory_frac=0.95,
                            dropped_frame_rate=0.2, desync_ms=100)
    assert not hot.healthy(thr)
    assert set(hot.violations(thr)) == {"thermal", "memory", "dropped_frames", "desync"}


def test_privacy_buffer_never_retains_beyond_window():
    buf = PrivacyRingBuffer(window=100, privacy_mode=True)
    for _ in range(10):
        buf.push(np.random.randn(50))
    assert buf.retained_samples <= 100                       # bounded retention


def test_privacy_clear_zeroises_and_drops():
    buf = PrivacyRingBuffer(window=100)
    buf.push(np.random.randn(80))
    assert buf.retained_samples == 80
    buf.clear()
    assert buf.retained_samples == 0
    assert buf.snapshot().size == 0                          # no residual audio


def test_privacy_buffer_rejects_bad_window():
    with pytest.raises(ValueError):
        PrivacyRingBuffer(window=0)
