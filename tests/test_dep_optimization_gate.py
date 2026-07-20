"""Adversarial tests for the optimization gate (Doc-13, stage 13e)."""

import pytest
import torch
import torch.nn.functional as F

from signtranslator.deployment.optimization_gate import (
    max_abs_error, max_rel_error, certify_optimization,
    online_softmax, online_softmax_attention,
)
from signtranslator.deployment.quantization import (
    symmetric_int8_quantize, symmetric_int8_dequantize,
)

torch.manual_seed(0)


def test_online_softmax_equals_full_softmax():
    x = torch.randn(37, dtype=torch.float64) * 5.0        # wide range
    for bs in (1, 4, 8, 37, 64):
        ref = torch.softmax(x, dim=0)
        got = online_softmax(x, block_size=bs)
        assert torch.allclose(got, ref, atol=1e-12)


def test_online_softmax_attention_equals_standard():
    n, d = 23, 6
    scores = torch.randn(n, dtype=torch.float64) * 3.0
    values = torch.randn(n, d, dtype=torch.float64)
    ref = torch.softmax(scores, dim=0) @ values
    for bs in (1, 5, 23):
        got = online_softmax_attention(scores, values, block_size=bs)
        assert torch.allclose(got, ref, atol=1e-12)       # exact chunked attention


def test_online_softmax_numerically_stable_large_logits():
    x = torch.tensor([1000.0, 1001.0, 999.0], dtype=torch.float64)
    got = online_softmax(x, block_size=1)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.softmax(x, 0), atol=1e-12)


def test_error_helpers():
    a = torch.tensor([1.0, 2.0, 4.0])
    b = torch.tensor([1.0, 2.0, 4.4])
    assert max_abs_error(a, b) == pytest.approx(0.4)
    assert max_rel_error(a, b) == pytest.approx(0.1)


def test_certify_accepts_equivalent_and_quality_preserving():
    eager = torch.randn(50)
    optimized = eager + 1e-5 * torch.randn(50)             # tiny perturbation
    cert = certify_optimization(eager, optimized,
                                quality_eager=0.90, quality_optimized=0.895,
                                atol=1e-3, quality_tolerance=0.01)
    assert cert.numerically_equivalent and cert.quality_preserved and cert.accepted


def test_certify_rejects_numerically_divergent_transform():
    eager = torch.randn(50)
    optimized = eager + 0.5                                 # large deviation
    cert = certify_optimization(eager, optimized,
                                quality_eager=0.9, quality_optimized=0.9)
    assert not cert.numerically_equivalent and not cert.accepted


def test_certify_rejects_quality_regression():
    eager = torch.randn(50)
    optimized = eager.clone()                              # numerically identical
    cert = certify_optimization(eager, optimized,
                                quality_eager=0.90, quality_optimized=0.70,
                                quality_tolerance=0.01)
    assert cert.numerically_equivalent and not cert.quality_preserved
    assert not cert.accepted                               # quality regressed


def test_int8_transform_passes_gate_at_reasonable_tolerance():
    # a real fake-quant transform: quantize -> dequantize, gated on abs error.
    x = torch.randn(200, dtype=torch.float64)
    q, scale = symmetric_int8_quantize(x)
    xhat = symmetric_int8_dequantize(q, scale)
    cert = certify_optimization(x, xhat, quality_eager=0.9, quality_optimized=0.9,
                                atol=scale)                # within one quantum
    assert cert.accepted
