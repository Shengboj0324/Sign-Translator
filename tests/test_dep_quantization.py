"""Adversarial tests for quantization math (Doc-13, stage 13d)."""

import pytest
import torch

from signtranslator.deployment.quantization import (
    FP16_MACHINE_EPS, fp16_round_trip, fp16_max_relative_error,
    symmetric_int8_quantize, symmetric_int8_dequantize,
    affine_uint8_quantize, affine_uint8_dequantize, quantization_error,
    calibrate_minmax, calibrate_percentile, range_coverage,
)

torch.manual_seed(0)


def test_fp16_relative_error_within_bound():
    x = torch.linspace(0.1, 10.0, 500, dtype=torch.float64)
    assert fp16_max_relative_error(x) <= FP16_MACHINE_EPS + 1e-12


def test_fp16_round_trip_exact_for_representable():
    x = torch.tensor([0.5, 1.0, 2.0, 0.25], dtype=torch.float64)   # exact in fp16
    assert torch.allclose(fp16_round_trip(x), x, atol=0.0)


def test_symmetric_int8_error_within_half_scale():
    x = torch.randn(1000, dtype=torch.float64)
    q, scale = symmetric_int8_quantize(x)
    xhat = symmetric_int8_dequantize(q, scale)
    assert q.abs().max() <= 127
    # in-range rounding error bounded by scale/2.
    assert quantization_error(x, xhat) <= scale / 2 + 1e-12


def test_symmetric_int8_range_endpoints():
    x = torch.tensor([-4.0, 4.0, 0.0, 2.0], dtype=torch.float64)
    q, scale = symmetric_int8_quantize(x)
    assert scale == pytest.approx(4.0 / 127)
    assert q.max() == 127 and q.min() == -127            # extremes map to +/-127


def test_affine_uint8_error_within_half_scale():
    x = torch.rand(1000, dtype=torch.float64) * 6.0 - 1.0     # asymmetric range
    q, scale, zp = affine_uint8_quantize(x)
    xhat = affine_uint8_dequantize(q, scale, zp)
    assert q.min() >= 0 and q.max() <= 255
    assert quantization_error(x, xhat) <= scale / 2 + 1e-12


def test_affine_handles_constant_tensor():
    x = torch.full((10,), 3.0, dtype=torch.float64)
    q, scale, zp = affine_uint8_quantize(x)
    assert torch.allclose(affine_uint8_dequantize(q, scale, zp),
                          torch.zeros_like(x))              # degenerate range


def test_percentile_calibration_shrinks_range():
    x = torch.randn(10000, dtype=torch.float64)
    x[0] = 100.0                                            # a gross outlier
    lo_mm, hi_mm = calibrate_minmax(x)
    lo_p, hi_p = calibrate_percentile(x, 99.9)
    assert hi_p < hi_mm                                     # percentile clips the tail
    # a smaller max-abs -> a smaller symmetric scale.
    assert max(abs(lo_p), abs(hi_p)) < max(abs(lo_mm), abs(hi_mm))


def test_range_coverage_certificate():
    x = torch.randn(1000, dtype=torch.float64)
    lo, hi = calibrate_minmax(x)
    assert range_coverage(x, lo, hi) == 1.0                # min/max covers all
    assert range_coverage(x, -0.5, 0.5) < 1.0             # a tight range clips
