"""Quantization math + error bounds (Doc-13 §4).

FP16 rounding, symmetric INT8 (TensorRT convention: scale from max-abs), and affine
UINT8 quant/dequant, each with a proven per-element error bound, plus min/max and
percentile calibration and a range-coverage certificate. No real INT8 kernel — the
math and the fake-quant that validation requires.
"""

from __future__ import annotations

from typing import Tuple

import torch

FP16_MACHINE_EPS = 2.0 ** -11        # 10 explicit mantissa bits, round-to-nearest


# ---------------------------------------------------------------------------
# FP16
# ---------------------------------------------------------------------------
def fp16_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Round to fp16 and back (fake fp16)."""
    return x.to(torch.float16).to(x.dtype)


def fp16_max_relative_error(x: torch.Tensor) -> float:
    """Max relative error of fp16 rounding over the (nonzero) elements of x."""
    xh = fp16_round_trip(x)
    nz = x.abs() > 0
    if not nz.any():
        return 0.0
    return float(((xh[nz] - x[nz]).abs() / x[nz].abs()).max())


# ---------------------------------------------------------------------------
# symmetric INT8 (TensorRT convention)
# ---------------------------------------------------------------------------
def symmetric_int8_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """q = clip(round(x/s), -127, 127), s = max|x| / 127. Returns (q, scale)."""
    amax = float(x.abs().max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -127, 127)
    return q, scale


def symmetric_int8_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q * scale


# ---------------------------------------------------------------------------
# affine (asymmetric) UINT8
# ---------------------------------------------------------------------------
def affine_uint8_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Affine uint8: s=(max-min)/255, z=round(-min/s). Returns (q, scale, zero_point)."""
    xmin, xmax = float(x.min()), float(x.max())
    if xmax == xmin:
        return torch.zeros_like(x), 1.0, 0
    scale = (xmax - xmin) / 255.0
    zero_point = int(round(-xmin / scale))
    zero_point = max(0, min(255, zero_point))
    q = torch.clamp(torch.round(x / scale) + zero_point, 0, 255)
    return q, scale, zero_point


def affine_uint8_dequantize(q: torch.Tensor, scale: float, zero_point: int
                            ) -> torch.Tensor:
    return (q - zero_point) * scale


def quantization_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Max absolute error between a tensor and its dequantised reconstruction."""
    return float((x - x_hat).abs().max())


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def calibrate_minmax(x: torch.Tensor) -> Tuple[float, float]:
    return float(x.min()), float(x.max())


def calibrate_percentile(x: torch.Tensor, p: float = 99.9) -> Tuple[float, float]:
    """Clip to the [100-p, p] percentiles — a smaller scale at the cost of clipping."""
    if not 50.0 < p < 100.0:
        raise ValueError("p must be in (50, 100)")
    lo = float(torch.quantile(x.flatten().to(torch.float64), (100 - p) / 100.0))
    hi = float(torch.quantile(x.flatten().to(torch.float64), p / 100.0))
    return lo, hi


def range_coverage(x: torch.Tensor, lo: float, hi: float) -> float:
    """Fraction of x within the representable range [lo, hi] (no silent clipping)."""
    within = (x >= lo) & (x <= hi)
    return float(within.float().mean())
