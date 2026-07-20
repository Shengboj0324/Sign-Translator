"""Frame pacing and SO(3) interpolation (docs/AVATAR_RENDER.md §6).

Resample a timestamped parameter stream to a target frame rate: rotations by the
Doc-06 **SLERP** (constant-speed geodesic, exact endpoints), translations and
expression channels by linear interpolation. The lip/non-manual (expression)
channel rides the same target timeline, so it stays synchronised. Resampling is a
pure function of the inputs (deterministic replay).
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..motion_transformer.streaming import slerp


def target_timeline(t_start: float, t_end: float, fps: float) -> torch.Tensor:
    """Frame timestamps at ``fps`` over ``[t_start, t_end]`` (inclusive of start)."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    n = int(torch.floor(torch.tensor((t_end - t_start) * fps)).item()) + 1
    return t_start + torch.arange(n, dtype=torch.float64) / fps


def _bracket(key_times: torch.Tensor, query: torch.Tensor):
    """Left index of the bracketing keyframe pair for each query time, plus alpha."""
    K = key_times.shape[0]
    idx = torch.searchsorted(key_times, query, right=True).clamp(1, K - 1) - 1
    t0, t1 = key_times[idx], key_times[idx + 1]
    alpha = ((query - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return idx, alpha


def resample_rotations(key_times: torch.Tensor, key_R: torch.Tensor,
                       query_times: torch.Tensor) -> torch.Tensor:
    """SLERP-resample rotations. ``key_R`` (K, 3, 3) -> (Q, 3, 3)."""
    idx, alpha = _bracket(key_times, query_times)
    return slerp(key_R[idx], key_R[idx + 1], alpha)


def resample_linear(key_times: torch.Tensor, key_vals: torch.Tensor,
                    query_times: torch.Tensor) -> torch.Tensor:
    """Linearly resample a vector channel. ``key_vals`` (K, D) -> (Q, D)."""
    idx, alpha = _bracket(key_times, query_times)
    a = alpha.unsqueeze(-1)
    return (1 - a) * key_vals[idx] + a * key_vals[idx + 1]


def pace(key_times: torch.Tensor, key_R: torch.Tensor, key_trans: torch.Tensor,
         key_expr: torch.Tensor, fps: float
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample a (rotations, translations, expression) stream to ``fps``.

    Returns (query_times, R (Q,3,3), trans (Q,D), expr (Q,E)); the expression rides
    the SAME timeline as the rotations, so lip/non-manual stays synchronised.
    """
    q = target_timeline(float(key_times[0]), float(key_times[-1]), fps)
    R = resample_rotations(key_times, key_R, q)
    trans = resample_linear(key_times, key_trans, q)
    expr = resample_linear(key_times, key_expr, q)
    return q, R, trans, expr
