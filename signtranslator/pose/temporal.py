"""Observed-support derivatives on an explicit clock measured in seconds.

These are Cartesian derivatives, not angular velocities of rotation matrices.
No interpolation crosses unavailable samples. Acceleration uses differences of
interval velocities divided by separation of interval midpoints.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def time_intervals(timestamps: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """Positive seconds between adjacent supported frames; inactive pairs use one.

    Returned inactive values are computational placeholders only. Callers must
    retain pair support. Valid frames must form a prefix, matching corpus padding.
    """
    if (timestamps.ndim != 2 or timestamps.dtype not in (torch.float32, torch.float64)
            or frame_mask.shape != timestamps.shape or frame_mask.dtype != torch.bool):
        raise ValueError('timestamps and boolean frame_mask must have shape (N,T)')
    if timestamps.device != frame_mask.device:
        raise ValueError('timestamps and frame_mask must share a device')
    if timestamps.shape[1] < 1 or timestamps.shape[0] == 0:
        raise ValueError('clock must contain a nonempty batch and at least one frame')
    if bool((frame_mask[:, 1:] & ~frame_mask[:, :-1]).any()):
        raise ValueError('valid clock frames must form a prefix; missing joints use validity masks')
    if not bool(torch.isfinite(timestamps[frame_mask]).all()):
        raise ValueError('supported timestamps must be finite seconds')
    # Sanitize padding before arithmetic: NaN multiplied by zero is still NaN.
    safe = torch.where(frame_mask, timestamps, 0)
    dt = safe[:, 1:] - safe[:, :-1]
    pairs = frame_mask[:, 1:] & frame_mask[:, :-1]
    if bool((~torch.isfinite(dt[pairs]) | (dt[pairs] <= 0)).any()):
        raise ValueError('supported timestamps must be strictly increasing with finite intervals')
    return torch.where(pairs, dt, 1)


def validate_max_gap_seconds(value):
    """Validate a caller-supplied source policy; no universal cutoff is assumed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('max_gap_seconds must be a finite positive real scalar')
    try:
        limit = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError('max_gap_seconds must be a finite positive real scalar') from exc
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('max_gap_seconds must be a finite positive real scalar')
    return limit


def supported_time_intervals(timestamps, frame_mask, *, max_gap_seconds=None):
    """Return finite computational intervals and explicit adjacency support.

    A gap exactly at the declared limit is accepted. Larger gaps are unavailable,
    not interpolated. Clock ordering remains validated even across rejected gaps.
    """
    max_gap_seconds = validate_max_gap_seconds(max_gap_seconds)
    dt = time_intervals(timestamps, frame_mask)
    pairs = frame_mask[:, 1:] & frame_mask[:, :-1]
    if max_gap_seconds is not None:
        pairs = pairs & (dt <= max_gap_seconds)
    return torch.where(pairs, dt, 1), pairs


def cartesian_derivatives(values: torch.Tensor, timestamps: torch.Tensor,
                          validity: torch.Tensor, *, frame_mask: torch.Tensor | None = None,
                          max_gap_seconds=None):
    """Return velocity, velocity support, acceleration and acceleration support.

    Values are (N,C,T,V), validity (N,T,V), timestamps (N,T). Velocity is at
    interval midpoints; acceleration is at the middle observation in the local
    quadratic interpolant. Unavailable entries are zero placeholders with false
    support, never observations. A two-frame sequence has empty acceleration.
    """
    if values.ndim != 4 or values.dtype not in (torch.float32, torch.float64):
        raise ValueError('values must be floating (N,C,T,V)')
    n, _, t, v = values.shape
    if t < 2:
        raise ValueError('Cartesian derivatives require at least two frames')
    if validity.shape != (n,t,v) or validity.dtype != torch.bool:
        raise ValueError('validity must be boolean (N,T,V)')
    if timestamps.shape != (n,t) or validity.device != values.device or timestamps.device != values.device:
        raise ValueError('clock shape and devices must match motion')
    if frame_mask is None:
        frame_mask = torch.ones((n,t),dtype=torch.bool,device=values.device)
    dt, clock_pairs = supported_time_intervals(
        timestamps, frame_mask, max_gap_seconds=max_gap_seconds)
    valid = validity & frame_mask.unsqueeze(-1)
    support = valid.unsqueeze(1).expand_as(values)
    if not bool(torch.isfinite(values[support]).all()):
        raise ValueError('observed motion must be finite')
    safe = torch.where(support, values, 0)
    pair = valid[:,1:] & valid[:,:-1] & clock_pairs.unsqueeze(-1)
    pair_support = pair.unsqueeze(1)
    velocity = (torch.where(pair_support, safe[:,:,1:], 0)
                - torch.where(pair_support, safe[:,:,:-1], 0)) / dt[:,None,:,None]
    triple = pair[:,1:] & pair[:,:-1]
    triple_support = triple.unsqueeze(1)
    # Half-sum avoids overflow in dt_left + dt_right at large finite intervals.
    midpoint_dt = dt[:,None,1:,None] / 2 + dt[:,None,:-1,None] / 2
    acceleration = (torch.where(triple_support, velocity[:,:,1:], 0)
                    - torch.where(triple_support, velocity[:,:,:-1], 0)) / midpoint_dt
    if not bool(torch.isfinite(velocity).all() & torch.isfinite(acceleration).all()):
        raise FloatingPointError('temporal derivative exceeds finite numerical range')
    return velocity, pair, acceleration, triple
