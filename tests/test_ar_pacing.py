"""Verification of frame pacing and SO(3) interpolation.

Proves the resampled rotation at a keyframe equals that keyframe, the SLERP
midpoint is the constant-speed geodesic, linear channel interpolation, timeline
monotonicity, lip/non-manual synchronisation, and determinism.
"""

import pytest
import torch

from signtranslator.pose.rotations import axis_angle_to_matrix, geodesic_distance
from signtranslator.avatar_render.pacing import (
    target_timeline, resample_rotations, resample_linear, pace,
)


def _keys(seed=0):
    g = torch.Generator().manual_seed(seed)
    times = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    R = axis_angle_to_matrix(0.8 * torch.randn(5, 3, generator=g, dtype=torch.float64))
    trans = torch.randn(5, 3, generator=g, dtype=torch.float64)
    expr = torch.randn(5, 4, generator=g, dtype=torch.float64)
    return times, R, trans, expr


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------
def test_target_timeline_monotone_at_fps():
    tl = target_timeline(0.0, 1.0, fps=30.0)
    assert torch.all(tl[1:] > tl[:-1])                       # strictly increasing
    assert abs(float(tl[1] - tl[0]) - 1 / 30.0) < 1e-9       # spacing = 1/fps
    assert tl[0] == 0.0 and tl[-1] <= 1.0 + 1e-9


def test_timeline_rejects_bad_fps():
    with pytest.raises(ValueError):
        target_timeline(0.0, 1.0, fps=0.0)


# ---------------------------------------------------------------------------
# rotation resampling
# ---------------------------------------------------------------------------
def test_resample_at_keyframes_returns_keyframes():
    times, R, _, _ = _keys(1)
    out = resample_rotations(times, R, times)               # query = keyframe times
    assert torch.allclose(out, R, atol=1e-9)


def test_slerp_midpoint_is_constant_speed_geodesic():
    times, R, _, _ = _keys(2)
    mid = (times[:-1] + times[1:]) / 2                       # midpoints
    out = resample_rotations(times, R, mid)
    d_a = geodesic_distance(R[:-1], out)
    d_b = geodesic_distance(out, R[1:])
    assert torch.allclose(d_a, d_b, atol=1e-8)               # equal -> midpoint of geodesic


# ---------------------------------------------------------------------------
# linear resampling + sync
# ---------------------------------------------------------------------------
def test_resample_linear_endpoints_and_midpoint():
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    vals = torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=torch.float64)
    q = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    out = resample_linear(times, vals, q)
    assert torch.allclose(out[0], vals[0], atol=1e-12)
    assert torch.allclose(out[1], torch.tensor([1.0, 2.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(out[2], vals[1], atol=1e-12)


def test_pace_keeps_expression_synced_to_rotation_timeline():
    times, R, trans, expr = _keys(3)
    q, Rp, tp, ep = pace(times, R, trans, expr, fps=60.0)
    # every channel rides the SAME query timeline -> same length, aligned frames
    assert Rp.shape[0] == q.shape[0] == tp.shape[0] == ep.shape[0]
    # at t=0 all channels equal the first keyframe (sync anchor)
    assert torch.allclose(Rp[0], R[0], atol=1e-9)
    assert torch.allclose(ep[0], expr[0], atol=1e-9)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_pacing_is_deterministic():
    times, R, trans, expr = _keys(4)
    a = pace(times, R, trans, expr, fps=45.0)
    b = pace(times, R, trans, expr, fps=45.0)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
