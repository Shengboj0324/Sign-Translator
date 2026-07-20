"""Verification of streaming attention and SO(3) chunk blending.

Proves the bounded-right-context latency (output t depends only on inputs <= t+R),
that SLERP is a constant-speed geodesic with exact endpoints, the crossfade
boundary constraints, and overlap-add length/continuity.
"""

import math

import pytest
import torch

from signtranslator.pose.rotations import (
    axis_angle_to_matrix, is_rotation_matrix, geodesic_distance,
)
from signtranslator.motion_transformer.streaming import (
    bounded_right_context_mask, streaming_latency_frames, slerp,
    crossfade_rotations, overlap_add_rotations,
)


def _rand_rots(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return axis_angle_to_matrix(0.9 * torch.randn(n, 3, generator=g, dtype=torch.float64))


# ---------------------------------------------------------------------------
# streaming mask + latency
# ---------------------------------------------------------------------------
def test_bounded_mask_allowed_positions():
    m = bounded_right_context_mask(5, right_context=1)
    # query 2 may see keys <= 3
    assert m[2, 3] == 0.0 and m[2, 4] == float("-inf")
    assert m[2, 0] == 0.0                                    # unbounded past
    # with a left context, the past is bounded too
    m2 = bounded_right_context_mask(5, right_context=1, left_context=1)
    assert m2[3, 1] == float("-inf") and m2[3, 2] == 0.0


def test_output_depends_only_on_inputs_within_right_context():
    """A masked single-head attention: output t must not depend on inputs t+R+1.."""
    torch.manual_seed(0)
    T, d, R = 8, 4, 2
    x = torch.randn(T, d, dtype=torch.float64, requires_grad=True)
    mask = bounded_right_context_mask(T, right_context=R)
    scores = x @ x.t() / math.sqrt(d) + mask
    attn = torch.softmax(scores, dim=-1)
    out = attn @ x
    grad = torch.autograd.grad(out[2].sum(), x, retain_graph=True)[0]
    # frames beyond t + R = 4 must have zero influence on output[2]
    assert grad[5:].abs().sum().item() < 1e-9


def test_latency_formula():
    assert streaming_latency_frames(chunk_size=8, right_context=2) == 10


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------
def test_slerp_endpoints():
    Ra, Rb = _rand_rots(6, 1), _rand_rots(6, 2)
    assert torch.allclose(slerp(Ra, Rb, 0.0), Ra, atol=1e-10)
    assert torch.allclose(slerp(Ra, Rb, 1.0), Rb, atol=1e-9)


def test_slerp_is_constant_speed_geodesic():
    Ra, Rb = _rand_rots(20, 3), _rand_rots(20, 4)
    d_full = geodesic_distance(Ra, Rb)                       # (20,)
    for a in (0.25, 0.5, 0.75):
        d_a = geodesic_distance(Ra, slerp(Ra, Rb, a))
        assert torch.allclose(d_a, a * d_full, atol=1e-8)    # constant-speed geodesic


def test_slerp_stays_on_so3():
    Ra, Rb = _rand_rots(10, 5), _rand_rots(10, 6)
    R = slerp(Ra, Rb, 0.37)
    assert is_rotation_matrix(R, atol=1e-9).all()


# ---------------------------------------------------------------------------
# crossfade + overlap-add
# ---------------------------------------------------------------------------
def test_crossfade_matches_boundaries():
    Ra = _rand_rots(5, 7)
    Rb = _rand_rots(5, 8)
    blended = crossfade_rotations(Ra, Rb)
    assert torch.allclose(blended[0], Ra[0], atol=1e-10)     # start = old chunk
    assert torch.allclose(blended[-1], Rb[-1], atol=1e-9)    # end = new chunk
    assert is_rotation_matrix(blended, atol=1e-9).all()


def test_overlap_add_length_and_endpoints():
    chunks = [_rand_rots(6, 10), _rand_rots(6, 11), _rand_rots(6, 12)]
    overlap = 2
    stitched = overlap_add_rotations(chunks, overlap)
    expected_len = sum(c.shape[0] for c in chunks) - overlap * (len(chunks) - 1)
    assert stitched.shape[0] == expected_len                 # 18 - 4 = 14
    assert torch.allclose(stitched[0], chunks[0][0], atol=1e-10)      # begins at chunk 0
    assert torch.allclose(stitched[-1], chunks[-1][-1], atol=1e-9)    # ends at last chunk
    assert is_rotation_matrix(stitched, atol=1e-9).all()


def test_overlap_add_zero_overlap_is_concatenation():
    chunks = [_rand_rots(4, 20), _rand_rots(4, 21)]
    stitched = overlap_add_rotations(chunks, overlap=0)
    assert stitched.shape[0] == 8
    assert torch.allclose(stitched[:4], chunks[0], atol=1e-12)
    assert torch.allclose(stitched[4:], chunks[1], atol=1e-12)


def test_overlap_add_rejects_bad_args():
    with pytest.raises(ValueError):
        overlap_add_rotations([], overlap=1)
    with pytest.raises(ValueError):
        overlap_add_rotations([_rand_rots(4)], overlap=-1)
