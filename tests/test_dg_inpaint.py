"""Verification of diffusion inpainting (RePaint-style).

Proves the known/unknown merge, the exact anchor at t=0, that the known region
ignores the model's proposal, the noise-free forward anchor, the streaming-overlap
mask, and that a full inpaint loop preserves the known region exactly.
"""

import pytest
import torch

from signtranslator.diffusion_gen.schedule import NoiseSchedule
from signtranslator.diffusion_gen.inpaint import (
    forward_diffuse_known, merge_known_unknown, inpaint_step, streaming_overlap_mask,
)


def _sched():
    return NoiseSchedule(num_timesteps=200)


# ---------------------------------------------------------------------------
# merge + anchors
# ---------------------------------------------------------------------------
def test_merge_selects_known_and_unknown_regions():
    known = torch.ones(4, 5, dtype=torch.float64)
    sampled = torch.full((4, 5), -1.0, dtype=torch.float64)
    mask = torch.zeros(4, 5, dtype=torch.float64); mask[:2] = 1.0     # first 2 rows known
    out = merge_known_unknown(known, sampled, mask)
    assert torch.allclose(out[:2], known[:2])
    assert torch.allclose(out[2:], sampled[2:])


def test_forward_diffuse_known_noise_free_anchor():
    s = _sched()
    x0 = torch.randn(3, 4, dtype=torch.float64)
    t = torch.full((3,), 100)
    anchor = forward_diffuse_known(s, x0, t, eps=None)       # eps=0
    a = s.a(t, x0.shape)
    assert torch.allclose(anchor, a * x0, atol=1e-12)        # = sqrt(abar) x0


def test_inpaint_step_exact_anchor_at_t0():
    s = _sched()
    x0_known = torch.randn(6, 4, dtype=torch.float64)
    mask = torch.zeros(6, 4, dtype=torch.float64); mask[:3] = 1.0
    x_prev = torch.randn(6, 4, dtype=torch.float64)
    out = inpaint_step(s, torch.randn(6, 4, dtype=torch.float64),
                       torch.zeros(6, dtype=torch.long), x0_known, mask, x_prev)
    assert torch.allclose(out[:3], x0_known[:3], atol=1e-12)  # known == x0 exactly at t=0


def test_known_region_ignores_model_proposal():
    s = _sched()
    torch.manual_seed(0)
    x0_known = torch.randn(6, 4, dtype=torch.float64)
    mask = torch.zeros(6, 4, dtype=torch.float64); mask[:3] = 1.0
    t = torch.full((6,), 50)
    g1 = torch.Generator().manual_seed(1)
    out_a = inpaint_step(s, torch.zeros(6, 4, dtype=torch.float64), t, x0_known, mask,
                         x_prev_sampled=torch.randn(6, 4, dtype=torch.float64), generator=g1)
    g2 = torch.Generator().manual_seed(1)                    # same noise
    out_b = inpaint_step(s, torch.zeros(6, 4, dtype=torch.float64), t, x0_known, mask,
                         x_prev_sampled=torch.full((6, 4), 99.0, dtype=torch.float64), generator=g2)
    # the known region is identical regardless of the (very different) proposals
    assert torch.allclose(out_a[:3], out_b[:3], atol=1e-12)
    assert not torch.allclose(out_a[3:], out_b[3:])          # unknown region differs


# ---------------------------------------------------------------------------
# streaming overlap mask
# ---------------------------------------------------------------------------
def test_streaming_overlap_mask():
    m = streaming_overlap_mask(num_frames=10, overlap=3, data_shape=(4,))
    assert m.shape == (10, 4)
    assert torch.all(m[:3] == 1.0) and torch.all(m[3:] == 0.0)


# ---------------------------------------------------------------------------
# full inpaint loop preserves the known region
# ---------------------------------------------------------------------------
def test_full_inpaint_loop_preserves_known_region():
    s = _sched()
    torch.manual_seed(2)
    x0_known = torch.randn(8, 4, dtype=torch.float64)        # the "true" clip
    mask = torch.zeros(8, 4, dtype=torch.float64); mask[:4] = 1.0   # first half fixed
    x = torch.randn(8, 4, dtype=torch.float64)               # x_T
    g = torch.Generator().manual_seed(3)
    for step in reversed(range(s.num_timesteps)):
        # a trivial "model": propose a small pull toward 0 for the unknown region
        x_prev = 0.99 * x
        x = inpaint_step(s, x, torch.full((8,), step), x0_known, mask, x_prev, generator=g)
    assert torch.allclose(x[:4], x0_known[:4], atol=1e-12)   # known region exact at the end
