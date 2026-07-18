"""Verification of the temporal resampler and the gated projection.

The gated projection's defining algebra is asserted directly: the convexity
collapse (W1 == W2 makes the gate irrelevant) and the elementwise convex bound.
"""

import pytest
import torch

from signtranslator.speech.projection import (
    TemporalResampler, GatedProjection, SpeechPathways, SpeechProjector,
)


# ---------------------------------------------------------------------------
# Temporal resampler
# ---------------------------------------------------------------------------
def test_linear_resampler_identity_at_same_length():
    r = TemporalResampler(dim=8, mode="linear", ratio=1.0)
    x = torch.randn(2, 20, 8)
    assert torch.allclose(r(x), x, atol=1e-6)


def test_linear_resampler_hits_target_length():
    r = TemporalResampler(dim=8, mode="linear")
    x = torch.randn(2, 20, 8)
    assert r(x, target_length=37).shape == (2, 37, 8)
    assert r(x, target_length=5).shape == (2, 5, 8)


def test_linear_resampler_ratio_sets_output_length():
    r = TemporalResampler(dim=4, mode="linear", ratio=0.5)
    assert r.output_length(20) == 10
    assert r(torch.randn(1, 20, 4)).shape == (1, 10, 4)


def test_linear_resampler_interpolates_exactly():
    """Upsampling a linear ramp must stay on the line (align_corners=True)."""
    t = torch.linspace(0.0, 1.0, 5).view(1, 5, 1)
    out = TemporalResampler(dim=1, mode="linear")(t, target_length=9)
    assert torch.allclose(out.flatten(), torch.linspace(0.0, 1.0, 9), atol=1e-6)


def test_conv_resampler_downsamples_by_stride():
    r = TemporalResampler(dim=6, mode="conv", stride=2, kernel_size=3)
    x = torch.randn(2, 21, 6)
    out = r(x)
    assert out.shape == (2, r.output_length(21), 6)
    assert out.shape[1] == 11                       # (21 + 2 - 3)//2 + 1


def test_conv_resampler_rejects_target_length():
    r = TemporalResampler(dim=6, mode="conv")
    with pytest.raises(ValueError):
        r(torch.randn(1, 10, 6), target_length=5)


def test_resampler_validates_arguments():
    with pytest.raises(ValueError):
        TemporalResampler(dim=4, mode="cubic")
    with pytest.raises(ValueError):
        TemporalResampler(dim=4, ratio=0.0)
    with pytest.raises(ValueError):
        TemporalResampler(dim=4)(torch.randn(3, 4))          # wrong rank
    with pytest.raises(ValueError):
        TemporalResampler(dim=4)(torch.randn(1, 5, 7))       # wrong feature dim


# ---------------------------------------------------------------------------
# Gated projection -- structural algebra
# ---------------------------------------------------------------------------
def test_gate_is_bounded_in_unit_interval():
    """G in [0,1] always -- this closed bound is what the convexity argument needs.

    Strict interiority holds mathematically but *not* in float32: sigmoid(x)
    rounds to exactly 1.0 for x >~ 17. That saturated endpoint is the
    "selects one projection" case covered separately, and it does not break the
    convex bound.
    """
    proj = GatedProjection(16, 32)
    g_large = proj.gate(torch.randn(4, 10, 16) * 10.0)
    assert torch.all(g_large >= 0.0) and torch.all(g_large <= 1.0)
    # Unsaturated inputs must land strictly inside.
    g_small = proj.gate(torch.randn(4, 10, 16) * 0.1)
    assert torch.all(g_small > 0.0) and torch.all(g_small < 1.0)


def test_convexity_collapse_when_projections_are_equal():
    """W1 == W2 => output == W1 x for EVERY gate value.

    The strongest available structural check: it holds identically, so it
    cannot be satisfied by accident (e.g. a gate stuck near 0.5).
    """
    torch.manual_seed(0)
    proj = GatedProjection(16, 24)
    with torch.no_grad():
        proj.w2.weight.copy_(proj.w1.weight)
        proj.w2.bias.copy_(proj.w1.bias)
        # Deliberately extreme gate logits: the identity must survive them.
        proj.wg.weight.normal_(0, 5.0)
        proj.wg.bias.normal_(0, 5.0)
    x = torch.randn(3, 7, 16)
    assert torch.allclose(proj(x), proj.w1(x), atol=1e-6)


def test_output_is_elementwise_between_the_two_projections():
    torch.manual_seed(1)
    proj = GatedProjection(12, 20)
    x = torch.randn(5, 9, 12)
    out, a, b = proj(x), proj.w1(x), proj.w2(x)
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    assert torch.all(out >= lo - 1e-6) and torch.all(out <= hi + 1e-6)


def test_matches_explicit_formula():
    """Recompute G (*) W1 x + (1-G) (*) W2 x by hand."""
    torch.manual_seed(2)
    proj = GatedProjection(8, 8)
    x = torch.randn(2, 4, 8)
    g = torch.sigmoid(proj.wg(x))
    expected = g * proj.w1(x) + (1.0 - g) * proj.w2(x)
    assert torch.allclose(proj(x), expected, atol=1e-7)


def test_saturated_gate_selects_one_projection():
    """Driving the gate to +inf must recover exactly W1 x."""
    torch.manual_seed(3)
    proj = GatedProjection(8, 8)
    with torch.no_grad():
        proj.wg.weight.zero_()
        proj.wg.bias.fill_(30.0)                     # sigmoid(30) == 1 in fp32
    x = torch.randn(2, 5, 8)
    assert torch.allclose(proj(x), proj.w1(x), atol=1e-6)
    with torch.no_grad():
        proj.wg.bias.fill_(-30.0)
    assert torch.allclose(proj(x), proj.w2(x), atol=1e-6)


def test_all_three_matrices_receive_gradient():
    proj = GatedProjection(10, 10)
    proj(torch.randn(3, 6, 10)).sum().backward()
    for name in ("w1", "w2", "wg"):
        w = getattr(proj, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, name


def test_projection_shape():
    assert GatedProjection(16, 40)(torch.randn(2, 11, 16)).shape == (2, 11, 40)


# ---------------------------------------------------------------------------
# Pathways
# ---------------------------------------------------------------------------
def test_pathways_keep_modalities_separate():
    """The spec requires the three pathways be retained, not pre-fused."""
    p = SpeechPathways(acoustic=torch.randn(2, 5, 8),
                       prosody=torch.randn(2, 5, 4))
    assert p.lexical is None
    assert p.acoustic.shape[-1] == 8 and p.prosody.shape[-1] == 4


def test_as_fused_concatenates_on_request_only():
    p = SpeechPathways(acoustic=torch.randn(2, 5, 8),
                       prosody=torch.randn(2, 5, 4))
    assert p.as_fused().shape == (2, 5, 12)
    assert SpeechPathways(acoustic=torch.randn(2, 5, 8)).as_fused().shape == (2, 5, 8)


def test_as_fused_rejects_misaligned_grids():
    p = SpeechPathways(acoustic=torch.randn(2, 5, 8),
                       prosody=torch.randn(2, 7, 4))
    with pytest.raises(ValueError):
        p.as_fused()


# ---------------------------------------------------------------------------
# Projector integration
# ---------------------------------------------------------------------------
def test_projector_maps_encoder_width_to_planner_width():
    sp = SpeechProjector(encoder_dim=32, planner_dim=64)
    out = sp(torch.randn(2, 30, 32))
    assert out.acoustic.shape == (2, 30, 64)


def test_projector_resamples_and_aligns_prosody():
    """Prosody is computed on a different frame grid and must be re-aligned."""
    sp = SpeechProjector(encoder_dim=16, planner_dim=8)
    out = sp(torch.randn(1, 40, 16), prosody=torch.randn(1, 27, 4),
             target_length=20)
    assert out.acoustic.shape == (1, 20, 8)
    assert out.prosody.shape == (1, 20, 4)
    assert out.as_fused().shape == (1, 20, 12)


def test_projector_with_conv_resampler():
    sp = SpeechProjector(encoder_dim=16, planner_dim=8,
                         resample_mode="conv", stride=2)
    out = sp(torch.randn(2, 21, 16))
    assert out.acoustic.shape == (2, 11, 8)


def test_projector_rejects_bad_prosody_rank():
    sp = SpeechProjector(encoder_dim=16, planner_dim=8)
    with pytest.raises(ValueError):
        sp(torch.randn(1, 10, 16), prosody=torch.randn(10, 4))


def test_projector_is_differentiable_end_to_end():
    sp = SpeechProjector(encoder_dim=16, planner_dim=8)
    x = torch.randn(2, 12, 16, requires_grad=True)
    sp(x).acoustic.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
