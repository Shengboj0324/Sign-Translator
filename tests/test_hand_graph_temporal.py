"""Verification of the temporal pyramid and confidence-aware masked conv.

Proves the receptive-field formula (via autograd dependency), length preservation,
and the masked-convolution properties: occluded frames contribute nothing, an
all-occluded window yields 0 (no NaN), and full confidence recovers the plain
normalised convolution.
"""

import pytest
import torch

from signtranslator.hand_graph.temporal import (
    TemporalPyramid, masked_normalized_conv1d, MaskedTemporalConv,
    to_time_series, from_time_series,
)


# ---------------------------------------------------------------------------
# temporal pyramid
# ---------------------------------------------------------------------------
def test_pyramid_preserves_length_and_receptive_field():
    pyr = TemporalPyramid(channels=4, kernel_size=3, dilations=(1, 2, 4)).double()
    x = torch.randn(2, 4, 20, dtype=torch.float64)
    y = pyr(x)
    assert y.shape == x.shape                                # length preserved
    assert pyr.receptive_field() == 1 + (3 - 1) * 4          # = 9


def test_pyramid_receptive_field_matches_autograd_dependency():
    pyr = TemporalPyramid(channels=1, kernel_size=3, dilations=(1, 2, 4),
                          residual=False).double()
    T = 21
    x = torch.zeros(1, 1, T, dtype=torch.float64, requires_grad=True)
    y = pyr(x)
    center = T // 2
    grad = torch.autograd.grad(y[0, 0, center], x)[0][0, 0]  # (T,)
    half = (pyr.receptive_field() - 1) // 2                  # 4
    influenced = (grad.abs() > 0).nonzero().flatten().tolist()
    assert min(influenced) == center - half
    assert max(influenced) == center + half


def test_pyramid_gradients_flow():
    pyr = TemporalPyramid(channels=3, kernel_size=3).double()
    x = torch.randn(1, 3, 15, dtype=torch.float64, requires_grad=True)
    pyr(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in pyr.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        TemporalPyramid(channels=2, kernel_size=4)


# ---------------------------------------------------------------------------
# masked normalized convolution
# ---------------------------------------------------------------------------
def test_masked_conv_excludes_occluded_frames():
    # box kernel over 3 frames; middle frame occluded -> averaged over the 2 visible
    x = torch.tensor([[[1.0, 100.0, 3.0]]], dtype=torch.float64)   # (1,1,3)
    conf = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64)     # frame 1 occluded
    w = torch.ones(3, dtype=torch.float64)
    y = masked_normalized_conv1d(x, w, conf)
    # at t=1 window covers frames 0,1,2 with confidences 1,0,1 -> (1*1 + 0 + 1*3)/(1+0+1)=2
    assert abs(float(y[0, 0, 1]) - 2.0) < 1e-12
    # the huge occluded value (100) never enters
    assert float(y[0, 0, 1]) < 5.0


def test_masked_conv_all_occluded_window_is_zero_no_nan():
    x = torch.randn(1, 2, 5, dtype=torch.float64)
    conf = torch.zeros(1, 5, dtype=torch.float64)                  # everything occluded
    w = torch.ones(3, dtype=torch.float64)
    y = masked_normalized_conv1d(x, w, conf)
    assert torch.isfinite(y).all()
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_masked_conv_full_confidence_matches_plain_normalized_conv():
    x = torch.randn(2, 3, 10, dtype=torch.float64)
    conf = torch.ones(2, 10, dtype=torch.float64)
    w = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    y = masked_normalized_conv1d(x, w, conf)
    # with c=1 everywhere, denominator = conv(1, w) (edge-aware), so y = conv(x,w)/conv(1,w)
    import torch.nn.functional as F
    num = F.conv1d(x, w.view(1, 1, 3).expand(3, 1, 3), padding=1, groups=3)
    den = F.conv1d(torch.ones(2, 1, 10, dtype=torch.float64), w.view(1, 1, 3), padding=1)
    assert torch.allclose(y, num / den, atol=1e-12)


def test_masked_temporal_conv_module_runs_and_grads():
    mod = MaskedTemporalConv(channels=3, kernel_size=5).double()
    x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    conf = torch.rand(2, 12, dtype=torch.float64)
    out = mod(x, conf)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.pow(2).sum().backward()
    assert torch.isfinite(mod.raw.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------
def test_time_series_layout_round_trip():
    x = torch.randn(2, 5, 8, 27)                             # (N, C, T, V)
    ts = to_time_series(x)
    assert ts.shape == (2 * 27, 5, 8)
    back = from_time_series(ts, n=2, v=27)
    assert torch.equal(back, x)
