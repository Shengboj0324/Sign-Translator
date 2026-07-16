"""Tests for the Transformer motion denoiser."""

import torch

from signtranslator.models import MotionDenoiser
from signtranslator.models.denoiser import timestep_embedding


def test_output_shape_matches_input():
    net = MotionDenoiser(num_joints=6, in_channels=3, cond_dim=16, hidden_dim=32,
                         num_layers=2, num_heads=2)
    x = torch.randn(4, 3, 20, 6)
    t = torch.randint(0, 1000, (4,))
    cond = torch.randn(4, 16)
    out = net(x, t, cond)
    assert out.shape == x.shape


def test_zero_initialised_output():
    """Zero-init output projection => the denoiser predicts exactly zero noise
    at initialisation, a standard diffusion-training stabiliser."""
    net = MotionDenoiser(num_joints=6, in_channels=3, cond_dim=16, hidden_dim=32,
                         num_layers=2, num_heads=2)
    x = torch.randn(2, 3, 10, 6)
    t = torch.randint(0, 1000, (2,))
    out = net(x, t, torch.randn(2, 16))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_conditioning_is_optional():
    net = MotionDenoiser(num_joints=6, in_channels=3, cond_dim=16, hidden_dim=32,
                         num_layers=2, num_heads=2)
    x = torch.randn(2, 3, 10, 6)
    t = torch.randint(0, 1000, (2,))
    assert net(x, t, None).shape == x.shape


def test_timestep_embedding_shape_and_finiteness():
    emb = timestep_embedding(torch.arange(16), dim=32)
    assert emb.shape == (16, 32)
    assert torch.isfinite(emb).all()
    # t=0 -> [cos(0)=1, ..., sin(0)=0, ...]
    assert torch.allclose(emb[0, :16], torch.ones(16))
