"""Verification of the temporal DiT denoiser (adaLN-Zero + cross-attention).

Proves the adaLN-Zero identity at init (each block is identity, the whole model
outputs 0), and — after activating the zero-initialised modulation — dependence on
the timestep, the conditioning vector, and the cross-attention condition tokens.
"""

import pytest
import torch

from signtranslator.diffusion_gen.dit import DiTBlock, TemporalDiT, modulate


def _activate(model, scale=0.05, seed=0):
    """Give the zero-init modulation MLPs small random weights so gates != 0."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if m.weight.abs().sum() == 0:                # a zero-init layer
                    m.weight.copy_(scale * torch.randn(m.weight.shape, generator=g))
                    m.bias.copy_(scale * torch.randn(m.bias.shape, generator=g))


# ---------------------------------------------------------------------------
# adaLN-Zero identity at init
# ---------------------------------------------------------------------------
def test_dit_block_is_identity_at_init():
    torch.manual_seed(0)
    blk = DiTBlock(dim=16, cond_dim=16, num_heads=4, cross=True)
    x = torch.randn(2, 6, 16)
    c = torch.randn(2, 16)
    cond_tokens = torch.randn(2, 3, 16)
    out = blk(x, c, cond_tokens)
    assert torch.allclose(out, x, atol=1e-6)                 # zero gates -> identity


def test_dit_outputs_zero_at_init():
    torch.manual_seed(1)
    dit = TemporalDiT(in_dim=12, dim=32, depth=3, num_heads=4)
    x = torch.randn(2, 8, 12)
    t = torch.randint(0, 1000, (2,))
    out = dit(x, t)
    assert out.shape == (2, 8, 12)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)   # zero-init head


# ---------------------------------------------------------------------------
# conditioning dependence (after activation)
# ---------------------------------------------------------------------------
def test_output_depends_on_timestep():
    torch.manual_seed(2)
    dit = TemporalDiT(in_dim=12, dim=32, depth=2, num_heads=4)
    _activate(dit)
    x = torch.randn(1, 8, 12)
    out_a = dit(x, torch.tensor([10]))
    out_b = dit(x, torch.tensor([900]))
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_output_depends_on_condition_vector():
    torch.manual_seed(3)
    dit = TemporalDiT(in_dim=12, dim=32, depth=2, num_heads=4, cond_dim=32)
    _activate(dit)
    x = torch.randn(1, 8, 12)
    t = torch.tensor([100])
    out_a = dit(x, t, cond_vec=torch.randn(1, 32))
    out_b = dit(x, t, cond_vec=torch.randn(1, 32))
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_output_depends_on_cross_attention_tokens():
    torch.manual_seed(4)
    dit = TemporalDiT(in_dim=12, dim=32, depth=2, num_heads=4, cross=True)
    _activate(dit)
    x = torch.randn(1, 8, 12)
    t = torch.tensor([100])
    out_a = dit(x, t, cond_tokens=torch.randn(1, 4, 32))
    out_b = dit(x, t, cond_tokens=torch.randn(1, 4, 32))
    assert not torch.allclose(out_a, out_b, atol=1e-5)


# ---------------------------------------------------------------------------
# gradient flow
# ---------------------------------------------------------------------------
def test_gradients_flow_through_dit():
    torch.manual_seed(5)
    dit = TemporalDiT(in_dim=12, dim=32, depth=2, num_heads=4)
    _activate(dit)
    x = torch.randn(2, 8, 12, requires_grad=True)
    out = dit(x, torch.randint(0, 1000, (2,)), cond_tokens=torch.randn(2, 3, 32))
    out.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    n = sum(1 for p in dit.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n > 0


def test_modulate_formula():
    x = torch.ones(1, 3, 4)
    shift = torch.full((1, 4), 2.0)
    scale = torch.full((1, 4), 0.5)
    out = modulate(x, shift, scale)
    assert torch.allclose(out, torch.full((1, 3, 4), 1.0 * 1.5 + 2.0))
