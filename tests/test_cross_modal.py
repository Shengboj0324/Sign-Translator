"""Tests for cross-modal attention denoiser + classifier-free guidance."""

import torch

from signtranslator.models.denoiser import CrossModalDenoiser
from signtranslator.models.guided_diffusion import GuidedMotionDiffusion
from signtranslator.models.encoders import StubTextEncoder


def _denoiser(context_dim=32):
    return CrossModalDenoiser(num_joints=6, in_channels=3, context_dim=context_dim,
                              hidden_dim=32, num_layers=2, num_heads=2)


def _context(n=4, L=5, dim=32):
    memory = torch.randn(n, L, dim)
    mask = torch.ones(n, L, dtype=torch.bool)
    return memory, mask


def test_cross_modal_output_shape():
    net = _denoiser()
    x = torch.randn(4, 3, 16, 6)
    t = torch.randint(0, 100, (4,))
    out = net(x, t, cond=_context())
    assert out.shape == x.shape


def test_unconditional_path_runs_without_context():
    net = _denoiser()
    x = torch.randn(2, 3, 12, 6)
    t = torch.randint(0, 100, (2,))
    assert net(x, t, cond=None).shape == x.shape


def test_zero_init_output():
    net = _denoiser()
    x = torch.randn(2, 3, 12, 6)
    t = torch.randint(0, 100, (2,))
    out = net(x, t, cond=_context(n=2))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_drop_yields_same_output_as_no_context():
    """A fully-dropped sample must produce exactly the unconditional prediction,
    since it can only attend to the null token."""
    net = _denoiser().eval()
    # Move output projection off zero so predictions are non-trivial.
    torch.nn.init.normal_(net.output_proj.weight, std=0.02)
    x = torch.randn(3, 3, 10, 6)
    t = torch.randint(0, 100, (3,))
    mem, mask = _context(n=3)
    drop_all = torch.ones(3, dtype=torch.bool)
    dropped = net(x, t, cond=(mem, mask), drop=drop_all)
    uncond = net(x, t, cond=None)
    assert torch.allclose(dropped, uncond, atol=1e-5)


def test_conditioning_changes_output():
    net = _denoiser().eval()
    torch.nn.init.normal_(net.output_proj.weight, std=0.05)
    x = torch.randn(3, 3, 10, 6)
    t = torch.randint(0, 100, (3,))
    cond_out = net(x, t, cond=_context(n=3))
    uncond_out = net(x, t, cond=None)
    assert not torch.allclose(cond_out, uncond_out, atol=1e-4)


def test_guided_diffusion_training_loss_finite():
    net = _denoiser()
    diff = GuidedMotionDiffusion(net, num_timesteps=50, cond_drop_prob=0.2)
    x0 = torch.randn(4, 3, 16, 6)
    loss = diff(x0, cond=_context())
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_guidance_scale_one_equals_plain_conditional():
    net = _denoiser().eval()
    torch.nn.init.normal_(net.output_proj.weight, std=0.02)
    diff = GuidedMotionDiffusion(net, num_timesteps=50)
    x = torch.randn(2, 3, 10, 6)
    # Low t: abar is close to 1, so the x0 estimate is well-scaled and the
    # sampling-stability clamp does not engage, making the eps round-trip exact.
    t = torch.full((2,), 5, dtype=torch.long)
    cond = _context(n=2)
    plain = net(x, t, cond=cond)
    guided = diff._guided_eps(x, t, cond, guidance_scale=1.0)
    assert torch.allclose(plain, guided, atol=1e-4)


def test_guidance_extrapolates_between_cond_and_uncond():
    net = _denoiser().eval()
    torch.nn.init.normal_(net.output_proj.weight, std=0.05)
    diff = GuidedMotionDiffusion(net, num_timesteps=50)
    x = torch.randn(2, 3, 10, 6)
    t = torch.full((2,), 5, dtype=torch.long)   # low noise: clamp does not engage
    cond = _context(n=2)
    w = 3.0
    eps_c = net(x, t, cond=cond)
    eps_u = net(x, t, cond=None)
    expected = eps_u + w * (eps_c - eps_u)
    got = diff._guided_eps(x, t, cond, guidance_scale=w)
    assert torch.allclose(got, expected, atol=1e-3)


def test_guided_sampling_shapes():
    net = _denoiser()
    diff = GuidedMotionDiffusion(net, num_timesteps=40)
    cond = _context(n=2)
    shape = (2, 3, 12, 6)
    out = diff.ddim_sample(shape, cond=cond, num_steps=8, guidance_scale=2.0)
    assert out.shape == shape and torch.isfinite(out).all()


def test_cross_attention_uses_text_encoder_memory():
    """Integration: real text-encoder sequence output feeds the denoiser."""
    enc = StubTextEncoder(vocab_size=64, embed_dim=32, num_layers=2, num_heads=2)
    tokens = torch.randint(1, 64, (3, 7))
    memory, mask = enc.encode_sequence(tokens)
    assert memory.shape == (3, 7, 32)
    net = CrossModalDenoiser(num_joints=6, in_channels=3, context_dim=32,
                             hidden_dim=32, num_layers=2, num_heads=2)
    x = torch.randn(3, 3, 10, 6)
    t = torch.randint(0, 50, (3,))
    assert net(x, t, cond=(memory, mask)).shape == x.shape
