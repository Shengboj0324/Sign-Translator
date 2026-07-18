"""Tests for x0/eps parameterization, velocity loss, and guidance equivalence."""

import torch
import pytest

from signtranslator.models import MotionDenoiser, CrossModalDenoiser
from signtranslator.models.diffusion import GaussianMotionDiffusion
from signtranslator.models.guided_diffusion import GuidedMotionDiffusion


def _denoiser():
    return MotionDenoiser(num_joints=4, in_channels=3, cond_dim=8, hidden_dim=32,
                          num_layers=2, num_heads=2)


def _cross():
    return CrossModalDenoiser(num_joints=4, in_channels=3, context_dim=8,
                              hidden_dim=32, num_layers=2, num_heads=2)


def _ctx(n=3, L=5, d=8):
    return torch.randn(n, L, d), torch.ones(n, L, dtype=torch.bool)


def test_predict_noise_from_start_inverts_predict_start():
    d = GaussianMotionDiffusion(_denoiser(), num_timesteps=100, schedule="linear")
    x0 = torch.randn(4, 3, 8, 4)
    noise = torch.randn_like(x0)
    for step in (1, 40, 99):
        t = torch.full((4,), step, dtype=torch.long)
        x_t = d.q_sample(x0, t, noise=noise)
        eps = d.predict_noise_from_start(x_t, t, x0)
        assert torch.allclose(eps, noise, atol=1e-4)
        back = d.predict_start_from_noise(x_t, t, eps)
        assert torch.allclose(back, x0, atol=1e-4)


@pytest.mark.parametrize("param", ["eps", "x0"])
def test_model_predictions_are_mutually_consistent(param):
    """Whatever the network predicts, the returned (eps, x0) pair must satisfy
    the forward-process relation x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
    torch.manual_seed(0)
    net = _denoiser()
    torch.nn.init.normal_(net.output_proj.weight, std=0.02)  # non-trivial output
    d = GaussianMotionDiffusion(net, num_timesteps=100, schedule="linear",
                                parameterization=param)
    x_t = torch.randn(3, 3, 8, 4)
    t = torch.full((3,), 30, dtype=torch.long)
    cond = torch.randn(3, 8)
    eps, x0 = d.model_predictions(x_t, t, cond)
    recon = d.q_sample(x0, t, noise=eps)
    assert torch.allclose(recon, x_t, atol=1e-3)


def test_x0_parameterization_loss_is_reconstruction_error():
    """With x0 prediction and zero-init output the loss equals E||x0||^2."""
    net = _denoiser()  # output_proj zero-init => predicts x0 = 0
    d = GaussianMotionDiffusion(net, num_timesteps=50, parameterization="x0",
                                velocity_weight=0.0)
    x0 = torch.randn(4, 3, 8, 4)
    t = torch.full((4,), 10, dtype=torch.long)
    loss = d.p_losses(x0, t, cond=torch.randn(4, 8))
    assert torch.allclose(loss, x0.pow(2).mean(), atol=1e-5)


def test_velocity_term_increases_loss_and_is_correct():
    net = _denoiser()
    base = GaussianMotionDiffusion(net, num_timesteps=50, parameterization="x0",
                                   velocity_weight=0.0)
    withv = GaussianMotionDiffusion(net, num_timesteps=50, parameterization="x0",
                                    velocity_weight=1.0)
    x0 = torch.randn(4, 3, 8, 4)
    t = torch.full((4,), 10, dtype=torch.long)
    noise = torch.randn_like(x0)
    l0 = base.p_losses(x0, t, cond=torch.randn(4, 8), noise=noise)
    l1 = withv.p_losses(x0, t, cond=torch.randn(4, 8), noise=noise)
    # Zero-init predicts 0, so velocity term = E||d x0||^2 exactly.
    vel = (x0[:, :, 1:] - x0[:, :, :-1]).pow(2).mean()
    assert torch.allclose(l1 - l0, vel, atol=1e-5)


def test_velocity_helper_matches_first_difference():
    x = torch.randn(2, 3, 6, 4)
    v = GaussianMotionDiffusion._velocity(x)
    assert v.shape == (2, 3, 5, 4)
    assert torch.allclose(v[:, :, 0], x[:, :, 1] - x[:, :, 0], atol=1e-6)


def test_guidance_in_x0_space_matches_eps_space():
    """CFG combined on x0 must equal CFG combined on eps (affine equivalence)."""
    torch.manual_seed(0)
    net = _cross()
    torch.nn.init.normal_(net.output_proj.weight, std=0.05)
    d = GuidedMotionDiffusion(net, num_timesteps=60, parameterization="x0")
    d.eval()  # deterministic: dropout would make the two passes incomparable
    x = torch.randn(3, 3, 8, 4)
    t = torch.full((3,), 25, dtype=torch.long)
    cond = _ctx(n=3)
    w = 2.5

    all_drop = torch.ones(3, dtype=torch.bool)
    eps_c, _ = d.model_predictions(x, t, cond, drop=None)
    eps_u, _ = d.model_predictions(x, t, cond, drop=all_drop)
    eps_expected = eps_u + w * (eps_c - eps_u)

    eps_got, _ = d._guided_predictions(x, t, cond, guidance_scale=w)
    assert torch.allclose(eps_got, eps_expected, atol=1e-3)


def test_guided_sampling_runs_under_x0_parameterization():
    d = GuidedMotionDiffusion(_cross(), num_timesteps=40, parameterization="x0",
                              velocity_weight=1.0)
    out = d.ddim_sample((2, 3, 10, 4), cond=_ctx(n=2), num_steps=6, guidance_scale=2.0)
    assert out.shape == (2, 3, 10, 4) and torch.isfinite(out).all()
    ddpm = d.sample((2, 3, 10, 4), cond=_ctx(n=2), guidance_scale=2.0)
    assert ddpm.shape == (2, 3, 10, 4) and torch.isfinite(ddpm).all()


def test_invalid_parameterization_rejected():
    with pytest.raises(ValueError):
        GaussianMotionDiffusion(_denoiser(), parameterization="v")
