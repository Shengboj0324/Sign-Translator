"""Mathematical verification of the Gaussian-diffusion process.

These tests check the closed-form identities of DDPM directly, independent of
any learned network, so a regression in the schedule or coefficient algebra is
caught immediately.
"""

import torch
import pytest

from signtranslator.models import GaussianMotionDiffusion, MotionDenoiser, make_beta_schedule


class _IdentityNoise(torch.nn.Module):
    """A trivial denoiser returning zeros, used only for shape/sampling tests."""

    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)


def _diffusion(T=200, schedule="cosine", denoiser=None):
    denoiser = denoiser or _IdentityNoise()
    return GaussianMotionDiffusion(denoiser, num_timesteps=T, schedule=schedule)


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_beta_schedule_valid_range(schedule):
    betas = make_beta_schedule(schedule, 500)
    assert betas.shape == (500,)
    assert (betas > 0).all() and (betas < 1).all()


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alphas_cumprod_monotone_decreasing(schedule):
    d = _diffusion(schedule=schedule)
    abar = d.alphas_cumprod
    # abar_t is a cumulative product of numbers in (0,1) -> strictly decreasing,
    # starts below 1 and ends near 0.
    assert (abar[1:] <= abar[:-1] + 1e-6).all()
    assert abar[0] < 1.0 and abar[-1] < abar[0]
    assert abar[-1] > 0.0


def test_q_sample_matches_analytic_mean_and_variance():
    """Empirical moments of q(x_t|x_0) must match N(sqrt(abar)x0, (1-abar)I)."""
    torch.manual_seed(0)
    d = _diffusion(T=100)
    x0 = torch.full((40000, 1, 4, 3), 2.0)  # constant signal, large batch
    t = torch.full((40000,), 50, dtype=torch.long)
    xt = d.q_sample(x0, t)
    abar = float(d.alphas_cumprod[50])
    exp_mean = (abar ** 0.5) * 2.0
    exp_var = 1.0 - abar
    assert abs(xt.mean().item() - exp_mean) < 1e-2
    assert abs(xt.var(unbiased=False).item() - exp_var) < 1e-2


def test_predict_start_inverts_q_sample_exactly():
    """x0 = (x_t - sqrt(1-abar) eps)/sqrt(abar) must recover the exact x0.

    Uses the linear schedule, where abar stays well away from 0 across all
    timesteps so the inversion is well-conditioned in float32. (Near abar->0
    the inversion is genuinely ill-conditioned -- see docs/MATH.md.)"""
    torch.manual_seed(1)
    d = _diffusion(T=100, schedule="linear")
    x0 = torch.randn(8, 1, 6, 3)
    noise = torch.randn_like(x0)
    for step in (1, 37, 99):
        t = torch.full((8,), step, dtype=torch.long)
        xt = d.q_sample(x0, t, noise=noise)
        recovered = d.predict_start_from_noise(xt, t, noise)
        assert torch.allclose(recovered, x0, atol=1e-4)


def test_posterior_variance_positive_and_matches_formula():
    d = _diffusion(T=100)
    betas, abar = d.betas, d.alphas_cumprod
    abar_prev = d.alphas_cumprod_prev
    manual = betas * (1.0 - abar_prev) / (1.0 - abar)
    assert torch.allclose(d.posterior_variance, manual, atol=1e-6)
    assert (d.posterior_variance[1:] > 0).all()


def test_posterior_mean_coefficients_convexity():
    """The two posterior-mean coefficients must match the DDPM identities.

    The reference is computed in float64 to avoid comparing two different
    float32 rounding paths of the same expression."""
    T = 100
    d = _diffusion(T=T)
    betas = make_beta_schedule("cosine", T).double()
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = torch.nn.functional.pad(abar[:-1], (1, 0), value=1.0)
    c1 = betas * torch.sqrt(abar_prev) / (1.0 - abar)
    c2 = (1.0 - abar_prev) * torch.sqrt(alphas) / (1.0 - abar)
    assert torch.allclose(d.posterior_mean_coef1.double(), c1, atol=1e-5)
    assert torch.allclose(d.posterior_mean_coef2.double(), c2, atol=1e-5)


def test_p_losses_is_finite_scalar():
    denoiser = MotionDenoiser(num_joints=4, in_channels=3, cond_dim=8, hidden_dim=32,
                              num_layers=2, num_heads=2)
    d = _diffusion(T=100, denoiser=denoiser)
    x0 = torch.randn(5, 3, 16, 4)
    cond = torch.randn(5, 8)
    loss = d(x0, cond=cond)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sampling_shapes():
    denoiser = MotionDenoiser(num_joints=4, in_channels=3, cond_dim=8, hidden_dim=32,
                              num_layers=2, num_heads=2)
    d = _diffusion(T=50, denoiser=denoiser)
    cond = torch.randn(2, 8)
    shape = (2, 3, 12, 4)
    ddpm = d.sample(shape, cond=cond)
    ddim = d.ddim_sample(shape, cond=cond, num_steps=10)
    assert ddpm.shape == shape and torch.isfinite(ddpm).all()
    assert ddim.shape == shape and torch.isfinite(ddim).all()
