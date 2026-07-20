"""Verification of the noise schedule and parameterization algebra.

Proves (float64): a²+b²=1, all ε/x₀/v conversions round-trip exactly, the
v-inversions, the loss reweighting identities, the DDPM posterior cross-checked
against the audited GaussianMotionDiffusion, and the forward marginal statistics.
"""

import pytest
import torch

from signtranslator.diffusion_gen.schedule import NoiseSchedule


def _sched(T=1000):
    return NoiseSchedule(num_timesteps=T, schedule="cosine")


def _batch(n=8, d=5, T=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, d, generator=g, dtype=torch.float64)
    eps = torch.randn(n, d, generator=g, dtype=torch.float64)
    t = torch.randint(0, T, (n,), generator=g)
    return x0, eps, t


# ---------------------------------------------------------------------------
# schedule sanity
# ---------------------------------------------------------------------------
def test_alpha_bar_monotone_from_one_to_zero():
    s = _sched()
    ab = s.alpha_bar
    assert ab[0] < 1.0 and ab[0] > 0.9                       # starts near 1
    assert ab[-1] < 0.05                                     # ends near 0
    assert torch.all(ab[1:] <= ab[:-1] + 1e-12)              # monotone non-increasing


def test_a_squared_plus_b_squared_is_one():
    s = _sched()
    assert torch.allclose(s.sqrt_ab ** 2 + s.sqrt_1m_ab ** 2,
                          torch.ones_like(s.sqrt_ab), atol=1e-12)


# ---------------------------------------------------------------------------
# parameterization round-trips
# ---------------------------------------------------------------------------
def test_eps_x0_round_trip():
    s = _sched()
    x0, eps, t = _batch()
    x_t = s.q_sample(x0, t, eps)
    assert torch.allclose(s.x0_from_eps(x_t, t, eps), x0, atol=1e-9)
    assert torch.allclose(s.eps_from_x0(x_t, t, x0), eps, atol=1e-9)


def test_v_inversions_exact():
    s = _sched()
    x0, eps, t = _batch(seed=1)
    x_t = s.q_sample(x0, t, eps)
    v = s.v_from_x0_eps(t, x0, eps)
    # x0 = a x_t - b v ; eps = b x_t + a v
    assert torch.allclose(s.x0_from_v(x_t, t, v), x0, atol=1e-9)
    assert torch.allclose(s.eps_from_v(x_t, t, v), eps, atol=1e-9)


def test_to_x0_dispatch_consistent_across_parameterizations():
    s = _sched()
    x0, eps, t = _batch(seed=2)
    x_t = s.q_sample(x0, t, eps)
    v = s.v_from_x0_eps(t, x0, eps)
    assert torch.allclose(s.to_x0(x_t, t, eps, "eps"), x0, atol=1e-9)
    assert torch.allclose(s.to_x0(x_t, t, v, "v"), x0, atol=1e-9)
    assert torch.allclose(s.to_x0(x_t, t, x0, "x0"), x0, atol=1e-12)


# ---------------------------------------------------------------------------
# loss reweighting identities
# ---------------------------------------------------------------------------
def test_eps_loss_is_snr_weighted_x0_loss():
    s = _sched()
    x0, eps, t = _batch(seed=3)
    x_t = s.q_sample(x0, t, eps)
    x0_pred = x0 + 0.1 * torch.randn_like(x0)                # a wrong x0
    eps_pred = s.eps_from_x0(x_t, t, x0_pred)
    lhs = ((eps - eps_pred) ** 2).sum(-1)
    rhs = s.snr(t) * ((x0 - x0_pred) ** 2).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-7)


def test_v_loss_reweights_x0_loss():
    s = _sched()
    x0, eps, t = _batch(seed=4)
    x_t = s.q_sample(x0, t, eps)
    x0_pred = x0 + 0.1 * torch.randn_like(x0)
    v = s.v_from_x0_eps(t, x0, eps)
    eps_pred = s.eps_from_x0(x_t, t, x0_pred)
    v_pred = s.v_from_x0_eps(t, x0_pred, eps_pred)
    ab = s.alpha_bar[t]
    lhs = ((v - v_pred) ** 2).sum(-1)
    rhs = (1.0 / (1.0 - ab)) * ((x0 - x0_pred) ** 2).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-7)


# ---------------------------------------------------------------------------
# posterior cross-check against the audited core
# ---------------------------------------------------------------------------
def test_posterior_matches_gaussian_motion_diffusion():
    import torch.nn as nn
    from signtranslator.models.diffusion import GaussianMotionDiffusion
    s = _sched()
    gmd = GaussianMotionDiffusion(denoiser=nn.Identity(), num_timesteps=1000,
                                  schedule="cosine")
    x0, eps, t = _batch(seed=5)
    x_t = s.q_sample(x0, t, eps)
    mean, var = s.posterior_mean_variance(x0, x_t, t)
    gmean, gvar, _ = gmd.q_posterior_mean_variance(x0, x_t, t)
    assert torch.allclose(mean, gmean.to(mean.dtype), atol=1e-8)
    assert torch.allclose(var, gvar.to(var.dtype), atol=1e-8)


# ---------------------------------------------------------------------------
# forward marginal statistics
# ---------------------------------------------------------------------------
def test_forward_marginal_has_correct_mean_and_variance():
    s = _sched()
    torch.manual_seed(6)
    x0 = torch.ones(20000, 1, dtype=torch.float64) * 2.0
    t = torch.full((20000,), 500)
    eps = torch.randn(20000, 1, dtype=torch.float64)
    x_t = s.q_sample(x0, t, eps)
    a = float(s.sqrt_ab[500]); b = float(s.sqrt_1m_ab[500])
    assert abs(float(x_t.mean()) - a * 2.0) < 0.05           # mean = a x0
    assert abs(float(x_t.var()) - b ** 2) < 0.05             # var = 1 - abar
