"""Verification of classifier-free guidance.

Proves the guidance formula and w=0 recovery, parameterization-equivariance
(ε-space CFG maps to x₀-space CFG), condition dropout, the guidance-annealing
schedule, and the diversity-reduction property (guided Gaussian variance shrinks
as w grows), demonstrated on real guided scores using the module's own function.
"""

import pytest
import torch

from signtranslator.diffusion_gen.schedule import NoiseSchedule
from signtranslator.diffusion_gen.guidance import (
    drop_condition_mask, apply_condition_dropout, classifier_free_guidance,
    guidance_weight_schedule,
)


# ---------------------------------------------------------------------------
# formula
# ---------------------------------------------------------------------------
def test_guidance_formula_and_w0_recovery():
    ec = torch.randn(4, 5, dtype=torch.float64)
    eu = torch.randn(4, 5, dtype=torch.float64)
    assert torch.allclose(classifier_free_guidance(ec, eu, 0.0), ec, atol=1e-12)
    w = 2.5
    assert torch.allclose(classifier_free_guidance(ec, eu, w),
                          (1 + w) * ec - w * eu, atol=1e-12)


# ---------------------------------------------------------------------------
# parameterization equivariance
# ---------------------------------------------------------------------------
def test_cfg_is_parameterization_equivariant():
    s = NoiseSchedule()
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(6, 4, generator=g, dtype=torch.float64)
    eps = torch.randn(6, 4, generator=g, dtype=torch.float64)
    t = torch.randint(0, 1000, (6,), generator=g)
    x_t = s.q_sample(x0, t, eps)
    eps_c = eps + 0.2 * torch.randn(6, 4, generator=g, dtype=torch.float64)
    eps_u = eps + 0.2 * torch.randn(6, 4, generator=g, dtype=torch.float64)
    w = 3.0
    # guide in eps space, then convert to x0
    x0_via_eps = s.x0_from_eps(x_t, t, classifier_free_guidance(eps_c, eps_u, w))
    # guide directly in x0 space
    x0_c = s.x0_from_eps(x_t, t, eps_c); x0_u = s.x0_from_eps(x_t, t, eps_u)
    x0_direct = classifier_free_guidance(x0_c, x0_u, w)
    assert torch.allclose(x0_via_eps, x0_direct, atol=1e-9)   # equivariant


# ---------------------------------------------------------------------------
# condition dropout
# ---------------------------------------------------------------------------
def test_condition_dropout_replaces_dropped_with_null():
    cond = torch.randn(5, 8, dtype=torch.float64)
    null = torch.zeros(8, dtype=torch.float64)
    mask = torch.tensor([True, False, True, False, False])
    out = apply_condition_dropout(cond, null, mask)
    assert torch.allclose(out[0], null) and torch.allclose(out[2], null)
    assert torch.allclose(out[1], cond[1]) and torch.allclose(out[4], cond[4])


def test_drop_mask_probability():
    g = torch.Generator().manual_seed(0)
    m = drop_condition_mask(20000, p_uncond=0.2, generator=g)
    assert abs(float(m.float().mean()) - 0.2) < 0.02


# ---------------------------------------------------------------------------
# guidance annealing (innovation)
# ---------------------------------------------------------------------------
def test_guidance_annealing_high_early_low_late():
    t = torch.tensor([0, 500, 999])
    w = guidance_weight_schedule(t, num_timesteps=1000, w_high=4.0, w_low=0.5)
    assert abs(float(w[0]) - 0.5) < 1e-6                      # t=0 (clean) -> low
    assert abs(float(w[2]) - 4.0) < 1e-2                      # t=T (noisy) -> high
    assert w[0] < w[1] < w[2]                                 # monotone in t


# ---------------------------------------------------------------------------
# diversity reduction (the CFG property, on real guided scores)
# ---------------------------------------------------------------------------
def test_guidance_shrinks_guided_gaussian_variance():
    """A sharper conditional guided against a broad unconditional yields a guided
    distribution whose variance shrinks as w grows -- the diversity/multimodality
    trade-off the document warns about."""
    xs = torch.linspace(-3.0, 4.0, 2000, dtype=torch.float64)
    mu_c, sc, mu_0, s0 = 1.0, 0.3, 0.0, 1.0
    score_c = -(xs - mu_c) / sc ** 2
    score_0 = -(xs - mu_0) / s0 ** 2
    variances = []
    for w in (0.0, 2.0, 4.0):
        s_hat = classifier_free_guidance(score_c, score_0, w)    # guided score (linear in xs)
        slope = (s_hat[1] - s_hat[0]) / (xs[1] - xs[0])          # = -1/var_hat
        variances.append(float(-1.0 / slope))
    assert variances[0] > variances[1] > variances[2]           # variance shrinks
    assert abs(variances[0] - sc ** 2) < 1e-6                    # w=0 -> conditional var
