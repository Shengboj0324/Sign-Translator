"""Noise schedule and the ε/x₀/v parameterization algebra (docs/DIFFUSION_GEN.md §1-2).

Reuses the audited cosine ``make_beta_schedule`` from ``models/diffusion.py`` and
adds the full parameterization triangle with proofs:

    x_t = a x₀ + b ε,     a = √ᾱ_t,  b = √(1−ᾱ_t),  a² + b² = 1,
    v   = a ε − b x₀,     x₀ = a x_t − b v,   ε = b x_t + a v.

All six conversions round-trip exactly; the three losses are reweightings of one
another (``‖ε−ε_θ‖² = SNR·‖Δx₀‖²``, ``‖v−v_θ‖² = (1/(1−ᾱ))·‖Δx₀‖²``).
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..models.diffusion import make_beta_schedule


def _gather(coef: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Index ``coef`` by timestep ``t`` and broadcast to ``shape``."""
    out = coef.to(t.device)[t].to(torch.get_default_dtype())
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class NoiseSchedule:
    """Holds ᾱ_t and the derived coefficients; provides the parameterization maps."""

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine") -> None:
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(schedule, num_timesteps)          # float64 (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)                     # (T,)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=alpha_bar.dtype),
                                    alpha_bar[:-1]])
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_ab = torch.sqrt(alpha_bar)                        # a
        self.sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar)              # b
        # posterior coefficients (DDPM)
        self.posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.post_coef_x0 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        self.post_coef_xt = ((1.0 - alpha_bar_prev) * torch.sqrt(alphas)
                             / (1.0 - alpha_bar))

    # -- coefficients -------------------------------------------------------
    def a(self, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return _gather(self.sqrt_ab, t, shape)

    def b(self, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return _gather(self.sqrt_1m_ab, t, shape)

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar.to(t.device)[t]
        return ab / (1.0 - ab)

    # -- forward ------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        return self.a(t, x0.shape) * x0 + self.b(t, x0.shape) * eps

    # -- parameterization conversions --------------------------------------
    def x0_from_eps(self, x_t, t, eps):
        a, b = self.a(t, x_t.shape), self.b(t, x_t.shape)
        return (x_t - b * eps) / a

    def eps_from_x0(self, x_t, t, x0):
        a, b = self.a(t, x_t.shape), self.b(t, x_t.shape)
        return (x_t - a * x0) / b

    def v_from_x0_eps(self, t, x0, eps):
        a, b = self.a(t, x0.shape), self.b(t, x0.shape)
        return a * eps - b * x0

    def x0_from_v(self, x_t, t, v):
        a, b = self.a(t, x_t.shape), self.b(t, x_t.shape)
        return a * x_t - b * v

    def eps_from_v(self, x_t, t, v):
        a, b = self.a(t, x_t.shape), self.b(t, x_t.shape)
        return b * x_t + a * v

    def to_x0(self, x_t, t, pred, param: str):
        """Convert a model prediction (``param`` in {eps, x0, v}) to x₀."""
        if param == "x0":
            return pred
        if param == "eps":
            return self.x0_from_eps(x_t, t, pred)
        if param == "v":
            return self.x0_from_v(x_t, t, pred)
        raise ValueError(f"unknown parameterization {param}")

    def target_for(self, x0, t, eps, param: str):
        """The regression target for a given parameterization."""
        if param == "x0":
            return x0
        if param == "eps":
            return eps
        if param == "v":
            return self.v_from_x0_eps(t, x0, eps)
        raise ValueError(f"unknown parameterization {param}")

    # -- posterior ----------------------------------------------------------
    def posterior_mean_variance(self, x0: torch.Tensor, x_t: torch.Tensor,
                                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = (_gather(self.post_coef_x0, t, x_t.shape) * x0
                + _gather(self.post_coef_xt, t, x_t.shape) * x_t)
        var = _gather(self.posterior_var, t, x_t.shape)
        return mean, var
