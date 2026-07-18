"""Classifier-free guidance (CFG) for cross-modal motion diffusion.

Extends :class:`GaussianMotionDiffusion` with

  * **condition dropout** during training: each sample's conditioning is replaced
    by the learned null context with probability ``cond_drop_prob``, so the same
    network learns both the conditional ``ε_θ(x_t, t, c)`` and the unconditional
    ``ε_θ(x_t, t, ∅)`` noise predictors (Ho & Salimans, 2022).

  * **guided sampling**: predictions are extrapolated

        ε̂ = ε_uncond + w · (ε_cond − ε_uncond),

    where ``w = guidance_scale``. ``w = 1`` recovers the plain conditional model;
    ``w > 1`` sharpens adherence to the conditioning at some cost to diversity.

This requires a denoiser that accepts a ``drop`` argument selecting per-sample
unconditional context (e.g. :class:`CrossModalDenoiser`).
"""

from __future__ import annotations

from typing import Optional

import torch

from .diffusion import GaussianMotionDiffusion, _extract


class GuidedMotionDiffusion(GaussianMotionDiffusion):
    def __init__(self, denoiser, cond_drop_prob: float = 0.1, **kwargs) -> None:
        super().__init__(denoiser, **kwargs)
        if not 0.0 <= cond_drop_prob < 1.0:
            raise ValueError("cond_drop_prob must be in [0, 1)")
        self.cond_drop_prob = cond_drop_prob

    # -- training with condition dropout -----------------------------------
    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, cond=None,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        drop = None
        if cond is not None and self.cond_drop_prob > 0:
            drop = torch.rand(x_start.shape[0], device=x_start.device) < self.cond_drop_prob
        # Delegate to the base loss so eps/x0 parameterization and the velocity
        # term are handled in exactly one place.
        return super().p_losses(x_start, t, cond=cond, noise=noise, drop=drop)

    # -- guided prediction --------------------------------------------------
    def _guided_predictions(self, x: torch.Tensor, t: torch.Tensor, cond,
                            guidance_scale: float):
        """Return CFG-combined ``(eps, x_start)``.

        Guidance may be applied in either space: for fixed ``(x_t, t)``, x_0 and
        eps are affine functions of one another, so
        ``x0_u + w (x0_c - x0_u)`` corresponds exactly to
        ``eps_u + w (eps_c - eps_u)``. We combine x_0 and derive eps from it.
        """
        if cond is None or guidance_scale == 1.0:
            return self.model_predictions(x, t, cond)
        all_drop = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        _, x0_cond = self.model_predictions(x, t, cond, drop=None)
        _, x0_uncond = self.model_predictions(x, t, cond, drop=all_drop)
        x_start = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        x_start = x_start.clamp(-10.0, 10.0)
        return self.predict_noise_from_start(x, t, x_start), x_start

    def _guided_eps(self, x: torch.Tensor, t: torch.Tensor, cond,
                    guidance_scale: float) -> torch.Tensor:
        """Backwards-compatible accessor returning only the guided noise."""
        return self._guided_predictions(x, t, cond, guidance_scale)[0]

    # -- guided sampling ----------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, cond=None,
                 guidance_scale: float = 1.0) -> torch.Tensor:
        _, x_start = self._guided_predictions(x_t, t, cond, guidance_scale)
        mean, _, log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, shape, cond=None, guidance_scale: float = 1.0, device=None):
        device = device or self.betas.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond=cond, guidance_scale=guidance_scale)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, num_steps: int = 50, eta: float = 0.0,
                    guidance_scale: float = 1.0, device=None):
        device = device or self.betas.device
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps,
                                      device=device).round().long().flip(0)
        x = torch.randn(shape, device=device)
        for k, i in enumerate(step_indices):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            eps_hat, x_start = self._guided_predictions(x, t, cond, guidance_scale)
            abar_t = _extract(self.alphas_cumprod, t, x.shape)
            if k < len(step_indices) - 1:
                t_next = torch.full((shape[0],), int(step_indices[k + 1]),
                                    device=device, dtype=torch.long)
                abar_next = _extract(self.alphas_cumprod, t_next, x.shape)
            else:
                abar_next = torch.ones_like(abar_t)
            sigma = eta * torch.sqrt(
                (1 - abar_next) / (1 - abar_t) * (1 - abar_t / abar_next)
            )
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = (abar_next.sqrt() * x_start
                 + torch.sqrt((1 - abar_next - sigma ** 2).clamp(min=0.0)) * eps_hat
                 + sigma * noise)
        return x
