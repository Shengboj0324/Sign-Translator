"""Gaussian diffusion for conditional 3D motion generation (DDPM / DDIM).

We follow Ho et al. (2020) "Denoising Diffusion Probabilistic Models" and
Nichol & Dhariwal (2021) for the cosine schedule.

Forward process (fixed):
    q(x_t | x_0) = N(x_t; sqrt(abar_t) x_0, (1 - abar_t) I),
    x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I).

Reverse process (learned): a network eps_theta predicts eps. The exact
posterior used both for the training target and the sampler is
    q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_tilde_t(x_t, x_0), beta_tilde_t I),
    beta_tilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t,
    mu_tilde_t   = (sqrt(abar_{t-1}) beta_t)/(1 - abar_t) x_0
                 + (sqrt(alpha_t)(1 - abar_{t-1}))/(1 - abar_t) x_t.

Training minimises the simplified objective E|| eps - eps_theta(x_t, t, c) ||^2.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule: str, num_timesteps: int,
                       beta_start: float = 1e-4, beta_end: float = 2e-2,
                       cosine_s: float = 0.008) -> torch.Tensor:
    """Return a 1-D tensor of betas of length ``num_timesteps``."""
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    if schedule == "cosine":
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
        alpha_bar = torch.cos((t + cosine_s) / (1 + cosine_s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule: {schedule}")


def _extract(coef: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gather per-sample coefficients coef[t] and broadcast to ``shape``."""
    out = coef.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianMotionDiffusion(nn.Module):
    """Wraps a denoiser with the forward/reverse Gaussian-diffusion math."""

    def __init__(self, denoiser: nn.Module, num_timesteps: int = 1000,
                 schedule: str = "cosine", beta_start: float = 1e-4,
                 beta_end: float = 2e-2, parameterization: str = "eps",
                 velocity_weight: float = 0.0, high_t_frac: float = 0.0,
                 high_t_start: float = 0.7) -> None:
        super().__init__()
        if parameterization not in {"eps", "x0"}:
            raise ValueError("parameterization must be 'eps' or 'x0'")
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        # "eps": network predicts the noise (Ho et al.).
        # "x0" : network predicts the clean signal directly. For *motion*
        #        diffusion this is markedly higher-fidelity (cf. MDM), because
        #        capacity is spent on the signal rather than on noise at high t.
        self.parameterization = parameterization
        self.velocity_weight = velocity_weight
        # Uniform timestep sampling spends most gradient on *easy* low-noise
        # denoising, where x_t already reveals the signal. Sampling, however,
        # starts at t=T where the model must synthesise from the conditioning
        # alone. ``high_t_frac`` re-balances training toward that high-noise
        # regime (draw t from the top ``1-high_t_start`` fraction of the
        # schedule with probability ``high_t_frac``).
        if not 0.0 <= high_t_frac <= 1.0:
            raise ValueError("high_t_frac must be in [0, 1]")
        if not 0.0 <= high_t_start < 1.0:
            raise ValueError("high_t_start must be in [0, 1)")
        self.high_t_frac = high_t_frac
        self.high_t_start = high_t_start

        betas = make_beta_schedule(schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Posterior variance beta_tilde_t (clamped so log is finite at t=0).
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        def reg(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.to(torch.float32))

        reg("betas", betas)
        reg("alphas_cumprod", alphas_cumprod)
        reg("alphas_cumprod_prev", alphas_cumprod_prev)
        reg("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        reg("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        reg("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        reg("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        reg("posterior_variance", posterior_variance)
        reg("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))
        reg("posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        reg("posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    # -- forward process ----------------------------------------------------
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (_extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:
        """Recover x_0 estimate from x_t and predicted noise."""
        return (_extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor,
                                 x_start: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`predict_start_from_noise`:
        eps = (x_t - sqrt(abar) x_0) / sqrt(1 - abar)."""
        return ((x_t - _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start)
                / _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape))

    def model_predictions(self, x_t: torch.Tensor, t: torch.Tensor, cond=None,
                          clip_x_start: bool = True, **denoiser_kwargs):
        """Run the denoiser and return ``(eps, x_start)`` for either parameterization.

        Centralising this conversion means every sampler and loss works
        identically whichever quantity the network predicts.
        """
        out = self.denoiser(x_t, t, cond, **denoiser_kwargs)
        if self.parameterization == "eps":
            eps = out
            x_start = self.predict_start_from_noise(x_t, t, eps)
            if clip_x_start:
                x_start = x_start.clamp(-10.0, 10.0)
                eps = self.predict_noise_from_start(x_t, t, x_start)
        else:  # "x0"
            x_start = out.clamp(-10.0, 10.0) if clip_x_start else out
            eps = self.predict_noise_from_start(x_t, t, x_start)
        return eps, x_start

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor,
                                  t: torch.Tensor):
        """Return (mean, variance, log_variance) of q(x_{t-1} | x_t, x_0)."""
        mean = (_extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance, t, x_t.shape)
        return mean, var, log_var

    # -- training loss ------------------------------------------------------
    @staticmethod
    def _velocity(x: torch.Tensor) -> torch.Tensor:
        """First temporal difference along the frame axis of (N, C, T, V)."""
        return x[:, :, 1:] - x[:, :, :-1]

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor,
                 cond: Optional[torch.Tensor] = None,
                 noise: Optional[torch.Tensor] = None,
                 **denoiser_kwargs) -> torch.Tensor:
        """Training loss.

        ``eps``: the simplified DDPM objective E|| eps - eps_theta ||^2.
        ``x0`` : E|| x_0 - x_theta ||^2 plus an optional **velocity** term
        E|| dx_0 - dx_theta ||^2 on the first temporal difference. The velocity
        term supervises temporal structure directly, which matters when a
        downstream sequence model (here CTC recognition) reads the motion.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        out = self.denoiser(x_t, t, cond, **denoiser_kwargs)

        if self.parameterization == "eps":
            return F.mse_loss(out, noise)

        loss = F.mse_loss(out, x_start)
        if self.velocity_weight > 0:
            loss = loss + self.velocity_weight * F.mse_loss(
                self._velocity(out), self._velocity(x_start))
        return loss

    def sample_timesteps(self, n: int, device) -> torch.Tensor:
        """Draw training timesteps, optionally emphasising the high-noise regime."""
        t = torch.randint(0, self.num_timesteps, (n,), device=device)
        if self.high_t_frac > 0.0:
            lo = int(self.num_timesteps * self.high_t_start)
            lo = min(lo, self.num_timesteps - 1)
            t_high = torch.randint(lo, self.num_timesteps, (n,), device=device)
            take_high = torch.rand(n, device=device) < self.high_t_frac
            t = torch.where(take_high, t_high, t)
        return t

    def forward(self, x_start: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample random timesteps and return the diffusion loss."""
        t = self.sample_timesteps(x_start.shape[0], x_start.device)
        return self.p_losses(x_start, t, cond=cond)

    # -- reverse process / sampling ----------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor,
                 cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One ancestral DDPM reverse step x_t -> x_{t-1}."""
        _, x_start = self.model_predictions(x_t, t, cond)
        mean, _, log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        noise = torch.randn_like(x_t)
        # No noise is added at t == 0.
        nonzero = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(self, shape, cond: Optional[torch.Tensor] = None,
               device: Optional[torch.device] = None) -> torch.Tensor:
        """Full ancestral sampling loop from pure noise to x_0."""
        device = device or self.betas.device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond=cond)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond: Optional[torch.Tensor] = None,
                    num_steps: int = 50, eta: float = 0.0,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """Deterministic (eta=0) / stochastic DDIM sampler with fewer steps."""
        device = device or self.betas.device
        step_indices = torch.linspace(0, self.num_timesteps - 1, num_steps,
                                      device=device).round().long().flip(0)
        x = torch.randn(shape, device=device)
        for k, i in enumerate(step_indices):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            eps_hat, x_start = self.model_predictions(x, t, cond)
            abar_t = _extract(self.alphas_cumprod, t, x.shape)
            if k < len(step_indices) - 1:
                j = step_indices[k + 1]
                t_next = torch.full((shape[0],), int(j), device=device, dtype=torch.long)
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
