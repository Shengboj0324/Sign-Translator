"""The conditional diffusion motion generator (docs/DIFFUSION_GEN.md §9-10).

Ties together the temporal DiT denoiser, the noise schedule, classifier-free
guidance, and (optionally) constraint projection into a trainer + sampler. The
denoiser predicts ε/x₀/v; training uses condition dropout; sampling uses guided
DDPM ancestral steps with an optional feasibility projection of the predicted x₀.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .schedule import NoiseSchedule
from .guidance import classifier_free_guidance, drop_condition_mask


class DiffusionMotionGenerator(nn.Module):
    def __init__(self, denoiser: nn.Module, in_dim: int,
                 num_timesteps: int = 1000, param: str = "x0",
                 p_uncond: float = 0.1, schedule: str = "cosine") -> None:
        super().__init__()
        if param not in ("eps", "x0", "v"):
            raise ValueError("param must be eps/x0/v")
        self.denoiser = denoiser
        self.in_dim = in_dim
        self.param = param
        self.p_uncond = p_uncond
        self.schedule = NoiseSchedule(num_timesteps, schedule)
        self.num_timesteps = num_timesteps

    # -- training -----------------------------------------------------------
    def training_loss(self, x0: torch.Tensor, cond_vec: Optional[torch.Tensor] = None,
                      cond_tokens: Optional[torch.Tensor] = None,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """MSE on the ``param`` target with condition dropout. ``x0`` (N, T, D)."""
        N = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (N,), generator=generator,
                          device=x0.device)
        eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
        x_t = self.schedule.q_sample(x0, t, eps)
        # classifier-free dropout of the conditioning
        cv, ct = cond_vec, cond_tokens
        if cond_vec is not None or cond_tokens is not None:
            drop = drop_condition_mask(N, self.p_uncond, generator, x0.device)
            if cond_vec is not None:
                cv = cond_vec.clone(); cv[drop] = 0.0
            if cond_tokens is not None:
                ct = cond_tokens.clone(); ct[drop] = 0.0
        pred = self.denoiser(x_t, t, cv, ct)
        target = self.schedule.target_for(x0, t, eps, self.param)
        return ((pred - target) ** 2).mean()

    # -- sampling -----------------------------------------------------------
    def _predict_x0(self, x_t, t, cond_vec, cond_tokens, w):
        x0_c = self.schedule.to_x0(x_t, t, self.denoiser(x_t, t, cond_vec, cond_tokens),
                                   self.param)
        if w == 0.0 or (cond_vec is None and cond_tokens is None):
            return x0_c
        # unconditional prediction (null conditioning) -> guide in x0 space
        x0_u = self.schedule.to_x0(x_t, t, self.denoiser(x_t, t, None, None), self.param)
        return classifier_free_guidance(x0_c, x0_u, w)

    @torch.no_grad()
    def sample(self, shape, cond_vec: Optional[torch.Tensor] = None,
               cond_tokens: Optional[torch.Tensor] = None, w: float = 0.0,
               project: Optional[Callable] = None,
               generator: Optional[torch.Generator] = None,
               device=None) -> torch.Tensor:
        """Guided DDPM ancestral sampling. ``project`` optionally maps the predicted
        x₀ onto the feasible set at each step (constraint-projected sampling)."""
        x_t = torch.randn(shape, generator=generator, device=device)
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=device)
            x0 = self._predict_x0(x_t, t, cond_vec, cond_tokens, w)
            if project is not None:
                x0 = project(x0)
            mean, var = self.schedule.posterior_mean_variance(x0, x_t, t)
            if step > 0:
                noise = torch.randn(shape, generator=generator, device=device)
                x_t = mean + torch.sqrt(var.clamp_min(1e-20)) * noise
            else:
                x_t = mean
        return x_t
