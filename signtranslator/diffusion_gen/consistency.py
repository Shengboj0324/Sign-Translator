"""Consistency and rectified-flow distillation for few-step sampling (§8).

Distilled **only after** the full diffusion model is good (per the document); here
we build and verify the machinery.

* **Consistency model** (Song et al., arXiv:2303.01469):
  ``f_θ(x, t) = c_skip(t) x + c_out(t) F_θ(x, t)`` with the boundary
  ``c_skip(t_min)=1, c_out(t_min)=0`` so ``f(x_{t_min}, t_min) = x_{t_min}`` — the
  boundary condition, proved. The self-consistency loss
  ``‖f_θ(x_{t_{n+1}}, t_{n+1}) − f_{θ⁻}(x_{t_n}, t_n)‖`` trains one/few-step mapping.

* **Rectified flow** (Liu et al.): ``x_t = (1−t) x₀ + t z`` for ``t ∈ [0,1]`` is a
  straight line with **constant velocity** ``dx/dt = z − x₀``, so
  ``x₀ = x_t − t·velocity`` (proved). A velocity model samples by an Euler ODE.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# consistency parameterisation
# ---------------------------------------------------------------------------
def consistency_coeffs(t: torch.Tensor, t_min: float = 0.002,
                       sigma_data: float = 0.5):
    """Karras-style skip/out coefficients with the boundary at ``t_min``.

    ``c_skip(t) = σ_data² / ((t − t_min)² + σ_data²)`` -> 1 at ``t=t_min``;
    ``c_out(t)  = σ_data (t − t_min) / √(σ_data² + t²)`` -> 0 at ``t=t_min``.
    """
    dt = t - t_min
    c_skip = sigma_data ** 2 / (dt ** 2 + sigma_data ** 2)
    c_out = sigma_data * dt / torch.sqrt(sigma_data ** 2 + t ** 2)
    return c_skip, c_out


class ConsistencyModel(nn.Module):
    """Wraps a raw network ``F`` into ``f(x,t)=c_skip x + c_out F(x,t)``."""

    def __init__(self, net: nn.Module, t_min: float = 0.002,
                 sigma_data: float = 0.5) -> None:
        super().__init__()
        self.net = net
        self.t_min = t_min
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond=None, cond_tokens=None) -> torch.Tensor:
        c_skip, c_out = consistency_coeffs(t, self.t_min, self.sigma_data)
        c_skip = c_skip.reshape((-1,) + (1,) * (x.dim() - 1))
        c_out = c_out.reshape((-1,) + (1,) * (x.dim() - 1))
        F = self.net(x, t, cond, cond_tokens) if cond_tokens is not None or cond is not None \
            else self.net(x, t)
        return c_skip * x + c_out * F


def self_consistency_loss(model: ConsistencyModel, target_model: ConsistencyModel,
                          x_high: torch.Tensor, t_high: torch.Tensor,
                          x_low: torch.Tensor, t_low: torch.Tensor) -> torch.Tensor:
    """‖f_θ(x_{t_{n+1}}, t_{n+1}) − f_{θ⁻}(x_{t_n}, t_n)‖² (target detached)."""
    pred = model(x_high, t_high)
    with torch.no_grad():
        tgt = target_model(x_low, t_low)
    return ((pred - tgt) ** 2).mean()


# ---------------------------------------------------------------------------
# rectified flow
# ---------------------------------------------------------------------------
def rectified_flow_interpolant(x0: torch.Tensor, z: torch.Tensor,
                               t: torch.Tensor) -> torch.Tensor:
    """x_t = (1−t) x₀ + t z, ``t`` a per-sample fraction in [0, 1]."""
    tt = t.reshape((-1,) + (1,) * (x0.dim() - 1))
    return (1.0 - tt) * x0 + tt * z


def rectified_flow_velocity(x0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The (constant, straight-line) target velocity ``dx/dt = z − x₀``."""
    return z - x0


def rectified_flow_x0_from_velocity(x_t: torch.Tensor, t: torch.Tensor,
                                    velocity: torch.Tensor) -> torch.Tensor:
    """Recover ``x₀ = x_t − t·velocity`` (straight path)."""
    tt = t.reshape((-1,) + (1,) * (x_t.dim() - 1))
    return x_t - tt * velocity


@torch.no_grad()
def rectified_flow_sample(velocity_fn: Callable, x1: torch.Tensor,
                          num_steps: int = 4) -> torch.Tensor:
    """Euler-integrate the reverse ODE from noise ``x1`` (t=1) to data (t=0).

    ``velocity_fn(x, t_scalar)`` returns the predicted velocity. Straight paths mean
    few steps suffice.
    """
    x = x1
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
        v = velocity_fn(x, t_tensor)
        x = x - dt * v                                       # move toward t=0
    return x
