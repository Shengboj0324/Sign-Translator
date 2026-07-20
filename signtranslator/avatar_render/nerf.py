"""NeRF volume rendering (docs/AVATAR_RENDER.md §5).

Mildenhall et al. (arXiv:2003.08934). The continuous integral
``C = ∫ T(t) σ(t) c(t) dt``, ``T(t) = exp(−∫ σ ds)`` discretises along ray samples
``t_i`` with intervals ``δ_i = t_{i+1} − t_i`` to

    α_i = 1 − exp(−σ_i δ_i),   T_i = exp(−Σ_{j<i} σ_j δ_j) = Π_{j<i}(1−α_j),
    C   = Σ_i T_i α_i c_i,     w_i = T_i α_i,

which is exactly the §4 alpha compositing with ``α_i = 1 − exp(−σ_i δ_i)``.
"""

from __future__ import annotations

from typing import Tuple

import torch


def deltas_from_samples(t_samples: torch.Tensor, far: float = 1e10) -> torch.Tensor:
    """Interval lengths δ_i = t_{i+1} − t_i along the ray; the last uses ``far``.

    ``t_samples`` (…, N) increasing. Returns (…, N).
    """
    d = t_samples[..., 1:] - t_samples[..., :-1]
    last = torch.full_like(t_samples[..., :1], far)
    return torch.cat((d, last), dim=-1)


def alphas_from_density(sigma: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """α_i = 1 − exp(−σ_i δ_i). ``sigma``/``delta`` (…, N)."""
    return 1.0 - torch.exp(-sigma.clamp_min(0.0) * delta)


def transmittance(alphas: torch.Tensor) -> torch.Tensor:
    """T_i = Π_{j<i}(1−α_j) (exclusive cumulative product). (…, N) -> (…, N)."""
    one_minus = 1.0 - alphas
    # exclusive cumprod: T_0 = 1, T_i = prod_{j<i}(1-alpha_j)
    cp = torch.cumprod(one_minus, dim=-1)
    T = torch.ones_like(alphas)
    T[..., 1:] = cp[..., :-1]
    return T


def volume_render(sigma: torch.Tensor, colors: torch.Tensor,
                  t_samples: torch.Tensor, far: float = 1e10
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render a ray. ``sigma`` (…,N), ``colors`` (…,N,C), ``t_samples`` (…,N).

    Returns (colour (…,C), weights (…,N), accumulated opacity (…,)).
    """
    delta = deltas_from_samples(t_samples, far)
    alphas = alphas_from_density(sigma, delta)               # (..., N)
    T = transmittance(alphas)
    w = T * alphas                                           # (..., N)
    color = (w.unsqueeze(-1) * colors).sum(dim=-2)           # (..., C)
    acc = w.sum(-1)                                          # accumulated opacity
    return color, w, acc


def expected_depth(weights: torch.Tensor, t_samples: torch.Tensor) -> torch.Tensor:
    """Expected ray-termination depth Σ_i w_i t_i (unnormalised by acc)."""
    return (weights * t_samples).sum(-1)
