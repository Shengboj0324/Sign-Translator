"""3D Gaussian Splatting rasterizer (docs/AVATAR_RENDER.md §4).

Kerbl et al. (arXiv:2308.04079). An anisotropic 3D Gaussian ``(μ, Σ, α, c)`` has
covariance ``Σ = R S Sᵀ Rᵀ`` (PSD by construction). It is projected to screen with
the EWA affine approximation ``Σ' = J W Σ Wᵀ Jᵀ`` (2×2 image block, ``J`` the
Jacobian of the perspective map), and Gaussians are alpha-composited front-to-back
in depth order (the "over" operator).
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..pose.rotations import quaternion_to_matrix


# ---------------------------------------------------------------------------
# 3D covariance
# ---------------------------------------------------------------------------
def covariance_3d(quaternion: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Σ = R S Sᵀ Rᵀ = (R S)(R S)ᵀ, PSD by construction. (…,4),(…,3) -> (…,3,3)."""
    R = quaternion_to_matrix(quaternion)                    # (..., 3, 3)
    S = torch.diag_embed(scale)                             # (..., 3, 3)
    M = R @ S
    return M @ M.transpose(-1, -2)


# ---------------------------------------------------------------------------
# projection to screen-space 2D covariance
# ---------------------------------------------------------------------------
def projection_jacobian(mu_cam: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    """Jacobian of ``(x,y,z) ↦ (fx x/z, fy y/z)`` at ``mu_cam``. (…,3) -> (…,2,3)."""
    x, y, z = mu_cam.unbind(-1)
    zero = torch.zeros_like(x)
    row0 = torch.stack((fx / z, zero, -fx * x / z ** 2), dim=-1)
    row1 = torch.stack((zero, fy / z, -fy * y / z ** 2), dim=-1)
    return torch.stack((row0, row1), dim=-2)                # (..., 2, 3)


def covariance_2d(cov3d: torch.Tensor, mu_cam: torch.Tensor,
                  W: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    """Σ' = (J W) Σ (J W)ᵀ, the 2×2 screen covariance. ``W`` (3,3) view rotation."""
    J = projection_jacobian(mu_cam, fx, fy)                 # (..., 2, 3)
    JW = J @ W                                              # (..., 2, 3)
    return JW @ cov3d @ JW.transpose(-1, -2)                # (..., 2, 2)


def gaussian_2d_value(p: torch.Tensor, mu2d: torch.Tensor,
                      cov2d: torch.Tensor) -> torch.Tensor:
    """exp(−½ (p−μ)ᵀ Σ'⁻¹ (p−μ)) for pixel ``p`` (…,2)."""
    d = (p - mu2d).unsqueeze(-1)                            # (..., 2, 1)
    inv = torch.linalg.inv(cov2d)
    m = (d.transpose(-1, -2) @ inv @ d).squeeze(-1).squeeze(-1)
    return torch.exp(-0.5 * m)


# ---------------------------------------------------------------------------
# alpha compositing (the "over" operator, front-to-back)
# ---------------------------------------------------------------------------
def alpha_composite(colors: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Front-to-back composite: C = Σ_i c_i α_i Π_{j<i}(1−α_j).

    ``colors`` (N, C) ordered front-to-back, ``alphas`` (N,) in [0,1]. Returns
    (composited colour (C,), accumulated opacity scalar = 1 − Π(1−α_i)).
    """
    C = torch.zeros(colors.shape[-1], dtype=colors.dtype, device=colors.device)
    T = torch.ones((), dtype=colors.dtype, device=colors.device)     # transmittance
    for i in range(colors.shape[0]):
        C = C + T * alphas[i] * colors[i]
        T = T * (1.0 - alphas[i])
    return C, 1.0 - T


def render_pixel(pixel: torch.Tensor, mu2d: torch.Tensor, cov2d: torch.Tensor,
                 opacity: torch.Tensor, colors: torch.Tensor,
                 depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Composite all Gaussians at one pixel in depth order (near -> far).

    ``mu2d`` (N,2), ``cov2d`` (N,2,2), ``opacity`` (N,), ``colors`` (N,C),
    ``depth`` (N,). Returns (colour, accumulated opacity).
    """
    order = torch.argsort(depth)                            # near first
    g = gaussian_2d_value(pixel.unsqueeze(0), mu2d, cov2d)  # (N,)
    alphas = (opacity * g).clamp(0.0, 1.0)[order]
    return alpha_composite(colors[order], alphas)
