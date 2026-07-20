"""Dual-quaternion skinning (docs/AVATAR_RENDER.md §3).

A dual quaternion ``q̂ = q_r + ε q_d`` (``ε²=0``) encodes a rigid transform:
``q_r`` unit quaternion (rotation), ``q_d = ½ (0,t) ⊗ q_r`` (translation ``t``).
Dual-quaternion linear blending (DLB) blends transforms *on the rotation manifold*,
avoiding the volume collapse ("candy-wrapper") of linear blend skinning.

Quaternions are (w, x, y, z) scalar-first, matching the pose layer. A dual
quaternion is stored as an (…, 8) tensor: real part first 4, dual part last 4.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ..pose.rotations import matrix_to_quaternion, quaternion_to_matrix


def quat_mul(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``q ⊗ p`` for (…, 4) scalar-first quaternions."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = p.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (w, −x, −y, −z)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


# ---------------------------------------------------------------------------
# transform <-> dual quaternion
# ---------------------------------------------------------------------------
def dq_from_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(R (…,3,3), t (…,3)) -> unit dual quaternion (…, 8)."""
    q_r = matrix_to_quaternion(R)                            # (..., 4)
    t_quat = torch.cat((torch.zeros_like(t[..., :1]), t), dim=-1)   # (0, t)
    q_d = 0.5 * quat_mul(t_quat, q_r)
    return torch.cat((q_r, q_d), dim=-1)


def dq_normalize(dq: torch.Tensor) -> torch.Tensor:
    """Divide by the norm of the real part so ``‖q_r‖ = 1``."""
    n = torch.linalg.norm(dq[..., :4], dim=-1, keepdim=True).clamp_min(1e-12)
    return dq / n


def transform_from_dq(dq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit dual quaternion (…, 8) -> (R (…,3,3), t (…,3))."""
    dq = dq_normalize(dq)
    q_r, q_d = dq[..., :4], dq[..., 4:]
    R = quaternion_to_matrix(q_r)
    t_quat = 2.0 * quat_mul(q_d, quat_conj(q_r))
    return R, t_quat[..., 1:]


def apply_dq_to_point(dq: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply the rigid transform of ``dq`` to a point ``p`` (…, 3): R p + t."""
    R, t = transform_from_dq(dq)
    return torch.einsum("...ab,...b->...a", R, p) + t


# ---------------------------------------------------------------------------
# dual-quaternion linear blending
# ---------------------------------------------------------------------------
def dlb(weights: torch.Tensor, dqs: torch.Tensor) -> torch.Tensor:
    """Blend ``K`` dual quaternions by weights (with antipodality), normalise.

    ``weights`` (K,), ``dqs`` (K, 8). Each ``q̂_k`` is sign-aligned to the reference
    ``q̂_0`` (flip if ``⟨q_{r,k}, q_{r,0}⟩ < 0``) because ``q̂`` and ``−q̂`` denote the
    same transform; blending unaligned signs would cancel. Returns the unit blended
    dual quaternion (8,).
    """
    ref = dqs[0, :4]
    dots = (dqs[:, :4] * ref).sum(-1)                        # (K,)
    signs = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    aligned = dqs * signs.unsqueeze(-1)
    b = (weights.unsqueeze(-1) * aligned).sum(0)             # (8,)
    return dq_normalize(b)
