"""Kinematic constraints: differentiable penalties + projection (§7).

Each penalty is ``≥ 0`` and zero iff the constraint is satisfied; each projection
maps onto the feasible set and is idempotent. The self-collision and contact terms
reuse the audited Doc-04 / Doc-05 primitives.

* **Joint limits** — angle magnitude ``|θ| ≤ θ_max``.
* **Self-collision** — Doc-04 sphere-proxy penalty.
* **Contact** — Doc-05 hard-contact indicator distance.
* **Temporal boundary** — match fixed boundary frames (streaming seams).

**Innovation — constraint-projected sampling:** after each denoising step, project
the predicted ``x₀`` onto the feasible set; the projection is idempotent so the
result is a fixed point on the feasible manifold.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..pose.fitting import self_collision_penalty


# ---------------------------------------------------------------------------
# joint limits (on axis-angle magnitude or any scalar angle channel)
# ---------------------------------------------------------------------------
def joint_limit_penalty(angles: torch.Tensor, theta_max: float) -> torch.Tensor:
    """Σ max(0, |θ| − θ_max)². Zero iff every |θ| ≤ θ_max."""
    excess = (angles.abs() - theta_max).clamp_min(0.0)
    return (excess ** 2).sum()


def project_joint_limits(angles: torch.Tensor, theta_max: float) -> torch.Tensor:
    """Clamp each angle to [−θ_max, θ_max]. Idempotent; result is feasible."""
    return angles.clamp(-theta_max, theta_max)


# ---------------------------------------------------------------------------
# self-collision (reuse Doc-04)
# ---------------------------------------------------------------------------
def collision_penalty(joints: torch.Tensor, radii: torch.Tensor,
                      adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sum of the Doc-04 sphere-proxy self-collision penalty over frames."""
    return self_collision_penalty(joints, radii, adjacency).sum()


# ---------------------------------------------------------------------------
# contact (reuse Doc-05 hard-contact indicator)
# ---------------------------------------------------------------------------
def contact_penalty(x_i: torch.Tensor, x_j: torch.Tensor, rho: float,
                    should_touch: torch.Tensor) -> torch.Tensor:
    """Penalise required contacts that are not made: for pairs with
    ``should_touch=1``, penalty = max(0, ‖x_i−x_j‖ − rho)² (they must be within rho).
    Zero iff every required contact is within the threshold."""
    d = torch.linalg.norm(x_i - x_j, dim=-1)
    gap = (d - rho).clamp_min(0.0)
    return ((gap ** 2) * should_touch.to(d.dtype)).sum()


# ---------------------------------------------------------------------------
# temporal boundary (streaming seams / fixed keyframes)
# ---------------------------------------------------------------------------
def temporal_boundary_penalty(x: torch.Tensor, boundary_target: torch.Tensor,
                              boundary_mask: torch.Tensor) -> torch.Tensor:
    """‖x − target‖² over the boundary frames (``boundary_mask`` selects them)."""
    m = boundary_mask.to(x.dtype)
    return (((x - boundary_target) ** 2) * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# combined + projection
# ---------------------------------------------------------------------------
def project_feasible(angles: torch.Tensor, theta_max: float) -> torch.Tensor:
    """Project a pose onto the feasible set (currently joint limits). Idempotent."""
    return project_joint_limits(angles, theta_max)
