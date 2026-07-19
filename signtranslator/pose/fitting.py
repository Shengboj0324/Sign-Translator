"""SMPL-X fitting objective: robust re-projection + prior + smoothness + collision.

Implements docs/HUMAN_REPRESENTATION.md §5:

    L_fit = lam_2D L_2D + lam_d L_depth + lam_p L_prior + lam_v ||dq||_1 + lam_c L_col.

Each term is a proper, testable object:

* ``GaussianPosePrior`` — Mahalanobis (theta - mu)^T Prec (theta - mu); >= 0,
  minimised (= 0) at the mean. (Production SMPLify-X uses the VPoser VAE prior;
  this is a legitimate, well-defined alternative, noted as such.)
* ``GMMPosePrior`` — negative log-likelihood of a Gaussian mixture, lower at the
  cluster centres.
* ``temporal_smoothness`` — sum_t ||q_{t+1} - q_t||_1, zero iff constant.
* ``self_collision_penalty`` — squared-hinge overlap of joint sphere proxies,
  zero iff no penetration.

The document warns monocular fitting is underdetermined; the tests demonstrate
exactly that (single-view fit drives re-projection to ~0 yet leaves 3D depth
ambiguous, whereas a multi-view fit recovers 3D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .camera import reprojection_loss


# ---------------------------------------------------------------------------
# pose priors
# ---------------------------------------------------------------------------
@dataclass
class GaussianPosePrior:
    """L(theta) = (theta - mu)^T Precision (theta - mu). Precision is SPD."""

    mean: torch.Tensor         # (D,)
    precision: torch.Tensor    # (D, D) symmetric positive-definite

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        d = theta - self.mean
        # supports (..., D)
        return torch.einsum("...i,ij,...j->...", d, self.precision, d)


@dataclass
class GMMPosePrior:
    """Negative log-likelihood of a Gaussian mixture over pose vectors."""

    weights: torch.Tensor          # (G,) sum to 1
    means: torch.Tensor            # (G, D)
    covariances: torch.Tensor      # (G, D, D) SPD

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        G, D = self.means.shape
        theta_e = theta.reshape(-1, D)                      # (N, D)
        comp = []
        for g in range(G):
            cov = self.covariances[g]
            L = torch.linalg.cholesky(cov)
            diff = theta_e - self.means[g]                  # (N, D)
            sol = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            maha = (diff * sol).sum(-1)                      # (N,)
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            log_norm = -0.5 * (D * torch.log(torch.tensor(2 * torch.pi,
                              dtype=theta.dtype)) + logdet)
            comp.append(torch.log(self.weights[g]) + log_norm - 0.5 * maha)
        log_prob = torch.logsumexp(torch.stack(comp, dim=-1), dim=-1)  # (N,)
        return (-log_prob).reshape(theta.shape[:-1])


# ---------------------------------------------------------------------------
# temporal smoothness
# ---------------------------------------------------------------------------
def temporal_smoothness(q: torch.Tensor) -> torch.Tensor:
    """sum_t ||q_{t+1} - q_t||_1 over the leading (time) axis. 0 iff constant."""
    if q.shape[0] < 2:
        return torch.zeros((), dtype=q.dtype, device=q.device)
    return (q[1:] - q[:-1]).abs().sum()


# ---------------------------------------------------------------------------
# self-collision (sphere proxies)
# ---------------------------------------------------------------------------
def self_collision_penalty(joints: torch.Tensor, radii: torch.Tensor,
                           adjacency: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
    """Squared-hinge overlap penalty over non-adjacent joint sphere pairs.

    ``joints`` (..., J, 3), ``radii`` (J,). ``adjacency`` (J, J) bool marks pairs
    that are allowed to touch (skeleton neighbours + self); those are excluded.
    Penalty per pair = max(0, (r_i + r_j) - ||c_i - c_j||)^2. Zero iff no overlap.
    """
    J = joints.shape[-2]
    dist = torch.cdist(joints, joints)                       # (..., J, J)
    r_sum = radii[:, None] + radii[None, :]                  # (J, J)
    overlap = torch.clamp(r_sum - dist, min=0.0)             # hinge
    # exclude self and adjacent (and double-counting: use strict upper triangle)
    mask = torch.triu(torch.ones(J, J, dtype=torch.bool, device=joints.device), 1)
    if adjacency is not None:
        mask = mask & (~adjacency)
    pen = (overlap ** 2) * mask
    return pen.sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# full objective
# ---------------------------------------------------------------------------
@dataclass
class FittingWeights:
    lam_2d: float = 1.0
    lam_depth: float = 0.0
    lam_prior: float = 1e-3
    lam_smooth: float = 1e-2
    lam_collision: float = 1.0


@dataclass
class FittingTerms:
    reproj: torch.Tensor
    depth: torch.Tensor
    prior: torch.Tensor
    smooth: torch.Tensor
    collision: torch.Tensor

    def total(self, w: FittingWeights) -> torch.Tensor:
        return (w.lam_2d * self.reproj + w.lam_depth * self.depth
                + w.lam_prior * self.prior + w.lam_smooth * self.smooth
                + w.lam_collision * self.collision)


def fitting_terms(projected: torch.Tensor, keypoints: torch.Tensor,
                  confidence: Optional[torch.Tensor],
                  motion_vec: torch.Tensor,
                  pose_vec_for_prior: Optional[torch.Tensor] = None,
                  prior: Optional[GaussianPosePrior] = None,
                  joints3d: Optional[torch.Tensor] = None,
                  radii: Optional[torch.Tensor] = None,
                  adjacency: Optional[torch.Tensor] = None,
                  depth_pred: Optional[torch.Tensor] = None,
                  depth_obs: Optional[torch.Tensor] = None,
                  sigma: float = 100.0) -> FittingTerms:
    """Assemble the five terms; unused terms default to 0."""
    zero = torch.zeros((), dtype=projected.dtype, device=projected.device)
    reproj = reprojection_loss(projected, keypoints, confidence, sigma=sigma)
    smooth = temporal_smoothness(motion_vec)
    prior_val = (prior(pose_vec_for_prior).sum()
                 if (prior is not None and pose_vec_for_prior is not None) else zero)
    collision = (self_collision_penalty(joints3d, radii, adjacency).sum()
                 if (joints3d is not None and radii is not None) else zero)
    depth = (torch.linalg.norm(depth_pred - depth_obs)
             if (depth_pred is not None and depth_obs is not None) else zero)
    return FittingTerms(reproj=reproj, depth=depth, prior=prior_val,
                        smooth=smooth, collision=collision)
