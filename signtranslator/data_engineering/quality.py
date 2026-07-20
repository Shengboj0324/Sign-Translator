"""Multi-view triangulation + weighted robust reprojection (Doc-10 §3).

Reuses the Doc-04 `PerspectiveCamera` (projection) and `reprojection_loss` /
`geman_mcclure` (the document's `e = c·ρ(‖Π(J)−k‖)`). Adds DLT triangulation and a
confidence-propagated 3D confidence so uncertainty is carried, not discarded.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..pose.camera import PerspectiveCamera, reprojection_loss


def projection_matrix(cam: PerspectiveCamera) -> torch.Tensor:
    """P = K [R | t]  (3,4) for a pinhole camera."""
    dt = cam.R.dtype
    K = torch.zeros((3, 3), dtype=dt)
    K[0, 0] = cam.fx; K[1, 1] = cam.fy
    K[0, 2] = cam.cx; K[1, 2] = cam.cy; K[2, 2] = 1.0
    Rt = torch.cat((cam.R, cam.t.reshape(3, 1)), dim=1)        # (3,4)
    return K @ Rt


def triangulate_dlt(cams: Sequence[PerspectiveCamera], obs_2d: torch.Tensor,
                    confidences: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Triangulate one 3D point from N views by the (weighted) DLT.

    ``obs_2d`` is (N, 2). Each view contributes two rows to A:
        u (p3·X) − (p1·X) = 0,   v (p3·X) − (p2·X) = 0,
    scaled by √c so the null-space solve is confidence-weighted. Returns the
    de-homogenised (3,) point (right singular vector of A with least σ).
    """
    n = len(cams)
    if obs_2d.shape != (n, 2):
        raise ValueError("obs_2d must be (N, 2) aligned with cams")
    dt = torch.float64
    if confidences is None:
        confidences = torch.ones(n, dtype=dt)
    rows = []
    for i, cam in enumerate(cams):
        P = projection_matrix(cam).to(dt)
        u, v = obs_2d[i, 0].to(dt), obs_2d[i, 1].to(dt)
        w = torch.sqrt(confidences[i].to(dt).clamp_min(0.0))
        rows.append(w * (u * P[2] - P[0]))
        rows.append(w * (v * P[2] - P[1]))
    A = torch.stack(rows, dim=0)                               # (2N, 4)
    _, _, Vh = torch.linalg.svd(A)
    X = Vh[-1]                                                 # least-σ direction
    return X[:3] / X[3]


def triangulation_confidence(residuals: torch.Tensor, confidences: torch.Tensor,
                             tau: float = 1.0) -> torch.Tensor:
    """3D confidence in [0,1]: ↑ with observation confidence, ↓ with residual.

    conf = mean(c) · 1/(1 + mean(‖resid‖)/τ). Monotone increasing in each c_i
    and strictly decreasing in the mean residual (both proved in tests).
    """
    c_mean = confidences.to(torch.float64).mean()
    r_mean = residuals.to(torch.float64).mean()
    quality = 1.0 / (1.0 + r_mean / tau)
    return (c_mean * quality).clamp(0.0, 1.0)


def weighted_reprojection_residual(cam: PerspectiveCamera, X: torch.Tensor,
                                   keypoint: torch.Tensor,
                                   confidence: torch.Tensor,
                                   sigma: float = 100.0) -> torch.Tensor:
    """The document's e = c·ρ(‖Π(X) − k‖) via the audited Doc-04 primitive."""
    proj, _ = cam.project(X)
    return reprojection_loss(proj.reshape(1, 2), keypoint.reshape(1, 2),
                             confidence.reshape(1), sigma=sigma, robust=True)
