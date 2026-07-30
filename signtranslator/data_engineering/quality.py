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
    device = cam.R.device
    tensors = (cam.fx, cam.fy, cam.cx, cam.cy, cam.R, cam.t)
    if any(not torch.isfinite(torch.as_tensor(value)).all() for value in tensors):
        raise ValueError("camera parameters must be finite")
    if cam.R.shape != (3, 3) or cam.t.shape != (3,):
        raise ValueError("camera extrinsics must have R=(3,3) and t=(3,)")
    if float(cam.fx) <= 0 or float(cam.fy) <= 0:
        raise ValueError("camera focal lengths must be positive")
    K = torch.zeros((3, 3), dtype=dt, device=device)
    K[0, 0] = cam.fx; K[1, 1] = cam.fy
    K[0, 2] = cam.cx; K[1, 2] = cam.cy; K[2, 2] = 1.0
    Rt = torch.cat((cam.R, cam.t.reshape(3, 1)), dim=1)        # (3,4)
    return K @ Rt


def triangulate_dlt(cams: Sequence[PerspectiveCamera], obs_2d: torch.Tensor,
                    confidences: Optional[torch.Tensor] = None,
                    *, rank_tolerance: float = 1e-10,
                    homogeneous_epsilon: float = 1e-12) -> torch.Tensor:
    """Triangulate one 3D point from N views by the (weighted) DLT.

    ``obs_2d`` is (N, 2). Each view contributes two rows to A:
        u (p3·X) − (p1·X) = 0,   v (p3·X) − (p2·X) = 0,
    scaled by √c so the null-space solve is confidence-weighted. Returns the
    de-homogenised (3,) point (right singular vector of A with least σ).
    """
    n = len(cams)
    if n < 2:
        raise ValueError("triangulation requires at least two camera views")
    if obs_2d.shape != (n, 2):
        raise ValueError("obs_2d must be (N, 2) aligned with cams")
    if not torch.isfinite(obs_2d).all():
        raise ValueError("2D observations must be finite")
    if rank_tolerance <= 0 or homogeneous_epsilon <= 0:
        raise ValueError("numerical tolerances must be positive")
    dt = torch.float64
    if confidences is None:
        confidences = torch.ones(n, dtype=dt, device=obs_2d.device)
    if confidences.shape != (n,):
        raise ValueError("confidences must be (N,) aligned with cams")
    if not torch.isfinite(confidences).all() or torch.any(
            (confidences < 0) | (confidences > 1)):
        raise ValueError("confidences must be finite and in [0, 1]")
    confidences = confidences.to(device=obs_2d.device)
    active = confidences > 0
    if int(active.sum()) < 2:
        raise ValueError("triangulation requires at least two positive-confidence views")
    rows = []
    for i, cam in enumerate(cams):
        if not bool(active[i]):
            continue
        P = projection_matrix(cam).to(device=obs_2d.device, dtype=dt)
        u, v = obs_2d[i, 0].to(dt), obs_2d[i, 1].to(dt)
        w = torch.sqrt(confidences[i].to(dt))
        row_u = u * P[2] - P[0]
        row_v = v * P[2] - P[1]
        # Row normalization removes arbitrary pixel/focal-length scale before
        # applying the statistical sqrt(confidence) weight.
        rows.append(w * row_u / torch.linalg.vector_norm(row_u).clamp_min(1e-15))
        rows.append(w * row_v / torch.linalg.vector_norm(row_v).clamp_min(1e-15))
    A = torch.stack(rows, dim=0)                               # (2N, 4)
    _, singular_values, Vh = torch.linalg.svd(A)
    # A unique projective point needs rank >= 3.  The third singular direction
    # must be numerically separated from zero; otherwise rays are coincident or
    # nearly parallel and dehomogenization would manufacture an unstable point.
    scale = singular_values[0].clamp_min(torch.finfo(dt).tiny)
    if singular_values.numel() < 3 or singular_values[-2] <= rank_tolerance * scale:
        raise ValueError("degenerate camera geometry: DLT system has rank below 3")
    X = Vh[-1]                                                 # least-σ direction
    if not torch.isfinite(X).all() or torch.abs(X[3]) <= homogeneous_epsilon:
        raise ValueError("triangulated point is at infinity or numerically unstable")
    point = X[:3] / X[3]
    if not torch.isfinite(point).all():
        raise ValueError("triangulation produced a non-finite point")
    for index, cam in enumerate(cams):
        if bool(active[index]):
            _, in_front = cam.project(point.to(dtype=cam.R.dtype, device=cam.R.device))
            if not bool(in_front):
                raise ValueError("triangulated point fails camera cheirality")
    return point


def triangulation_confidence(residuals: torch.Tensor, confidences: torch.Tensor,
                             tau: float = 1.0) -> torch.Tensor:
    """3D confidence in [0,1]: ↑ with observation confidence, ↓ with residual.

    conf = mean(c) · 1/(1 + mean(‖resid‖)/τ). Monotone increasing in each c_i
    and strictly decreasing in the mean residual (both proved in tests).
    """
    if residuals.ndim != 1 or confidences.shape != residuals.shape or residuals.numel() < 2:
        raise ValueError("residuals and confidences must be aligned vectors with N>=2")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if (not torch.isfinite(residuals).all() or torch.any(residuals < 0)
            or not torch.isfinite(confidences).all()
            or torch.any((confidences < 0) | (confidences > 1))):
        raise ValueError("residuals/confidences are outside their valid domains")
    active = confidences > 0
    if int(active.sum()) < 2:
        raise ValueError("confidence requires at least two observed views")
    c_mean = confidences[active].to(torch.float64).mean()
    r_mean = residuals[active].to(torch.float64).mean()
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
