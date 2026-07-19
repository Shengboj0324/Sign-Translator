"""Evaluation metrics for 3D human pose -- never MPJPE alone.

Per docs/HUMAN_REPRESENTATION.md §6 and the source document: *"small fingertip
errors can change meaning"*, so we evaluate joint rotations and surface landmarks,
not only body-joint position.

* ``mpjpe`` -- mean per-joint position error after root alignment.
* ``pa_mpjpe`` -- Procrustes (Kabsch) aligned MPJPE; the similarity transform is
  the closed-form optimum, with a reflection guard so ``det R = +1``.
* ``mean_geodesic_rotation_error`` -- mean SO(3) angle between predicted and true
  joint rotations (catches wrong orientation even when position looks fine).
* ``v2v`` -- mean vertex (surface) error.
* ``fingertip_weighted_mpjpe`` -- position error with fingertips up-weighted; the
  quantified reason MPJPE alone is inadequate for signing.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .rotations import geodesic_distance


def _root_align(joints: torch.Tensor, root: int) -> torch.Tensor:
    return joints - joints[..., root:root + 1, :]


def mpjpe(pred: torch.Tensor, gt: torch.Tensor, root: Optional[int] = 0
          ) -> torch.Tensor:
    """Mean per-joint position error. If ``root`` is given, align by subtracting it.

    ``pred``/``gt`` are (..., J, 3). Returns a scalar (mean over all leading dims
    and joints).
    """
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must share shape")
    if root is not None:
        pred = _root_align(pred, root)
        gt = _root_align(gt, root)
    return (pred - gt).norm(dim=-1).mean()


def kabsch(A: torch.Tensor, B: torch.Tensor):
    """Optimal similarity transform (s, R, t) mapping A onto B (least squares).

    Minimises ||s R A + t - B||^2 over rotation R (det +1), scale s > 0,
    translation t. A, B are (N, 3). Returns (s, R, t, A_aligned).
    """
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    H = A0.T @ B0                                             # cross-covariance
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))             # reflection guard
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=A.dtype, device=A.device))
    R = Vt.T @ D @ U.T
    varA = (A0 ** 2).sum()
    s = (S * torch.tensor([1.0, 1.0, d], dtype=A.dtype, device=A.device)).sum() / varA
    t = muB - s * (R @ muA)
    A_aligned = s * (A0 @ R.T) + muB
    return s, R, t, A_aligned


def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Procrustes-aligned MPJPE: align ``pred`` to ``gt`` by the optimal (s,R,t)."""
    if pred.shape != gt.shape or pred.dim() != 2:
        raise ValueError("pa_mpjpe expects (J, 3) tensors")
    _, _, _, aligned = kabsch(pred, gt)
    return (aligned - gt).norm(dim=-1).mean()


def mean_geodesic_rotation_error(R_pred: torch.Tensor, R_gt: torch.Tensor
                                 ) -> torch.Tensor:
    """Mean SO(3) geodesic angle (radians) between predicted and true rotations.

    ``R_pred``/``R_gt`` are (..., 3, 3). Returns the mean over leading dims.
    """
    return geodesic_distance(R_pred, R_gt).mean()


def v2v(pred_verts: torch.Tensor, gt_verts: torch.Tensor,
        root: Optional[int] = None) -> torch.Tensor:
    """Mean vertex-to-vertex (surface) error. Optional root alignment by index."""
    if pred_verts.shape != gt_verts.shape:
        raise ValueError("vertex sets must share shape")
    if root is not None:
        pred_verts = pred_verts - pred_verts[..., root:root + 1, :]
        gt_verts = gt_verts - gt_verts[..., root:root + 1, :]
    return (pred_verts - gt_verts).norm(dim=-1).mean()


def fingertip_weighted_mpjpe(pred: torch.Tensor, gt: torch.Tensor,
                             fingertip_idx: Sequence[int],
                             fingertip_weight: float = 10.0,
                             root: Optional[int] = 0) -> torch.Tensor:
    """Weighted per-joint error with fingertips up-weighted.

    Weighted mean = (sum_j w_j ||e_j||) / (sum_j w_j), fingertips get
    ``fingertip_weight``, everyone else 1.
    """
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must share shape")
    if root is not None:
        pred = _root_align(pred, root)
        gt = _root_align(gt, root)
    err = (pred - gt).norm(dim=-1)                            # (..., J)
    J = err.shape[-1]
    w = torch.ones(J, dtype=err.dtype, device=err.device)
    idx = torch.tensor(list(fingertip_idx), dtype=torch.long, device=err.device)
    w[idx] = fingertip_weight
    weighted = (err * w).sum(-1) / w.sum()
    return weighted.mean()
