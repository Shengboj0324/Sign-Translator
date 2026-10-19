"""Multi-view skeleton fitting: rigid bones, 2D reprojection, temporal smoothness.

Independent per-frame triangulation lets every bone change length and jitter from
frame to frame. Here the unknowns are what the canonical state encodes — bone
directions — on a skeleton whose bone lengths are shared by the whole session:

    joint_j(t) = joint_parent(t) + L_j · u_j(t),   |u_j(t)| = 1.

Each direction is parameterised in the tangent plane of its initial value,
``u = normalise(u0 + a e1 + b e2)``, which has no singularity near the solution
(unlike spherical angles at the poles). The objective, minimised with a robust loss
and a sparse Jacobian, is

* confidence-weighted reprojection error of every joint in both views (pixels);
* a temporal term ``λ |u(t+1) − 2u(t) + u(t−1)|`` on each direction's second difference
  (an acceleration penalty: constant angular velocity costs nothing);
* bone lengths fixed to the per-session robust (median) triangulated lengths, so the
  fit cannot trade length for direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Skeleton:
    joints: tuple[int, ...]            # keypoint indices in fitting order (parents first)
    parents: tuple[int, ...]           # index into ``joints`` of each joint's parent; -1 for the root


def _tangent_basis(u0: np.ndarray):
    helper = np.where(np.abs(u0[..., :1]) < .9, np.array([1., 0, 0]), np.array([0., 1, 0]))
    e1 = np.cross(u0, helper)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(u0, e1)
    return e1, e2


def fit_skeleton(points3d: np.ndarray, view1: np.ndarray, view2: np.ndarray, conf1: np.ndarray,
                 conf2: np.ndarray, p1: np.ndarray, p2: np.ndarray, skeleton: Skeleton, *,
                 smoothness: float = 20., max_nfev: int = 60) -> tuple[np.ndarray, dict]:
    """Refined (T, K, 3) points (NaN for joints outside the skeleton) and a report.

    ``points3d`` (T, K, 3) is the triangulated initialisation (NaN allowed);
    views are (T, K, 2) pixels with confidences (T, K); p1/p2 are 3x4 projections.
    """
    frames = len(points3d)
    joints, parents = np.array(skeleton.joints), np.array(skeleton.parents)
    n = len(joints)
    init = points3d[:, joints]
    # Root position: fill gaps by nearest finite frame.
    root = init[:, 0].copy()
    finite = np.isfinite(root).all(1)
    if not finite.any():
        raise ValueError("the root joint is never triangulated")
    last = np.flatnonzero(finite)
    fill = np.array([last[np.argmin(np.abs(last - t))] for t in range(frames)])
    root = root[fill]
    lengths = np.ones(n)
    u0 = np.zeros((frames, n, 3))
    for k in range(1, n):
        bone = init[:, k] - init[:, parents[k]]
        norm = np.linalg.norm(bone, axis=1)
        ok = np.isfinite(norm) & (norm > 0)
        if not ok.any():
            raise ValueError(f"joint {joints[k]} is never triangulated with its parent")
        lengths[k] = np.median(norm[ok])
        direction = np.where(ok[:, None], bone / np.where(ok, norm, 1.)[:, None], np.nan)
        good = np.flatnonzero(ok)
        nearest = np.array([good[np.argmin(np.abs(good - t))] for t in range(frames)])
        u0[:, k] = direction[nearest]
    e1, e2 = _tangent_basis(u0[:, 1:])
    w1 = np.sqrt(np.clip(conf1[:, joints], 0, 1))
    w2 = np.sqrt(np.clip(conf2[:, joints], 0, 1))
    x1, x2 = view1[:, joints], view2[:, joints]

    def unpack(x):
        r = x[:frames * 3].reshape(frames, 3)
        ab = x[frames * 3:].reshape(frames, n - 1, 2)
        u = u0[:, 1:] + ab[..., :1] * e1 + ab[..., 1:] * e2
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        pos = np.zeros((frames, n, 3))
        pos[:, 0] = r
        for k in range(1, n):
            pos[:, k] = pos[:, parents[k]] + lengths[k] * u[:, k - 1]
        return pos, u

    def project(p, pos):
        h = pos @ p[:, :3].T + p[:, 3]
        return h[..., :2] / h[..., 2:]

    def residuals(x):
        pos, u = unpack(x)
        r1 = ((project(p1, pos) - x1) * w1[..., None]).ravel()
        r2 = ((project(p2, pos) - x2) * w2[..., None]).ravel()
        smooth = smoothness * (u[2:] - 2 * u[1:-1] + u[:-2]).ravel() if frames > 2 else np.zeros(0)
        return np.concatenate((np.nan_to_num(r1), np.nan_to_num(r2), smooth))

    # Sparsity: a frame's reprojections depend on its root and the directions of the
    # joint's ancestors; smoothness couples three consecutive frames of one direction.
    ancestors = [[] for _ in range(n)]
    for k in range(1, n):
        ancestors[k] = ancestors[parents[k]] + [k]
    rows_per_frame = n * 2
    total_rows = 2 * frames * rows_per_frame + (max(frames - 2, 0)) * (n - 1) * 3
    sparsity = lil_matrix((total_rows, frames * 3 + frames * (n - 1) * 2), dtype=np.int8)
    for view in range(2):
        for t in range(frames):
            for k in range(n):
                for c in range(2):
                    row = view * frames * rows_per_frame + t * rows_per_frame + 2 * k + c
                    sparsity[row, t * 3:t * 3 + 3] = 1
                    for a in ancestors[k]:
                        col = frames * 3 + (t * (n - 1) + a - 1) * 2
                        sparsity[row, col:col + 2] = 1
    base = 2 * frames * rows_per_frame
    for t in range(max(frames - 2, 0)):
        for k in range(n - 1):
            for c in range(3):
                row = base + (t * (n - 1) + k) * 3 + c
                for dt in range(3):
                    col = frames * 3 + ((t + dt) * (n - 1) + k) * 2
                    sparsity[row, col:col + 2] = 1
    x0 = np.concatenate((root.ravel(), np.zeros(frames * (n - 1) * 2)))
    start = np.sqrt(np.mean(residuals(x0)[:base] ** 2))
    result = least_squares(residuals, x0, jac_sparsity=sparsity.tocsr(), method="trf", loss="soft_l1", f_scale=2.,
                           x_scale="jac", max_nfev=max_nfev)
    pos, _ = unpack(result.x)
    out = np.full_like(points3d, np.nan)
    out[:, joints] = pos
    report = dict(start_rms_px=float(start), end_rms_px=float(np.sqrt(np.mean(residuals(result.x)[:base] ** 2))),
                  evaluations=int(result.nfev), success=bool(result.success), lengths=lengths.tolist())
    return out, report


__all__ = ["Skeleton", "fit_skeleton"]
