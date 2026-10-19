"""Two-view self-calibration and triangulation using the signer as calibration object.

How2Sign recorded each session with a frontal and a lateral HD camera (1280×720,
30 fps) whose calibration was not released. This module recovers both cameras from
the signer's own 2D keypoints and triangulates 3D.

**Why plain epipolar self-calibration is not enough.** Both cameras look at the
signer, so their optical axes (nearly) intersect — exactly the configuration in which
focal lengths are not recoverable from the fundamental matrix (Bougnoux's
degeneracy). The signer resolves it: rigid bones keep their length while the pose
changes, and a wrong focal length distorts every pose differently, so bone lengths
would fluctuate over time. The calibration is therefore a bundle adjustment whose
residuals are (i) reprojection errors in pixels and (ii) the relative deviation of
each rigid bone's length from its per-session mean.

**Model.** Pinhole cameras with square pixels and the principal point at the image
centre (stated assumptions). Optionally (``distortion=True``) each camera has one
radial coefficient in Fitzgibbon's division model,
``x_u = c + (x_d − c) / (1 + λ |(x_d − c)/f|²)``, which is estimable from two views and
first-order equivalent to a Brown ``k1 = −λ``. Reprojection is always measured in the
observed (distorted) pixels — measured on undistorted pixels the objective could be
lowered by shrinking the whole image towards the centre — and ``|λ| ≤ 1`` and
``f ∈ [0.2, 10]·width`` are hard bounds. Gauge: camera 1 is
``[I | 0]``, the baseline has unit length (metric scale is not identifiable; canonical
rotations are scale-invariant, so they do not need it). Unknowns: ``log f1``,
``log f2``, the rotation vector of camera 2 and the baseline direction. 3D points are
eliminated by linear triangulation inside the residual (variable projection).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def intrinsics(focal: float, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])


def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(points - centre, axis=1))
    transform = np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1]])
    homogeneous = np.c_[points, np.ones(len(points))] @ transform.T
    return homogeneous, transform


def eight_point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Normalised eight-point fundamental matrix (rank 2), ``x2ᵀ F x1 = 0``."""
    if len(x1) < 8 or x1.shape != x2.shape:
        raise ValueError("at least 8 matching points are required")
    a, t1 = _normalise(x1)
    b, t2 = _normalise(x2)
    design = np.einsum("ni,nj->nij", b, a).reshape(len(a), 9)
    f = np.linalg.svd(design)[2][-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(f)
    f = u @ np.diag([s[0], s[1], 0.]) @ vt
    f = t2.T @ f @ t1
    return f / np.linalg.norm(f)


def sampson(f: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order geometric (Sampson) distance in pixels for each correspondence."""
    a = np.c_[x1, np.ones(len(x1))]
    b = np.c_[x2, np.ones(len(x2))]
    fa, ftb = a @ f.T, b @ f
    numerator = np.sum(b * fa, axis=1) ** 2
    return np.sqrt(numerator / (fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2))


def ransac_fundamental(x1, x2, *, threshold: float = 2., iterations: int = 2000, seed: int = 0):
    """Robust F and inlier mask; the final F is refit on all inliers."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        pick = rng.choice(len(x1), 8, replace=False)
        try:
            f = eight_point(x1[pick], x2[pick])
        except np.linalg.LinAlgError:
            continue
        inliers = sampson(f, x1, x2) < threshold
        if best is None or inliers.sum() > best[1].sum():
            best = (f, inliers)
    if best is None or best[1].sum() < 8:
        raise ValueError("no fundamental matrix with at least 8 inliers")
    inliers = best[1]
    return eight_point(x1[inliers], x2[inliers]), inliers


def undistort_division(points: np.ndarray, size, focal: float, coefficient: float) -> np.ndarray:
    """Division-model undistortion of (..., 2) pixels about the image centre."""
    if coefficient == 0:
        return points
    centre = np.array(size, dtype=float) / 2
    offset = points - centre
    r2 = np.sum((offset / focal) ** 2, axis=-1, keepdims=True)
    return centre + offset / (1 + coefficient * r2)


def distort_division(points: np.ndarray, size, focal: float, coefficient: float, iterations: int = 8) -> np.ndarray:
    """Inverse of :func:`undistort_division` by fixed-point iteration (contractive for |λ r²| < ½)."""
    if coefficient == 0:
        return points
    centre = np.array(size, dtype=float) / 2
    undistorted = points - centre
    distorted = undistorted
    for _ in range(iterations):
        distorted = undistorted * (1 + coefficient * np.sum((distorted / focal) ** 2, axis=-1, keepdims=True))
    return centre + distorted


def projections(params: np.ndarray, size1, size2) -> tuple[np.ndarray, np.ndarray]:
    f1, f2 = np.exp(params[:2])
    rotation = Rotation.from_rotvec(params[2:5]).as_matrix()
    theta, phi = params[5:7]
    direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    p1 = intrinsics(f1, size1) @ np.c_[np.eye(3), np.zeros(3)]
    p2 = intrinsics(f2, size2) @ np.c_[rotation, direction]
    return p1, p2


def triangulate(p1: np.ndarray, p2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of many points at once; returns (N, 3)."""
    rows = np.stack((x1[:, 0, None] * p1[2] - p1[0], x1[:, 1, None] * p1[2] - p1[1],
                     x2[:, 0, None] * p2[2] - p2[0], x2[:, 1, None] * p2[2] - p2[1]), axis=1)
    homogeneous = np.linalg.svd(rows)[2][:, -1]
    return homogeneous[:, :3] / homogeneous[:, 3:]


def project(p: np.ndarray, points: np.ndarray) -> np.ndarray:
    h = np.c_[points, np.ones(len(points))] @ p.T
    return h[:, :2] / h[:, 2:]


@dataclass(frozen=True)
class TwoViewObservations:
    """Matched 2D keypoints of one session: (T, J, 2) per view with a usable mask."""

    view1: np.ndarray
    view2: np.ndarray
    usable: np.ndarray            # (T, J)
    size1: tuple[int, int]
    size2: tuple[int, int]
    bones: tuple[tuple[int, int], ...]


def _undistorted(params, obs: TwoViewObservations, t_idx, j_idx):
    x1, x2 = obs.view1[t_idx, j_idx], obs.view2[t_idx, j_idx]
    if len(params) > 7:
        f1, f2 = np.exp(params[:2])
        x1 = undistort_division(x1, obs.size1, f1, params[7])
        x2 = undistort_division(x2, obs.size2, f2, params[8])
    return x1, x2


def _residuals(params, obs: TwoViewObservations, bone_weight: float):
    p1, p2 = projections(params, obs.size1, obs.size2)
    t_idx, j_idx = np.nonzero(obs.usable)
    x1, x2 = _undistorted(params, obs, t_idx, j_idx)
    points = triangulate(p1, p2, x1, x2)
    q1, q2 = project(p1, points), project(p2, points)
    if len(params) > 7:
        f1, f2 = np.exp(params[:2])
        q1 = distort_division(q1, obs.size1, f1, params[7])
        q2 = distort_division(q2, obs.size2, f2, params[8])
    reprojection = np.concatenate(((q1 - obs.view1[t_idx, j_idx]).ravel(), (q2 - obs.view2[t_idx, j_idx]).ravel()))
    grid = np.full(obs.usable.shape + (3,), np.nan)
    grid[t_idx, j_idx] = points
    bone_terms = []
    for a, b in obs.bones:
        lengths = np.linalg.norm(grid[:, a] - grid[:, b], axis=1)
        lengths = lengths[np.isfinite(lengths)]
        if len(lengths) >= 2:
            bone_terms.append(lengths / np.median(lengths) - 1.)
    bones = bone_weight * np.concatenate(bone_terms) if bone_terms else np.zeros(0)
    return np.concatenate((reprojection, bones))


def initial_parameters(obs: TwoViewObservations, focal_guess: float, f: np.ndarray) -> np.ndarray:
    """Pose of camera 2 from E = K2ᵀ F K1 under a focal guess, with the cheirality check."""
    k1, k2 = intrinsics(focal_guess, obs.size1), intrinsics(focal_guess, obs.size2)
    e = k2.T @ f @ k1
    u, _, vt = np.linalg.svd(e)
    u, vt = u * np.sign(np.linalg.det(u)), vt * np.sign(np.linalg.det(vt))
    w = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    t_idx, j_idx = np.nonzero(obs.usable)
    x1, x2 = obs.view1[t_idx, j_idx], obs.view2[t_idx, j_idx]
    best = None
    for rotation in (u @ w @ vt, u @ w.T @ vt):
        for translation in (u[:, 2], -u[:, 2]):
            p1 = k1 @ np.c_[np.eye(3), np.zeros(3)]
            p2 = k2 @ np.c_[rotation, translation]
            points = triangulate(p1, p2, x1, x2)
            depth2 = (points @ rotation.T + translation)[:, 2]
            in_front = np.mean((points[:, 2] > 0) & (depth2 > 0))
            if best is None or in_front > best[0]:
                best = (in_front, rotation, translation)
    _, rotation, translation = best
    theta = np.arccos(np.clip(translation[2], -1, 1))
    phi = np.arctan2(translation[1], translation[0])
    return np.r_[np.log(focal_guess), np.log(focal_guess), Rotation.from_matrix(rotation).as_rotvec(), theta, phi]


@dataclass(frozen=True)
class Calibration:
    focal1: float
    focal2: float
    rotation: np.ndarray       # camera 2 relative to camera 1
    baseline: np.ndarray       # unit vector
    median_reprojection_px: tuple[float, float]
    bone_length_cv: float      # median coefficient of variation of rigid bones
    inlier_fraction: float
    converged: bool
    distortion: tuple[float, float] = (0., 0.)   # division-model coefficient per camera

    def undistort(self, view1: np.ndarray, view2: np.ndarray, size1, size2):
        return (undistort_division(view1, size1, self.focal1, self.distortion[0]),
                undistort_division(view2, size2, self.focal2, self.distortion[1]))

    def matrices(self, size1, size2):
        p1 = intrinsics(self.focal1, size1) @ np.c_[np.eye(3), np.zeros(3)]
        p2 = intrinsics(self.focal2, size2) @ np.c_[self.rotation, self.baseline]
        return p1, p2


def bone_length_cv(points: np.ndarray, bones) -> float:
    """Median over bones of the coefficient of variation of length across frames."""
    values = []
    for a, b in bones:
        lengths = np.linalg.norm(points[:, a] - points[:, b], axis=1)
        lengths = lengths[np.isfinite(lengths)]
        if len(lengths) >= 2 and np.mean(lengths) > 0:
            values.append(np.std(lengths) / np.mean(lengths))
    return float(np.median(values)) if values else float("nan")


def calibrate(obs: TwoViewObservations, *, focal_guess: float | None = None, bone_weight: float = 50.,
              ransac_threshold: float = 3., seed: int = 0, distortion: bool = False,
              anchors: tuple[int, ...] = ()) -> Calibration:
    """Self-calibrate a camera pair; robust start, then bone-constrained bundle adjustment.

    With ``distortion`` the pinhole solution is refined once more with one division-model
    coefficient per camera, started at zero. ``anchors`` are extra view-consistent
    joints (not on a bone) allowed into the reprojection term.
    """
    if obs.view1.shape != obs.view2.shape or obs.usable.shape != obs.view1.shape[:2]:
        raise ValueError("views and mask must share (T, J)")
    # Only joints on a rigid bone (and the anchors) take part: on real data the remaining keypoints
    # (face contour, feet, ...) are not view-consistent — on CMU Panoptic under the
    # dome's own calibration their median two-view error is 6 px against 1.2 px for
    # arm and hand joints, and they would outnumber the bone joints four to one.
    on_bone = np.zeros(obs.usable.shape[1], bool)
    on_bone[[j for bone in obs.bones for j in bone] + list(anchors)] = True
    usable = obs.usable & on_bone
    t_idx, j_idx = np.nonzero(usable)
    x1, x2 = obs.view1[t_idx, j_idx], obs.view2[t_idx, j_idx]
    f, inliers = ransac_fundamental(x1, x2, threshold=ransac_threshold, seed=seed)
    usable[t_idx[~inliers], j_idx[~inliers]] = False
    clean = TwoViewObservations(obs.view1, obs.view2, usable, obs.size1, obs.size2, obs.bones)
    guess = focal_guess or float(max(obs.size1))
    start = initial_parameters(clean, guess, f)
    width = float(max(obs.size1))
    lower = np.r_[np.log(.2 * width), np.log(.2 * width), np.full(5, -np.inf)]
    upper = np.r_[np.log(10 * width), np.log(10 * width), np.full(5, np.inf)]
    start[:2] = np.clip(start[:2], lower[:2] + 1e-6, upper[:2] - 1e-6)
    result = least_squares(_residuals, start, args=(clean, bone_weight), method="trf", loss="soft_l1",
                           f_scale=2., x_scale="jac", max_nfev=400, bounds=(lower, upper))
    if distortion:
        result = least_squares(_residuals, np.r_[result.x, 0., 0.], args=(clean, bone_weight), method="trf",
                               loss="soft_l1", f_scale=2., x_scale="jac", max_nfev=400,
                               bounds=(np.r_[lower, -1., -1.], np.r_[upper, 1., 1.]))
    p1, p2 = projections(result.x, obs.size1, obs.size2)
    t_idx, j_idx = np.nonzero(usable)
    x1, x2 = _undistorted(result.x, obs, t_idx, j_idx)
    points = triangulate(p1, p2, x1, x2)
    grid = np.full(usable.shape + (3,), np.nan)
    grid[t_idx, j_idx] = points
    q1, q2 = project(p1, points), project(p2, points)
    if distortion:
        q1 = distort_division(q1, obs.size1, np.exp(result.x[0]), result.x[7])
        q2 = distort_division(q2, obs.size2, np.exp(result.x[1]), result.x[8])
    errors1 = np.linalg.norm(q1 - obs.view1[t_idx, j_idx], axis=1)
    errors2 = np.linalg.norm(q2 - obs.view2[t_idx, j_idx], axis=1)
    rotation = Rotation.from_rotvec(result.x[2:5]).as_matrix()
    theta, phi = result.x[5:7]
    return Calibration(float(np.exp(result.x[0])), float(np.exp(result.x[1])), rotation,
                       np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]),
                       (float(np.median(errors1)), float(np.median(errors2))), bone_length_cv(grid, obs.bones),
                       float(inliers.mean()), bool(result.success),
                       (float(result.x[7]), float(result.x[8])) if distortion else (0., 0.))


def triangulate_session(calibration: Calibration, obs: TwoViewObservations) -> np.ndarray:
    """(T, J, 3) points in camera-1 coordinates (unit baseline); NaN where unusable."""
    p1, p2 = calibration.matrices(obs.size1, obs.size2)
    view1, view2 = calibration.undistort(obs.view1, obs.view2, obs.size1, obs.size2)
    t_idx, j_idx = np.nonzero(obs.usable)
    grid = np.full(obs.usable.shape + (3,), np.nan)
    grid[t_idx, j_idx] = triangulate(p1, p2, view1[t_idx, j_idx], view2[t_idx, j_idx])
    return grid


def view_offset(signal1: np.ndarray, signal2: np.ndarray, *, max_lag: int = 15) -> tuple[int, float]:
    """Integer lag maximising the normalised correlation of two motion signals.

    Returns (lag, correlation) such that ``signal2[t + lag]`` best matches
    ``signal1[t]``; NaN samples are ignored pairwise.
    """
    best = (0, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        a = signal1[max(0, -lag):len(signal1) - max(0, lag)]
        b = signal2[max(0, lag):len(signal2) - max(0, -lag)][:len(a)]
        a = a[:len(b)]
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.sum() < 10 or a[finite].std() == 0 or b[finite].std() == 0:
            continue
        r = float(np.corrcoef(a[finite], b[finite])[0, 1])
        if r > best[1]:
            best = (lag, r)
    return best


__all__ = ["intrinsics", "eight_point", "sampson", "ransac_fundamental", "triangulate", "project",
           "TwoViewObservations", "Calibration", "calibrate", "triangulate_session", "bone_length_cv",
           "view_offset", "initial_parameters", "projections", "undistort_division", "distort_division"]
