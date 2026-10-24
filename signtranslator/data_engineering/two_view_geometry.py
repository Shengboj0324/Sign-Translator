"""Two-view self-calibration and triangulation using the signer as calibration object.

How2Sign recorded each session with a frontal and a lateral HD camera (1280×720,
24 fps H.264, measured on the acquired clips) whose calibration was not released. This module recovers both cameras from
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
    f = np.linalg.svd(design, full_matrices=len(design) < 9)[2][-1].reshape(3, 3)   # all of V, never the n x n U (16 GB at 45k)
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


def _eight_point_batch(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """:func:`eight_point` on a batch of minimal samples, (B, 8, 2) each -> (B, 3, 3)."""
    def normalise(points):
        centre = points.mean(axis=1)
        scale = np.sqrt(2) / np.mean(np.linalg.norm(points - centre[:, None], axis=2), axis=1)
        transform = np.zeros((len(points), 3, 3))
        transform[:, 0, 0] = transform[:, 1, 1] = scale
        transform[:, :2, 2] = -scale[:, None] * centre
        transform[:, 2, 2] = 1.
        homogeneous = np.concatenate((points, np.ones(points.shape[:2] + (1,))), axis=2)
        return homogeneous @ transform.transpose(0, 2, 1), transform
    a, t1 = normalise(x1)
    b, t2 = normalise(x2)
    design = np.einsum("bni,bnj->bnij", b, a).reshape(len(a), a.shape[1], 9)
    f = np.linalg.svd(design)[2][:, -1].reshape(-1, 3, 3)
    u, s, vt = np.linalg.svd(f)
    s[:, 2] = 0.
    f = (u * s[:, None, :]) @ vt
    f = t2.transpose(0, 2, 1) @ f @ t1
    return f / np.linalg.norm(f, axis=(1, 2), keepdims=True)


def _sampson_batch(f: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """:func:`sampson` for a batch of F, (B, 3, 3) -> (B, N)."""
    a = np.c_[x1, np.ones(len(x1))]
    b = np.c_[x2, np.ones(len(x2))]
    fa, ftb = a @ f.transpose(0, 2, 1), b @ f
    numerator = np.einsum("nj,bnj->bn", b, fa) ** 2
    return np.sqrt(numerator / (fa[..., 0] ** 2 + fa[..., 1] ** 2 + ftb[..., 0] ** 2 + ftb[..., 1] ** 2))


def ransac_fundamental(x1, x2, *, threshold: float = 2., iterations: int = 2000, seed: int = 0):
    """Robust F and inlier mask; the final F is refit on all inliers.

    The hypotheses are the same minimal samples, drawn in the same order, as a loop of
    :func:`eight_point` fits; they are solved and scored in batches (vectorised SVDs), and
    the first hypothesis with the most inliers wins, as in the loop.
    """
    rng = np.random.default_rng(seed)
    picks = np.stack([rng.choice(len(x1), 8, replace=False) for _ in range(iterations)]) if iterations else None
    best, best_count = None, -1
    chunk = max(1, min(iterations, 2_000_000 // max(len(x1), 1)))
    for begin in range(0, iterations, chunk):
        pick = picks[begin:begin + chunk]
        with np.errstate(invalid="ignore", divide="ignore"):
            f = _eight_point_batch(x1[pick], x2[pick])
            counts = (_sampson_batch(f, x1, x2) < threshold).sum(axis=1)
        finite = np.isfinite(f).all(axis=(1, 2))
        counts = np.where(finite, counts, -1)
        k = int(np.argmax(counts))
        if counts[k] > best_count:
            best, best_count = f[k], int(counts[k])
    if best is None or best_count < 8:
        raise ValueError("no fundamental matrix with at least 8 inliers")
    inliers = sampson(best, x1, x2) < threshold
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


def triangulate(p1: np.ndarray, p2: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                w1: np.ndarray | None = None, w2: np.ndarray | None = None) -> np.ndarray:
    """Linear (DLT) triangulation of many points at once; returns (N, 3).

    Optional per-point weights scale each view's two equations (e.g. keypoint
    confidences), so a joint seen poorly in one view is placed mostly by the other.
    """
    rows = np.stack((x1[:, 0, None] * p1[2] - p1[0], x1[:, 1, None] * p1[2] - p1[1],
                     x2[:, 0, None] * p2[2] - p2[0], x2[:, 1, None] * p2[2] - p2[1]), axis=1)
    if w1 is not None:
        # Rows are normalised so the weights, not the projections' scale, set each view's say.
        rows = rows / np.linalg.norm(rows, axis=2, keepdims=True) * np.stack((w1, w1, w2, w2), axis=1)[..., None]
    homogeneous = np.linalg.svd(rows)[2][:, -1]
    return homogeneous[:, :3] / homogeneous[:, 3:]


def project(p: np.ndarray, points: np.ndarray) -> np.ndarray:
    h = np.c_[points, np.ones(len(points))] @ p.T
    with np.errstate(divide="ignore", invalid="ignore"):   # a point on the camera plane projects to inf / NaN
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


def _cleaned(obs: TwoViewObservations, *, ransac_threshold: float, seed: int, anchors, ransac_joints):
    """Observations restricted to bone joints (and anchors) that agree with a robust epipolar start."""
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
    if ransac_joints is None:
        f, inliers = ransac_fundamental(x1, x2, threshold=ransac_threshold, seed=seed)
    else:
        subset = np.isin(j_idx, ransac_joints)
        f, _ = ransac_fundamental(x1[subset], x2[subset], threshold=ransac_threshold, seed=seed)
        inliers = sampson(f, x1, x2) < ransac_threshold
    usable[t_idx[~inliers], j_idx[~inliers]] = False
    return TwoViewObservations(obs.view1, obs.view2, usable, obs.size1, obs.size2, obs.bones), f, float(inliers.mean())


def _calibration(params, obs: TwoViewObservations, inlier_fraction: float, converged: bool) -> Calibration:
    """The Calibration described by one pair's parameter vector, with its diagnostics on ``obs``."""
    distortion = len(params) > 7
    p1, p2 = projections(params, obs.size1, obs.size2)
    t_idx, j_idx = np.nonzero(obs.usable)
    x1, x2 = _undistorted(params, obs, t_idx, j_idx)
    points = triangulate(p1, p2, x1, x2)
    grid = np.full(obs.usable.shape + (3,), np.nan)
    grid[t_idx, j_idx] = points
    q1, q2 = project(p1, points), project(p2, points)
    if distortion:
        q1 = distort_division(q1, obs.size1, np.exp(params[0]), params[7])
        q2 = distort_division(q2, obs.size2, np.exp(params[1]), params[8])
    errors1 = np.linalg.norm(q1 - obs.view1[t_idx, j_idx], axis=1)
    errors2 = np.linalg.norm(q2 - obs.view2[t_idx, j_idx], axis=1)
    theta, phi = params[5:7]
    return Calibration(float(np.exp(params[0])), float(np.exp(params[1])), Rotation.from_rotvec(params[2:5]).as_matrix(),
                       np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]),
                       (float(np.median(errors1)), float(np.median(errors2))), bone_length_cv(grid, obs.bones),
                       inlier_fraction, converged, (float(params[7]), float(params[8])) if distortion else (0., 0.))


def _focal_bounds(size):
    width = float(max(size))
    return np.log(.2 * width), np.log(10 * width)


def calibrate(obs: TwoViewObservations, *, focal_guess: float | None = None, bone_weight: float = 50.,
              ransac_threshold: float = 3., seed: int = 0, distortion: bool = False,
              anchors: tuple[int, ...] = (), ransac_joints: tuple[int, ...] | None = None,
              intrinsics: tuple[float, float, float, float] | None = None) -> Calibration:
    """Self-calibrate a camera pair; robust start, then bone-constrained bundle adjustment.

    With ``intrinsics`` = (f1, f2, λ1, λ2) — e.g. from :func:`calibrate_jointly` on many
    sessions of the same cameras — only this session's relative pose is estimated.

    With ``distortion`` the pinhole solution is refined once more with one division-model
    coefficient per camera, started at zero. ``anchors`` are extra view-consistent
    joints (not on a bone) allowed into the reprojection term. With ``ransac_joints``
    the robust epipolar start is fitted on those joints only (the most view-consistent
    ones), and every other observation is kept or dropped by its Sampson distance to
    that geometry — inconsistent keypoints (e.g. occluded fingers in a profile view)
    can then neither bias the start nor enter the bundle adjustment.
    """
    clean, f, inlier_fraction = _cleaned(obs, ransac_threshold=ransac_threshold, seed=seed, anchors=anchors,
                                         ransac_joints=ransac_joints)
    if intrinsics is not None:
        f1, f2, l1, l2 = intrinsics
        fixed = lambda pose: np.r_[np.log(f1), np.log(f2), pose, l1, l2]
        pose0 = initial_parameters(clean, float(np.sqrt(f1 * f2)), f)[2:7]
        result = least_squares(lambda pose: _residuals(fixed(pose), clean, bone_weight), pose0, method="trf",
                               loss="soft_l1", f_scale=2., x_scale="jac", max_nfev=400)
        return _calibration(fixed(result.x), clean, inlier_fraction, bool(result.success))
    start = initial_parameters(clean, focal_guess or float(max(obs.size1)), f)
    low, high = _focal_bounds(obs.size1)
    lower, upper = np.r_[low, low, np.full(5, -np.inf)], np.r_[high, high, np.full(5, np.inf)]
    start[:2] = np.clip(start[:2], lower[:2] + 1e-6, upper[:2] - 1e-6)
    result = least_squares(_residuals, start, args=(clean, bone_weight), method="trf", loss="soft_l1",
                           f_scale=2., x_scale="jac", max_nfev=400, bounds=(lower, upper))
    if distortion:
        result = least_squares(_residuals, np.r_[result.x, 0., 0.], args=(clean, bone_weight), method="trf",
                               loss="soft_l1", f_scale=2., x_scale="jac", max_nfev=400,
                               bounds=(np.r_[lower, -1., -1.], np.r_[upper, 1., 1.]))
    return _calibration(result.x, clean, inlier_fraction, bool(result.success))


def calibrate_jointly(sessions: list[TwoViewObservations], *, focal_guess: float | None = None,
                      bone_weight: float = 50., ransac_threshold: float = 3., seed: int = 0,
                      distortion: bool = False, anchors: tuple[int, ...] = (),
                      ransac_joints: tuple[int, ...] | None = None) -> list[Calibration]:
    """Calibrate several sessions recorded by the same two cameras at once.

    The intrinsics — both focal lengths and, with ``distortion``, both radial
    coefficients — are shared by every session; each session keeps its own relative
    pose (the cameras may be moved between sessions). Pooling sessions multiplies the
    pose variety that the bone-length term needs to resolve the focal lengths of
    fixating cameras, which one short session may not provide.
    """
    cleaned = [_cleaned(o, ransac_threshold=ransac_threshold, seed=seed, anchors=anchors,
                        ransac_joints=ransac_joints) for o in sessions]
    guess = focal_guess or float(max(sessions[0].size1))
    starts = [initial_parameters(c, guess, f) for c, f, _ in cleaned]
    low, high = _focal_bounds(sessions[0].size1)

    def unpack(x, n_shared):
        shared, poses = x[:n_shared], x[n_shared:].reshape(len(sessions), 5)
        return [np.r_[shared[:2], pose, shared[2:]] for pose in poses]

    def residuals(x, n_shared):
        return np.concatenate([_residuals(params, c, bone_weight)
                               for params, (c, _, _) in zip(unpack(x, n_shared), cleaned)])

    lower = np.r_[low, low, np.full(5 * len(sessions), -np.inf)]
    upper = np.r_[high, high, np.full(5 * len(sessions), np.inf)]
    x0 = np.r_[np.clip(np.log(guess), low + 1e-6, high - 1e-6), np.clip(np.log(guess), low + 1e-6, high - 1e-6),
               np.concatenate([s[2:7] for s in starts])]
    result = least_squares(residuals, x0, args=(2,), method="trf", loss="soft_l1", f_scale=2., x_scale="jac",
                           max_nfev=400, bounds=(lower, upper))
    n_shared = 2
    if distortion:                       # refine with one radial coefficient per camera, started at zero
        n_shared = 4
        x0 = np.r_[result.x[:2], 0., 0., result.x[2:]]
        lower = np.r_[lower[:2], -1., -1., lower[2:]]
        upper = np.r_[upper[:2], 1., 1., upper[2:]]
        result = least_squares(residuals, x0, args=(4,), method="trf", loss="soft_l1", f_scale=2., x_scale="jac",
                               max_nfev=400, bounds=(lower, upper))
    return [_calibration(params, c, inlier, bool(result.success))
            for params, (c, _, inlier) in zip(unpack(result.x, n_shared), cleaned)]


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
           "TwoViewObservations", "Calibration", "calibrate", "calibrate_jointly", "triangulate_session", "bone_length_cv",
           "view_offset", "initial_parameters", "projections", "undistort_division", "distort_division"]
