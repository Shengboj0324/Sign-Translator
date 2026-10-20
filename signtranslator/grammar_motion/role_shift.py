"""Role shift (constructed action) from torso, head and gaze.

**Features per frame** (degrees; z-scored against the clip's narrator baseline):
torso yaw ψ_T (azimuth of the chest's forward axis in the camera-aligned frame), head
yaw relative to the torso ψ_H − ψ_T, lateral torso position x_root (metres, when the
state carries metric translation) and the angle α_A between gaze and the addressee
direction (the camera, +z). The literature gives no ASL torso angles for role shift
(reports/ASL grammar and spatial reference.md), so shifted spans are defined
relative to the signer's own narrator baseline, not by absolute angles.

**Change points by exact PELT** (Killick, Fearnhead & Eckley 2012): minimise
``Σ_segments C(y_{a:b}) + β·(#segments)`` with the Gaussian mean-change cost
``C = Σ_t ‖y_t − ȳ_{a:b}‖²`` (features standardised), β = ``PENALTY``·log T (BIC-like);
pruning discards a candidate τ once ``F(τ) + C(y_{τ:t}) > F(t)``, which keeps the search
exact. A segment is *shifted* when its mean feature vector departs from the narrator
baseline (the clip's median) by more than ``SHIFT_Z`` robust σ and ``MIN_SHIFT_DEG``
degrees in any of torso yaw, relative head yaw or gaze-to-addressee angle, and lasts
at least ``MIN_SHIFT_S``; its
direction is the sign of its torso/head yaw (toward the signer's left or right).
Head and eyes lead the torso in role shift (McDonald et al.), so a shift's onset is
the earliest change point among the features' own segmentations within it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..generation.motion_metrics import _matrices
from .frames import chest_rotation

PENALTY = 3.
SHIFT_Z = 2.5
MIN_SHIFT_DEG = 5.                # the segment must also turn this far in absolute terms
MIN_SHIFT_S = .6                  # a quote or clause; shorter departures are head movement, not role shift
MIN_SEGMENT = 4


def pelt(y: np.ndarray, penalty: float, min_size: int = MIN_SEGMENT) -> list[int]:
    """Exact PELT change points (segment start indices after 0) for the Gaussian mean cost."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    cs = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(y, 0)])
    cs2 = np.r_[0., np.cumsum((y ** 2).sum(1))]

    def cost(a, b):                            # Σ‖y − mean‖² over y[a:b]
        m = b - a
        s = cs[b] - cs[a]
        return cs2[b] - cs2[a] - (s @ s) / m

    F = np.full(n + 1, np.inf); F[0] = -penalty
    last = np.zeros(n + 1, int)
    candidates = [0]
    for t in range(min_size, n + 1):
        options = [(F[s] + cost(s, t) + penalty, s) for s in candidates if t - s >= min_size]
        if options:
            F[t], last[t] = min(options)
        keep = [s for s in candidates if t - s < min_size or F[s] + cost(s, t) <= F[t]]
        candidates = keep + [t - min_size + 1] if t - min_size + 1 > 0 else keep
    points, t = [], n
    while t > 0:
        s = last[t]
        if s > 0:
            points.append(int(s))
        t = s
    return sorted(points)


def features(state, gaze: np.ndarray | None = None):
    """(T, k) feature matrix in degrees/metres, (T,) validity, and the feature names."""
    chest = chest_rotation(state)
    forward = chest[:, :, 2]
    torso = np.degrees(np.arctan2(forward[:, 0], forward[:, 2]))
    head = state.channels["head_rotation"]
    body = state.channels["body"]
    labels = list(body.labels)
    ok = body.valid[:, labels.index("Pelvis")] & body.valid[:, labels.index("Chest")] & head.valid[:, 0]  # legs are off-screen
    identity = np.array([1., 0., 0., 0., 1., 0.])
    hf = _matrices(np.where(head.valid[:, :1], head.values[:, 0], identity))[:, :, 2]
    head_yaw = np.degrees(np.arctan2(hf[:, 0], hf[:, 2]))
    cols, names = [torso, head_yaw - torso], ["torso_yaw_deg", "head_minus_torso_yaw_deg"]
    root = state.channels["root_translation"]
    if root.valid.any():
        cols.append(np.where(root.valid[:, 0], root.values[:, 0, 0], 0.)); names.append("torso_x_m")
        ok = ok & root.valid[:, 0]
    if gaze is not None:
        g_ok = np.isfinite(gaze).all(1) & (np.linalg.norm(gaze, axis=1) > 0)
        cols.append(np.degrees(np.arccos(np.clip(np.where(g_ok, gaze[:, 2], 1.), -1, 1)))); names.append("gaze_from_addressee_deg")
        ok = ok & g_ok
    return np.stack(cols, 1), ok, tuple(names)


@dataclass(frozen=True)
class RoleShift:
    start: int
    stop: int
    direction: str                # 'left' or 'right' (signer's), from the torso/head yaw sign
    deviation_z: dict             # feature -> mean z of the segment


def role_shifts(state, fps: float, gaze: np.ndarray | None = None) -> tuple[list[RoleShift], dict]:
    x, ok, names = features(state, gaze)
    report = dict(features=names, change_points=[], frames=int(ok.sum()))
    if ok.sum() < 2 * MIN_SEGMENT:
        return [], report
    idx = np.nonzero(ok)[0]
    v = x[idx]
    med = np.median(v, 0)
    floors = {"torso_yaw_deg": 3., "head_minus_torso_yaw_deg": 3., "torso_x_m": .01, "gaze_from_addressee_deg": 4.}
    sigma = np.maximum(1.4826 * np.median(np.abs(v - med), 0), [floors[n] for n in names])
    z = (v - med) / sigma
    cps = pelt(z, PENALTY * np.log(len(z)))
    report["change_points"] = [int(idx[c]) for c in cps]
    bounds = [0] + cps + [len(z)]
    shifts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mean = z[a:b].mean(0)
        dev = {n: float(m) for n, m in zip(names, mean)}
        angular = [abs(dev[n]) for n in names if n != "torso_x_m"]
        degrees = np.abs(v[a:b].mean(0) - med)[[i for i, n in enumerate(names) if n != "torso_x_m"]]
        if (max(angular) >= SHIFT_Z and degrees.max() >= MIN_SHIFT_DEG
                and (idx[b - 1] + 1 - idx[a]) / fps >= MIN_SHIFT_S):
            yaw = v[a:b, 0].mean() - med[0] + v[a:b, 1].mean() - med[1]
            shifts.append(RoleShift(int(idx[a]), int(idx[b - 1]) + 1, "left" if yaw > 0 else "right", dev))
    return shifts, report


__all__ = ["pelt", "features", "RoleShift", "role_shifts"]
