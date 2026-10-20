"""Index pointing, pointing rays, and referent loci in signing space.

**Pointing frames.** A frame of the dominant hand is a point when (i) the index
finger is extended — its three segments are nearly collinear and its tip is as far
from its knuckle as the finger is long — (ii) the middle, ring and little fingers
are flexed (each tip within 70% of its finger length of its knuckle), (iii) the wrist is nearly still for at least
``MIN_HOLD`` consecutive frames, and (iv) the wrist is forward of the chest. The
pointing direction is the unit vector from the index MCP to the fingertip.

**Least-squares locus of several pointings.** Rays ``r_i(s) = p_i + s d_i`` meet (in the
least-squares sense) at ``x* = (Σ A_i)⁻¹ Σ A_i p_i`` with ``A_i = I − d_i d_iᵀ``, the
point minimising the summed squared distances to the rays; it is ill-posed when the
rays are parallel, so the direction-only locus ``normalise(Σ d_i)`` is also returned.

**DP-vMF-means clustering of directions** (small-variance limit of a Dirichlet-process
mixture of von Mises–Fisher distributions; Straub et al. 2015, AISTATS): with cosine
threshold ``λ = cos(radius)``, each direction joins the most similar centroid when
the similarity exceeds ``λ`` and otherwise opens a new cluster; centroids are the
normalised sums of their members; the objective ``Σ_i (1 − d_iᵀμ_{z_i}) + (1 − λ)·K``
never increases across the alternating steps, so the iteration terminates. The
number of loci is therefore inferred, not fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import azimuth_elevation, unit

MIN_HOLD = 3                      # frames
STRAIGHT_COS = .85                # cosine between successive index segments for "extended"
REACH = .9                        # |tip − MCP| / finger length for "extended"
CURL = .7                         # |tip − MCP| / finger length below which a finger is "flexed"
MAX_WRIST_SPEED = 1.5             # shoulder widths per second for "still"
LOCUS_RADIUS_DEG = 15.            # DP-vMF-means cluster radius (Straub et al.'s λ = cos radius)


def _finger(joints, positions, side, name):
    idx = [joints.index(f"{side}Hand")] + [joints.index(f"{side}Hand{name}{k}") for k in (1, 2, 3, 4)]
    return positions[:, idx]                                  # (T, 5, 3) wrist, MCP, PIP, DIP, tip


def extension(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(straightness, reach) of a finger: the minimum cosine between its successive MCP→PIP→DIP→tip
    segments, and |tip − MCP| over the summed segment lengths (1 when extended, small when curled)."""
    steps = np.diff(chain[:, 1:], axis=1)                     # three segments MCP→PIP→DIP→tip
    seg = unit(steps)
    straight = np.minimum((seg[:, 0] * seg[:, 1]).sum(-1), (seg[:, 1] * seg[:, 2]).sum(-1))
    path = np.linalg.norm(steps, axis=-1).sum(-1)
    reach = np.linalg.norm(chain[:, -1] - chain[:, 1], axis=-1) / np.where(path > 0, path, 1.)
    return straight, reach


def pointing_frames(joints, positions: np.ndarray, valid: np.ndarray, times: np.ndarray, side: str) -> np.ndarray:
    """(T,) bool: frames where hand ``side`` holds an index point."""
    index = _finger(joints, positions, side, "Index")
    straight, reach = extension(index)
    extended = (straight >= STRAIGHT_COS) & (reach >= REACH)
    flexed = np.ones(len(positions), bool)
    for name in ("Middle", "Ring", "Pinky"):
        _, other = extension(_finger(joints, positions, side, name))
        flexed &= other < CURL
    wrist = positions[:, joints.index(f"{side}Hand")]
    speed = frame_speed(wrist, times)
    forward = wrist[:, 2] > 0
    ok = valid[:, joints.index(f"{side}HandIndex4")] & valid[:, joints.index(f"{side}Hand")]
    raw = extended & flexed & (speed <= MAX_WRIST_SPEED) & forward & ok
    return _min_run(raw, MIN_HOLD)


def frame_speed(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    """(T,) speed at each frame: the mean of the adjacent finite-difference speeds (one-sided at the ends)."""
    t = len(points)
    if t < 2:
        return np.zeros(t)
    dt = np.diff(times)
    v = np.linalg.norm(np.diff(points, axis=0), axis=-1) / np.where(dt > 0, dt, np.inf)
    total, count = np.zeros(t), np.zeros(t)
    total[1:] += v; total[:-1] += v
    count[1:] += 1; count[:-1] += 1
    return total / count


def _min_run(mask: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros_like(mask)
    start = None
    for t, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = t
        elif not m and start is not None:
            if t - start >= length:
                out[start:t] = True
            start = None
    return out


@dataclass(frozen=True)
class PointingEvent:
    start: int                    # first frame (inclusive)
    stop: int                     # last frame + 1
    origin: np.ndarray            # (3,) mean index MCP, body frame
    direction: np.ndarray         # (3,) unit mean pointing direction, body frame


def pointing_events(joints, positions, valid, times, side) -> list[PointingEvent]:
    mask = pointing_frames(joints, positions, valid, times, side)
    index = _finger(joints, positions, side, "Index")
    events, t = [], 0
    while t < len(mask):
        if mask[t]:
            u = t
            while u < len(mask) and mask[u]:
                u += 1
            d = unit(index[t:u, 4] - index[t:u, 1]).sum(0)
            events.append(PointingEvent(t, u, index[t:u, 1].mean(0), unit(d)))
            t = u
        else:
            t += 1
    return events


def least_squares_locus(origins: np.ndarray, directions: np.ndarray, *, ridge: float = 1e-9):
    """(x*, condition number) of the point nearest all rays; x* is None when the rays are parallel."""
    d = unit(directions)
    A = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    M, b = A.sum(0), np.einsum("nij,nj->i", A, origins)
    eig = np.linalg.eigvalsh(M)
    condition = float(eig[-1] / max(eig[0], 1e-15))
    if eig[0] < 1e-6 * max(eig[-1], 1e-12):
        return None, condition
    return np.linalg.solve(M + ridge * np.eye(3), b), condition


def dp_vmf_means(directions: np.ndarray, *, radius_deg: float = LOCUS_RADIUS_DEG, max_iter: int = 100):
    """(labels, centroids, objective) of DP-vMF-means on unit vectors (Straub et al. 2015)."""
    d = unit(np.asarray(directions, float))
    lam = np.cos(np.radians(radius_deg))
    if len(d) == 0:
        return np.zeros(0, int), np.zeros((0, 3)), 0.
    centroids = [d[0]]
    labels = np.zeros(len(d), int)
    previous = np.inf
    for _ in range(max_iter):
        for i, x in enumerate(d):                              # assignment, opening clusters as needed
            sims = np.array([x @ c for c in centroids])
            k = int(np.argmax(sims))
            if sims[k] >= lam:
                labels[i] = k
            else:
                centroids.append(x)
                labels[i] = len(centroids) - 1
        used = np.unique(labels)                               # drop empty clusters, relabel
        remap = {k: j for j, k in enumerate(used)}
        labels = np.array([remap[k] for k in labels])
        centroids = [unit(d[labels == j].sum(0)) for j in range(len(used))]
        objective = float(sum(1 - d[i] @ centroids[labels[i]] for i in range(len(d))) + (1 - lam) * len(centroids))
        if objective >= previous - 1e-12:
            break
        previous = objective
    return labels, np.array(centroids), objective


@dataclass(frozen=True)
class Locus:
    direction: np.ndarray         # unit, body frame
    azimuth_deg: float
    elevation_deg: float
    point: np.ndarray | None      # least-squares ray intersection, shoulder widths, if well posed
    events: tuple[int, ...]       # indices of the pointing events assigned to it


def loci_from_events(events: list[PointingEvent], *, radius_deg: float = LOCUS_RADIUS_DEG) -> list[Locus]:
    if not events:
        return []
    dirs = np.stack([e.direction for e in events])
    labels, centroids, _ = dp_vmf_means(dirs, radius_deg=radius_deg)
    out = []
    for k, c in enumerate(centroids):
        members = tuple(int(i) for i in np.nonzero(labels == k)[0])
        point = None
        if len(members) >= 2:
            point, _ = least_squares_locus(np.stack([events[i].origin for i in members]), dirs[list(members)])
        az, el = np.degrees(azimuth_elevation(c))
        out.append(Locus(c, float(az), float(el), point, members))
    return out


__all__ = ["pointing_frames", "pointing_events", "PointingEvent", "least_squares_locus", "dp_vmf_means",
           "Locus", "loci_from_events", "extension", "frame_speed"]
