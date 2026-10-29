"""Composition beyond joining isolated signs, part 1 (docs/NEXT_PHASE_PLAN.md, "Next lever after the oracle finding",
2026-10-29 17:38 UTC, options 2 and 5): one signing space per sentence, minimum-jerk bridges, tightened cores. No
training; deterministic; every knob off by default (the composition path is then unchanged).

**Space normalisation** (``space_normalise``) works on the canonical rotations, in the chest frame of the forward
kinematics (``canonical_skeleton``: nominal bone lengths, the wrist at shoulder + R_upper (0.28 rest_upper) + R_upper
R_fore (0.25 rest_fore)), per word clip of a composition:
1. **torso**: the word's mean torso orientation M_k (the chordal mean of Pelvis x Chest over its core's valid frames)
   is moved to the target M* (the chordal mean over the sentence's cores, or How2Sign train's): every frame's Pelvis
   rotation is pre-multiplied by M* M_k^T, so the word's own torso motion is kept;
2. **dominant-hand centroid**: on the sentence's dominant side (the side whose wrist height in the chest frame varies
   most over the cores), the word's mean shoulder-to-wrist direction m_k is rotated onto the target m* (the normalised
   mean over the sentence's cores, or How2Sign train's) by the minimal rotation Q_k, applied to the upper arm in the
   chest frame (the whole arm turns rigidly about the shoulder: the word's hand trajectory keeps its shape);
3. **excursion**: the sentence's RMS angle of the dominant wrist direction about m* (over its cores, after step 2) is
   scaled to How2Sign train's median clip RMS angle (``stats``): each frame's direction is turned along its great
   circle from m* by (s - 1) x its angle, s = target / current clipped to ``scale_clip``.
Steps 1-2 use the sentence's own targets with ``target="sentence"`` and How2Sign train's with ``target="how2sign"``;
step 3 always needs the statistics (``space_statistics``, from How2Sign *train* clips only).

**Minimum-jerk bridges** (``minimum_jerk_bridge``): the frames between two cores follow the quintic with the cores'
boundary values and velocities (per-frame differences of the resampled cores' last two and first two frames) and zero
boundary accelerations: p(t) = H0 p0 + H1 T v0 + H4 T v1 + H5 p1, t = j / T, T = frames + 1, with H0 = 1 - 10t^3 +
15t^4 - 6t^5, H1 = t - 6t^3 + 8t^4 - 3t^5, H4 = -4t^3 + 7t^4 - 3t^5, H5 = 10t^3 - 15t^4 + 6t^5. With velocities equal to
the chord (p1 - p0) / T it is the linear bridge exactly. Values are put back on each channel's manifold as the linear
bridge does; a channel valid on one side only takes that side.

**Tightening** (``tighten_window``): a core whose window came from the median sign position (reasons ``entry`` /
``all``) is cut again by the activity rule (``compositional.core_window``) inside that window; never extended (the
result lies inside the window), and kept as it was when the rule cannot cut.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..data_engineering.video_landmarks import BODY_JOINTS
from .canonical_skeleton import _LENGTH
from .motion_metrics import _matrices

SIDES = ("Left", "Right")
SCALE_CLIP = (.5, 2.)
TARGETS = ("sentence", "how2sign")


# ---- rotations ---------------------------------------------------------------------------------------------------

def to_6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6), the first two columns (``pose.rotations.matrix_to_rotation_6d``)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def chordal_mean(R: np.ndarray) -> np.ndarray:
    """The rotation nearest the mean of (N, 3, 3) rotations (SVD projection, det +1)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).mean(0))
    D = np.diag([1., 1., np.sign(np.linalg.det(U @ Vt)) or 1.])
    return U @ D @ Vt


def axis_angle(axis: np.ndarray, angle) -> np.ndarray:
    """(..., 3, 3) rotations about unit ``axis`` (..., 3) by ``angle`` (...) radians (Rodrigues)."""
    axis = np.asarray(axis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    K = np.zeros(axis.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -z, y, z, -x, -y, x
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` to unit vector ``b`` (antiparallel: half a turn about an axis
    perpendicular to ``a``)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.clip(a @ b, -1., 1.))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1., 0., 0.] if abs(a[0]) < .9 else [0., 1., 0.])
        return axis_angle(perp / np.linalg.norm(perp), np.pi)
    return axis_angle(axis / s, np.arctan2(s, c))


# ---- kinematics in the chest frame ---------------------------------------------------------------------------------

def _rest(name: str) -> np.ndarray:
    return np.asarray(next(j for j in BODY_JOINTS if j.name == name).rest, dtype=np.float64)


def _body(state):
    body = state.channels["body"]
    return body, {label: k for k, label in enumerate(body.labels)}


def wrist_from_shoulder(state, side: str) -> tuple[np.ndarray, np.ndarray]:
    """((T, 3) wrist minus shoulder in the chest frame, (T,) valid: upper arm and forearm both valid)."""
    body, index = _body(state)
    up, fore = index[f"{side}UpperArm"], index[f"{side}Forearm"]
    Ru, Rf = _matrices(body.values[:, up]), _matrices(body.values[:, fore])
    p = Ru @ (_LENGTH["UpperArm"] * _rest(f"{side}UpperArm")) + Ru @ Rf @ (_LENGTH["Forearm"] * _rest(f"{side}Forearm"))
    return p, body.valid[:, up] & body.valid[:, fore]


def torso(state) -> tuple[np.ndarray, np.ndarray]:
    """((T, 3, 3) Pelvis x Chest, (T,) valid)."""
    body, index = _body(state)
    P, C = _matrices(body.values[:, index["Pelvis"]]), _matrices(body.values[:, index["Chest"]])
    return P @ C, body.valid[:, index["Pelvis"]] & body.valid[:, index["Chest"]]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def dominant_side(states_and_frames) -> str:
    """The side whose wrist height (chest frame) varies most over the given (state, frame indices) pairs."""
    best = None
    for side in SIDES:
        h = []
        for state, idx in states_and_frames:
            p, ok = wrist_from_shoulder(state, side)
            h += list(p[idx][ok[idx], 1])
        spread = float(np.ptp(h)) if h else -1.
        if best is None or spread > best[0]:
            best = (spread, side)
    return best[1]


def rms_angle(directions: np.ndarray, centre: np.ndarray) -> float:
    """sqrt(mean angle^2) of unit ``directions`` (N, 3) from unit ``centre``."""
    if not len(directions):
        return 0.
    theta = np.arccos(np.clip(directions @ centre, -1., 1.))
    return float(np.sqrt(np.mean(theta ** 2)))


# ---- How2Sign train statistics -------------------------------------------------------------------------------------

def clip_statistics(state) -> dict | None:
    """One clip's dominant side, mean dominant direction, RMS excursion angle and mean torso; None when the clip's
    dominant arm or torso is never valid."""
    frames = np.arange(len(state.timestamps))
    side = dominant_side([(state, frames)])
    p, ok = wrist_from_shoulder(state, side)
    T, tok = torso(state)
    if not ok.any() or not tok.any():
        return None
    u = _unit(p[ok])
    m = _unit(u.mean(0))
    return dict(side=side, direction=m.tolist(), rms_angle=rms_angle(u, m), torso=to_6d(chordal_mean(T[tok])).tolist())


def space_statistics(rows: list[dict]) -> dict:
    """How2Sign train targets from ``clip_statistics`` rows: the median RMS excursion angle, per dominant side the
    normalised mean direction, and the chordal mean torso."""
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("no clip with a valid dominant arm and torso")
    direction = {}
    for side in SIDES:
        d = [r["direction"] for r in rows if r["side"] == side]
        direction[side] = _unit(np.mean(d, 0)).round(6).tolist() if d else None
    return dict(clips=len(rows), rms_angle=round(float(np.median([r["rms_angle"] for r in rows])), 6),
                direction=direction, sides={s: sum(r["side"] == s for r in rows) for s in SIDES},
                torso=to_6d(chordal_mean(_matrices(np.asarray([r["torso"] for r in rows])))).round(6).tolist())


# ---- normalisation of a composition's word clips -------------------------------------------------------------------

def _set_body(state, updates: dict) -> object:
    """``state`` with body joints replaced on their valid frames: {label: (T, 3, 3)}."""
    body, index = _body(state)
    values = body.values.copy()
    for label, R in updates.items():
        k = index[label]
        ok = body.valid[:, k]
        values[ok, k] = to_6d(R[ok]).astype(values.dtype)
    return replace(state, channels=dict(state.channels, body=replace(body, values=values)))


def space_normalise(pieces, stats: dict | None = None, *, target: str = "sentence", scale_clip=SCALE_CLIP):
    """(pieces, info): the word clips of one composition (``{"state", "lo", "hi", ...}``) in one signing space
    (module docstring). ``stats`` (``space_statistics``) enables the excursion scaling and is required for
    ``target="how2sign"``. Core windows are unchanged."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if target == "how2sign" and not stats:
        raise ValueError("the how2sign target needs the statistics")
    cores = [np.arange(p["lo"], p["hi"] + 1) for p in pieces]
    # 1. torso
    torsos = [torso(p["state"]) for p in pieces]
    pooled = [T[idx][ok[idx]] for (T, ok), idx in zip(torsos, cores)]
    if target == "how2sign":
        M_star = _matrices(np.asarray(stats["torso"]))
    else:
        M_star = chordal_mean(np.concatenate(pooled)) if sum(len(x) for x in pooled) else np.eye(3)
    out, torso_angles = [], []
    for p, (T, ok), idx, sel in zip(pieces, torsos, cores, pooled):
        if not len(sel):
            out.append(dict(p)); torso_angles.append(0.)
            continue
        C = M_star @ chordal_mean(sel).T
        torso_angles.append(float(np.degrees(np.arccos(np.clip((np.trace(C) - 1) / 2, -1., 1.)))))
        body, index = _body(p["state"])
        P = _matrices(body.values[:, index["Pelvis"]])
        out.append(dict(p, state=_set_body(p["state"], {"Pelvis": C @ P})))
    # 2. the dominant-hand centroid
    side = dominant_side([(p["state"], idx) for p, idx in zip(out, cores)])
    arms = [wrist_from_shoulder(p["state"], side) for p in out]
    dirs = [_unit(w[idx][ok[idx]]) for (w, ok), idx in zip(arms, cores)]
    if target == "how2sign" and stats["direction"].get(side):
        m_star = _unit(np.asarray(stats["direction"][side], dtype=np.float64))
    else:
        m_star = _unit(np.concatenate(dirs).mean(0)) if sum(len(d) for d in dirs) else None
    Q = [minimal_rotation(_unit(d.mean(0)), m_star) if len(d) and m_star is not None else np.eye(3) for d in dirs]
    # 3. the excursion
    scale, current = 1., None
    if stats and m_star is not None:
        aligned = np.concatenate([d @ q.T for d, q in zip(dirs, Q) if len(d)]) if any(len(d) for d in dirs) else []
        current = rms_angle(np.asarray(aligned), m_star) if len(aligned) else 0.
        scale = float(np.clip(stats["rms_angle"] / current, *scale_clip)) if current > 1e-9 else 1.
    final = []
    for p, q, (w, ok) in zip(out, Q, arms):
        body, index = _body(p["state"])
        Ru = _matrices(body.values[:, index[f"{side}UpperArm"]])
        G = np.broadcast_to(q, Ru.shape).copy()
        if scale != 1.:
            u = _unit(w) @ q.T                                                    # (T, 3) aligned directions
            axis = np.cross(np.broadcast_to(m_star, u.shape), u)
            norm = np.linalg.norm(axis, axis=-1)
            theta = np.arctan2(norm, u @ m_star)
            turn = np.where(norm > 1e-9, (scale - 1) * theta, 0.)
            E = axis_angle(axis / np.clip(norm, 1e-12, None)[:, None], turn)
            G = E @ G
        final.append(dict(p, state=_set_body(p["state"], {f"{side}UpperArm": G @ Ru})))
    info = dict(side=side, target=target, scale=round(scale, 4),
                current_rms_angle=round(current, 6) if current is not None else None,
                torso_correction_degrees=[round(a, 3) for a in torso_angles],
                centroid_rotation_degrees=[round(float(np.degrees(np.arccos(np.clip((np.trace(q) - 1) / 2, -1, 1)))),
                                                 3) for q in Q])
    return final, info


# ---- minimum-jerk bridges ------------------------------------------------------------------------------------------

def quintic(p0, v0, p1, v1, frames: int) -> np.ndarray:
    """(frames, ...) interior samples of the minimum-jerk quintic from p0 (velocity v0 per frame) to p1 (v1), zero
    boundary accelerations, over frames + 1 intervals (module docstring)."""
    T = frames + 1
    t = (np.arange(1, frames + 1, dtype=np.float64) / T).reshape(-1, *([1] * np.ndim(p0)))
    H0 = 1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5
    H1 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
    H4 = -4 * t ** 3 + 7 * t ** 4 - 3 * t ** 5
    H5 = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    return H0 * p0 + H1 * T * v0 + H4 * T * v1 + H5 * p1


def minimum_jerk_bridge(a, b, frames: int):
    """``frames`` frames from ``a``'s end to ``b``'s start along the minimum-jerk quintic of each channel, matching
    the boundary values and per-frame velocities (``a``'s last two frames, ``b``'s first two; zero when a clip has one
    frame or those frames are not both valid). Manifold projection and validity as ``compositional.blend_channel``."""
    from ..pose.multichannel import CHANNELS
    from .compositional import _gram_schmidt, crossfade_weights, take_frames
    tail = take_frames(a, np.full(frames, len(a.timestamps) - 1, dtype=int))
    channels = {}
    for name, ca in a.channels.items():
        cb = b.channels[name]
        kind = CHANNELS[name][0]
        p0, p1 = ca.values[-1].astype(np.float64), cb.values[0].astype(np.float64)
        v0 = (p0 - ca.values[-2]) * (ca.valid[-1] & ca.valid[-2])[..., None] if len(ca.values) > 1 else 0 * p0
        v1 = (cb.values[1] - p1) * (cb.valid[1] & cb.valid[0])[..., None] if len(cb.values) > 1 else 0 * p1
        v = quintic(p0, v0, p1, v1, frames)
        if kind == "rotation_6d_columns":
            v, bad = _gram_schmidt(v)
            w = crossfade_weights(frames).reshape(-1, *([1] * (v.ndim - 1)))
            v = np.where(bad[..., None], np.where(w > .5, p1, p0), v)
        elif kind == "unit_vector":
            v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6, None)
        elif kind == "bounded_scalar":
            v = np.clip(v, 0., 1.)
        va, vb = ca.valid[-1], cb.valid[0]
        valid = np.broadcast_to(va | vb, (frames, *va.shape)).copy()
        v = np.where((va & ~vb)[..., None], p0, v)
        v = np.where((vb & ~va)[..., None], p1, v)
        v = np.where(valid[..., None], v, 0.)
        w = crossfade_weights(frames)[:, None]
        wa, wb = (1 - w) * va, w * vb
        conf = np.where(valid, (wa * ca.confidence[-1] + wb * cb.confidence[0]) / np.where(valid, wa + wb, 1.), 0.)
        method = ca.inference_method or ("compositional minimum-jerk bridge" if valid.any() else None)
        channels[name] = replace(tail.channels[name], values=v.astype(ca.values.dtype), valid=valid,
                                 observed=np.zeros_like(valid), confidence=conf.astype(ca.confidence.dtype),
                                 inference_method=method)
    return replace(tail, channels=channels)


# ---- tightening ----------------------------------------------------------------------------------------------------

TIGHTEN_REASONS = ("entry", "all")


def tighten_window(state, window: dict, rule=None) -> dict:
    """A median-position core (``window["reason"]`` in ``TIGHTEN_REASONS``) cut again by the activity rule inside it:
    the new window lies inside the old one (never extended); unchanged when the rule cannot cut or the reason is
    another one. Adds ``tightened`` (frames removed) and marks the reason."""
    from .compositional import core_window, take_frames
    if window.get("reason") not in TIGHTEN_REASONS:
        return dict(window, tightened=0)
    lo, hi = int(window["lo"]), int(window["hi"])
    inner = core_window(take_frames(state, np.arange(lo, hi + 1)), rule)
    if inner["whole"]:
        return dict(window, reason=f"{window['reason']}_untightened", tightened=0)
    new_lo, new_hi = lo + inner["lo"], lo + inner["hi"]
    return dict(window, lo=new_lo, hi=new_hi, reason=f"{window['reason']}_tightened",
                tightened=int((hi - lo) - (new_hi - new_lo)))


__all__ = ["SIDES", "SCALE_CLIP", "TARGETS", "TIGHTEN_REASONS", "to_6d", "chordal_mean", "axis_angle",
           "minimal_rotation", "wrist_from_shoulder", "torso", "dominant_side", "rms_angle", "clip_statistics",
           "space_statistics", "space_normalise", "quintic", "minimum_jerk_bridge", "tighten_window"]
