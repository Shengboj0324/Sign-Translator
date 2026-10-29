"""Composition beyond joining isolated signs, part 1 (docs/NEXT_PHASE_PLAN.md, "Next lever after the oracle finding",
2026-10-29 17:38 UTC, options 2 and 5): one signing space per sentence, minimum-jerk bridges, tightened cores. No
training; deterministic; every knob off by default (the composition path is then unchanged).

**Space normalisation** (``space_normalise``) works on the canonical rotations, in the chest frame of the forward
kinematics (``canonical_skeleton``: nominal bone lengths, the wrist at shoulder + R_upper (0.28 rest_upper) + R_upper
R_fore (0.25 rest_fore)), per word clip of a composition:
1. **dominant-hand centroid**: on the **right** arm, the word's mean shoulder-to-wrist direction m_k is rotated onto the
   target m* (the normalised mean over the sentence's cores, or How2Sign train's) by the minimal rotation Q_k, applied
   to the upper arm in the chest frame (the whole arm turns rigidly about the shoulder: the word's hand trajectory
   keeps its shape);
2. **excursion**: the sentence's RMS angle of the right wrist direction about m* (over its cores, after step 1) is
   scaled to How2Sign train's median clip RMS angle (``stats``): each frame's direction is turned along its great
   circle from m* by (s - 1) x its angle, s = target / current clipped to ``scale_clip``.
Step 1 uses the sentence's own target with ``target="sentence"`` and How2Sign train's with ``target="how2sign"``; step 2
always needs the statistics (``space_statistics``, from How2Sign *train* clips only).

**The dominant side is the right one, always** (``DOMINANT``). The states are right-dominant by construction: the
corpus loader mirrors left-handed records (``corpus_manifest.load_sample``), How2Sign's signers are right-dominant
and so is the generator. A per-clip cue does not work: the wrist-height range picked the left arm in about a third of
How2Sign train clips, because the non-dominant wrist moves as much in two-handed signs. Each side's wrist speed is
reported as a diagnostic only (``wrist_speeds``), never used.

**No torso correction.** Rotating the torso (Pelvis) would turn the arms and shoulders but not the head channels,
which live in MediaPipe's face-geometry camera frame (``head_rotation``, ``head_translation`` in metres from the
camera); the head-torso relation would shift, and rotating the head consistently needs the hips' position in that
camera frame, which the canonical state does not hold. So the torso is left as recorded; the torso orientation in the
statistics is a diagnostic.

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


DOMINANT = "Right"                    # the states are right-dominant by construction (module docstring)


def wrist_speeds(states_and_frames) -> dict:
    """Diagnostic only: each side's mean wrist speed (chest frame, metres per frame) over the given (state, frame
    indices) pairs, on consecutive valid frames."""
    out = {}
    for side in SIDES:
        steps = []
        for state, idx in states_and_frames:
            p, ok = wrist_from_shoulder(state, side)
            idx = np.asarray(idx)
            both = ok[idx[1:]] & ok[idx[:-1]] & (np.diff(idx) == 1)
            steps += list(np.linalg.norm(p[idx[1:]] - p[idx[:-1]], axis=-1)[both])
        out[side] = round(float(np.mean(steps)), 6) if steps else None
    return out


def rms_angle(directions: np.ndarray, centre: np.ndarray) -> float:
    """sqrt(mean angle^2) of unit ``directions`` (N, 3) from unit ``centre``."""
    if not len(directions):
        return 0.
    theta = np.arccos(np.clip(directions @ centre, -1., 1.))
    return float(np.sqrt(np.mean(theta ** 2)))


# ---- How2Sign train statistics -------------------------------------------------------------------------------------

def clip_statistics(state) -> dict | None:
    """One clip's right-arm mean direction and RMS excursion angle, with its mean torso and both wrists' speeds as
    diagnostics; None when the right arm is never valid."""
    frames = np.arange(len(state.timestamps))
    p, ok = wrist_from_shoulder(state, DOMINANT)
    if not ok.any():
        return None
    u = _unit(p[ok])
    m = _unit(u.mean(0))
    T, tok = torso(state)
    return dict(side=DOMINANT, direction=m.tolist(), rms_angle=rms_angle(u, m),
                torso=to_6d(chordal_mean(T[tok])).tolist() if tok.any() else None,
                wrist_speeds=wrist_speeds([(state, frames)]))


def space_statistics(rows: list[dict]) -> dict:
    """How2Sign train targets from ``clip_statistics`` rows: the median right-arm RMS excursion angle and the
    normalised mean right-arm direction; diagnostics: the chordal-mean torso (not applied) and the median wrist speed
    of each side."""
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("no clip with a valid right arm")
    torsos = [r["torso"] for r in rows if r.get("torso")]
    speeds = {s: [r["wrist_speeds"][s] for r in rows if (r.get("wrist_speeds") or {}).get(s) is not None]
              for s in SIDES}
    return dict(clips=len(rows), side=DOMINANT, rms_angle=round(float(np.median([r["rms_angle"] for r in rows])), 6),
                direction={DOMINANT: _unit(np.mean([r["direction"] for r in rows], 0)).round(6).tolist()},
                diagnostics=dict(
                    torso_not_applied=to_6d(chordal_mean(_matrices(np.asarray(torsos)))).round(6).tolist()
                    if torsos else None,
                    wrist_speed_median={s: round(float(np.median(v)), 6) if v else None for s, v in speeds.items()}))


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
    out, side = [dict(p) for p in pieces], DOMINANT
    # 1. the dominant (right) hand's centroid
    arms = [wrist_from_shoulder(p["state"], side) for p in out]
    dirs = [_unit(w[idx][ok[idx]]) for (w, ok), idx in zip(arms, cores)]
    if target == "how2sign" and stats["direction"].get(side):
        m_star = _unit(np.asarray(stats["direction"][side], dtype=np.float64))
    else:
        m_star = _unit(np.concatenate(dirs).mean(0)) if sum(len(d) for d in dirs) else None
    Q = [minimal_rotation(_unit(d.mean(0)), m_star) if len(d) and m_star is not None else np.eye(3) for d in dirs]
    # 2. the excursion
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
                wrist_speeds=wrist_speeds([(p["state"], idx) for p, idx in zip(pieces, cores)]),
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
           "minimal_rotation", "wrist_from_shoulder", "torso", "DOMINANT", "wrist_speeds", "rms_angle",
           "clip_statistics",
           "space_statistics", "space_normalise", "quintic", "minimum_jerk_bridge", "tighten_window"]
