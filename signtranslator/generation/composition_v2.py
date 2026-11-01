"""Round 20, composition v2: timing, bridges and filtering for composed sentences (docs/NEXT_PHASE_PLAN.md, "Round 20:
composition v2, the method to M1", declared 2026-10-30 19:09 UTC). Evaluation tooling only: nothing here trains, and
nothing here is used unless a round-20 option asks for it (the round-15 composition path is unchanged).

**Continuous-pace durations** (``continuous_seconds``): a sign's duration in continuous signing is its ASLLRP train
sentence median (the stage-2 model's ``timing["sentence_seconds"]``, entries with at least ``min_signs`` timed
occurrences), else its isolated annotated duration (``compositional_grammar.isolated_sign_seconds``) times
``CONTINUOUS_RATIO`` = 0.385, the continuous-over-isolated ratio recorded in the round-20 evidence (ASLLRP signs last
0.234 s against their annotated isolated forms; the F12 model's own fitted ratio, 0.3926, is in ``timing["ratio"]``).
An entry with neither (``isolated_sign_seconds``' "default" source: the median ASLLRP *sentence* sign, already a
continuous duration) takes that default unshrunk ("default"), never x the ratio a second time. No clipping.

**Velocity-bounded bridges** (``bridge_frames``, after Sign Stitching, Walsh et al., arXiv 2405.07663, step 5: linear
bridges whose length keeps the bridge no faster than the signs it joins). For each side, with the wrist taken in the
chest frame (``composition_space.wrist_from_shoulder``, metres), d the distance from the end of core k to the start of
core k + 1 and v the mean of the two boundary speeds (the last / first ``steps`` frame-to-frame wrist speeds of the
*resampled* cores, metres per second), floored at ``min_speed``: a linear bridge of n frames moves d / (n + 1) per frame,
so n >= d x fps / v - 1. The join gets max(base gap, the largest n over the tracked sides), capped at ``max_gap``
frames (never below the base gap). The base gap is ASLLRP's median gap (the stage-2 model's ``timing["gap"]``) times the
tempo. This is our reading of the paper's rule (its text does not give a formula); the bound and its floor are
reported per join.

**The continuous plan** (``continuous_plan``): sign k gets round(seconds_k x tempo x 24) frames (at least 1); the
joins as above; then the sentence is fitted like ``compositional_grammar.plan_frames``: ``pad`` fills the reference
with the first word's approach and the last word's return (lead share as the model's), a body longer than the reference
is compressed (``allocate``, at least one frame per sign; more signs than frames: truncated); ``none`` keeps the
body's own length (the evaluation never uses it: its ``@predicted`` form pads or compresses to the predicted seconds).
``tempo`` > 1 is slower.

**Butterworth low-pass** (``lowpass_state``): 4th-order Butterworth (second-order sections), zero-phase (``sosfiltfilt``)
on every channel's values along time. Forward-backward filtering squares the magnitude response: at the nominal cutoff
the gain is 0.5 (-6 dB), and the effective -3 dB point of a 6 Hz setting at 24 fps is about 5.3 Hz. The state must be
on a uniform 24 fps clock (checked from its timestamps). The filter runs per run of consecutive valid frames (shorter runs than two frames untouched; the
padding is shortened on short runs), then each value put back on its manifold: 6D rotations re-orthonormalised
(Gram-Schmidt; a degenerate frame keeps its input), unit vectors renormalised, bounded scalars clipped to [0, 1].
Validity, observation and confidence are unchanged.

**Predicted sentence duration** (``fit_duration_model``, report-only): least squares of How2Sign *train* sentence seconds
on their normalised token count; deployment cannot read the reference clip's duration, so the ``@predicted`` forms
replace it by this text-only prediction, under either plan: the same words, with ``--retime reference`` the
``--timing`` plan and with ``--retime continuous`` the continuous plan, filling round(predicted x 24) frames. Only the
sentence's total changes; the main form and its controls keep the reference's duration. A sentence with no selected word
is the blank at the predicted length in that form, and ``blank@predicted`` is its comparator, so the form is text-only.

**Arm scale** (``arm_scale_state``, post-S2 plan A1; off unless asked for): each listed side's upper arm and forearm rotations
are pulled toward their mean over the state's valid frames by the rotation power R -> M (M^T R)^f, f the scale. Timing, torso, face,
validity and the hand channels' own values are untouched; the hands themselves do move, because they ride on the wrist the arm carries.
f = 1 leaves the side exactly as it is. This is the amplitude change of the motion diagnosis (ledger 2026-10-31 22:58). It shrinks
motion: it can pass a speed band by moving less, so it is read only beside real clips scaled the same way, and only as a sensitivity.

**Left-arm hold** (``hold_left_arm``, post-S2 A1's ``hold`` arm; off unless asked for). A word annotated one-handed has no use for the
left arm, yet composition sends that arm from each two-handed word down to the clip's rest and back up, inside a bridge. For each stretch
between two anchor cores (any core not annotated one-handed) that holds a one-handed core, the left upper arm and forearm rotations follow
a minimum-jerk slerp from the last pose of the anchor before to the first pose of the anchor after. At a sentence's edge the neighbouring
anchor's pose is held; a sentence with no anchor holds the ready pose from How2Sign train. Everything else stays bit-exact: the right arm,
the anchors, the hands' own values, every frame outside a stretch, and the plan. The wrist's chest-frame speed is capped at ``cap``: a
stretch whose minimum-jerk peak exceeds it runs at constant speed, which has the lowest peak its frames allow; if even that exceeds the cap
the plan's frames cannot satisfy it, and the stretch is counted as ``cap_bound``, never stretched and never left with a jump. No learned
stage and no gloss; the annotation is read from the manifest's isolated records.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

FPS = 24.
CONTINUOUS_RATIO = .385            # ASLLRP continuous sign / isolated annotated sign (round-20 evidence)
SIDES = ("Left", "Right")
MIN_SPEED = .3                     # m/s: the floor of a bridge's speed bound (a still boundary never forces a long bridge)
MAX_GAP_SECONDS = .6               # the longest velocity-bounded bridge
H2S = "how2sign-two-view-text-v1"
TOTAL_FITS = ("pad", "none")


# ---- durations -----------------------------------------------------------------------------------------------------

def continuous_seconds(entry: str, timing: dict, typical, ratio: float = CONTINUOUS_RATIO) -> tuple[float, str]:
    """(seconds, source) of one entry's sign at continuous pace (module docstring): the ASLLRP sentence median
    ("asllrp"), else ``ratio`` x its isolated annotated seconds ("isolated_<source>")."""
    from .compositional_grammar import isolated_sign_seconds
    if entry in (timing.get("sentence_seconds") or {}):
        return float(timing["sentence_seconds"][entry]), "asllrp"
    seconds, source = isolated_sign_seconds(entry, timing, typical)
    if source == "default":                    # already the ASLLRP sentence-sign median: a continuous duration
        return float(seconds), "default"
    return float(ratio) * float(seconds), f"isolated_{source}"


# ---- velocity-bounded bridges ---------------------------------------------------------------------------------------

def _wrist(state, side):
    from .composition_space import wrist_from_shoulder
    return wrist_from_shoulder(state, side)


def boundary_speed(state, side: str, *, end: bool, steps: int = 3, fps: float = FPS) -> float | None:
    """The mean wrist speed (m/s, chest frame) over the last (``end``) or first ``steps`` frame-to-frame steps of
    ``state`` whose two frames are both tracked; None when there is none."""
    p, ok = _wrist(state, side)
    T = len(p)
    if T < 2:
        return None
    idx = np.arange(max(T - 1 - steps, 0), T - 1) if end else np.arange(0, min(steps, T - 1))
    good = ok[idx] & ok[idx + 1]
    if not good.any():
        return None
    return float(np.linalg.norm(p[idx + 1][good] - p[idx][good], axis=-1).mean() * fps)


def bridge_frames(a, b, base: int, *, min_speed: float = MIN_SPEED, max_frames: int | None = None,
                  steps: int = 3, fps: float = FPS) -> tuple[int, dict]:
    """(frames, info) of the join from ``a``'s last frame to ``b``'s first (module docstring)."""
    need, sides = 0, {}
    for side in SIDES:
        pa, oka = _wrist(a, side)
        pb, okb = _wrist(b, side)
        if not (len(pa) and len(pb) and oka[-1] and okb[0]):
            continue
        d = float(np.linalg.norm(pa[-1] - pb[0]))
        speeds = [v for v in (boundary_speed(a, side, end=True, steps=steps, fps=fps),
                              boundary_speed(b, side, end=False, steps=steps, fps=fps)) if v is not None]
        bound = max(float(np.mean(speeds)) if speeds else 0., float(min_speed))
        n = max(int(np.ceil(d * fps / bound - 1 - 1e-9)), 0)
        sides[side] = dict(distance=round(d, 4), bound=round(bound, 4), frames=n)
        need = max(need, n)
    frames = max(int(base), need)
    if max_frames is not None:
        frames = min(frames, max(int(max_frames), int(base)))
    return frames, dict(base=int(base), needed=int(need), frames=int(frames), sides=sides)


def continuous_plan(cores, seconds, reference_seconds: float, *, gap_seconds: float, tempo: float = 1.,
                    total_fit: str = "pad", lead_share: float = .4, velocity: bool = True,
                    min_speed: float = MIN_SPEED, max_gap_seconds: float = MAX_GAP_SECONDS, fps: float = FPS) -> dict:
    """The frame plan of one sentence at continuous pace (module docstring), in ``compositional.render``'s format
    (``cores``, ``gaps``, ``lead``, ``trail``, ``frames``, ``fit``, ...). ``cores``: the word cores in sentence order
    (states); ``seconds``: each sign's continuous seconds (``continuous_seconds``)."""
    from .compositional import time_scale
    from .compositional_grammar import allocate
    if total_fit not in TOTAL_FITS:
        raise ValueError(f"total_fit must be one of {TOTAL_FITS}")
    if not len(cores) or len(cores) != len(seconds):
        raise ValueError("one duration per core, at least one core")
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    n = len(cores)
    core_frames = [max(1, int(round(float(s) * tempo * fps))) for s in seconds]
    base = max(0, int(round(float(gap_seconds) * tempo * fps)))
    gaps, joins = [], []
    if velocity and n > 1:
        scaled = [time_scale(c, f) for c, f in zip(cores, core_frames)]
        cap = int(round(max_gap_seconds * fps))
        for k in range(n - 1):
            f, info = bridge_frames(scaled[k], scaled[k + 1], base, min_speed=min_speed, max_frames=cap, fps=fps)
            gaps.append(f)
            joins.append(info)
    else:
        gaps = [base] * (n - 1)
    body = sum(core_frames) + sum(gaps)
    R = max(1, int(round(reference_seconds * fps)))
    cores_out, gaps_out, lead, trail = list(core_frames), list(gaps), 0, 0
    if total_fit == "none":
        fit = "predicted"
    elif body <= R:
        pad = R - body
        lead = int(round(pad * lead_share))
        trail, fit = pad - lead, "padded"
    elif n > R:
        cores_out, gaps_out, fit = [1] * R + [0] * (n - R), [0] * (n - 1), "truncated"
    else:
        out = allocate(core_frames + gaps, R, [1] * n + [0] * (n - 1))
        cores_out, gaps_out, fit = out[:n], out[n:], "compressed"
    frames = lead + sum(cores_out) + sum(gaps_out) + trail
    if total_fit == "pad" and frames != R:
        raise AssertionError("the continuous plan does not fill the reference")
    return dict(cores=cores_out, gaps=gaps_out, lead=lead, trail=trail, fit=fit, body_planned=body,
                scale=round((sum(cores_out) + sum(gaps_out)) / body, 4) if body else None, frames=frames,
                overlong=fit == "truncated", truncated_signs=int(sum(c == 0 for c in cores_out)), tempo=tempo,
                base_gap=base, velocity_raised=int(sum(g > base for g in gaps)), joins=joins)


# ---- Butterworth low-pass -------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)) of the runs of True in a 1-D mask."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def lowpass_values(values: np.ndarray, valid: np.ndarray, hz: float, *, fps: float = FPS, order: int = 4) -> np.ndarray:
    """``values`` (T, J, D) filtered along time (module docstring), per joint over its runs of valid frames."""
    from scipy.signal import butter, sosfiltfilt
    if not 0 < hz < fps / 2:
        raise ValueError(f"the cutoff must be in (0, {fps / 2}) Hz")
    sos = butter(order, hz, fs=fps, output="sos")
    default = 3 * (2 * len(sos) + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    out = np.array(values, dtype=np.float64, copy=True)
    cols = np.asarray(valid, dtype=bool)
    if cols.ndim == 1:
        cols = cols[:, None]
    masks, inverse = np.unique(cols.T, axis=0, return_inverse=True)
    for m, mask in enumerate(masks):
        joints = np.flatnonzero(np.asarray(inverse).reshape(-1) == m)
        for a, b in _runs(mask):
            if b - a < 2:
                continue
            seg = out[a:b][:, joints]
            out[a:b, joints] = sosfiltfilt(sos, seg, axis=0, padlen=min(default, b - a - 1))
    return out


def lowpass_state(state, hz: float, *, fps: float = FPS, order: int = 4):
    """``state`` with every channel low-passed (module docstring); the clock, masks and confidences unchanged."""
    from ..pose.multichannel import CHANNELS
    from .compositional import _gram_schmidt
    t = np.asarray(state.timestamps, dtype=float)
    if len(t) > 1 and not np.allclose(np.diff(t), 1. / fps, rtol=.01, atol=0.):
        raise ValueError(f"the low-pass needs a uniform {fps:g} fps clock")
    channels = {}
    for name, c in state.channels.items():
        kind = CHANNELS[name][0]
        if not c.valid.any():
            channels[name] = c
            continue
        v = lowpass_values(c.values, c.valid, hz, fps=fps, order=order)
        if kind == "rotation_6d_columns":
            v, bad = _gram_schmidt(v)
            v = np.where(bad[..., None], c.values, v)
        elif kind == "unit_vector":
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            v = np.where(norm > 1e-6, v / np.clip(norm, 1e-6, None), c.values)
        elif kind == "bounded_scalar":
            v = np.clip(v, 0., 1.)
        v = np.where(c.valid[..., None], v, c.values)
        channels[name] = replace(c, values=v.astype(c.values.dtype))
    return replace(state, channels=channels)


# ---- arm scale ------------------------------------------------------------------------------------------------------

def _rotation_power(R: np.ndarray, f: float) -> np.ndarray:
    """R^f of (..., 3, 3) rotations, by the axis-angle form."""
    from scipy.spatial.transform import Rotation
    shape = R.shape
    vectors = Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec()
    return Rotation.from_rotvec(vectors * f).as_matrix().reshape(shape)


def arm_scale_state(state, *, left: float = 1., right: float = 1.):
    """``state`` with each side's arm rotations scaled about their mean over its valid frames (module docstring)."""
    from .composition_space import chordal_mean, to_6d
    from .motion_metrics import _matrices
    if left <= 0 or right <= 0:
        raise ValueError("an arm scale must be positive")
    body = state.channels["body"]
    labels = list(body.labels)
    values = None
    for side, f in (("Left", float(left)), ("Right", float(right))):
        if f == 1.:
            continue
        for joint in (f"{side}UpperArm", f"{side}Forearm"):
            k = labels.index(joint)
            ok = body.valid[:, k]
            if ok.sum() < 2:
                continue
            if values is None:
                values = body.values.astype(np.float64)
            R = _matrices(values[:, k])[ok]
            mean = chordal_mean(R)
            relative = np.einsum("ba,tbc->tac", mean, R)
            values[ok, k] = to_6d(np.einsum("ab,tbc->tac", mean, _rotation_power(relative, f)))
    if values is None:
        return state
    return replace(state, channels=dict(state.channels, body=replace(body, values=values.astype(body.values.dtype))))


# ---- left-arm hold ---------------------------------------------------------------------------------------------------

HOLD_JOINTS = ("LeftUpperArm", "LeftForearm")


def frame_labels(plan: dict) -> np.ndarray:
    """Per frame of a rendered plan: the core index k >= 0, -1 for a bridge, -2 for the lead-in or lead-out."""
    labels = [-2] * plan["lead"]
    for k, c in enumerate(plan["cores"]):
        labels += [k] * c
        if k < len(plan["gaps"]):
            labels += [-1] * plan["gaps"][k]
    labels += [-2] * plan["trail"]
    return np.asarray(labels, dtype=int)


def minimum_jerk(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def hold_stretches(labels: np.ndarray, one_handed) -> list[tuple[int, int]]:
    """[(first, last)] inclusive frame ranges: each maximal stretch without an anchor core's frame that holds a one-handed core's frame.
    ``one_handed[k]``: True for an annotated one-handed word; False or None (no annotation) makes the core an anchor."""
    labels = np.asarray(labels)
    is_one = np.array([k >= 0 and one_handed[k] is True for k in labels], dtype=bool)
    anchor = np.array([k >= 0 and one_handed[k] is not True for k in labels], dtype=bool)
    out, start = [], None
    for i in range(len(labels) + 1):
        if i == len(labels) or anchor[i]:
            if start is not None and is_one[start:i].any():
                out.append((start, i - 1))
            start = None
        elif start is None:
            start = i
    return out


def _slerp(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(n, 3, 3) from rotation ``a`` toward ``b`` by the fractions ``s``, along the geodesic."""
    from scipy.spatial.transform import Rotation
    vector = Rotation.from_matrix(a.T @ b).as_rotvec()
    return a @ Rotation.from_rotvec(np.outer(s, vector)).as_matrix()


def _anchor_pose(R, valid, start: int, step: int):
    """The nearest valid pose from ``start`` going by ``step``, inside the run of frames ``valid`` marks; None past either end."""
    i = start
    while 0 <= i < len(valid):
        if valid[i]:
            return R[i]
        i += step
    return None


def hold_left_arm(state, plan: dict, one_handed, *, ready: dict, cap: float, fps: float = FPS):
    """(state, info): ``state`` with the left arm held across one-handed stretches (module docstring). ``ready``: {joint: 6D values} of
    How2Sign train's ready pose; ``cap``: the wrist's chest-frame speed in metres per second."""
    from .composition_space import to_6d, wrist_of_rotations
    from .motion_metrics import _matrices
    labels = frame_labels(plan)
    body = state.channels["body"]
    if len(labels) != len(body.values):
        raise ValueError("the plan and the state differ in length")
    index = {j: list(body.labels).index(j) for j in HOLD_JOINTS}
    stretches = hold_stretches(labels, list(one_handed))
    info = dict(stretches=len(stretches), frames=int(sum(b - a + 1 for a, b in stretches)), minimum_jerk=0, constant_speed=0, held=0, ready_pose=0,
                cap_bound=0, cap=float(cap), peak=0.)
    if not stretches:
        return state, info
    values = body.values.copy()
    rotations = {j: _matrices(body.values[:, k].astype(np.float64)) for j, k in index.items()}
    anchors = np.array([k >= 0 and one_handed[k] is not True for k in labels], dtype=bool)

    def planned(a, b, profile):
        out, kinds = {}, set()
        n = b - a + 1
        for j, k in index.items():
            ok = body.valid[:, k] & anchors
            before = _anchor_pose(rotations[j], ok, a - 1, -1) if a > 0 else None
            after = _anchor_pose(rotations[j], ok, b + 1, 1) if b < len(labels) - 1 else None
            if before is None and after is None:
                pose = _matrices(np.asarray(ready[j], dtype=np.float64))
                out[j] = np.repeat(pose[None], n, 0)
                kinds.add("ready_pose")
            elif before is None or after is None:
                pose = before if after is None else after
                out[j] = np.repeat(pose[None], n, 0)
                kinds.add("held")
            else:
                t = np.arange(1, n + 1) / (n + 1)
                out[j] = _slerp(before, after, minimum_jerk(t) if profile == "minimum_jerk" else t)
                kinds.add("moved")
        # a stretch where one joint moves is a moving stretch, whatever the other does: the cap applies to it
        return out, "moved" if "moved" in kinds else "held" if "held" in kinds else "ready_pose"

    def peak_of(a, b, poses):
        """The wrist's peak chest-frame speed over the stretch, its entry step and its exit step included."""
        lo, hi = max(a - 1, 0), min(b + 1, len(labels) - 1)
        upper, fore = (rotations[j][lo:hi + 1].copy() for j in HOLD_JOINTS)
        upper[a - lo:b - lo + 1], fore[a - lo:b - lo + 1] = poses[HOLD_JOINTS[0]], poses[HOLD_JOINTS[1]]
        p = wrist_of_rotations(upper, fore, "Left")
        seen = body.valid[lo:hi + 1, index[HOLD_JOINTS[0]]] & body.valid[lo:hi + 1, index[HOLD_JOINTS[1]]]
        good = seen[1:] & seen[:-1]
        return float(np.linalg.norm(p[1:] - p[:-1], axis=-1)[good].max() * fps) if good.any() else 0.

    for a, b in stretches:
        poses, kind = planned(a, b, "minimum_jerk")
        peak = peak_of(a, b, poses)
        if kind == "moved":
            if peak > cap:
                poses, _ = planned(a, b, "constant_speed")
                peak = peak_of(a, b, poses)
                info["constant_speed"] += 1
                info["cap_bound"] += int(peak > cap)
            else:
                info["minimum_jerk"] += 1
        else:
            info[kind] += 1
        info["peak"] = max(info["peak"], round(peak, 4))
        for j, k in index.items():
            ok = body.valid[a:b + 1, k]
            values[a:b + 1, k][ok] = to_6d(poses[j])[ok].astype(values.dtype)
    return replace(state, channels=dict(state.channels, body=replace(body, values=values))), info


# ---- predicted sentence duration (report-only) -----------------------------------------------------------------------

def fit_duration_model(manifest, corpus: str = H2S, partition: str = "train", *, components=None,
                       signer_balanced: bool = False) -> dict:
    """Least squares seconds = intercept + slope x tokens over ``corpus`` ``partition`` records with a text and a
    duration (``continuous.seconds``); tokens by ``compositional.normalise``.

    Stage S2's probe P3 (keyword-only; the defaults reproduce the fit above exactly): ``components`` restricts the fit to
    those signer components, and ``signer_balanced`` gives every component equal total weight. How2Sign train is almost
    all one signer (c02, 0.366 s/token; val 0.28-0.33), so an unweighted fit carries that signer's tempo."""
    from .compositional import normalise
    x, y, group = [], [], []
    for r in manifest["records"]:
        if r.get("corpus") != corpus or r.get("partition") != partition or not r.get("text"):
            continue
        if components is not None and r.get("signer_component") not in components:
            continue
        s = (r.get("continuous") or {}).get("seconds")
        if s:
            x.append(len(normalise(r["text"])))
            y.append(float(s))
            group.append(r.get("signer_component"))
    if len(x) < 2 or len(set(x)) < 2:
        raise ValueError("too few records to fit the duration model")
    x, y = np.asarray(x, float), np.asarray(y, float)
    weights = None
    extra = {}
    if signer_balanced:
        names, inverse, counts = np.unique(np.asarray(group, dtype=str), return_inverse=True, return_counts=True)
        if len(names) < 2:
            raise ValueError("a signer-balanced fit needs at least two signer components")
        weights = np.sqrt(1. / counts[inverse])                    # np.polyfit weights the residual, so the square root
        extra = dict(signer_balanced=True, components={str(n): int(c) for n, c in zip(names, counts)})
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    residual = y - (intercept + slope * x)
    return dict(corpus=corpus, partition=partition, records=len(x), intercept=round(float(intercept), 4),
                slope=round(float(slope), 4), median_abs_error=round(float(np.median(np.abs(residual))), 4), **extra)


def predicted_seconds(model: dict, text: str, floor: float = .5) -> float:
    from .compositional import normalise
    return max(float(model["intercept"]) + float(model["slope"]) * len(normalise(text)), floor)


__all__ = ["FPS", "CONTINUOUS_RATIO", "SIDES", "MIN_SPEED", "MAX_GAP_SECONDS", "TOTAL_FITS", "continuous_seconds",
           "boundary_speed", "bridge_frames", "continuous_plan", "lowpass_values", "lowpass_state", "arm_scale_state", "frame_labels",
           "minimum_jerk", "hold_stretches", "hold_left_arm", "HOLD_JOINTS",
           "fit_duration_model", "predicted_seconds"]
