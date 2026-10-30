"""Round 20, composition v2: timing, bridges and filtering for composed sentences (docs/NEXT_PHASE_PLAN.md, "Round 20:
composition v2, the method to M1", declared 2026-10-30 19:09 UTC). Evaluation tooling only: nothing here trains, and
nothing here is used unless a round-20 option asks for it (the round-15 composition path is unchanged).

**Continuous-pace durations** (``continuous_seconds``): a sign's duration in continuous signing is its ASLLRP train
sentence median (the stage-2 model's ``timing["sentence_seconds"]``, entries with at least ``min_signs`` timed
occurrences), else its isolated annotated duration (``compositional_grammar.isolated_sign_seconds``) times
``CONTINUOUS_RATIO`` = 0.385, the continuous-over-isolated ratio recorded in the round-20 evidence (ASLLRP signs last
0.234 s against their annotated isolated forms; the F12 model's own fitted ratio, 0.3926, is in ``timing["ratio"]``).
No clipping.

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
predicted length (the report-only predicted-duration form). ``tempo`` > 1 is slower.

**Butterworth low-pass** (``lowpass_state``): 4th-order Butterworth (second-order sections), zero-phase (``sosfiltfilt``)
on every channel's values along time, per run of consecutive valid frames (shorter runs than two frames untouched; the
padding is shortened on short runs), then each value put back on its manifold: 6D rotations re-orthonormalised
(Gram-Schmidt; a degenerate frame keeps its input), unit vectors renormalised, bounded scalars clipped to [0, 1].
Validity, observation and confidence are unchanged.

**Predicted sentence duration** (``fit_duration_model``, report-only): least squares of How2Sign *train* sentence seconds
on their normalised token count; deployment cannot read the reference clip's duration, so the ``@predicted`` forms
replace it by this text-only prediction.
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


# ---- predicted sentence duration (report-only) -----------------------------------------------------------------------

def fit_duration_model(manifest, corpus: str = H2S, partition: str = "train") -> dict:
    """Least squares seconds = intercept + slope x tokens over ``corpus`` ``partition`` records with a text and a
    duration (``continuous.seconds``); tokens by ``compositional.normalise``."""
    from .compositional import normalise
    x, y = [], []
    for r in manifest["records"]:
        if r.get("corpus") != corpus or r.get("partition") != partition or not r.get("text"):
            continue
        s = (r.get("continuous") or {}).get("seconds")
        if s:
            x.append(len(normalise(r["text"])))
            y.append(float(s))
    if len(x) < 2 or len(set(x)) < 2:
        raise ValueError("too few records to fit the duration model")
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    residual = np.asarray(y) - (intercept + slope * np.asarray(x, float))
    return dict(corpus=corpus, partition=partition, records=len(x), intercept=round(float(intercept), 4),
                slope=round(float(slope), 4), median_abs_error=round(float(np.median(np.abs(residual))), 4))


def predicted_seconds(model: dict, text: str, floor: float = .5) -> float:
    from .compositional import normalise
    return max(float(model["intercept"]) + float(model["slope"]) * len(normalise(text)), floor)


__all__ = ["FPS", "CONTINUOUS_RATIO", "SIDES", "MIN_SPEED", "MAX_GAP_SECONDS", "TOTAL_FITS", "continuous_seconds",
           "boundary_speed", "bridge_frames", "continuous_plan", "lowpass_values", "lowpass_state",
           "fit_duration_model", "predicted_seconds"]
