"""Non-manual markers from face blendshapes and head rotation.

**Feature tracks** (MediaPipe's 52 ARKit-named blendshapes; the action-unit
correlates are from Dachkovsky, Healy & Sandler 2013 and the mapping to blendshapes
is derived, since no published ASL-to-blendshape mapping exists):

* brow raise (AU1+2; yes/no questions, topics, conditionals) = mean(browInnerUp,
  browOuterUpLeft, browOuterUpRight);
* brow furrow (AU4; wh-questions) = mean(browDownLeft, browDownRight) − k·raise, the
  raise term removing the AU1+4 "worried" configuration (k = ``FURROW_RAISE_COUPLING``);
* squint (AU6+AU10) = mean(cheekSquint*, eyeSquint*);
* mouth morphemes: MM = mean(mouthPress*, mouthClose), PUFF = cheekPuff,
  OO = max(mouthPucker, mouthFunnel), CHA/open = jawOpen.

Each track is z-scored against a robust neutral baseline — median and MAD-derived
σ — pooled over many clips of the same signer (:func:`pooled_baselines`; a single
clip's own median would absorb a marker that covers most of a short sentence, e.g. a
wh-furrow over a whole question), so a signer's resting face is zero, then low-pass
filtered (zero-phase moving average of ≈ 1/``SMOOTH_HZ`` s), and segmented by
hysteresis (enter at ``ON_Z``, leave at ``OFF_Z``); each span's edges are then moved
to the half-level crossings of the unfiltered track (smoothing blurs edges by half
its width) and spans under ``MIN_SPAN_S`` are dropped, which keeps the 0.2–0.3 s
pre-sign brow raise of Schlenker & Lamberton detectable.

**Head gestures** (Neidle et al. 2009 headshake detector, recast for rotations):
from the head's yaw ψ(t) and pitch, a headshake is a window with at least
``SHAKE_REVERSALS`` sign changes of dψ/dt between excursions of at least
``SHAKE_AMPLITUDE_DEG`` (peak to peak), each half-cycle within 0.1–0.6 s; a nod is the
same on pitch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..generation.motion_metrics import _matrices

FURROW_RAISE_COUPLING = .5
SMOOTH_HZ = 3.5
ON_Z, OFF_Z = 2.5, 1.5
MIN_SPAN_S = .15
SHAKE_REVERSALS = 3
SHAKE_AMPLITUDE_DEG = 6.
HALF_CYCLE_S = (.1, .6)

TRACKS = {
    "brow_raise": (("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), "mean"),
    "brow_furrow": (("browDownLeft", "browDownRight"), "mean"),
    "squint": (("cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight"), "mean"),
    "mm": (("mouthPressLeft", "mouthPressRight", "mouthClose"), "mean"),
    "puff": (("cheekPuff",), "mean"),
    "oo": (("mouthPucker", "mouthFunnel"), "max"),
    "jaw_open": (("jawOpen",), "mean"),
}


def raw_tracks(face) -> tuple[dict, np.ndarray]:
    """name -> (T,) raw track, and (T,) validity, from a ``face`` channel."""
    labels = list(face.labels)
    values = face.values[..., 0]
    out = {}
    for name, (parts, how) in TRACKS.items():
        cols = values[:, [labels.index(p) for p in parts]]
        out[name] = cols.max(1) if how == "max" else cols.mean(1)
    out["brow_furrow"] = out["brow_furrow"] - FURROW_RAISE_COUPLING * out["brow_raise"]
    return out, face.valid.all(axis=1)


def robust_z(x: np.ndarray, valid: np.ndarray, baseline: tuple[float, float] | None = None):
    """(z-scores, (median, σ)); σ = 1.4826·MAD, floored to keep a flat track finite."""
    if baseline is None:
        v = x[valid]
        med = float(np.median(v)) if len(v) else 0.
        sigma = 1.4826 * float(np.median(np.abs(v - med))) if len(v) else 1.
        baseline = (med, max(sigma, 1e-3))
    return np.where(valid, (x - baseline[0]) / baseline[1], 0.), baseline


def pooled_baselines(faces) -> dict:
    """name -> (median, σ) over the valid frames of several ``face`` channels (one signer or one reference set)."""
    pooled = {}
    for face in faces:
        raw, valid = raw_tracks(face)
        for name, x in raw.items():
            pooled.setdefault(name, []).append(x[valid])
    out = {}
    for name, parts in pooled.items():
        v = np.concatenate(parts) if parts else np.zeros(0)
        med = float(np.median(v)) if len(v) else 0.
        out[name] = (med, max(1.4826 * float(np.median(np.abs(v - med))) if len(v) else 1., 1e-3))
    return out


def smooth(x: np.ndarray, valid: np.ndarray, fps: float, hz: float = SMOOTH_HZ) -> np.ndarray:
    """Zero-phase moving average over ≈ fps/hz frames (windows clipped at the clip's ends), ignoring invalid frames."""
    width = max(1, int(round(fps / hz)))
    x = np.where(valid, x, 0.)
    if width == 1 or len(x) == 0:
        return x
    half = width // 2
    cs = np.r_[0., np.cumsum(x)]
    cv = np.r_[0., np.cumsum(valid.astype(float))]
    lo = np.clip(np.arange(len(x)) - half, 0, len(x))
    hi = np.clip(np.arange(len(x)) + (width - half), 0, len(x))
    num, den = cs[hi] - cs[lo], cv[hi] - cv[lo]
    return np.where(den > 0, num / np.where(den > 0, den, 1.), 0.)


def hysteresis_spans(z: np.ndarray, valid: np.ndarray, fps: float, *, on: float = ON_Z, off: float = OFF_Z,
                     min_s: float = MIN_SPAN_S) -> list[tuple[int, int]]:
    """[start, stop) frame spans entering above ``on`` and leaving below ``off``, at least ``min_s`` long."""
    spans, start = [], None
    for t in range(len(z)):
        if start is None and valid[t] and z[t] >= on:
            start = t
        elif start is not None and (not valid[t] or z[t] < off):
            spans.append((start, t)); start = None
    if start is not None:
        spans.append((start, len(z)))
    return [(a, b) for a, b in spans if (b - a) / fps >= min_s]


def refine_edges(raw_z: np.ndarray, valid: np.ndarray, spans, fps: float, *, min_s: float = MIN_SPAN_S):
    """Move each span's edges to the half-level crossings of the unsmoothed z within one filter width.

    Smoothing locates spans robustly but blurs their edges by half its width; the edge
    is placed where the raw signal first (last) reaches half the span's 90th-percentile level,
    and spans shorter than ``min_s`` after refinement are dropped.
    """
    width = max(1, int(round(fps / SMOOTH_HZ)))
    out = []
    for a, b in spans:
        inside = raw_z[a:b][valid[a:b]]
        if not len(inside):
            continue
        half = float(np.percentile(inside, 90)) / 2
        lo, hi = max(0, a - width), min(len(raw_z), b + width)
        above = np.nonzero(valid[lo:hi] & (raw_z[lo:hi] >= half))[0]
        if not len(above):
            continue
        start, stop = lo + int(above[0]), lo + int(above[-1]) + 1
        if (stop - start) / fps >= min_s:
            out.append((start, stop))
    return out


@dataclass(frozen=True)
class NMMTracks:
    z: dict                      # name -> (T,) smoothed robust z-score
    spans: dict                  # name -> [(start, stop), ...]
    baselines: dict              # name -> (median, σ)
    valid: np.ndarray


def nmm_tracks(face, fps: float, baselines: dict | None = None) -> NMMTracks:
    raw, valid = raw_tracks(face)
    z, spans, used = {}, {}, {}
    for name, x in raw.items():
        zz, used[name] = robust_z(x, valid, (baselines or {}).get(name))
        z[name] = smooth(zz, valid, fps)
        spans[name] = refine_edges(zz, valid, hysteresis_spans(z[name], valid, fps, min_s=0.), fps)
    return NMMTracks(z, spans, used, valid)


def head_angles_deg(head_rotation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw, pitch, valid) of the head's forward axis in its camera-aligned frame, degrees."""
    ok = head_rotation.valid[:, 0]
    identity = np.array([1., 0., 0., 0., 1., 0.])
    R = _matrices(np.where(ok[:, None], head_rotation.values[:, 0], identity))
    f = R[:, :, 2]
    return np.degrees(np.arctan2(f[:, 0], f[:, 2])), np.degrees(np.arcsin(np.clip(f[:, 1], -1, 1))), ok


def oscillation_spans(angle_deg: np.ndarray, valid: np.ndarray, fps: float, *,
                      reversals: int = SHAKE_REVERSALS, amplitude: float = SHAKE_AMPLITUDE_DEG,
                      half_cycle_s=HALF_CYCLE_S) -> list[tuple[int, int]]:
    """Spans of back-and-forth rotation: ≥ ``reversals`` extrema, successive extrema ≥ ``amplitude`` apart."""
    a = smooth(angle_deg, valid, fps, hz=8.)
    d = np.diff(a)
    extrema = [t for t in range(1, len(d)) if valid[t] and np.sign(d[t]) != np.sign(d[t - 1]) and d[t] != 0]
    spans, run = [], []
    lo, hi = half_cycle_s[0] * fps, half_cycle_s[1] * fps
    for t in extrema:
        if run and (lo <= t - run[-1] <= hi) and abs(a[t] - a[run[-1]]) >= amplitude:
            run.append(t)
        else:
            if len(run) >= reversals:
                spans.append((run[0], run[-1] + 1))
            run = [t]
    if len(run) >= reversals:
        spans.append((run[0], run[-1] + 1))
    return spans


def head_gestures(head_rotation, fps: float) -> dict:
    yaw, pitch, ok = head_angles_deg(head_rotation)
    return dict(headshake=oscillation_spans(yaw, ok, fps), nod=oscillation_spans(pitch, ok, fps),
                yaw_deg=yaw, pitch_deg=pitch, valid=ok)


def span_fraction(spans, frames: int) -> float:
    return float(sum(b - a for a, b in spans) / frames) if frames else 0.


__all__ = ["TRACKS", "raw_tracks", "robust_z", "pooled_baselines", "smooth", "hysteresis_spans", "refine_edges", "NMMTracks", "nmm_tracks",
           "head_angles_deg", "oscillation_spans", "head_gestures", "span_fraction"]
