"""Evaluation of generated canonical motion against held-out reference motion.

Per-frame error alone rewards the conditional mean: a model that averages
plausible signings scores well on it while under-articulating every one of them.
These metrics therefore pair accuracy with articulation and realism:

* ``geodesic_deg`` — median geodesic angle between generated and reference joint
  rotations on the shared clock (lower is better); for translation channels
  ``position_cm``, the median Euclidean distance, and for unit-vector channels
  (gaze) ``angle_deg``, the median angle between the directions;
* ``speed_ratio`` — median speed of generated joints over that of the reference
  (deg/s, or m/s for translations, between consecutive valid frames; 1 is
  realistic, < 1 is the damped, averaged motion of a regressor);
* ``variation_ratio`` — median per-joint temporal standard deviation of the 6D
  values, generated over reference;
* ``frechet`` — Fréchet distance between the distributions of per-clip features
  (mean pose, temporal variation, mean angular speed per joint), each feature
  standardised by the reference, in a PCA basis fitted on the reference clips (the
  text-to-motion "FID"; lower is better).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg

from ..pose.multichannel import CHANNELS, MultichannelMotion

ROTATION_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")


def _matrices(values6: np.ndarray) -> np.ndarray:
    """Gram–Schmidt rotation matrices from 6D column pairs (…, 6) -> (…, 3, 3)."""
    a, b = values6[..., :3], values6[..., 3:]
    r1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=-1)


def geodesic_degrees(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Angle in degrees between corresponding rotations given as 6D values."""
    relative = np.einsum("...ji,...jk->...ik", _matrices(values_a), _matrices(values_b))
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def unit_angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between corresponding directions (…, 3), robust near 0 and 180°."""
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(-1)))


def distance(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-element distance of a channel: degrees (rotations, directions) or metres (translations)."""
    kind = CHANNELS[name][0]
    if kind == "rotation_6d_columns":
        return geodesic_degrees(a, b)
    if kind == "unit_vector":
        return unit_angle_degrees(a, b)
    if kind == "translation":
        return np.linalg.norm(a - b, axis=-1)
    raise ValueError(f"{name}: no motion distance for {kind} channels")


ERROR_NAME = {"rotation_6d_columns": ("geodesic_deg", 1.), "unit_vector": ("angle_deg", 1.),
              "translation": ("position_cm", 100.)}


def angular_speeds(state: MultichannelMotion, name: str) -> np.ndarray:
    """Speed between consecutive frames where an element is valid in both (deg/s, or m/s); NaN elsewhere."""
    channel = state.channels[name]
    dt = np.diff(state.timestamps)[:, None]
    angle = distance(name, channel.values[1:], channel.values[:-1])
    both = channel.valid[1:] & channel.valid[:-1]
    return np.where(both, angle / np.where(dt > 0, dt, np.nan), np.nan)


def clip_features(state: MultichannelMotion, channels=ROTATION_CHANNELS) -> np.ndarray:
    """Mean 6D pose, temporal std of 6D values and mean angular speed of every joint."""
    parts = []
    for name in channels:
        channel = state.channels[name]
        values = np.where(channel.valid[..., None], channel.values, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(values, axis=0)
            spread = np.nanstd(values, axis=0)
            speed = np.nanmean(angular_speeds(state, name), axis=0)
        parts += [mean.ravel(), spread.ravel(), speed.ravel()]
    return np.concatenate(parts)


def frechet_distance(generated: np.ndarray, reference: np.ndarray, *, dimensions: int = 16) -> float:
    """Fréchet distance of Gaussian fits, in the reference's leading PCA components.

    Rows with missing features are dropped; missing columns are filled with the
    reference mean so that both sets share one basis.
    """
    fill = np.nanmean(reference, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.)
    reference = np.where(np.isfinite(reference), reference, fill)
    generated = np.where(np.isfinite(generated), generated, fill)
    # Features mix units (6D components, degrees per second): standardise every column
    # by the reference's mean and spread so that no unit dominates the basis.
    centre, spread = reference.mean(0), reference.std(0)
    spread = np.where(spread > 1e-12, spread, 1.)
    reference, generated = (reference - centre) / spread, (generated - centre) / spread
    centre = reference.mean(0)
    _, _, vt = np.linalg.svd(reference - centre, full_matrices=False)
    basis = vt[:min(dimensions, len(vt))].T
    a, b = (generated - centre) @ basis, (reference - centre) @ basis
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    with warnings.catch_warnings():          # as disp=False did: a singular product is handled by .real below
        warnings.simplefilter("ignore", linalg.LinAlgWarning)
        root = linalg.sqrtm(cov_a @ cov_b)   # SciPy 1.17 deprecates disp; same algorithm, same values
    root = root.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def compare(generated: list[MultichannelMotion], reference: list[MultichannelMotion],
            channels=ROTATION_CHANNELS) -> dict:
    """Metrics of generated clips against their references (paired by position, same clock)."""
    if len(generated) != len(reference) or not generated:
        raise ValueError("generated and reference clips must be paired and nonempty")
    out = {}
    for name in channels:
        errors, speed_g, speed_r, spread_g, spread_r = [], [], [], [], []
        for g, r in zip(generated, reference):
            frames = min(len(g.timestamps), len(r.timestamps))
            gv, rv = g.channels[name].values[:frames], r.channels[name].values[:frames]
            valid = r.channels[name].valid[:frames] & g.channels[name].valid[:frames]
            if valid.any():
                errors.append(distance(name, gv, rv)[valid])
            speed_g.append(np.nanmedian(angular_speeds(g, name)) if g.channels[name].valid.any() else np.nan)
            speed_r.append(np.nanmedian(angular_speeds(r, name)) if r.channels[name].valid.any() else np.nan)
            for state, sink in ((g, spread_g), (r, spread_r)):
                ch = state.channels[name]
                values = np.where(ch.valid[..., None], ch.values, np.nan)
                with np.errstate(all="ignore"):
                    sink.append(np.nanmedian(np.nanstd(values, axis=0)))
        error_name, factor = ERROR_NAME[CHANNELS[name][0]]
        with np.errstate(all="ignore"):
            out[name] = {
                error_name: factor * float(np.median(np.concatenate(errors))) if errors else float("nan"),
                "speed_ratio": float(np.nanmedian(speed_g) / np.nanmedian(speed_r)),
                "variation_ratio": float(np.nanmedian(spread_g) / np.nanmedian(spread_r))}
    out["frechet"] = frechet_distance(np.stack([clip_features(g, channels) for g in generated]),
                                      np.stack([clip_features(r, channels) for r in reference]))
    return out


def high_frequency_rms(state: MultichannelMotion, name: str, *, cutoff_hz: float = 6., min_valid: float = .9) -> float | None:
    """Amplitude of a channel's motion at or above ``cutoff_hz`` (the visible shake): per element with valid frames on
    at least ``min_valid`` of the clip, the RMS of its mean-removed valid samples' spectral components f >= cutoff
    (Parseval: sqrt( 2 Σ_{f ≥ cutoff} |X(f)|² ) / N), and the median over elements, in the channel's own units
    (6D rotation components, metres). Signing lives below ~3 Hz; energy near Nyquist (12 Hz at 24 fps) is
    frame-to-frame jitter. An absolute amplitude, not a share: smoothing lowers it monotonically, while a share can
    rise when smoothing also removes mid-band motion. ``None`` when no element has enough valid frames."""
    if name not in state.channels:
        return None
    channel = state.channels[name]
    times = np.asarray(state.timestamps, float)
    fps = 1. / float(np.median(np.diff(times))) if len(times) > 1 else 30.
    values = np.asarray(channel.values, float)
    valid = np.asarray(channel.valid, bool)
    out = []
    for j in range(values.shape[1]):
        ok = valid[:, j]
        if ok.mean() < min_valid or ok.sum() < 16:
            continue
        x = values[ok, j].reshape(int(ok.sum()), -1)
        x = x - x.mean(0)
        spectrum = np.fft.rfft(x, axis=0)
        f = np.fft.rfftfreq(len(x), 1. / fps)
        out.append(float(np.sqrt(2. * (np.abs(spectrum[f >= cutoff_hz]) ** 2).sum()) / len(x)))
    return float(np.median(out)) if out else None


__all__ = ["geodesic_degrees", "unit_angle_degrees", "distance", "angular_speeds", "clip_features", "frechet_distance",
           "compare", "high_frequency_rms", "ROTATION_CHANNELS"]
