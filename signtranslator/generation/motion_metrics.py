"""Evaluation of generated canonical motion against held-out reference motion.

Per-frame error alone rewards the conditional mean: a model that averages
plausible signings scores well on it while under-articulating every one of them.
These metrics therefore pair accuracy with articulation and realism:

* ``geodesic_deg`` — median geodesic angle between generated and reference joint
  rotations on the shared clock (lower is better);
* ``speed_ratio`` — median angular speed of generated joints over that of the
  reference (deg/s between consecutive valid frames; 1 is realistic, < 1 is the
  damped, averaged motion of a regressor);
* ``variation_ratio`` — median per-joint temporal standard deviation of the 6D
  values, generated over reference;
* ``frechet`` — Fréchet distance between the distributions of per-clip features
  (mean pose, temporal variation, mean angular speed per joint), in a PCA basis
  fitted on the reference clips (the text-to-motion "FID"; lower is better).
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from ..pose.multichannel import MultichannelMotion

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


def angular_speeds(state: MultichannelMotion, name: str) -> np.ndarray:
    """Degrees per second between consecutive frames where a joint is valid in both; NaN elsewhere."""
    channel = state.channels[name]
    dt = np.diff(state.timestamps)[:, None]
    angle = geodesic_degrees(channel.values[1:], channel.values[:-1])
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
    centre = reference.mean(0)
    _, _, vt = np.linalg.svd(reference - centre, full_matrices=False)
    basis = vt[:min(dimensions, len(vt))].T
    a, b = (generated - centre) @ basis, (reference - centre) @ basis
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
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
                errors.append(geodesic_degrees(gv, rv)[valid])
            speed_g.append(np.nanmedian(angular_speeds(g, name)) if g.channels[name].valid.any() else np.nan)
            speed_r.append(np.nanmedian(angular_speeds(r, name)) if r.channels[name].valid.any() else np.nan)
            for state, sink in ((g, spread_g), (r, spread_r)):
                ch = state.channels[name]
                values = np.where(ch.valid[..., None], ch.values, np.nan)
                with np.errstate(all="ignore"):
                    sink.append(np.nanmedian(np.nanstd(values, axis=0)))
        with np.errstate(all="ignore"):
            out[name] = dict(
                geodesic_deg=float(np.median(np.concatenate(errors))) if errors else float("nan"),
                speed_ratio=float(np.nanmedian(speed_g) / np.nanmedian(speed_r)),
                variation_ratio=float(np.nanmedian(spread_g) / np.nanmedian(spread_r)))
    out["frechet"] = frechet_distance(np.stack([clip_features(g, channels) for g in generated]),
                                      np.stack([clip_features(r, channels) for r in reference]))
    return out


__all__ = ["geodesic_degrees", "angular_speeds", "clip_features", "frechet_distance", "compare", "ROTATION_CHANNELS"]
