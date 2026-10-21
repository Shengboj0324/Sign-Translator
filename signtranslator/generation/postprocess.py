"""Post-processing of generated states: temporal smoothing of rotation channels.

The subset round's selected generator turned the head 1.87× faster than the references
(validation; 3.29× on test). The head-rotation codes of the face part jitter from step to
step at 30 code steps/s. A short centred moving average of the rotation, re-projected onto
SO(3), removes jitter faster than the window while keeping slower motion (nods,
headshakes, turns at ~1–3 Hz):

    R̄_t = Π_SO(3)( Σ_{|k| ≤ h, valid} R_{t+k} / n_t ),

where Π is the projection onto SO(3) by the polar factor (SVD, with the determinant sign
fixed). This is the chordal L2 mean, which equals the geodesic mean to second order for
small spreads. Only valid frames enter a window, and invalid frames stay as they are. The
window ``seconds`` is a sampling setting, chosen on validation like guidance and
temperature.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..pose.multichannel import MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d, rotation_6d_to_matrix


def project_to_rotation(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) nearest rotation matrices (polar factor, det +1)."""
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


def smooth_rotation_channel(state: MultichannelMotion, name: str, seconds: float) -> MultichannelMotion:
    """``state`` with channel ``name`` (6D rotations) smoothed by a centred window of ``seconds``."""
    if seconds <= 0:
        return state
    channel = state.channels[name]
    times = np.asarray(state.timestamps, float)
    fps = 1. / float(np.median(np.diff(times))) if len(times) > 1 else 30.
    half = max(int(round(seconds * fps / 2)), 0)
    if half == 0:
        return state
    values = np.asarray(channel.values, float)                      # (T, J, 6)
    valid = np.asarray(channel.valid, bool)                         # (T, J)
    identity = np.array([1., 0., 0., 0., 1., 0.])
    safe = np.where(valid[..., None], values, identity)
    mats = rotation_6d_to_matrix(torch.from_numpy(safe)).numpy()    # (T, J, 3, 3)
    weights = valid.astype(float)
    t = len(times)
    csum = np.concatenate([np.zeros((1,) + mats.shape[1:]), np.cumsum(mats * weights[..., None, None], 0)], 0)
    wsum = np.concatenate([np.zeros((1,) + weights.shape[1:]), np.cumsum(weights, 0)], 0)
    lo = np.clip(np.arange(t) - half, 0, t)
    hi = np.clip(np.arange(t) + half + 1, 0, t)
    total = csum[hi] - csum[lo]
    count = (wsum[hi] - wsum[lo])[..., None, None]
    mean = np.where(count > 0, total / np.maximum(count, 1e-12), mats)
    smoothed = matrix_to_rotation_6d(torch.from_numpy(project_to_rotation(mean))).numpy()
    out = np.where(valid[..., None], smoothed, values)
    return replace(state, channels={**state.channels, name: replace(channel, values=out.astype(values.dtype))})


__all__ = ["project_to_rotation", "smooth_rotation_channel"]
