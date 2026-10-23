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


def damp_rotation_channel(state: MultichannelMotion, name: str, factor: float) -> MultichannelMotion:
    """``state`` with channel ``name``'s rotations pulled toward the clip's mean rotation: per joint,

        R'_t = R̄ · exp( factor · log( R̄ᵀ R_t ) ),

    with R̄ the chordal mean of the valid frames (``project_to_rotation``). ``factor`` 1 is the identity. Below 1 it
    shrinks every excursion (nods, shakes, turns) by that factor while keeping their timing. Smoothing removes speed;
    damping removes amplitude: round 3's answer to the full prior's head over-activity on unseen signers (R6: nod rate
    19.8x, head variation 2.3x). Invalid frames are unchanged."""
    if factor == 1:
        return state
    from ..pose.rotations import axis_angle_to_matrix, matrix_to_axis_angle
    channel = state.channels[name]
    values = np.asarray(channel.values, float)
    valid = np.asarray(channel.valid, bool)
    identity = np.array([1., 0., 0., 0., 1., 0.])
    mats = rotation_6d_to_matrix(torch.from_numpy(np.where(valid[..., None], values, identity))).numpy()
    w = valid.astype(float)[..., None, None]
    mean = project_to_rotation((mats * w).sum(0) / np.maximum(w.sum(0), 1e-12))            # (J, 3, 3)
    rel = np.einsum("jba,tjbc->tjac", mean, mats)                                            # R̄ᵀ R_t
    aa = matrix_to_axis_angle(torch.from_numpy(rel)).numpy() * factor
    damped = np.einsum("jab,tjbc->tjac", mean, axis_angle_to_matrix(torch.from_numpy(aa)).numpy())
    out = np.where(valid[..., None], matrix_to_rotation_6d(torch.from_numpy(damped)).numpy(), values)
    return replace(state, channels={**state.channels, name: replace(channel, values=out.astype(values.dtype))})


__all__ = ["estimate_face_calibration", "calibrate_face", "damp_rotation_channel", "project_to_rotation",
           "smooth_rotation_channel"]


# --- face expression calibration ------------------------------------------------------------------------------------
# The full-corpus prior samples face codes near the neutral face: on validation its brow-down / inner-brow / pucker
# blendshapes vary ~3x less within a clip than the real signing, while the tokenizer reconstructs them at 0.89-0.97x
# (the loss is in sampling, not in the codes). A per-blendshape affine map restores the real range:
#
#     v' = clip( c_real + g * (v - c_gen), 0, 1 ),   g = clip( s_real / s_gen, 1, g_max ),
#
# with c the pooled median and s the mean within-clip standard deviation of each blendshape over a calibration set of
# generated clips and the real clips of the same sentences. It is a declared post-processing setting, chosen on
# validation like head smoothing and confirmed once on the test partition (strict review R6); it never reduces a
# blendshape's range (g >= 1).


def estimate_face_calibration(generated, references, *, max_gain: float = 4.) -> dict:
    def stats(states):
        labels = states[0].channels["face"].labels
        per_clip, pooled = [], []
        for s in states:
            f = s.channels["face"]
            ok = f.valid[:, 0].astype(bool)
            if ok.sum() >= 2:
                v = f.values[ok, :, 0]
                per_clip.append(v.std(0)); pooled.append(v)
        return labels, np.median(np.concatenate(pooled), 0), np.mean(per_clip, 0)
    labels, c_gen, s_gen = stats(generated)
    labels_r, c_real, s_real = stats(references)
    if tuple(labels) != tuple(labels_r):
        raise ValueError("generated and reference face channels use different blendshape bases")
    gain = np.clip(s_real / np.maximum(s_gen, 1e-6), 1., max_gain)
    return dict(labels=list(labels), center_generated=c_gen.round(6).tolist(), center_real=c_real.round(6).tolist(),
                gain=gain.round(4).tolist(), max_gain=max_gain, clips=dict(generated=len(generated), real=len(references)))


def calibrate_face(state: MultichannelMotion, calibration: dict) -> MultichannelMotion:
    f = state.channels["face"]
    if list(f.labels) != calibration["labels"]:
        raise ValueError("the calibration was estimated on another blendshape basis")
    c_gen, c_real, g = (np.asarray(calibration[k], dtype=f.values.dtype) for k in ("center_generated", "center_real", "gain"))
    values = np.clip(c_real[None, :, None] + g[None, :, None] * (f.values - c_gen[None, :, None]), 0., 1.)
    values = np.where(f.valid[..., None], values, 0.).astype(f.values.dtype)
    return replace(state, channels=dict(state.channels, face=replace(f, values=values)))
