"""Resample a ``MultichannelMotion`` to a uniform frame rate.

The corpora arrive at different and sometimes variable rates (How2Sign 24 fps, ASL Citizen 25-30 fps, ASLLVD 60 fps).
Models that read frames, not seconds, see the same sign at different speeds across corpora; resampling every state to
one rate removes that difference before training or scoring.

New frames sit at ``t0 + k / fps`` for every ``k`` with the time inside the clip. Each value is interpolated linearly
between the two bracketing source frames, then put back on its manifold: 6D rotation columns are re-orthonormalised
(Gram-Schmidt on the two columns), unit vectors renormalised, bounded scalars clipped to [0, 1]. A new frame is valid
(and observed) only where both bracketing source frames are; its confidence is the smaller of the two. An invalid frame
carries the canonical absent payload: zero values and zero confidence.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .multichannel import CHANNELS, MotionChannel, MultichannelMotion


def _orthonormal_6d(values: np.ndarray) -> np.ndarray:
    a, b = values[..., 0:3], values[..., 3:6]
    a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9, None)
    return np.concatenate([a, b], axis=-1)


def resample_motion(state: MultichannelMotion, fps: float) -> MultichannelMotion:
    if not fps > 0:
        raise ValueError("fps must be positive")
    t = state.timestamps
    if len(t) == 1:
        return state
    new_t = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fps + 1e-9)) + 1, dtype=np.float64) / fps
    hi = np.clip(np.searchsorted(t, new_t, side="right"), 1, len(t) - 1)
    lo = hi - 1
    w = np.clip((new_t - t[lo]) / (t[hi] - t[lo]), 0., 1.)
    channels = {}
    for name, ch in state.channels.items():
        kind = CHANNELS[name][0]
        wv = w.reshape(-1, *([1] * (ch.values.ndim - 1)))
        values = (1 - wv) * ch.values[lo] + wv * ch.values[hi]
        if kind == "rotation_6d_columns":
            values = _orthonormal_6d(values)
        elif kind == "unit_vector":
            values = values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-9, None)
        elif kind == "bounded_scalar":
            values = np.clip(values, 0., 1.)
        exact = (w == 0.)[:, None]                      # a new frame on a source frame needs only that frame
        valid = np.where(exact, ch.valid[lo], ch.valid[lo] & ch.valid[hi])
        observed = np.where(exact, ch.observed[lo], ch.observed[lo] & ch.observed[hi]) & valid
        confidence = np.where(exact, ch.confidence[lo], np.minimum(ch.confidence[lo], ch.confidence[hi]))
        confidence = np.where(valid, confidence, 0.)       # the contract: invalid values carry no confidence
        values = np.where(valid[..., None], values, 0.)    # and an unavailable payload is exactly zero
        channels[name] = replace(ch, values=values.astype(ch.values.dtype), valid=valid, observed=observed,
                                 confidence=confidence.astype(ch.confidence.dtype))
    return replace(state, timestamps=new_t, channels=channels)


__all__ = ["resample_motion"]
