"""Gaze toward loci and the agreement-window gaze test.

With eye point ``e(t)`` and gaze ``g(t)`` in the body frame, the angle to locus k is
``α_k(t) = arccos⟨g(t), unit(L_k − e(t))⟩`` (for direction-only loci, ``L_k − e`` is
replaced by the locus direction itself, i.e. the locus at infinity); the addressee
is +z. A frame's gaze target is the nearest of the loci and the addressee when that
angle is below ``TARGET_DEG`` (tuning; 8–15° in the report), else "other".

Thompson, Emmorey & Kluender (2006) found gaze on the object locus for 73.8% of
agreeing verbs, the syntactic object for 82.5% of backwards verbs, the locative for
72.2% / 68.2% of transitive / intransitive spatial verbs and only 11.1% of plain
verbs, starting ≈160 ms before verb onset. :func:`verb_gaze_marked` applies that
window: gaze is marked when the target locus holds for at least half the frames of
``[onset − 160 ms, onset + duration/2]``. These proportions are the reference bands
for generated motion (``EXPECTED_GAZE_RATE``).
"""
from __future__ import annotations

import numpy as np

from .frames import unit

TARGET_DEG = 12.
GAZE_LEAD_S = .16
EXPECTED_GAZE_RATE = {"agreeing": .738, "backwards": .825, "spatial_transitive": .722,
                      "spatial_intransitive": .682, "plain": .111}


def angles_to(gaze: np.ndarray, eye: np.ndarray | None, targets: np.ndarray, *, at_infinity: bool) -> np.ndarray:
    """(T, K) degrees between gaze and each target (points, or directions if ``at_infinity``)."""
    g = unit(gaze)
    if at_infinity or eye is None:
        d = unit(targets)[None].repeat(len(g), 0)
    else:
        d = unit(targets[None] - eye[:, None])
    return np.degrees(np.arccos(np.clip(np.einsum("tc,tkc->tk", g, d), -1, 1)))


def gaze_targets(gaze, eye, loci_dirs: np.ndarray, valid: np.ndarray, *, threshold: float = TARGET_DEG) -> np.ndarray:
    """(T,) target per frame: locus index k ≥ 0, −1 addressee, −2 other, −3 invalid."""
    targets = np.vstack([loci_dirs.reshape(-1, 3), [[0., 0., 1.]]])
    a = angles_to(gaze, eye, targets, at_infinity=True)
    best = a.argmin(1)
    out = np.where(a[np.arange(len(a)), best] <= threshold, best, -2)
    out = np.where(out == len(targets) - 1, -1, out)
    return np.where(valid, out, -3)


def verb_gaze_marked(targets: np.ndarray, locus: int, onset: int, stop: int, fps: float) -> bool:
    lead = int(round(GAZE_LEAD_S * fps))
    a, b = max(0, onset - lead), max(onset + 1, onset + (stop - onset) // 2)
    window = targets[a:b]
    window = window[window != -3]
    return bool(len(window)) and float(np.mean(window == locus)) >= .5


def gaze_locus_loss(gaze, target_dirs, weights):
    """Differentiable Σ_t w_t·α_t² (radians²) toward per-frame target directions (torch tensors)."""
    import torch
    g = torch.nn.functional.normalize(gaze, dim=-1)
    d = torch.nn.functional.normalize(target_dirs, dim=-1)
    cos = (g * d).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return (weights * torch.acos(cos) ** 2).sum() / weights.sum().clamp_min(1e-9)


__all__ = ["TARGET_DEG", "GAZE_LEAD_S", "EXPECTED_GAZE_RATE", "angles_to", "gaze_targets", "verb_gaze_marked",
           "gaze_locus_loss"]
