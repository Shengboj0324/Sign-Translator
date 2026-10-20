"""Body position in space: metric scale and session-world translations from two-view 3D.

Two-view self-calibration recovers geometry only up to scale: the triangulated
points are in camera-1 coordinates with a unit baseline. Rotations are unaffected,
but positions — where the torso and head are, how far the hands travel — need
metres. The camera rig gives no scale, so the signer's own body does.

**Anthropometric scale.** For each cue *i* (interpupillary distance, hand length,
forearm and upper-arm length) the session's robust (median) reconstructed length
``l_i`` and the population reference ``L_i`` give ``y_i = log(L_i / l_i)``, an
estimate of ``log σ`` whose error is the sum of

- the signer's deviation from the population, ``N(0, Σ_pop)`` — log SDs and
  correlations from ANSUR II (equal-sex mixture, ``scripts/derive_anthropometric_priors.py``),
  strongly correlated between limb cues (0.74–0.81), weakly with the eyes (0.35–0.43);
- a definition term ``τ_i`` — keypoints are joint centres or image features, the
  survey measures skin landmarks;
- a reconstruction term ``ρ_i`` — calibration and keypoint error of that length
  (largest for the 6 cm eye baseline);
- the sampling error of the median, ``(1.2533 · s / √n_eff)²`` with ``s`` the robust
  spread of the per-frame log lengths and ``n_eff = n / 10`` for temporal correlation;

so ``log σ`` is the generalised-least-squares mean

    log σ̂ = (1ᵀ C⁻¹ y) / (1ᵀ C⁻¹ 1),     var(log σ̂) = 1 / (1ᵀ C⁻¹ 1),

``C = D Σ_R D + diag(τ² + ρ² + sampling)``, and ``χ² = rᵀ C⁻¹ r`` of the residuals
``r = y − log σ̂`` (``n − 1`` degrees of freedom) flags sessions whose cues disagree.
On three CMU Panoptic sequences with metric dome ground truth the scale errors are
+6.4%, +5.0% and −3.6% against a predicted SD of 6.7% (``ρ_i`` were set from the
same sequences' length errors, so this checks consistency rather than tests on held-out data). The
floor is the population term: one signer's body can be off by about 6% from the population, whatever is observed.
The scale is per session; relative motion within it is exact.

**Session world frame.** Camera-1 axes turned y-up (``video_landmarks.Y_UP``; the
camera is upright but not gravity-levelled), origin at the session's median torso
anchor (shoulder-joint midpoint), metres. ``root_translation`` is the torso anchor
in that frame — leans, shifts and role-shift movement — and ``head_translation``
the cyclopean eye point (midpoint of the eyes), which is also the gaze origin.

References: ANSUR II public data (Gordon et al., US Army NSRDEC, 2012/2017), archived
in R2; upper-arm joint-centre length 278.4 mm — de Leva (1996), J. Biomech. 29(9),
mean of the male (281.7) and female (275.1) values, because ANSUR's acromion–radiale
length is measured from the acromion, about 4 cm above the shoulder joint centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .video_landmarks import Y_UP

# Goliath indices.
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1, 2, 5, 6
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 7, 8, 62, 41
LEFT_MIDDLE, RIGHT_MIDDLE = (53, 52, 51, 50), (32, 31, 30, 29)        # MCP, PIP, DIP, tip

# cue -> keypoint chains whose summed segment lengths measure it (one per body side)
CUES = {
    "interpupillary": ((LEFT_EYE, RIGHT_EYE),),
    "hand_length": ((LEFT_WRIST, *LEFT_MIDDLE), (RIGHT_WRIST, *RIGHT_MIDDLE)),
    "forearm": ((LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST)),
    "upper_arm": ((LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW)),
}
# cue -> (reference length m, population log SD, definition τ, reconstruction ρ)
POPULATION = {
    "interpupillary": (.06275, .0592, .03, .12),
    "hand_length": (.18681, .0626, .03, .035),
    "forearm": (.2538, .0799, .06, .06),
    "upper_arm": (.2784, .0655, .06, .06),
}
ANSUR_MEDIAN = {"interpupillary": .06275, "hand_length": .18681, "forearm": .2538, "upper_arm": None}
CORRELATION = {
    ("interpupillary", "upper_arm"): .35, ("interpupillary", "forearm"): .43, ("interpupillary", "hand_length"): .43,
    ("upper_arm", "forearm"): .81, ("upper_arm", "hand_length"): .74, ("forearm", "hand_length"): .78,
}
TEMPORAL_DECORRELATION = 10.          # frames per effectively independent sample
MIN_CUE_FRAMES = 20


@dataclass(frozen=True)
class MetricScale:
    scale: float                       # metres per reconstruction unit
    relative_sd: float                 # predicted SD of log(scale / true scale)
    chi2: float                        # cue disagreement, n − 1 degrees of freedom
    cues: dict                         # cue -> (median length in metres at this scale, samples used)


def _correlation(a: str, b: str) -> float:
    return 1. if a == b else CORRELATION.get((a, b), CORRELATION.get((b, a), 0.))


def chain_lengths(points3d: np.ndarray, chains) -> np.ndarray:
    """Per-frame summed segment lengths of each chain, concatenated (NaN where a joint is missing)."""
    return np.concatenate([sum(np.linalg.norm(points3d[:, c[i]] - points3d[:, c[i + 1]], axis=1)
                               for i in range(len(c) - 1)) for c in chains])


def metric_scale(points3d: np.ndarray) -> MetricScale:
    """GLS combination of the anthropometric cues over (T, K, 3) points (NaN where unusable)."""
    names, y, medians = [], [], {}
    pop_sd, extra = [], []
    for name, chains in CUES.items():
        reference, sd, definition, reconstruction = POPULATION[name]
        lengths = chain_lengths(points3d, chains)
        lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
        if len(lengths) < MIN_CUE_FRAMES:
            continue
        logs = np.log(lengths)
        median = float(np.median(logs))
        spread = 1.4826 * float(np.median(np.abs(logs - median)))       # robust SD of log lengths
        sampling = (1.2533 * spread / np.sqrt(max(len(logs) / TEMPORAL_DECORRELATION, 1.))) ** 2
        names.append(name)
        y.append(np.log(reference) - median)
        pop_sd.append(sd)
        extra.append(definition ** 2 + reconstruction ** 2 + sampling)
        medians[name] = (median, len(lengths))
    if not names:
        raise ValueError("no anthropometric cue is observed often enough to set the scale")
    y, pop_sd = np.array(y), np.array(pop_sd)
    correlation = np.array([[_correlation(a, b) for b in names] for a in names])
    covariance = np.outer(pop_sd, pop_sd) * correlation + np.diag(extra)
    weights = np.linalg.solve(covariance, np.ones(len(names)))
    precision = float(weights.sum())
    log_scale = float(weights @ y / precision)
    residual = y - log_scale
    chi2 = float(residual @ np.linalg.solve(covariance, residual))
    cues = {n: (float(np.exp(m + log_scale)), k) for n, (m, k) in medians.items()}
    return MetricScale(float(np.exp(log_scale)), float(np.sqrt(1. / precision)), chi2, cues)


def _midpoint(points3d: np.ndarray, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    mid = (points3d[:, a] + points3d[:, b]) / 2
    return mid, np.isfinite(mid).all(axis=1)


@dataclass(frozen=True)
class SessionFrame:
    """Session-world frame: scale, rotation into y-up, origin (reconstruction units)."""
    scale: float
    origin: np.ndarray                 # (3,) camera-1 reconstruction coordinates

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points - self.origin) @ Y_UP.T


def session_frame(points3d: np.ndarray, scale: MetricScale) -> SessionFrame:
    """Origin at the median torso anchor of every frame of the session."""
    anchor, ok = _midpoint(points3d, LEFT_SHOULDER, RIGHT_SHOULDER)
    if not ok.any():
        raise ValueError("the torso anchor (both shoulders) is never triangulated")
    return SessionFrame(scale.scale, np.median(anchor[ok], axis=0))


def body_translations(points3d: np.ndarray, frame: SessionFrame):
    """(root (T,3), root valid, head (T,3), head valid) in metres in the session-world frame."""
    root, root_ok = _midpoint(points3d, LEFT_SHOULDER, RIGHT_SHOULDER)
    head, head_ok = _midpoint(points3d, LEFT_EYE, RIGHT_EYE)
    to = lambda p, ok: np.where(ok[:, None], frame.to_world(np.where(ok[:, None], p, 0.)), 0.)
    return to(root, root_ok), root_ok, to(head, head_ok), head_ok


__all__ = ["CUES", "CORRELATION", "MetricScale", "metric_scale", "SessionFrame", "session_frame", "body_translations"]
