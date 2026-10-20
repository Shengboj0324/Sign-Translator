"""Prosodic structure: movement units, holds, blinks and phrase-boundary evidence.

Without glosses, the manual stream is segmented into *movement units* at local
minima of the summed wrist speed (a hold is a run of frames below ``HOLD_SPEED``),
the standard movement–hold view of the sign stream. Tang et al. (2010) found
phrase-final signs about twice as long as phrase-internal ones (units are taken
within the *active range*, the frames with a wrist in signing space, because clip
edges include the hands' rise from and return to rest) and a blink at 48 of
48 ASL phrase right edges, and counted a boundary when at least two cues co-occur;
:func:`boundary_evidence` applies that rule to units: (1) duration ≥ 2× the clip's
median unit, (2) a blink beginning in the unit's right half or the following
transition, (3) a brow change |Δz| ≥ ``BROW_JUMP_Z``, (4) a head change ≥ ``HEAD_JUMP_DEG``.
"""
from __future__ import annotations

import numpy as np

from .nmm import hysteresis_spans, smooth

HOLD_SPEED = .35                 # shoulder widths per second
BROW_JUMP_Z = 1.5
HEAD_JUMP_DEG = 8.
BLINK_Z = 3.                     # robust z of the eyelid-closure score
REST_ELEVATION = -1.1            # wrist height (shoulder widths below the chest) under which a hand is at rest


def movement_units(speed: np.ndarray, valid: np.ndarray, fps: float, *, min_frames: int = 3,
                   span: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """[start, stop) units split at local minima of the smoothed speed, within ``span`` (the active range)."""
    s = smooth(speed, valid, fps, hz=6.)
    lo, hi = span if span is not None else (0, len(s))
    minima = [t for t in range(max(1, lo), min(len(s) - 1, hi)) if valid[t] and s[t] <= s[t - 1] and s[t] < s[t + 1]]
    cuts = [lo] + minima + [hi]
    units = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_frames]
    return units


def blink_onsets(blink: np.ndarray, valid: np.ndarray, fps: float) -> list[int]:
    """Onsets of eyelid closures: the closure score's robust z (per clip) above ``BLINK_Z``.

    Absolute scores are unreliable at How2Sign's face resolution (MediaPipe's eyeBlink
    rarely passes 0.5 there), so a blink is a departure from the signer's own open-eye
    level, lasting at least 40 ms.
    """
    from .nmm import robust_z
    z, _ = robust_z(blink, valid)
    spans = hysteresis_spans(z, valid, fps, on=BLINK_Z, off=BLINK_Z * .5, min_s=.04)
    return [a for a, _ in spans]


def active_range(wrist_heights: np.ndarray, valid: np.ndarray) -> tuple[int, int] | None:
    """[first, last + 1) frames with at least one wrist in signing space (above ``REST_ELEVATION``)."""
    up = valid & (wrist_heights > REST_ELEVATION).any(axis=1)
    idx = np.nonzero(up)[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None


def boundary_evidence(units, blinks, brow_z: np.ndarray, head_yaw_deg: np.ndarray, *,
                      final_window: int = 0) -> list[dict]:
    """Per unit: the four cues and whether at least two hold (Tang et al.'s rule).

    The blink cue covers the unit's right half and the following transition; for the
    last unit the transition is ``final_window`` frames (the hands' return to rest).
    """
    if not units:
        return []
    median = float(np.median([b - a for a, b in units]))
    out = []
    for i, (a, b) in enumerate(units):
        nxt = units[i + 1][0] if i + 1 < len(units) else b + final_window
        right = (a + b) // 2
        cues = dict(lengthened=(b - a) >= 2 * median,
                    blink=any(right <= t < max(nxt, b) + 1 for t in blinks),
                    brow=abs(float(brow_z[min(b, len(brow_z) - 1)] - brow_z[a])) >= BROW_JUMP_Z,
                    head=abs(float(head_yaw_deg[min(b, len(head_yaw_deg) - 1)] - head_yaw_deg[a])) >= HEAD_JUMP_DEG)
        out.append(dict(unit=(a, b), boundary=sum(cues.values()) >= 2, **cues))
    return out


def final_lengthening(units) -> float | None:
    """Duration of the last unit over the median of the others (Tang et al.: ≈ 2 at sentence ends)."""
    if len(units) < 3:
        return None
    d = [b - a for a, b in units]
    return float(d[-1] / np.median(d[:-1]))


__all__ = ["movement_units", "blink_onsets", "active_range", "boundary_evidence", "final_lengthening"]
