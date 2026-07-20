"""Masking strategies + interpolation-defeating certificate (Doc-11 §1).

A clip is a (T frames x P part-streams) token grid. Random point masking is easy
because adjacent frames interpolate; span, part (VideoMAE-tube analogue), and
semantic-boundary masks defeat that. `mask_interpolation_error_floor` turns "too
easy" into a checkable property: the best temporal-interpolation predictor's error
floor per masked token, exact for a known-curvature signal.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

Mask = np.ndarray  # bool (T, P); True == masked/hidden


def _empty(T: int, P: int) -> Mask:
    if T <= 0 or P <= 0:
        raise ValueError("T and P must be positive")
    return np.zeros((T, P), dtype=bool)


def random_point_mask(T: int, P: int, ratio: float, seed: int = 0) -> Mask:
    """The 'too easy' baseline: independent per-token masking at a given ratio."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0,1]")
    rng = np.random.default_rng(seed)
    return rng.random((T, P)) < ratio


def span_mask(T: int, P: int, span_len: int, num_spans: int,
              parts: Optional[Sequence[int]] = None, seed: int = 0) -> Mask:
    """Contiguous temporal runs of length ``span_len`` on selected parts."""
    if span_len < 1:
        raise ValueError("span_len must be >= 1")
    rng = np.random.default_rng(seed)
    m = _empty(T, P)
    parts = list(range(P)) if parts is None else list(parts)
    for p in parts:
        for _ in range(num_spans):
            if span_len >= T:
                m[:, p] = True
                continue
            start = int(rng.integers(0, T - span_len + 1))
            m[start:start + span_len, p] = True
    return m


def part_mask(T: int, P: int, parts_to_mask: Sequence[int]) -> Mask:
    """Mask entire part streams across all frames (the 'tube' analogue)."""
    m = _empty(T, P)
    for p in parts_to_mask:
        if not 0 <= p < P:
            raise ValueError("part index out of range")
        m[:, p] = True
    return m


def semantic_boundary_mask(T: int, P: int, boundaries: Sequence[int],
                           radius: int = 1,
                           parts: Optional[Sequence[int]] = None) -> Mask:
    """Mask tokens within ``radius`` frames of each SIR event boundary."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    m = _empty(T, P)
    parts = list(range(P)) if parts is None else list(parts)
    for b in boundaries:
        lo, hi = max(0, b - radius), min(T, b + radius + 1)
        for p in parts:
            m[lo:hi, p] = True
    return m


def mask_ratio(m: Mask) -> float:
    return float(m.mean())


# ---------------------------------------------------------------------------
# interpolation-defeating certificate (innovation)
# ---------------------------------------------------------------------------
def mask_interpolation_error_floor(m: Mask, curvature: float = 1.0) -> np.ndarray:
    """Per-token error floor of the best temporal linear-interpolation predictor.

    For a masked token at frame ``t`` in part ``p`` with nearest VISIBLE neighbours
    at temporal distance ``a`` (left) and ``b`` (right) within the same part, a C²
    signal has interpolation error ``½|x''|·a·b`` (exact for a quadratic). Returns
    an array shaped like ``m``: 0 on visible tokens, ``0.5*curvature*a*b`` on masked
    tokens, and ``+inf`` when a masked token has no visible neighbour on one side
    within its part (e.g. a fully-masked part — temporal interpolation is impossible
    and cross-part inference is forced).
    """
    T, P = m.shape
    floor = np.zeros((T, P), dtype=np.float64)
    for p in range(P):
        col = m[:, p]
        for t in range(T):
            if not col[t]:
                continue
            # nearest visible to the left
            a = None
            for d in range(1, t + 1):
                if not col[t - d]:
                    a = d
                    break
            # nearest visible to the right
            b = None
            for d in range(1, T - t):
                if not col[t + d]:
                    b = d
                    break
            if a is None or b is None:
                floor[t, p] = np.inf
            else:
                floor[t, p] = 0.5 * abs(curvature) * a * b
    return floor


def worst_masked_floor(m: Mask, curvature: float = 1.0) -> float:
    """The largest per-token interpolation floor over masked tokens (0 if none)."""
    floor = mask_interpolation_error_floor(m, curvature)
    masked = floor[m]
    return float(masked.max()) if masked.size else 0.0


def typical_masked_floor(m: Mask, curvature: float = 1.0) -> float:
    """MEDIAN per-token interpolation floor over masked tokens (0 if none).

    The robust statistic for comparing strategies: random point masking leaves most
    masked tokens with adjacent visible neighbours (a=b=1, small floor), while span
    and part masks raise the *typical* floor. The worst-token floor can be inflated
    to +inf by a single edge-masked token, so use the median to characterise a
    strategy's usual difficulty.
    """
    floor = mask_interpolation_error_floor(m, curvature)
    masked = floor[m]
    return float(np.median(masked)) if masked.size else 0.0


def is_certified_hard(m: Mask, threshold: float, curvature: float = 1.0) -> bool:
    """A mask is certified hard iff its worst masked-token floor exceeds threshold."""
    return worst_masked_floor(m, curvature) > threshold


def linear_interpolate_reconstruction(signal: np.ndarray, m: Mask) -> np.ndarray:
    """Reconstruct masked tokens by temporal linear interpolation per part.

    ``signal`` is (T, P). Masked tokens with visible neighbours on both sides are
    linearly interpolated; a masked token missing a neighbour keeps its value
    (cannot be interpolated). Used to VERIFY the error floor empirically.
    """
    T, P = m.shape
    out = signal.astype(np.float64).copy()
    for p in range(P):
        col = m[:, p]
        for t in range(T):
            if not col[t]:
                continue
            a = next((d for d in range(1, t + 1) if not col[t - d]), None)
            b = next((d for d in range(1, T - t) if not col[t + d]), None)
            if a is not None and b is not None:
                left, right = signal[t - a, p], signal[t + b, p]
                out[t, p] = (b * left + a * right) / (a + b)
    return out
