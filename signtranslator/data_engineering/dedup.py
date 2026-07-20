"""Deduplication: perceptual hashing + transcript similarity (Doc-10 §4).

Average/difference perceptual hashes with Hamming distance detect near-duplicate
frames; Jaccard n-gram and normalised edit distance detect near-duplicate
transcripts. Pairs whose distance falls in a near-threshold band are FLAGGED for
manual inspection rather than auto-removed.
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# perceptual hashing
# ---------------------------------------------------------------------------
def _downsample(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Block-mean downsample a 2D grayscale array to (out_h, out_w)."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("img must be 2D grayscale")
    rows = np.array_split(img, out_h, axis=0)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for i, r in enumerate(rows):
        cols = np.array_split(r, out_w, axis=1)
        out[i] = [c.mean() for c in cols]
    return out


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits.reshape(-1):
        v = (v << 1) | int(b)
    return v


def average_hash(img: np.ndarray, size: int = 8) -> int:
    """aHash: downsample to size×size, bit = pixel > mean. 64-bit for size=8."""
    small = _downsample(img, size, size)
    return _bits_to_int(small > small.mean())


def difference_hash(img: np.ndarray, size: int = 8) -> int:
    """dHash: downsample to size×(size+1); bit = left > right. 64-bit for size=8."""
    small = _downsample(img, size, size + 1)
    return _bits_to_int(small[:, :-1] > small[:, 1:])


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# transcript similarity
# ---------------------------------------------------------------------------
def _ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(a: Sequence[str], b: Sequence[str], n: int = 1) -> float:
    """|A∩B| / |A∪B| over token n-grams. 1.0 identical, 0.0 disjoint."""
    A, B = _ngrams(a, n), _ngrams(b, n)
    if not A and not B:
        return 1.0
    union = A | B
    return len(A & B) / len(union)


def normalized_edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Levenshtein(a, b) / max(len) over tokens, in [0, 1]."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb] / max(la, lb)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def cluster_duplicates(hashes: Sequence[int], threshold: int) -> List[List[int]]:
    """Group indices whose perceptual hashes are within ``threshold`` Hamming.

    Union-find over the near-duplicate graph; returns index groups (sorted).
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values()]


def near_threshold_pairs(hashes: Sequence[int], tau: int,
                         delta: int) -> List[Tuple[int, int, int]]:
    """Pairs whose Hamming distance is within [tau-delta, tau+delta].

    These are the ambiguous cases the document routes to MANUAL inspection
    rather than auto-deciding. Returns (i, j, distance).
    """
    out: List[Tuple[int, int, int]] = []
    lo, hi = tau - delta, tau + delta
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hamming_distance(hashes[i], hashes[j])
            if lo <= d <= hi:
                out.append((i, j, d))
    return out
