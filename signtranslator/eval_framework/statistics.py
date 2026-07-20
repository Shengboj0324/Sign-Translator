"""Statistical rigor for trained comparisons (Doc-12 §3).

Paired permutation test + paired t-statistic, exact sign test, percentile bootstrap
CI, multi-seed aggregation, and the minimum-meaningful-effect gate. Self-contained
(no special functions): the paired test is an exact sign-flip permutation, the sign
test an exact binomial tail.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


def paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D of equal length")
    return a - b


def paired_t_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """t = mean(d) / (sd(d)/sqrt(n)), ddof=1 (descriptive)."""
    d = paired_differences(a, b)
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 pairs")
    dbar = d.mean()
    s = d.std(ddof=1)
    if s == 0.0:
        return 0.0 if dbar == 0.0 else math.copysign(math.inf, dbar)
    return float(dbar / (s / math.sqrt(n)))


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float],
                              max_exact: int = 16, num_samples: int = 20000,
                              seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value on the paired differences.

    Under the symmetric null each |d_i| keeps its magnitude but its sign is random.
    p = P(|mean(±|d|)| >= |mean(d)|). Exact enumeration of 2^n flips for n<=max_exact,
    otherwise Monte-Carlo (the observed assignment is always included).
    """
    d = paired_differences(a, b)
    n = d.size
    if n == 0:
        raise ValueError("no pairs")
    mag = np.abs(d)
    observed = abs(d.mean())
    tol = 1e-12
    if n <= max_exact:
        count = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            if abs(np.dot(signs, mag) / n) >= observed - tol:
                count += 1
        return count / total
    rng = np.random.default_rng(seed)
    hits = 1                                    # include observed
    for _ in range(num_samples - 1):
        s = rng.choice((-1.0, 1.0), size=n)
        if abs(np.dot(s, mag) / n) >= observed - tol:
            hits += 1
    return hits / num_samples


def sign_test_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided sign test on the count of positive paired differences."""
    d = paired_differences(a, b)
    nonzero = d[d != 0.0]
    n = nonzero.size
    if n == 0:
        return 1.0
    k = int((nonzero > 0).sum())
    def cdf_le(x):
        return sum(math.comb(n, i) for i in range(0, x + 1)) / (2 ** n)
    p_le = cdf_le(k)
    p_ge = 1.0 - cdf_le(k - 1)
    return min(1.0, 2.0 * min(p_le, p_ge))


def bootstrap_ci(values: Sequence[float], stat_fn: Callable[[np.ndarray], float]
                 = np.mean, alpha: float = 0.05, num_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic of ``values``."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("empty sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    rng = np.random.default_rng(seed)
    stats = np.empty(num_boot)
    for i in range(num_boot):
        stats[i] = stat_fn(rng.choice(x, size=x.size, replace=True))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


@dataclass(frozen=True)
class SeedSummary:
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n_seeds: int


def aggregate_seeds(values: Sequence[float], alpha: float = 0.05,
                    seed: int = 0) -> SeedSummary:
    """Mean, sd, and bootstrap CI across (>=3 recommended) seed results."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 1:
        raise ValueError("need at least one seed")
    lo, hi = bootstrap_ci(x, np.mean, alpha=alpha, seed=seed)
    return SeedSummary(float(x.mean()),
                       float(x.std(ddof=1)) if x.size > 1 else 0.0,
                       lo, hi, int(x.size))


def significant_and_meaningful(effect: float, min_effect: float,
                               pvalue: float, alpha: float = 0.05) -> bool:
    """A result counts only if it is BOTH significant AND >= the minimum effect.

    Either condition alone is insufficient (the document's pre-registered minimum
    meaningful effect combined with a significance test).
    """
    if min_effect < 0 or not 0.0 < alpha < 1.0:
        raise ValueError("bad min_effect/alpha")
    return (abs(effect) >= min_effect) and (pvalue <= alpha)
