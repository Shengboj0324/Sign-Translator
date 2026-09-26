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
from typing import Callable, Sequence, Tuple

import numpy as np


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a non-empty real numeric vector")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite binary64 values")
    return result


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds binary64 range") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = _finite_vector(a, "a")
    b = _finite_vector(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must be 1-D of equal length")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = a - b
    if not np.isfinite(differences).all():
        raise ValueError("paired differences exceed finite binary64 range")
    return differences


def paired_t_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """t = mean(d) / (sd(d)/sqrt(n)), ddof=1 (descriptive)."""
    d = paired_differences(a, b)
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 pairs")
    # The t statistic is scale invariant. Normalize to avoid overflow in
    # sums/squares without changing constant-difference behavior.
    scale = np.max(np.abs(d))
    if scale == 0.0:
        return 0.0
    d = d / scale
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
    max_exact = _integer(max_exact, "max_exact")
    if max_exact > 20:
        raise ValueError("max_exact must be <= 20 to bound exact enumeration")
    num_samples = _integer(num_samples, "num_samples", minimum=2)
    seed = _integer(seed, "seed")
    d = paired_differences(a, b)
    n = d.size
    scale = np.max(np.abs(d))
    if scale == 0.0:
        return 1.0
    d = d / scale
    mag = np.abs(d)
    observed = abs(d.mean())
    # Relative to normalized data, not an absolute 1e-12 in user units.
    tol = 8 * np.finfo(np.float64).eps
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
    # Symmetry selects the SMALL tail directly; subtracting a CDF near one
    # destroys representable small probabilities. Integer binomial arithmetic
    # postpones rounding until the final division. Values below binary64's
    # subnormal range may still underflow legitimately.
    tail = 0
    coefficient = 1
    for i in range(min(k, n - k) + 1):
        if i:
            coefficient = coefficient * (n - i + 1) // i
        tail += coefficient
    return min(1.0, (2 * tail) / (1 << n))


def bootstrap_ci(values: Sequence[float], stat_fn: Callable[[np.ndarray], float]
                 = np.mean, alpha: float = 0.05, num_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic of ``values``."""
    x = _finite_vector(values, "values")
    alpha = _scalar(alpha, "alpha")
    num_boot = _integer(num_boot, "num_boot", minimum=2)
    seed = _integer(seed, "seed")
    if not callable(stat_fn):
        raise ValueError("stat_fn must be callable")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    rng = np.random.default_rng(seed)
    stats = np.empty(num_boot)
    for i in range(num_boot):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            statistic = stat_fn(rng.choice(x, size=x.size, replace=True))
        stats[i] = _scalar(statistic, "bootstrap statistic")
    with np.errstate(over="ignore", invalid="ignore"):
        lo = _scalar(np.percentile(stats, 100 * alpha / 2), "lower bootstrap quantile")
        hi = _scalar(np.percentile(stats, 100 * (1 - alpha / 2)), "upper bootstrap quantile")
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
    x = _finite_vector(values, "seed values")
    lo, hi = bootstrap_ci(x, np.mean, alpha=alpha, seed=seed)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = _scalar(x.mean(), "seed mean")
        std = _scalar(x.std(ddof=1), "seed standard deviation") if x.size > 1 else 0.0
    return SeedSummary(mean, std, lo, hi, int(x.size))


def significant_and_meaningful(effect: float, min_effect: float,
                               pvalue: float, alpha: float = 0.05) -> bool:
    """A result counts only if it is BOTH significant AND >= the minimum effect.

    Either condition alone is insufficient (the document's pre-registered minimum
    meaningful effect combined with a significance test).
    """
    effect = _scalar(effect, "effect")
    min_effect = _scalar(min_effect, "min_effect")
    pvalue = _scalar(pvalue, "pvalue")
    alpha = _scalar(alpha, "alpha")
    if not 0.0 <= pvalue <= 1.0:
        raise ValueError("pvalue must be in [0,1]")
    if min_effect < 0 or not 0.0 < alpha < 1.0:
        raise ValueError("bad min_effect/alpha")
    return (abs(effect) >= min_effect) and (pvalue <= alpha)
