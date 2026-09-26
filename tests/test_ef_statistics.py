"""Adversarial tests for statistical rigor (Doc-12, stage 12c)."""

import math

import numpy as np
import pytest

from signtranslator.eval_framework.statistics import (
    paired_differences, paired_t_statistic, paired_permutation_pvalue,
    sign_test_pvalue, bootstrap_ci, aggregate_seeds, significant_and_meaningful,
)


def test_paired_permutation_exact_hand_case():
    # d = [1,2,3]: only +++ and --- reach |mean|>=2 out of 8 flips => p = 0.25.
    a, b = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    assert paired_permutation_pvalue(a, b) == pytest.approx(0.25, abs=1e-12)


def test_sign_test_exact_hand_case():
    # 3 positive differences of 3: two-sided = 2 * 1/8 = 0.25.
    a, b = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    assert sign_test_pvalue(a, b) == pytest.approx(0.25, abs=1e-12)


def test_permutation_pvalue_is_a_probability():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 1.0, 12)
    b = rng.normal(0.0, 1.0, 12)
    p = paired_permutation_pvalue(a, b)
    assert 0.0 <= p <= 1.0


def test_large_consistent_effect_is_significant():
    # b uniformly larger than a by ~1 across many pairs -> tiny p-value.
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, 40)
    b = a + 1.0
    assert paired_permutation_pvalue(a, b, max_exact=10) < 0.01
    assert sign_test_pvalue(a, b) < 0.01


def test_no_effect_is_not_significant():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = list(a)                                          # identical -> all ties
    assert sign_test_pvalue(a, b) == 1.0


def test_t_statistic_sign_and_zero():
    assert paired_t_statistic([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) > 0   # a > b
    assert paired_t_statistic([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_bootstrap_ci_degenerate_constant():
    lo, hi = bootstrap_ci([5.0, 5.0, 5.0, 5.0])
    assert lo == hi == 5.0


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(10.0, 1.0, 200)
    lo, hi = bootstrap_ci(x, seed=3)
    assert lo < x.mean() < hi


def test_aggregate_seeds():
    s = aggregate_seeds([0.80, 0.82, 0.78], seed=0)
    assert s.n_seeds == 3
    assert s.mean == pytest.approx(0.80, abs=1e-9)
    assert s.ci_low <= s.mean <= s.ci_high


def test_significant_and_meaningful_requires_both():
    # significant but tiny effect -> not meaningful.
    assert not significant_and_meaningful(effect=0.001, min_effect=0.02, pvalue=0.001)
    # meaningful effect but not significant -> insufficient.
    assert not significant_and_meaningful(effect=0.05, min_effect=0.02, pvalue=0.20)
    # both -> counts.
    assert significant_and_meaningful(effect=0.05, min_effect=0.02, pvalue=0.001)


@pytest.mark.parametrize("wins,n", [(60, 60), (0, 60), (83, 100), (500, 1000)])
def test_sign_tail_matches_independent_integer_oracle(wins, n):
    # Closed binomial sum, independently evaluated with math.comb.
    expected = min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, n-wins)+1)) / 2**n)
    actual = sign_test_pvalue([1.] * wins + [-1.] * (n-wins), [0.] * n)
    assert actual == expected
    assert actual > 0


@pytest.mark.parametrize("factor", [1e-100, 1., 1e100])
def test_permutation_scale_invariance(factor):
    assert paired_permutation_pvalue(np.array([1., 2., 3.]) * factor, [0.]*3) == .25


@pytest.mark.parametrize("bad", [[float('nan'), 1], [float('inf'), 1], [], [[1., 2.]], [True, False], ['1','2'], [1j,2j]])
@pytest.mark.parametrize("fn", [paired_differences, paired_t_statistic, paired_permutation_pvalue, sign_test_pvalue])
def test_paired_helpers_reject_invalid_evidence(fn, bad):
    with pytest.raises(ValueError):
        fn(bad, [0., 0.])


@pytest.mark.parametrize("kwargs", [{'num_samples':0}, {'num_samples':1}, {'num_samples':2.5}, {'max_exact':True}, {'max_exact':-1}, {'max_exact':21}, {'seed':-1}])
def test_permutation_rejects_invalid_controls_even_for_exact_case(kwargs):
    with pytest.raises(ValueError):
        paired_permutation_pvalue([1.,2.],[0.,0.],**kwargs)


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -1., 1.1, True])
def test_significance_gate_rejects_invalid_probability(bad):
    with pytest.raises(ValueError):
        significant_and_meaningful(1., .1, bad)


@pytest.mark.parametrize("kwargs", [{'num_boot':0}, {'num_boot':True}, {'alpha':float('nan')}, {'seed':-1}, {'stat_fn':lambda x: float('nan')}, {'stat_fn':lambda x: [1.,2.]}])
def test_bootstrap_rejects_invalid_controls_or_statistics(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([1.,2.],**kwargs)


def test_overflowing_pairs_rejected_and_large_t_is_stable():
    with pytest.raises(ValueError):
        paired_differences([1e308],[-1e308])
    assert paired_t_statistic([1e300,2e300,3e300],[0.]*3) == pytest.approx(math.sqrt(12))
