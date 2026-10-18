"""Split-conformal bounds and abstention: exact rank, guarantee, no hidden reads."""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from signtranslator.generation.conformal_abstention import (
    conformal_quantile, evaluate_abstention, fit_conformal_bound, span_error_and_difficulty,
)
from signtranslator.generation.representation_comparison import interpolation_baseline
from tests.test_generation_masked_motion import hidden_frames, state


def test_quantile_rank_is_exact_including_the_infinite_case():
    scores = np.arange(1., 10.)  # n = 9
    assert conformal_quantile(scores, .1) == 9.          # k = ceil(10 * 0.9) = 9
    assert conformal_quantile(scores[:8], .1) == math.inf  # k = ceil(9 * 0.9) = 9 > 8
    assert conformal_quantile(np.arange(1., 20.), .1) == 18.  # k = ceil(20 * 0.9) = 18
    assert conformal_quantile(np.arange(1., 20.)[::-1], .1) == 18.  # order-free
    with pytest.raises(ValueError):
        conformal_quantile([], .1)
    with pytest.raises(ValueError):
        conformal_quantile([1., np.nan], .1)


def test_coverage_guarantee_holds_on_exchangeable_heteroscedastic_data():
    rng = np.random.default_rng(0)
    alpha, n, trials = .1, 50, 4000
    covered, violations = [], []
    for _ in range(trials):
        difficulty = rng.gamma(2., 1., n + 1)
        errors = np.abs(rng.normal(0, .5 + difficulty))
        bound = fit_conformal_bound(errors[:n], difficulty[:n], alpha=alpha, floor=.1)
        limit = bound.bound(difficulty[n:])[0]
        covered.append(errors[n] <= limit)
        violations.append(limit <= 2. and errors[n] > 2.)
    coverage = np.mean(covered)
    se = math.sqrt(.1 * .9 / trials)
    assert 1 - alpha - 3 * se <= coverage <= 1 - alpha + 1 / (n + 1) + 3 * se
    assert np.mean(violations) <= alpha + 3 * se  # P(accept and error > tau) <= alpha


def test_abstention_summary_counts_exactly():
    bound = fit_conformal_bound(np.array([1., 2., 3., 4.]), np.zeros(4), alpha=.2, floor=1.)
    assert bound.quantile == 4.  # k = ceil(5 * 0.8) = 4
    result = evaluate_abstention(np.array([1., 5., 3.]), np.array([0., 0., 1.]), bound, tolerance=4.)
    # bounds 4, 4, 8: accept the first two; the second misses the tolerance.
    assert result["abstention_rate"] == pytest.approx(1 / 3)
    assert result["accepted_violation_rate"] == pytest.approx(1 / 3)
    assert result["coverage"] == pytest.approx(2 / 3)


def test_span_difficulty_never_reads_hidden_targets():
    s = state(3, curvature=6.)
    hidden = hidden_frames()
    recon = interpolation_baseline(s, hidden, allow_inferred=True)
    values, covered = recon["body"]
    first = span_error_and_difficulty(s, "body", hidden["body"], values, covered, allow_inferred=True)
    channel = s.channels["body"]
    poisoned = np.where(hidden["body"][..., None], np.roll(channel.values, 3, axis=0), channel.values)
    s2 = replace(s, channels={**s.channels, "body": replace(channel, values=poisoned)})
    second = span_error_and_difficulty(s2, "body", hidden["body"], values, covered, allow_inferred=True)
    assert first[1] == second[1] and first[0] != second[0]
    assert first[0] > 0 and first[1] > 0


def test_unscored_span_returns_none_and_inputs_are_checked():
    s = state()
    hidden = hidden_frames((0, 1))  # no left neighbour: nothing covered or scored
    values, covered = interpolation_baseline(s, hidden, allow_inferred=True)["body"]
    assert span_error_and_difficulty(s, "body", hidden["body"], values, covered, allow_inferred=True) is None
    with pytest.raises(ValueError):
        fit_conformal_bound(np.ones(3), np.ones(3), alpha=.1, floor=0.)
    with pytest.raises(ValueError):
        fit_conformal_bound(-np.ones(3), np.ones(3), alpha=.1, floor=.1)
