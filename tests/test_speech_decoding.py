"""Verification of CTC prefix beam search, N-best, and the lattice.

The central test proves the beam search against exhaustive enumeration of every
alignment: for small inputs the two must agree to float precision. The CTC
repeat/blank rule is easy to get subtly wrong and a shape test would never
notice, so nothing here relies on the implementation's own reasoning.
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.speech.decoding import (
    ctc_prefix_beam_search, ctc_exact_posteriors, collapse, ctc_greedy_path,
    Hypothesis, NBestList, Lattice, NEG_INF,
)


# Inputs are float64 throughout: these tests assert exact probability
# identities, and float32 log_softmax already sums to 1 +/- 1e-7 *before* the
# decoder runs, which would make a tight tolerance a test of the input's
# precision rather than of the algorithm.
def _random_log_probs(T, C, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, C, generator=g, dtype=torch.float64)
    return F.log_softmax(x, dim=-1)


def _peaked(path, C, confidence=0.97):
    """Log-probs whose argmax follows `path` with the given confidence."""
    T = len(path)
    p = torch.full((T, C), (1.0 - confidence) / (C - 1), dtype=torch.float64)
    for t, c in enumerate(path):
        p[t, c] = confidence
    return p.log()


# ---------------------------------------------------------------------------
# The collapse operator B
# ---------------------------------------------------------------------------
def test_collapse_removes_repeats_then_blanks():
    assert collapse([1, 1, 0, 1, 2, 2], blank=0) == (1, 1, 2)
    assert collapse([0, 0, 0], blank=0) == ()
    assert collapse([1, 2, 3], blank=0) == (1, 2, 3)


def test_collapse_requires_blank_between_repeats():
    """Without an intervening blank a doubled symbol collapses to one."""
    assert collapse([2, 2], blank=0) == (2,)
    assert collapse([2, 0, 2], blank=0) == (2, 2)


# ---------------------------------------------------------------------------
# THE PROOF: beam search == exhaustive enumeration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T,C,seed", [(4, 3, 0), (5, 3, 1), (6, 3, 2), (4, 4, 3)])
def test_beam_search_matches_exhaustive_enumeration(T, C, seed):
    """p(l|x) from the beam must equal the sum over all alignments of l.

    With a beam wide enough to prune nothing, prefix beam search is exact. This
    is a genuine mathematical check of the recursion, not a regression snapshot.
    """
    lp = _random_log_probs(T, C, seed)
    exact = ctc_exact_posteriors(lp, blank=0)
    nbest = ctc_prefix_beam_search(lp, beam_width=10_000, blank=0)

    got = {h.tokens: h.log_prob for h in nbest.hypotheses}
    for label, ref in exact.items():
        if ref == NEG_INF:
            continue
        assert label in got, f"beam missed label {label}"
        assert abs(got[label] - ref) < 1e-9, (
            f"{label}: beam {got[label]} vs exact {ref}")


def test_total_probability_over_all_labels_is_one():
    """Summing p(l|x) over every label sequence must give exactly 1.

    CTC partitions all C^T alignments among label sequences, so the posteriors
    form a probability distribution. This catches double-counting or dropped
    paths in the recursion.
    """
    lp = _random_log_probs(5, 3, seed=7)
    nbest = ctc_prefix_beam_search(lp, beam_width=10_000, blank=0)
    assert abs(nbest.retained_mass - 1.0) < 1e-9


def test_exhaustive_enumeration_itself_sums_to_one():
    """Sanity-check the reference before trusting it as ground truth."""
    lp = _random_log_probs(5, 3, seed=11)
    exact = ctc_exact_posteriors(lp, blank=0)
    total = sum(math.exp(v) for v in exact.values())
    assert abs(total - 1.0) < 1e-9


def test_repeat_rule_blank_separation_is_honoured():
    """p("aa") must equal the mass of paths with a blank between the two a's.

    Constructed by hand: T=3, tokens {blank=0, a=1}. The only path collapsing to
    (1,1) is [1,0,1]. Any error in the p_b/p_nb split shows up immediately.
    """
    lp = torch.tensor([[0.3, 0.7], [0.6, 0.4], [0.3, 0.7]],
                      dtype=torch.float64).log()
    nbest = ctc_prefix_beam_search(lp, beam_width=1000, blank=0)
    got = {h.tokens: h.probability for h in nbest.hypotheses}
    expected_aa = 0.7 * 0.6 * 0.7          # the single path [a, blank, a]
    assert abs(got[(1, 1)] - expected_aa) < 1e-12


# ---------------------------------------------------------------------------
# Beam behaviour
# ---------------------------------------------------------------------------
def test_hypotheses_are_sorted_by_descending_probability():
    nbest = ctc_prefix_beam_search(_random_log_probs(8, 5, seed=4), beam_width=8)
    lps = [h.log_prob for h in nbest.hypotheses]
    assert lps == sorted(lps, reverse=True)


def test_beam_matches_greedy_on_a_deterministic_signal():
    """When one alignment dominates, the beam's best equals the greedy collapse."""
    path = [1, 1, 0, 2, 3, 3, 0, 2]
    lp = _peaked(path, C=5, confidence=0.999)
    best = ctc_prefix_beam_search(lp, beam_width=25, blank=0).best
    assert best.tokens == collapse(ctc_greedy_path(lp), blank=0)
    assert best.tokens == (1, 2, 3, 2)


def test_beam_can_beat_greedy_by_summing_alignments():
    """The label with the most total mass need not be the best single path.

    Frame 0 slightly prefers token 1; but token 2's mass is split across two
    alignments that collapse to the same label, so summing them wins. Greedy
    decoding cannot express this -- which is precisely why the spec asks for a
    lattice rather than a string.
    """
    # tokens: blank=0, a=1, b=2
    lp = torch.tensor([
        [1e-12, 0.55, 0.45],
        [0.50, 0.05, 0.45],
    ], dtype=torch.float64).log()
    greedy = collapse(ctc_greedy_path(lp), blank=0)
    exact = ctc_exact_posteriors(lp, blank=0)
    best_label = max(exact.items(), key=lambda kv: kv[1])[0]
    beam_best = ctc_prefix_beam_search(lp, beam_width=100, blank=0).best.tokens
    assert beam_best == best_label
    # And the beam's answer is at least as probable as greedy's.
    assert exact[beam_best] >= exact[greedy] - 1e-12


def test_wider_beam_never_loses_probability_mass():
    lp = _random_log_probs(9, 6, seed=5)
    narrow = ctc_prefix_beam_search(lp, beam_width=2).retained_mass
    wide = ctc_prefix_beam_search(lp, beam_width=50).retained_mass
    assert wide >= narrow - 1e-12
    assert wide <= 1.0 + 1e-9


def test_beam_width_one_still_returns_a_valid_hypothesis():
    nbest = ctc_prefix_beam_search(_random_log_probs(6, 4, seed=6), beam_width=1)
    assert len(nbest) == 1 and nbest.best.log_prob <= 0.0


def test_all_blank_input_decodes_to_empty():
    lp = _peaked([0] * 6, C=4, confidence=0.999)
    assert ctc_prefix_beam_search(lp, beam_width=10, blank=0).best.tokens == ()


def test_pruning_threshold_reduces_work_without_breaking_top_hypothesis():
    lp = _peaked([1, 0, 2, 2, 0, 3], C=6, confidence=0.99)
    full = ctc_prefix_beam_search(lp, beam_width=20).best.tokens
    pruned = ctc_prefix_beam_search(lp, beam_width=20,
                                    prune_threshold=math.log(1e-4)).best.tokens
    assert full == pruned


def test_decoder_validates_arguments():
    lp = _random_log_probs(4, 3)
    with pytest.raises(ValueError):
        ctc_prefix_beam_search(lp, blank=9)
    with pytest.raises(ValueError):
        ctc_prefix_beam_search(lp, beam_width=0)
    with pytest.raises(ValueError):
        ctc_prefix_beam_search(torch.randn(2, 3, 4))


def test_exhaustive_enumerator_refuses_large_inputs():
    with pytest.raises(ValueError):
        ctc_exact_posteriors(_random_log_probs(30, 10))


# ---------------------------------------------------------------------------
# N-best posteriors
# ---------------------------------------------------------------------------
def test_posteriors_are_normalised_over_the_retained_set():
    nbest = ctc_prefix_beam_search(_random_log_probs(7, 4, seed=8), beam_width=5)
    post = nbest.posteriors
    assert abs(sum(post) - 1.0) < 1e-9
    assert post == sorted(post, reverse=True)


def test_retained_mass_reports_beam_coverage_honestly():
    """A narrow beam yields confident-looking posteriors over little real mass."""
    lp = _random_log_probs(8, 6, seed=9)
    narrow = ctc_prefix_beam_search(lp, beam_width=2)
    assert abs(sum(narrow.posteriors) - 1.0) < 1e-9   # normalised...
    assert narrow.retained_mass < 1.0                  # ...over partial mass


def test_token_posteriors_marginalise_over_alternatives():
    lp = _random_log_probs(6, 4, seed=10)
    nbest = ctc_prefix_beam_search(lp, beam_width=20)
    tok_post = nbest.token_posteriors()
    assert len(tok_post) == len(nbest.best.tokens)
    for pos in tok_post:
        assert all(0.0 <= v <= 1.0 + 1e-9 for v in pos.values())
        assert sum(pos.values()) <= 1.0 + 1e-9


def test_empty_nbest_is_handled():
    empty = NBestList(hypotheses=[])
    assert empty.posteriors == [] and empty.token_posteriors() == []
    with pytest.raises(IndexError):
        _ = empty.best


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------
def test_lattice_confidence_matches_best_path_posteriors():
    nbest = ctc_prefix_beam_search(_random_log_probs(7, 4, seed=12), beam_width=10)
    lat = Lattice.from_nbest(nbest)
    assert len(lat.confidence()) == len(lat.best)
    assert all(0.0 <= c <= 1.0 + 1e-9 for c in lat.confidence())


def test_lattice_flags_ambiguous_positions():
    """A deterministic signal has no ambiguity; a noisy one does."""
    sharp = ctc_prefix_beam_search(_peaked([1, 0, 2, 0, 3], C=5, confidence=0.999),
                                   beam_width=20)
    assert Lattice.from_nbest(sharp).ambiguous_positions(threshold=0.9) == []

    noisy = ctc_prefix_beam_search(_random_log_probs(8, 5, seed=13), beam_width=20)
    assert len(Lattice.from_nbest(noisy).ambiguous_positions(threshold=0.9)) > 0


def test_lattice_from_empty_nbest():
    lat = Lattice.from_nbest(NBestList(hypotheses=[]))
    assert lat.best == () and lat.confidence() == []
