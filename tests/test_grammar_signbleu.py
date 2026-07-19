"""Verification of SignBLEU (multi-channel) and inter-rater kappa.

SignBLEU's defining property vs. single-stream BLEU: it can *see* a dropped
channel. Kappa's three regimes (perfect=1, chance~0, below-chance<0) are checked
against hand-computed values.
"""

import math

import pytest

from signtranslator.grammar.signbleu import (
    sign_bleu, modified_precision, within_channel_ngrams, blended_ngrams,
    cohens_kappa, fleiss_kappa, GrammaticalityRating, agreement_on_grammaticality,
)


def _utt(manual, nonmanual=None):
    """Build a multi-channel utterance from token lists (time = index)."""
    utt = {"manual": [(t, i) for i, t in enumerate(manual)]}
    if nonmanual is not None:
        utt["nonmanual"] = [(t, i) for i, t in enumerate(nonmanual)]
    return utt


# ---------------------------------------------------------------------------
# n-gram extraction
# ---------------------------------------------------------------------------
def test_within_channel_ngrams_are_per_channel():
    utt = _utt([1, 2, 3], nonmanual=[9, 9])
    unigrams = within_channel_ngrams(utt, 1)
    assert unigrams[("manual", 1)] == 1
    assert unigrams[("nonmanual", 9)] == 2                # counted in its channel
    bigrams = within_channel_ngrams(utt, 2)
    assert bigrams[("manual", 1, 2)] == 1


def test_blended_grams_capture_co_temporal_cross_channel_tokens():
    utt = _utt([1, 2], nonmanual=[9, 9])                  # both channels at t=0,1
    blends = blended_ngrams(utt)
    assert blends[("blend", ("manual", 1), ("nonmanual", 9))] == 1
    assert blends[("blend", ("manual", 2), ("nonmanual", 9))] == 1


# ---------------------------------------------------------------------------
# SignBLEU
# ---------------------------------------------------------------------------
def test_identical_multichannel_scores_one():
    utt = _utt([1, 2, 3, 4], nonmanual=[9, 9, 0, 0])
    result = sign_bleu(utt, utt, max_n=3)
    assert abs(result.score - 1.0) < 1e-6
    assert all(abs(p - 1.0) < 1e-6 for p in result.precisions)
    assert result.brevity_penalty == 1.0


def test_dropping_a_channel_lowers_the_score():
    """The whole point: SignBLEU sees the missing non-manual channel.

    Single-stream BLEU over just the manual glosses would score these two as
    identical; SignBLEU must not.
    """
    ref = _utt([1, 2, 3], nonmanual=[9, 9, 9])
    hyp_missing_nm = _utt([1, 2, 3])                      # manual identical, no NM
    full = sign_bleu(ref, ref, max_n=2).score
    dropped = sign_bleu(hyp_missing_nm, ref, max_n=2).score
    assert dropped < full
    # and the manual-only BLEU-style score would be perfect for the manual stream
    manual_only_ref = _utt([1, 2, 3])
    assert abs(sign_bleu(manual_only_ref, manual_only_ref, max_n=2).score - 1.0) < 1e-6


def test_clipping_caps_repeated_grams():
    """A hypothesis repeating a token more than the reference gets clipped."""
    ref = _utt([1, 2])
    hyp = _utt([1, 1, 1])                                 # token 1 x3, ref has x1
    clipped, total = modified_precision(hyp, ref, 1)
    assert clipped == 1 and total == 3                    # only one '1' credited


def test_brevity_penalty_punishes_short_output():
    ref = _utt([1, 2, 3, 4, 5, 6])
    short = _utt([1, 2])                                  # correct but too short
    result = sign_bleu(short, ref, max_n=1)
    assert result.brevity_penalty < 1.0
    assert result.score < 1.0
    # BP = exp(1 - r/c) with r=6, c=2
    assert abs(result.brevity_penalty - math.exp(1 - 6 / 2)) < 1e-9


def test_no_overlap_scores_zero():
    assert sign_bleu(_utt([7, 8, 9]), _utt([1, 2, 3]), max_n=1).score == 0.0


def test_sign_bleu_rejects_bad_order():
    with pytest.raises(ValueError):
        sign_bleu(_utt([1]), _utt([1]), max_n=0)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------
def test_cohens_kappa_perfect_agreement():
    a = [1, 0, 1, 1, 0]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-12


def test_cohens_kappa_matches_hand_computation():
    # classic 2x2 example: 10 items
    a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    b = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    # p_o = agreements / 10
    po = sum(1 for x, y in zip(a, b) if x == y) / 10
    # marginals: a: 5 ones/5 zeros; b: 5 ones/5 zeros -> p_e = .5*.5 + .5*.5 = .5
    expected = (po - 0.5) / (1 - 0.5)
    assert abs(cohens_kappa(a, b) - expected) < 1e-12


def test_cohens_kappa_can_be_negative_below_chance():
    """Systematic disagreement drives kappa below 0."""
    a = [1, 0, 1, 0, 1, 0]
    b = [0, 1, 0, 1, 0, 1]                                # always disagree
    assert cohens_kappa(a, b) < 0.0


def test_cohens_kappa_near_zero_at_chance():
    """Independent random-ish raters agree near the chance level -> kappa ~ 0."""
    a = [0, 1] * 50
    b = [0, 0, 1, 1] * 25                                 # uncorrelated pattern
    assert abs(cohens_kappa(a, b)) < 0.15


def test_cohens_kappa_validates_input():
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohens_kappa([], [])


# ---------------------------------------------------------------------------
# Fleiss' kappa
# ---------------------------------------------------------------------------
def test_fleiss_kappa_perfect_agreement():
    # 3 items, 4 raters, 2 categories; all raters agree on each item
    ratings = [[4, 0], [0, 4], [4, 0]]
    assert abs(fleiss_kappa(ratings, num_categories=2) - 1.0) < 1e-9


def test_fleiss_kappa_below_chance_is_negative():
    # maximal disagreement: every item split evenly
    ratings = [[2, 2], [2, 2], [2, 2]]
    assert fleiss_kappa(ratings, num_categories=2) < 0.0


def test_fleiss_kappa_validates_rater_counts():
    with pytest.raises(ValueError):
        fleiss_kappa([[4, 0], [3, 0]], num_categories=2)   # unequal rater counts
    with pytest.raises(ValueError):
        fleiss_kappa([], num_categories=2)


# ---------------------------------------------------------------------------
# Grammaticality instrument
# ---------------------------------------------------------------------------
def test_agreement_instrument_on_two_raters():
    ratings = [
        GrammaticalityRating(0, 1, grammatical=True, meaning_preserved=True),
        GrammaticalityRating(0, 2, grammatical=True, meaning_preserved=True),
        GrammaticalityRating(1, 1, grammatical=False, meaning_preserved=True),
        GrammaticalityRating(1, 2, grammatical=False, meaning_preserved=False),
    ]
    # both raters agree on grammaticality of both items -> kappa 1
    assert abs(agreement_on_grammaticality(ratings) - 1.0) < 1e-9


def test_agreement_instrument_requires_two_raters():
    ratings = [GrammaticalityRating(0, 1, True, True),
               GrammaticalityRating(0, 2, True, True),
               GrammaticalityRating(0, 3, True, True)]
    with pytest.raises(ValueError):
        agreement_on_grammaticality(ratings)
