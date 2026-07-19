"""SignBLEU: multi-channel evaluation, and inter-rater agreement (kappa).

Standard BLEU scores a single token stream. Sign is multi-channel -- manual signs
plus co-occurring non-manual markers -- so a single-stream metric is blind to a
whole channel. SignBLEU (Kim et al., 2024, arXiv:2406.06648) forms n-grams that
blend *within* and *across* channels and scores modified n-gram precision with a
brevity penalty.

This implementation:

* takes each channel as a time-ordered token stream;
* builds (a) within-channel n-grams and (b) **blended** grams pairing co-temporal
  tokens across channels at each time step;
* computes modified (clipped) precision, the geometric mean over n = 1..N, and a
  brevity penalty ``BP = min(1, exp(1 - r/c))``.

Properties proved in the tests: an identical multi-channel hypothesis/reference
scores 1; dropping a whole channel lowers the score (a single-stream BLEU could
not detect that); clipping caps repeated grams; BP penalises short output.

Also here: **Cohen's** and **Fleiss'** kappa for the human grammaticality
evaluation the document requires. Both equal 1 at perfect agreement, ~0 at
chance, and can go negative below chance.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# A multi-channel utterance: channel name -> list of (token, time_index).
Channel = List[Tuple[int, int]]
MultiChannel = Dict[str, Channel]


# ---------------------------------------------------------------------------
# Gram extraction
# ---------------------------------------------------------------------------
def _channel_tokens_by_time(channel: Channel) -> List[int]:
    return [tok for tok, _ in sorted(channel, key=lambda x: x[1])]


def within_channel_ngrams(utt: MultiChannel, n: int) -> Counter:
    """n-grams formed inside each channel independently."""
    grams: Counter = Counter()
    for name, channel in utt.items():
        toks = _channel_tokens_by_time(channel)
        for i in range(len(toks) - n + 1):
            grams[(name,) + tuple(toks[i:i + n])] += 1
    return grams


def blended_ngrams(utt: MultiChannel) -> Counter:
    """Blended 1-grams: the multiset of co-temporal tokens across channels.

    At each time index we pair the tokens active in every channel, so a
    manual+non-manual co-occurrence is a distinct gram. This is exactly what a
    single-stream metric cannot see.
    """
    by_time: Dict[int, Dict[str, int]] = {}
    for name, channel in utt.items():
        for tok, t in channel:
            by_time.setdefault(t, {})[name] = tok
    grams: Counter = Counter()
    for t, chans in by_time.items():
        if len(chans) >= 2:
            key = tuple(sorted(chans.items()))           # e.g. (("manual",5),("nm",0))
            grams[("blend",) + key] += 1
    return grams


def all_ngrams(utt: MultiChannel, n: int) -> Counter:
    grams = within_channel_ngrams(utt, n)
    if n == 1:
        grams += blended_ngrams(utt)                     # blend is a co-temporal 1-gram
    return grams


# ---------------------------------------------------------------------------
# Modified precision + SignBLEU
# ---------------------------------------------------------------------------
def modified_precision(hyp: MultiChannel, ref: MultiChannel, n: int
                       ) -> Tuple[int, int]:
    """Return (clipped_matches, total_hyp_grams) for order ``n``."""
    hyp_grams = all_ngrams(hyp, n)
    ref_grams = all_ngrams(ref, n)
    clipped = sum(min(c, ref_grams.get(g, 0)) for g, c in hyp_grams.items())
    total = sum(hyp_grams.values())
    return clipped, total


def _total_tokens(utt: MultiChannel) -> int:
    return sum(len(ch) for ch in utt.values())


@dataclass
class SignBLEUResult:
    score: float
    precisions: List[float]
    brevity_penalty: float
    hyp_len: int
    ref_len: int


def sign_bleu(hyp: MultiChannel, ref: MultiChannel, max_n: int = 3,
              smooth: float = 1e-9) -> SignBLEUResult:
    """Multi-channel SignBLEU of one hypothesis against one reference."""
    if max_n < 1:
        raise ValueError("max_n must be >= 1")
    precisions: List[float] = []
    base_clipped = 0
    for n in range(1, max_n + 1):
        clipped, total = modified_precision(hyp, ref, n)
        if n == 1:
            base_clipped = clipped
        precisions.append((clipped + smooth) / (total + smooth) if total
                          else 0.0)

    c, r = _total_tokens(hyp), _total_tokens(ref)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)

    # BLEU convention: no unigram overlap means nothing matched -> exactly 0.
    # Smoothing only softens the higher-order "one missing n-gram zeroes all"
    # harshness; it must not turn a genuine no-match into a positive score.
    if base_clipped == 0 or not all(p > 0 for p in precisions):
        score = 0.0
    else:
        log_mean = sum(math.log(p) for p in precisions) / len(precisions)
        score = bp * math.exp(log_mean)
    return SignBLEUResult(score=score, precisions=precisions,
                          brevity_penalty=bp, hyp_len=c, ref_len=r)


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------
def cohens_kappa(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Cohen's kappa for two raters over categorical labels.

    kappa = (p_o - p_e) / (1 - p_e). 1 = perfect, 0 = chance, negative below.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same items")
    n = len(rater_a)
    if n == 0:
        raise ValueError("no items")
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    cats = set(rater_a) | set(rater_b)
    ca, cb = Counter(rater_a), Counter(rater_b)
    p_e = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in cats)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_o >= 1.0 - 1e-12 else 0.0        # degenerate: all one label
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings: Sequence[Sequence[int]], num_categories: int) -> float:
    """Fleiss' kappa for a fixed number of raters over many items.

    ``ratings[i][k]`` = number of raters who assigned item ``i`` to category
    ``k``. Every item must have the same number of raters.
    """
    if not ratings:
        raise ValueError("no items")
    n_items = len(ratings)
    n_raters = sum(ratings[0])
    if n_raters < 2:
        raise ValueError("need at least 2 raters")
    for row in ratings:
        if len(row) != num_categories or sum(row) != n_raters:
            raise ValueError("every item must have the same rater count and categories")

    # P_i: agreement within item i
    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in ratings]
    p_bar = sum(p_i) / n_items
    # p_j: proportion of all assignments to category j
    p_j = [sum(row[j] for row in ratings) / (n_items * n_raters)
           for j in range(num_categories)]
    p_e = sum(pj * pj for pj in p_j)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if p_bar >= 1.0 - 1e-12 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


@dataclass
class GrammaticalityRating:
    """One rater's judgment of an item, for the human-evaluation instrument."""

    item_id: int
    rater_id: int
    grammatical: bool
    meaning_preserved: bool


def agreement_on_grammaticality(ratings: Sequence[GrammaticalityRating]) -> float:
    """Cohen's kappa on grammaticality between exactly two raters.

    Provided as the *instrument* the document asks for; the judgments themselves
    can only come from fluent signers.
    """
    raters = sorted({r.rater_id for r in ratings})
    if len(raters) != 2:
        raise ValueError("this instrument compares exactly two raters")
    a_id, b_id = raters
    by_item: Dict[int, Dict[int, bool]] = {}
    for r in ratings:
        by_item.setdefault(r.item_id, {})[r.rater_id] = r.grammatical
    items = [i for i, d in by_item.items() if a_id in d and b_id in d]
    a = [int(by_item[i][a_id]) for i in items]
    b = [int(by_item[i][b_id]) for i in items]
    return cohens_kappa(a, b)
