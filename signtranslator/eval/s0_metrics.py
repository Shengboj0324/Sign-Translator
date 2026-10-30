"""Road to 3.8, part 2, stage S0: the measurement panel beside BLEU-4 (docs/NEXT_PHASE_PLAN.md, declared 2026-10-30
19:32 UTC; background in ``reports/Route to near real signing.md``, "Measure progress with likelihood and faithfulness").

Every function here is pure (NumPy, or PyTorch on tensors handed in), so each metric is tested on synthetic data
(``tests/test_s0_metrics.py``). The scripts ``scripts/s0_ceilings.py`` and ``scripts/s1_text_kl.py`` apply them.

**Reference likelihood gain over blank (ΔNLL).** For sentence i read by a back-translator from motion x_i:
``NLL(ref_i | x_i)`` is the teacher-forced negative log-likelihood of the reference's tokens (the translator's own
tokenisation, end-of-sequence token included), divided by the number of those tokens (token-length normalised).
``ΔNLL_i = NLL(ref_i | blank_i) − NLL(ref_i | x_i)``: **positive when the motion makes the reference more likely than
the blank control does** (blank_i: the same clip with every channel removed, only its duration left). Blank's own
ΔNLL is 0 by definition; real's is the instrument's range, so ΔNLL_x / ΔNLL_real is the gap fraction.

**100-way sentence retrieval.** Clip i is scored against k candidate references: its own and k − 1 distractors drawn
without replacement from the other clips' references (``retrieval_candidates``; one draw per clip, shared by every
form, so ranks are paired across forms). Lower NLL is better. The own reference's rank counts ties at the mid-rank
(1 + #better + #tied/2), so a constant score gives the chance rank (k + 1)/2 instead of the worst rank. Chance top-1
is 1/k. The PMI variant ranks NLL(c | x_i) − NLL(c | blank_i), removing each candidate's template prior.

**ρ_faith** (Hong & Košecká, arXiv 2606.01643): the distance of an output to its own reference, divided by its mean
distance to random other references, in a frozen motion autoencoder's latent space; averaged over clips. 1.0 = no
closer to its own reference than to a random one; 0 = identical to its reference. Our latent space and choices
(documented in ``reports/s0s1_dev.md``):
- d2's per-part encoder outputs (continuous, before quantisation), 24 Hz, every frame of the clip;
- each latent dimension standardised by its mean and standard deviation over the *real* clips' frames (so every part
  and dimension counts equally, and the scale is fixed by real signing, not by the form read);
- **primary: DTW** between the two latent sequences, frame cost the Euclidean distance divided by sqrt(dims), the
  accumulated cost divided by (n + m) (``dtw_distances``), so different lengths and tempi compare;
- secondary: the distance between time-averaged latents (``mean_distances``), blind to order;
- K random references per clip (``random_references``, never the clip itself; the same draw for every form).

**Text KL (S1).** ``categorical_kl`` and ``true_code_gain`` read a prior's teacher-forced logits with and without its
text (``scripts/s1_text_kl.py``).
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Sequence

import numpy as np


# ---- bootstrap helpers ---------------------------------------------------------------------------------------------

def bootstrap_mean(x, *, n: int = 1000, seed: int = 0) -> dict:
    """{mean, ci95, n} of ``x`` with a percentile bootstrap over its entries (resampled with replacement)."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return dict(mean=None, ci95=None, n=0)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return dict(mean=round(float(x.mean()), 5), ci95=[round(float(np.percentile(means, 2.5)), 5),
                                                     round(float(np.percentile(means, 97.5)), 5)], n=int(len(x)))


def paired_mean_difference(x, y, *, n: int = 1000, seed: int = 0) -> dict:
    """{difference, ci95} of mean(x) − mean(y), the same entries resampled for both (x and y paired by position)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape")
    out = bootstrap_mean(x - y, n=n, seed=seed)
    return dict(difference=out["mean"], ci95=out["ci95"])


def ratio_of_means(x, y, *, n: int = 1000, seed: int = 0) -> dict:
    """{ratio, ci95} of mean(x) / mean(y) with the same resample of the paired entries (e.g. ΔNLL_form / ΔNLL_real)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape")
    if not len(x) or y.mean() == 0:
        return dict(ratio=None, ci95=None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    num, den = x[idx].mean(1), y[idx].mean(1)
    ratios = np.where(den != 0, num / np.where(den == 0, 1., den), np.nan)
    ratios = ratios[np.isfinite(ratios)]
    return dict(ratio=round(float(x.mean() / y.mean()), 5),
                ci95=[round(float(np.percentile(ratios, 2.5)), 5), round(float(np.percentile(ratios, 97.5)), 5)])


def gap_fraction(x: float, blank: float, real: float) -> float | None:
    """(x − blank) / (real − blank): 0 at blank, 1 at real; None when real equals blank."""
    if real == blank:
        return None
    return (x - blank) / (real - blank)


# ---- likelihood ----------------------------------------------------------------------------------------------------

def delta_nll(nll_blank, nll_motion) -> np.ndarray:
    """Per sentence NLL(ref | blank) − NLL(ref | motion): positive when the motion makes the reference more likely."""
    a, b = np.asarray(nll_blank, np.float64), np.asarray(nll_motion, np.float64)
    if a.shape != b.shape:
        raise ValueError("blank and motion NLLs must be paired")
    return a - b


def token_nll(logits, labels, ignore_index: int = -100):
    """(summed NLL (B,), token counts (B,)) of ``labels`` (B, L) under ``logits`` (B, L, V), ignoring ``ignore_index``.
    Torch tensors in, float64 tensors out (on the logits' device)."""
    import torch
    import torch.nn.functional as F
    keep = labels != ignore_index
    per = F.cross_entropy(logits.double().transpose(1, 2), labels.masked_fill(~keep, 0), reduction="none")
    return (per * keep).sum(1), keep.sum(1).double()


# ---- retrieval -----------------------------------------------------------------------------------------------------

def retrieval_candidates(n: int, k: int = 100, *, seed: int = 0) -> np.ndarray:
    """(n, k') candidate indices, column 0 the clip itself, the rest drawn without replacement from the other n − 1
    clips; k' = min(k, n). Each row's draw depends only on (seed, row), not on n's other rows."""
    k = min(k, n)
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        others = np.delete(np.arange(n), i)
        out[i, 0] = i
        out[i, 1:] = rng.choice(others, size=k - 1, replace=False)
    return out


def mid_ranks(scores, own: int = 0, *, lower_is_better: bool = True) -> np.ndarray:
    """1-based rank of column ``own`` in every row of ``scores`` (n, k), ties at the mid-rank: 1 + #better + #tied/2
    (the own entry excluded from the tie count). A constant row ranks (k + 1)/2, the chance mean."""
    s = np.asarray(scores, np.float64)
    if not lower_is_better:
        s = -s
    target = s[:, own][:, None]
    better = (s < target).sum(1)
    tied = (s == target).sum(1) - 1
    return 1. + better + tied / 2.


def retrieval_summary(ranks, k: int) -> dict:
    """top-1/5/10, mean rank and mean reciprocal rank of mid-ranks; with their chance values for k candidates."""
    r = np.asarray(ranks, np.float64)
    if not len(r):
        return dict(k=k, clips=0)
    return dict(k=int(k), clips=int(len(r)), top1=round(float((r <= 1).mean()), 4), top5=round(float((r <= 5).mean()), 4),
                top10=round(float((r <= 10).mean()), 4), mean_rank=round(float(r.mean()), 3),
                mrr=round(float((1. / r).mean()), 4),
                chance=dict(top1=round(1 / k, 4), top5=round(min(5, k) / k, 4), mean_rank=(k + 1) / 2,
                            mrr=round(float(np.mean(1. / np.arange(1, k + 1))), 4)))


# ---- chrF ----------------------------------------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> Counter:
    s = "".join(text.split())                       # sacreBLEU chrF: whitespace removed
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf_statistics(hyp: str, ref: str, order: int = 6) -> np.ndarray:
    """(order, 3) per n-gram order: hypothesis n-grams, reference n-grams, matches (clipped)."""
    out = np.zeros((order, 3), np.float64)
    for n in range(1, order + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        out[n - 1] = sum(h.values()), sum(r.values()), sum((h & r).values())
    return out


def chrf_from_statistics(stats, beta: float = 2.):
    """sacreBLEU's chrF (character order 6, word order 0, beta 2) from summed statistics (..., order, 3): the F-beta
    of every order with both sides non-empty, averaged over those orders, times 100 (0 when none). Vectorised over
    leading axes (a float for a single (order, 3) table)."""
    stats = np.asarray(stats, np.float64)
    n_hyp, n_ref, n_match = stats[..., 0], stats[..., 1], stats[..., 2]
    ok = (n_hyp > 0) & (n_ref > 0)
    prec = np.where(ok, n_match / np.where(ok, n_hyp, 1.), 0.)
    rec = np.where(ok, n_match / np.where(ok, n_ref, 1.), 0.)
    factor = beta ** 2
    denom = factor * prec + rec
    f = np.where(ok & (denom > 0), (1 + factor) * prec * rec / np.where(denom > 0, denom, 1.), 0.)
    effective = ok.sum(-1)
    out = np.where(effective > 0, 100. * f.sum(-1) / np.maximum(effective, 1), 0.)
    return float(out) if out.ndim == 0 else out


def sentence_statistics(hyps: Sequence[str], refs: Sequence[str], order: int = 6) -> np.ndarray:
    if len(hyps) != len(refs):
        raise ValueError("hypotheses and references must be paired")
    return np.stack([chrf_statistics(h, r, order) for h, r in zip(hyps, refs)]) if len(hyps) else \
        np.zeros((0, order, 3))


def corpus_chrf(hyps, refs, order: int = 6) -> float:
    """Corpus chrF: statistics summed over sentences first (sacreBLEU's corpus score)."""
    return chrf_from_statistics(sentence_statistics(hyps, refs, order).sum(0))


def chrf_ci(hyps, refs, *, n: int = 1000, seed: int = 0) -> tuple[float, list[float]]:
    stats = sentence_statistics(hyps, refs)
    idx = np.random.default_rng(seed).integers(0, len(stats), (n, len(stats)))
    vals = chrf_from_statistics(stats[idx].sum(1))
    return round(chrf_from_statistics(stats.sum(0)), 3), [round(float(np.percentile(vals, 2.5)), 3),
                                                          round(float(np.percentile(vals, 97.5)), 3)]


def paired_chrf_difference(hyps_a, hyps_b, refs, *, n: int = 1000, seed: int = 0) -> dict:
    a, b = sentence_statistics(hyps_a, refs), sentence_statistics(hyps_b, refs)
    idx = np.random.default_rng(seed).integers(0, len(refs), (n, len(refs)))
    diffs = chrf_from_statistics(a[idx].sum(1)) - chrf_from_statistics(b[idx].sum(1))
    return dict(difference=round(chrf_from_statistics(a.sum(0)) - chrf_from_statistics(b.sum(0)), 3),
                ci95=[round(float(np.quantile(diffs, .025)), 3), round(float(np.quantile(diffs, .975)), 3)])


# ---- BLEU (the repository's corpus_bleu, as per-sentence statistics for fast paired bootstraps) ---------------------

def bleu_statistics(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> np.ndarray:
    """(n, 2·max_n + 2) per sentence: matches and totals per order, hypothesis and reference length, counted exactly
    as ``motion_to_text.corpus_bleu`` (its ``words`` tokens, clipped n-gram matches)."""
    from ..generation.motion_to_text import words
    if len(hyps) != len(refs):
        raise ValueError("hypotheses and references must be paired")
    out = np.zeros((len(hyps), 2 * max_n + 2), np.float64)
    for k, (hyp, ref) in enumerate(zip(hyps, refs)):
        h, r = words(hyp), words(ref)
        for n in range(1, max_n + 1):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            out[k, n - 1] = sum(min(c, rc[g]) for g, c in hc.items())
            out[k, max_n + n - 1] = max(len(h) - n + 1, 0)
        out[k, -2:] = len(h), len(r)
    return out


def bleu_from_statistics(stats, max_n: int = 4):
    """corpus_bleu's BLEU (0-100) from summed statistics (..., 2·max_n + 2); vectorised over leading axes."""
    stats = np.asarray(stats, np.float64)
    m, t = stats[..., :max_n], stats[..., max_n:2 * max_n]
    hyp_len, ref_len = stats[..., -2], stats[..., -1]
    prec = np.where(t > 0, m / np.where(t > 0, t, 1.), 0.)
    zero = (hyp_len == 0) | (prec.min(-1) == 0)
    logp = np.log(np.where(prec > 0, prec, 1.)).sum(-1) / max_n
    bp = np.where(hyp_len > ref_len, 1., np.exp(1 - ref_len / np.where(hyp_len > 0, hyp_len, 1.)))
    out = np.where(zero, 0., 100. * bp * np.exp(logp))
    return float(out) if out.ndim == 0 else out


def bleu_ci_fast(hyps, refs, *, n: int = 1000, seed: int = 0) -> tuple[float, list[float]]:
    """``strict_prior_review.bleu_ci`` (same seed, same draws, same percentiles), vectorised."""
    stats = bleu_statistics(hyps, refs)
    idx = np.random.default_rng(seed).integers(0, len(stats), (n, len(stats)))
    vals = bleu_from_statistics(stats[idx].sum(1))
    return round(float(bleu_from_statistics(stats.sum(0))), 3), [round(float(np.percentile(vals, 2.5)), 3),
                                                                 round(float(np.percentile(vals, 97.5)), 3)]


def paired_bleu_fast(stats_a, stats_b, *, n: int = 2000, seed: int = 0) -> dict:
    """``compare_pilot_lexicon.paired_bleu_difference`` (a minus b, same draws and quantiles) on sentence statistics."""
    a, b = np.asarray(stats_a), np.asarray(stats_b)
    if a.shape != b.shape:
        raise ValueError("the two forms must read the same clips")
    idx = np.random.default_rng(seed).integers(0, len(a), (n, len(a)))
    diffs = bleu_from_statistics(a[idx].sum(1)) - bleu_from_statistics(b[idx].sum(1))
    return dict(difference=round(float(bleu_from_statistics(a.sum(0)) - bleu_from_statistics(b.sum(0))), 4),
                ci95=[round(float(np.quantile(diffs, .025)), 4), round(float(np.quantile(diffs, .975)), 4)])


def length_ratio(stats) -> float | None:
    """Corpus hypothesis words / reference words (sacreBLEU's length ratio, on corpus_bleu's tokens)."""
    total = np.asarray(stats).sum(0)
    return round(float(total[-2] / total[-1]), 4) if total[-1] else None


# ---- ρ_faith -------------------------------------------------------------------------------------------------------

def random_references(n: int, K: int = 10, *, seed: int = 0) -> np.ndarray:
    """(n, K') indices of random other clips per clip (never itself, without replacement), K' = min(K, n − 1)."""
    K = min(K, n - 1)
    out = np.empty((n, K), dtype=np.int64)
    for i in range(n):
        rng = np.random.default_rng([seed, 7919, i])
        out[i] = rng.choice(np.delete(np.arange(n), i), size=K, replace=False)
    return out


def standardiser(latents: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) per latent dimension over every frame of ``latents`` (each (T, D)); std floored at 1e-6."""
    frames = np.concatenate([np.asarray(z, np.float64) for z in latents], 0)
    return frames.mean(0), np.maximum(frames.std(0), 1e-6)


def dtw_distances(a_list, b_list, *, device: str = "cpu", chunk: int = 64) -> np.ndarray:
    """DTW distance of each pair (a_list[i], b_list[i]) of (T, D) arrays: frame cost ||a_t − b_s|| / sqrt(D), steps
    (1,0), (0,1), (1,1), accumulated cost divided by (n + m). Batched over pairs by anti-diagonals (torch)."""
    import torch
    if len(a_list) != len(b_list):
        raise ValueError("pairs must be paired")
    out = np.zeros(len(a_list), np.float64)
    order = sorted(range(len(a_list)), key=lambda k: (len(a_list[k]) + len(b_list[k])))   # little padding per chunk
    for start in range(0, len(order), chunk):
        rows = order[start:start + chunk]
        A, B = [a_list[k] for k in rows], [b_list[k] for k in rows]
        na = torch.tensor([len(a) for a in A]); nb = torch.tensor([len(b) for b in B])
        if (na == 0).any() or (nb == 0).any():
            raise ValueError("empty latent sequence")
        D = A[0].shape[1]
        pa = torch.zeros(len(A), int(na.max()), D, dtype=torch.float64, device=device)
        pb = torch.zeros(len(B), int(nb.max()), D, dtype=torch.float64, device=device)
        for k, (a, b) in enumerate(zip(A, B)):
            pa[k, :len(a)] = torch.as_tensor(np.asarray(a), dtype=torch.float64)
            pb[k, :len(b)] = torch.as_tensor(np.asarray(b), dtype=torch.float64)
        cost = torch.cdist(pa, pb) / math.sqrt(D)                                    # (P, N, M)
        P, N, M = cost.shape
        acc = torch.full((P, N + 1, M + 1), float("inf"), dtype=torch.float64, device=device)
        acc[:, 0, 0] = 0.
        for d in range(2, N + M + 1):                                                # i + j = d (1-based cells)
            i = torch.arange(max(1, d - M), min(N, d - 1) + 1, device=device)
            j = d - i
            best = torch.minimum(torch.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
            acc[:, i, j] = best + cost[:, i - 1, j - 1]
        end = acc[torch.arange(P, device=device), na.to(device), nb.to(device)]
        out[rows] = (end / (na + nb).to(device=device, dtype=torch.float64)).cpu().numpy()
    return out


def dtw_reference(a: np.ndarray, b: np.ndarray) -> float:
    """The plain O(nm) DTW of ``dtw_distances`` for one pair (the tests' reference)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n, m, D = len(a), len(b), a.shape[1]
    acc = np.full((n + 1, m + 1), np.inf); acc[0, 0] = 0.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1]) / math.sqrt(D)
            acc[i, j] = c + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def mean_distances(a_list, b_list) -> np.ndarray:
    """||mean_t a − mean_t b|| / sqrt(D) for each pair (the order-blind variant)."""
    return np.array([np.linalg.norm(np.asarray(a).mean(0) - np.asarray(b).mean(0)) / math.sqrt(np.asarray(a).shape[1])
                     for a, b in zip(a_list, b_list)])


def rho_faith(outputs, references, *, K: int = 10, seed: int = 0, distance: str = "dtw", device: str = "cpu") -> dict:
    """Per-clip ρ_i = d(output_i, ref_i) / mean_k d(output_i, ref_{r_ik}) and its mean with a bootstrap 95% interval.
    ``outputs`` and ``references`` are paired lists of standardised (T, D) latent arrays; ``r`` = random_references."""
    n = len(outputs)
    if n != len(references) or n < 2:
        raise ValueError("rho_faith needs at least two paired clips")
    others = random_references(n, K, seed=seed)
    measure = (lambda a, b: dtw_distances(a, b, device=device)) if distance == "dtw" else mean_distances
    own = measure(list(outputs), list(references))
    pairs_a = [outputs[i] for i in range(n) for _ in range(others.shape[1])]
    pairs_b = [references[j] for i in range(n) for j in others[i]]
    rand = measure(pairs_a, pairs_b).reshape(n, others.shape[1]).mean(1)
    ratio = own / np.maximum(rand, 1e-12)
    return dict(distance=distance, K=int(others.shape[1]), **bootstrap_mean(ratio, seed=seed),
                median=round(float(np.median(ratio)), 5), own_distance=round(float(own.mean()), 5),
                random_distance=round(float(rand.mean()), 5), per_clip=[round(float(r), 5) for r in ratio])


# ---- text KL (S1) --------------------------------------------------------------------------------------------------

def categorical_kl(logits_p, logits_q):
    """KL(p ‖ q) in nats over the last axis of two logit tensors (torch, float64 inside): Σ p (log p − log q)."""
    lp, lq = logits_p.double().log_softmax(-1), logits_q.double().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


def true_code_gain(logits_p, logits_q, target):
    """log p(target) − log q(target) per position (nats; the cross-entropy gain of ``text_use_check.py``)."""
    lp, lq = logits_p.double().log_softmax(-1), logits_q.double().log_softmax(-1)
    idx = target[..., None]
    return (lp.gather(-1, idx) - lq.gather(-1, idx))[..., 0]


__all__ = ["bootstrap_mean", "paired_mean_difference", "ratio_of_means", "gap_fraction", "delta_nll", "token_nll",
           "retrieval_candidates", "mid_ranks", "retrieval_summary", "chrf_statistics", "chrf_from_statistics",
           "corpus_chrf", "chrf_ci", "paired_chrf_difference", "bleu_statistics", "bleu_from_statistics", "bleu_ci_fast",
           "paired_bleu_fast", "length_ratio", "random_references", "standardiser", "dtw_distances",
           "dtw_reference", "mean_distances", "rho_faith", "categorical_kl", "true_code_gain"]
