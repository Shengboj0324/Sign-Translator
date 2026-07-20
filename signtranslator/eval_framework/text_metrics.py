"""Reproducible text metrics: SacreBLEU + BERTScore (Doc-12 §5).

Corpus BLEU with clipped n-gram precision, brevity penalty, exponential smoothing,
and a reproducibility SIGNATURE that records the exact settings (Post 2018). BERTScore
greedy-match precision/recall/F1 with optional IDF weighting (Zhang et al. 2019).
Both are automatic metrics — the framework attaches the caveat that they do not
substitute for signer evaluation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# tokenisation (documented + named in the signature for reproducibility)
# ---------------------------------------------------------------------------
_PUNCT = re.compile(r"([.,!?;:()\"'])")
TOKENIZER_NAME = "basic_lc_v1"   # lowercase + split listed punctuation + whitespace


def tokenize(text: str) -> List[str]:
    """Deterministic tokeniser (its name is recorded in the BLEU signature)."""
    text = _PUNCT.sub(r" \1 ", text.lower())
    return text.split()


# ---------------------------------------------------------------------------
# SacreBLEU-style corpus BLEU
# ---------------------------------------------------------------------------
def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


@dataclass(frozen=True)
class BLEUSignature:
    tokenizer: str
    smooth: str
    max_ngram: int
    num_refs: int

    def __str__(self) -> str:
        return (f"BLEU|tok:{self.tokenizer}|smooth:{self.smooth}"
                f"|maxn:{self.max_ngram}|refs:{self.num_refs}")


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str],
                max_ngram: int = 4, smooth: str = "exp"):
    """Corpus BLEU with brevity penalty + exponential smoothing.

    BLEU = BP · exp(Σ_{n=1}^N (1/N) log p_n), p_n clipped n-gram precision,
    BP = min(1, exp(1 − r/c)). Returns (score in [0,1], BLEUSignature).
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be parallel")
    if max_ngram < 1:
        raise ValueError("max_ngram must be >= 1")
    num = [0] * (max_ngram + 1)
    den = [0] * (max_ngram + 1)
    c_len = r_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize(hyp), tokenize(ref)
        c_len += len(h)
        r_len += len(r)
        for n in range(1, max_ngram + 1):
            hc = _ngrams(h, n)
            rc = _ngrams(r, n)
            overlap = sum(min(cnt, rc[g]) for g, cnt in hc.items())
            num[n] += overlap
            den[n] += max(len(h) - n + 1, 0)

    if c_len == 0:
        return 0.0, BLEUSignature(TOKENIZER_NAME, smooth, max_ngram, 1)

    # per-order precisions with exponential-decay smoothing for zero numerators.
    log_sum = 0.0
    counted = 0
    smooth_val = 1.0
    for n in range(1, max_ngram + 1):
        if den[n] == 0:
            continue                                  # effective order: skip
        counted += 1
        if num[n] == 0:
            if smooth == "exp":
                smooth_val *= 2.0
                p = 1.0 / (smooth_val * den[n])
            elif smooth == "none":
                p = 0.0
            else:
                raise ValueError("smooth must be 'exp' or 'none'")
        else:
            p = num[n] / den[n]
        if p == 0.0:
            return 0.0, BLEUSignature(TOKENIZER_NAME, smooth, max_ngram, 1)
        log_sum += math.log(p)

    geo = math.exp(log_sum / counted)
    bp = 1.0 if c_len > r_len else math.exp(1.0 - r_len / c_len)
    return bp * geo, BLEUSignature(TOKENIZER_NAME, smooth, max_ngram, 1)


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------
def idf_weights(documents: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Smoothed IDF over a set of tokenised documents: log((N+1)/(df+1))+1."""
    N = len(documents)
    df: Counter = Counter()
    for doc in documents:
        for tok in set(doc):
            df[tok] += 1
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def bert_score(ref_emb: torch.Tensor, cand_emb: torch.Tensor,
               idf_ref: Optional[torch.Tensor] = None,
               idf_cand: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Greedy-match precision/recall/F1 over contextual token embeddings.

    ``ref_emb`` (m, d), ``cand_emb`` (k, d). Rows are L2-normalised so the dot is
    cosine. R = mean_i max_j x_i·x̂_j, P = mean_j max_i x_i·x̂_j (IDF-weighted means
    if weights given), F1 = 2PR/(P+R).
    """
    if ref_emb.ndim != 2 or cand_emb.ndim != 2:
        raise ValueError("embeddings must be (num_tokens, dim)")
    x = ref_emb / ref_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = cand_emb / cand_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = x @ y.t()                                   # (m, k)
    recall_terms = sim.max(dim=1).values              # best cand for each ref
    prec_terms = sim.max(dim=0).values                # best ref for each cand

    def _wmean(vals, w):
        if w is None:
            return float(vals.mean())
        w = w.to(vals.dtype)
        return float((w * vals).sum() / w.sum().clamp_min(1e-12))

    R = _wmean(recall_terms, idf_ref)
    P = _wmean(prec_terms, idf_cand)
    F = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
    return {"precision": P, "recall": R, "f1": F}
