"""Memorisation diagnostics of the sentence refiner (report-only; independent review of a1bdda3, item 2).

A refiner trained on How2Sign train sentences could score by reproducing training sentences rather than by refining
its input. Two reads, each against a chance level from forms that cannot have memorised:

**Text** (``TextIndex``): each back-translated hypothesis's best smoothed sentence BLEU against every How2Sign *train*
text, and whether it equals one exactly (``words`` tokens joined). Candidates are the ``candidates`` train texts that
share the most distinct tokens with the hypothesis (an inverted index); sentence BLEU (``sentence_bleu``) is computed on
those only. Reported for the refined forms, the unrefined ones and the real val clips (the chance level: their
hypotheses come from signing the translator never saw).

**Codes** (``CodeIndex``): every n-gram of ``n`` consecutive steps of the stage-1 codes (one code per part, the coarse
motion) of each training target, hashed; n-grams found in more than ``max_df`` of the targets (rest poses and other
common motion) are left out. A query code sequence reports ``seen_fraction`` (its informative n-grams found in any
target) and ``max_containment`` (the largest share found in one single target, and which). Reported for the refined
codes, the refiner's input codes and the real val clips' codes (the chance level).
"""
from __future__ import annotations

import collections
import math

import numpy as np

FNV = np.uint64(1099511628211)
OFFSET = np.uint64(14695981039346656037)


# ---- text -------------------------------------------------------------------------------------------------------

def _ngrams(tokens, n):
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hyp_tokens, ref_tokens, max_n: int = 4) -> float:
    """Sentence BLEU (0-100) over token lists, add-one smoothed above unigrams (Lin and Och 2004), brevity penalty."""
    if not hyp_tokens or not ref_tokens:
        return 0.
    logs = 0.
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp_tokens, n), _ngrams(ref_tokens, n)
        m = sum(min(c, r[g]) for g, c in h.items())
        t = max(len(hyp_tokens) - n + 1, 0)
        if n == 1:
            if m == 0:
                return 0.
            logs += math.log(m / t)
        else:
            logs += math.log((m + 1) / (t + 1))
    bp = 1. if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    return 100 * bp * math.exp(logs / max_n)


class TextIndex:
    """The How2Sign train texts, tokenised as BLEU tokenises (``motion_to_text.words``)."""

    def __init__(self, texts, candidates: int = 50):
        from .motion_to_text import words
        self.words = words
        self.tokens = [words(t) for t in texts]
        self.exact = {" ".join(t) for t in self.tokens}
        self.candidates = candidates
        postings = collections.defaultdict(list)
        for i, toks in enumerate(self.tokens):
            for w in set(toks):
                postings[w].append(i)
        self.postings = {w: np.asarray(v, dtype=np.int64) for w, v in postings.items()}

    def nearest(self, hypothesis: str) -> dict:
        h = self.words(hypothesis)
        lists = [self.postings[w] for w in set(h) if w in self.postings]
        if not h or not lists:
            return dict(best_bleu=0., nearest=None, exact=" ".join(h) in self.exact)
        shared = np.bincount(np.concatenate(lists), minlength=len(self.tokens))
        k = min(self.candidates, int((shared > 0).sum()))
        top = np.argsort(-shared, kind="stable")[:k]
        scores = [(sentence_bleu(h, self.tokens[i]), int(i)) for i in top]
        best, index = max(scores, key=lambda s: (s[0], -s[1]))
        return dict(best_bleu=round(best, 4), nearest=index, exact=" ".join(h) in self.exact)

    def summary(self, hypotheses) -> dict:
        rows = [self.nearest(h) for h in hypotheses]
        b = np.array([r["best_bleu"] for r in rows], dtype=np.float64)
        return dict(n=len(rows), mean_best_bleu=round(float(b.mean()), 4) if len(b) else None,
                    median_best_bleu=round(float(np.median(b)), 4) if len(b) else None,
                    share_best_bleu_ge_50=round(float((b >= 50).mean()), 4) if len(b) else None,
                    exact_match_rate=round(float(np.mean([r["exact"] for r in rows])), 4) if rows else None)


# ---- codes ------------------------------------------------------------------------------------------------------

def step_keys(codes: np.ndarray, parts: int, codebook: int) -> np.ndarray:
    """(S,) uint64 key of each step's stage-1 codes (the first ``parts`` code positions)."""
    c = np.asarray(codes)[:, :parts].astype(np.uint64)
    base = np.uint64(codebook + 1)
    key = np.zeros(len(c), dtype=np.uint64)
    for p in range(parts):
        key = key * base + c[:, p]
    return key


def ngram_hashes(keys: np.ndarray, n: int) -> np.ndarray:
    """(S - n + 1,) uint64 FNV-style hashes of every n consecutive keys (empty when S < n)."""
    S = len(keys)
    if S < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.full(S - n + 1, OFFSET, dtype=np.uint64)
    for j in range(n):
        h = (h ^ keys[j:S - n + 1 + j]) * FNV
    return h


class CodeIndex:
    """Hashed stage-1 code n-grams of training targets (module docstring)."""

    def __init__(self, hashes: np.ndarray, owners: np.ndarray, common: np.ndarray, *, targets: int, n: int,
                 parts: int, codebook: int, max_df: float):
        self.hashes, self.owners, self.common = hashes, owners, common
        self.targets, self.n, self.parts, self.codebook, self.max_df = targets, n, parts, codebook, max_df

    @classmethod
    def build(cls, targets, *, parts: int, codebook: int, n: int = 12, max_df: float = .01) -> "CodeIndex":
        """From (S_i, J) target code arrays: unique (hash, target) pairs sorted by hash, minus the n-grams found in
        more than ``max_df`` of the targets."""
        hs, os = [], []
        for t, codes in enumerate(targets):
            h = np.unique(ngram_hashes(step_keys(codes, parts, codebook), n))
            hs.append(h)
            os.append(np.full(len(h), t, dtype=np.int32))
        hashes = np.concatenate(hs) if hs else np.zeros(0, np.uint64)
        owners = np.concatenate(os) if os else np.zeros(0, np.int32)
        order = np.lexsort((owners, hashes))
        hashes, owners = hashes[order], owners[order]
        values, counts = np.unique(hashes, return_counts=True)
        common = values[counts > max(1., max_df * len(targets))]
        keep = ~np.isin(hashes, common)
        return cls(hashes[keep], owners[keep], common, targets=len(targets), n=n, parts=parts, codebook=codebook,
                   max_df=max_df)

    def query(self, codes) -> dict:
        q = np.unique(ngram_hashes(step_keys(codes, self.parts, self.codebook), self.n))
        q = q[~np.isin(q, self.common)]
        if not len(q):
            return dict(ngrams=0, seen_fraction=None, max_containment=None, nearest=None)
        lo = np.searchsorted(self.hashes, q, "left")
        hi = np.searchsorted(self.hashes, q, "right")
        hits = hi - lo
        total = int(hits.sum())
        seen = float((hits > 0).mean())
        if not total:
            return dict(ngrams=int(len(q)), seen_fraction=0., max_containment=0., nearest=None)
        starts = np.repeat(lo, hits)
        offsets = np.arange(total) - np.repeat(np.cumsum(hits) - hits, hits)
        per_target = np.bincount(self.owners[starts + offsets], minlength=self.targets)
        best = int(per_target.argmax())
        return dict(ngrams=int(len(q)), seen_fraction=round(seen, 4),
                    max_containment=round(float(per_target[best]) / len(q), 4), nearest=best)

    def to_arrays(self) -> tuple[dict, dict]:
        """(arrays, meta) for ``refiner.pack_pair`` (deterministic bytes)."""
        return (dict(hashes=self.hashes, owners=self.owners, common=self.common),
                dict(kind="refiner-target-ngrams", targets=self.targets, n=self.n, parts=self.parts,
                     codebook=self.codebook, max_df=self.max_df, common_ngrams=int(len(self.common))))

    @classmethod
    def from_arrays(cls, arrays: dict, meta: dict) -> "CodeIndex":
        if meta.get("kind") != "refiner-target-ngrams":
            raise ValueError("not a refiner target n-gram index")
        return cls(arrays["hashes"], arrays["owners"], arrays["common"], targets=meta["targets"], n=meta["n"],
                   parts=meta["parts"], codebook=meta["codebook"], max_df=meta["max_df"])


def code_summary(rows) -> dict:
    """Means over ``CodeIndex.query`` rows (rows without informative n-grams left out, and counted)."""
    ok = [r for r in rows if r.get("max_containment") is not None]
    return dict(n=len(rows), without_ngrams=len(rows) - len(ok),
                mean_seen_fraction=round(float(np.mean([r["seen_fraction"] for r in ok])), 4) if ok else None,
                mean_max_containment=round(float(np.mean([r["max_containment"] for r in ok])), 4) if ok else None,
                share_max_containment_ge_50=round(float(np.mean([r["max_containment"] >= .5 for r in ok])), 4)
                if ok else None)


__all__ = ["sentence_bleu", "TextIndex", "step_keys", "ngram_hashes", "CodeIndex", "code_summary"]
