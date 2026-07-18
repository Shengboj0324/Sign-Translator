"""CTC decoding: prefix beam search, N-best hypotheses, and a lattice.

The source specification requires the layer to emit *"a transcript lattice or
N-best hypotheses"*, not a single string, so the planner can weigh alternatives
and revise. Greedy (best-path) decoding cannot supply that: it returns one
alignment, and its score is the probability of that **alignment**, not of the
**label sequence**.

CTC defines the probability of a label sequence as a sum over every alignment
that collapses to it,

    p(l | x) = sum_{pi in B^-1(l)} prod_t y_t(pi_t)

where ``B`` removes repeats then blanks. Prefix beam search (Graves et al. 2006;
Hannun et al. 2014) accumulates that sum incrementally by tracking, for each
prefix, the probability mass ending in a blank (``p_b``) separately from that
ending in a non-blank (``p_nb``). The split is what makes the repeat rule
expressible: emitting the same symbol twice in a row collapses to one symbol
*unless* a blank separates them, so an extension by ``c == l[-1]`` may only draw
on ``p_b``.

Correctness here is not asserted, it is **proved**: :func:`ctc_exact_posteriors`
enumerates every path exhaustively for small inputs, and the tests require the
beam search to reproduce those probabilities.

All arithmetic is in log space; ``p_b`` and ``p_nb`` are stored separately and
combined with ``logaddexp``, which is stable when probabilities underflow.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

NEG_INF = -float("inf")
Prefix = Tuple[int, ...]


def _logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) that tolerates -inf on either side."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return float(np.logaddexp(a, b))


@dataclass
class Hypothesis:
    """One decoded label sequence and its log-probability."""

    tokens: Tuple[int, ...]
    log_prob: float
    log_p_blank: float = NEG_INF
    log_p_nonblank: float = NEG_INF

    @property
    def probability(self) -> float:
        return math.exp(self.log_prob) if self.log_prob > NEG_INF else 0.0

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class NBestList:
    """Ranked hypotheses with posteriors renormalised over the retained set.

    ``posteriors`` sum to 1 across the list. They are *conditional on the beam*:
    pruned mass is excluded, so a confident-looking posterior from a narrow beam
    is not evidence of a confident model. :attr:`retained_mass` reports how much
    total probability the list actually covers, which is the honest caveat.
    """

    hypotheses: List[Hypothesis]
    retained_mass: float = 0.0

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __getitem__(self, i: int) -> Hypothesis:
        return self.hypotheses[i]

    @property
    def best(self) -> Hypothesis:
        if not self.hypotheses:
            raise IndexError("empty N-best list")
        return self.hypotheses[0]

    @property
    def posteriors(self) -> List[float]:
        if not self.hypotheses:
            return []
        logs = [h.log_prob for h in self.hypotheses]
        m = max(logs)
        if m == NEG_INF:
            return [1.0 / len(logs)] * len(logs)
        exps = [math.exp(l - m) for l in logs]
        total = sum(exps)
        return [e / total for e in exps]

    def token_posteriors(self) -> List[Dict[int, float]]:
        """Marginal token posterior at each position of the best hypothesis.

        Position ``i`` marginalises over every hypothesis long enough to have a
        token there, which gives a per-token confidence that accounts for
        competing alternatives rather than only the top path.
        """
        if not self.hypotheses:
            return []
        post = self.posteriors
        out: List[Dict[int, float]] = []
        for i in range(len(self.best.tokens)):
            acc: Dict[int, float] = {}
            for h, p in zip(self.hypotheses, post):
                if i < len(h.tokens):
                    acc[h.tokens[i]] = acc.get(h.tokens[i], 0.0) + p
            out.append(acc)
        return out


def _as_numpy_log_probs(log_probs) -> np.ndarray:
    if isinstance(log_probs, torch.Tensor):
        arr = log_probs.detach().cpu().double().numpy()
    else:
        arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("log_probs must be (T, C)")
    return arr


def ctc_prefix_beam_search(log_probs, beam_width: int = 10, blank: int = 0,
                           prune_threshold: Optional[float] = None,
                           ) -> NBestList:
    """Prefix beam search over ``(T, C)`` per-frame log-probabilities.

    Args:
        beam_width: prefixes retained per frame. Larger is closer to exact.
        prune_threshold: optional per-frame log-prob floor; tokens below it are
            skipped. ``None`` considers every token, which is what the exactness
            proof requires.

    Returns an :class:`NBestList` ordered by descending log-probability.
    """
    lp = _as_numpy_log_probs(log_probs)
    T, C = lp.shape
    if not 0 <= blank < C:
        raise ValueError("blank index out of range")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    # prefix -> (log p ending in blank, log p ending in non-blank)
    beams: Dict[Prefix, Tuple[float, float]] = {(): (0.0, NEG_INF)}

    for t in range(T):
        nxt: Dict[Prefix, Tuple[float, float]] = {}

        def bump(pref: Prefix, d_blank: float = NEG_INF,
                 d_nonblank: float = NEG_INF) -> None:
            pb, pnb = nxt.get(pref, (NEG_INF, NEG_INF))
            nxt[pref] = (_logaddexp(pb, d_blank), _logaddexp(pnb, d_nonblank))

        candidates = range(C)
        if prune_threshold is not None:
            candidates = [c for c in range(C) if lp[t, c] >= prune_threshold]

        for prefix, (pb, pnb) in beams.items():
            p_total = _logaddexp(pb, pnb)
            last = prefix[-1] if prefix else None
            for c in candidates:
                y = float(lp[t, c])
                if c == blank:
                    # Blank never extends the label sequence; all mass moves to
                    # the "ends in blank" state of the same prefix.
                    bump(prefix, d_blank=p_total + y)
                elif c == last:
                    # Repeat of the final symbol: collapses onto the same prefix
                    # (from non-blank mass), and only extends the prefix when a
                    # blank intervened (from blank mass).
                    bump(prefix, d_nonblank=pnb + y)
                    bump(prefix + (c,), d_nonblank=pb + y)
                else:
                    bump(prefix + (c,), d_nonblank=p_total + y)

        ranked = sorted(nxt.items(),
                        key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                        reverse=True)
        beams = dict(ranked[:beam_width])

    hyps = [
        Hypothesis(tokens=p, log_prob=_logaddexp(pb, pnb),
                   log_p_blank=pb, log_p_nonblank=pnb)
        for p, (pb, pnb) in beams.items()
    ]
    hyps.sort(key=lambda h: h.log_prob, reverse=True)
    mass = 0.0
    for h in hyps:
        mass += h.probability
    return NBestList(hypotheses=hyps, retained_mass=mass)


def collapse(path: Sequence[int], blank: int = 0) -> Tuple[int, ...]:
    """The CTC collapse ``B``: remove repeats, then remove blanks."""
    out: List[int] = []
    prev = None
    for s in path:
        if s != prev:
            if s != blank:
                out.append(s)
            prev = s
        # a repeat of the previous symbol (blank or not) contributes nothing
    return tuple(out)


def ctc_exact_posteriors(log_probs, blank: int = 0) -> Dict[Prefix, float]:
    """Exact ``p(l | x)`` for every label sequence, by exhaustive enumeration.

    Sums over all ``C^T`` alignments. Exponential, so usable only for tiny
    inputs -- its purpose is to serve as ground truth for the beam search, which
    is otherwise easy to get subtly wrong (the repeat/blank rule especially).
    """
    lp = _as_numpy_log_probs(log_probs)
    T, C = lp.shape
    if C ** T > 2_000_000:
        raise ValueError("input too large for exhaustive enumeration")
    acc: Dict[Prefix, float] = {}
    for path in itertools.product(range(C), repeat=T):
        score = float(sum(lp[t, c] for t, c in enumerate(path)))
        label = collapse(path, blank)
        acc[label] = _logaddexp(acc.get(label, NEG_INF), score)
    return acc


def ctc_greedy_path(log_probs) -> List[int]:
    """Best-path (argmax per frame) alignment, before collapsing."""
    lp = _as_numpy_log_probs(log_probs)
    return [int(np.argmax(lp[t])) for t in range(lp.shape[0])]


@dataclass
class Lattice:
    """A compact alternatives structure over an N-best list.

    Node ``i`` holds the alternative tokens observed at position ``i`` with their
    marginal posteriors. This is the "revise uncommitted speech" substrate: the
    planner can see *where* the recogniser is unsure, not merely what it guessed.
    """

    positions: List[Dict[int, float]] = field(default_factory=list)
    best: Tuple[int, ...] = ()
    retained_mass: float = 0.0

    @staticmethod
    def from_nbest(nbest: NBestList) -> "Lattice":
        return Lattice(positions=nbest.token_posteriors(),
                       best=nbest.best.tokens if len(nbest) else (),
                       retained_mass=nbest.retained_mass)

    def confidence(self) -> List[float]:
        """Posterior of the chosen token at each position of the best path."""
        return [pos.get(tok, 0.0) for tok, pos in zip(self.best, self.positions)]

    def ambiguous_positions(self, threshold: float = 0.9) -> List[int]:
        """Positions whose chosen token falls below ``threshold`` confidence.

        These are exactly the places a fail-closed policy should consider
        pausing or fingerspelling rather than committing to a sign.
        """
        return [i for i, c in enumerate(self.confidence()) if c < threshold]
