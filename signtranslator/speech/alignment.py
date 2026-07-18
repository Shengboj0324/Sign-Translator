"""CTC forced alignment and word timestamps.

The specification requires **word timestamps** alongside the transcript, because
a sign plan has to be scheduled against the speech it renders. Timestamps come
from a Viterbi forced alignment of a known token sequence to the frame
posteriors.

Given targets ``l = [l_1..l_L]``, CTC alignment operates on the blank-extended
sequence

    l' = [blank, l_1, blank, l_2, ..., l_L, blank],   |l'| = 2L + 1

At each frame the path may stay in its state, advance one, or advance two --
the last only when it skips a blank *and* the two surrounding labels differ
(``l'[s] != l'[s-2]``). That final condition is the same repeat rule that
governs decoding: two identical labels in a row must be separated by a blank, so
the path may not jump over it.

The Viterbi recursion is

    a(t, s) = max_{p in pred(s)} a(t-1, p) + log y_t(l'[s])

Correctness is proved, not asserted: for small inputs the tests enumerate every
alignment, keep those collapsing to the target, and require the Viterbi score to
equal the maximum over that set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

NEG_INF = -float("inf")


# ---------------------------------------------------------------------------
# Frame <-> time
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameTimeMapper:
    """Converts model frame indices to seconds.

    Accounts for the analysis window, the hop, and any encoder subsampling, so a
    timestamp refers to the audio the frame actually observed rather than to an
    abstract index. Feature frame ``m`` spans samples ``[m*hop, m*hop + n_fft)``;
    a model frame ``u`` after subsampling by ``S`` covers feature frames
    ``[u*S, u*S + S)``.
    """

    hop_length: int = 160
    sample_rate: int = 16000
    n_fft: int = 400
    subsample: int = 1

    def __post_init__(self) -> None:
        if self.subsample < 1:
            raise ValueError("subsample must be >= 1")
        if self.hop_length < 1 or self.sample_rate < 1 or self.n_fft < 1:
            raise ValueError("hop_length, sample_rate, n_fft must be positive")

    def start_s(self, frame: int) -> float:
        return (frame * self.subsample * self.hop_length) / self.sample_rate

    def end_s(self, frame: int) -> float:
        last_feature_frame = frame * self.subsample + self.subsample - 1
        return (last_feature_frame * self.hop_length + self.n_fft) / self.sample_rate

    def center_s(self, frame: int) -> float:
        return 0.5 * (self.start_s(frame) + self.end_s(frame))

    def duration_s(self, num_frames: int) -> float:
        return self.end_s(num_frames - 1) if num_frames > 0 else 0.0


@dataclass
class TokenTiming:
    token: int
    start_frame: int
    end_frame: int          # inclusive
    start_s: float
    end_s: float
    score: float = 0.0      # mean log-posterior of the token over its frames

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------
def extended_targets(targets: Sequence[int], blank: int = 0) -> List[int]:
    """``[blank, l_1, blank, ..., l_L, blank]`` of length ``2L + 1``."""
    out = [blank]
    for tok in targets:
        out.append(int(tok))
        out.append(blank)
    return out


def minimum_frames_required(targets: Sequence[int]) -> int:
    """Frames needed to emit ``targets``: one per token plus one per repeat.

    Adjacent identical tokens need an intervening blank frame, so ``[a, a]``
    needs 3 frames, not 2. Alignment is impossible below this bound.
    """
    if not targets:
        return 0
    need = len(targets)
    for i in range(1, len(targets)):
        if targets[i] == targets[i - 1]:
            need += 1
    return need


def _as_numpy(log_probs) -> np.ndarray:
    if isinstance(log_probs, torch.Tensor):
        arr = log_probs.detach().cpu().double().numpy()
    else:
        arr = np.asarray(log_probs, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("log_probs must be (T, C)")
    return arr


@dataclass
class Alignment:
    """Result of forced alignment."""

    path: List[int]            # per-frame index into the extended sequence
    score: float               # log-probability of the best path
    extended: List[int]
    targets: Tuple[int, ...]

    def state_tokens(self) -> List[int]:
        """Per-frame emitted symbol (blank included)."""
        return [self.extended[s] for s in self.path]


def ctc_forced_alignment(log_probs, targets: Sequence[int], blank: int = 0
                         ) -> Alignment:
    """Viterbi-align ``targets`` to ``(T, C)`` frame log-probabilities."""
    lp = _as_numpy(log_probs)
    T, C = lp.shape
    targets = [int(t) for t in targets]
    if any(t == blank for t in targets):
        raise ValueError("targets must not contain the blank symbol")
    if any(not 0 <= t < C for t in targets):
        raise ValueError("target token out of range")
    need = minimum_frames_required(targets)
    if T < need:
        raise ValueError(
            f"cannot align {len(targets)} tokens into {T} frames; "
            f"at least {need} are required (repeats need a separating blank)")

    ext = extended_targets(targets, blank)
    S = len(ext)

    alpha = np.full((T, S), NEG_INF)
    back = np.full((T, S), -1, dtype=np.int64)

    alpha[0, 0] = lp[0, ext[0]]
    if S > 1:
        alpha[0, 1] = lp[0, ext[1]]

    for t in range(1, T):
        for s in range(S):
            best, arg = alpha[t - 1, s], s                 # stay
            if s >= 1 and alpha[t - 1, s - 1] > best:
                best, arg = alpha[t - 1, s - 1], s - 1     # advance
            # Skip the intervening blank only if the labels differ.
            if (s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]
                    and alpha[t - 1, s - 2] > best):
                best, arg = alpha[t - 1, s - 2], s - 2
            if best == NEG_INF:
                continue
            alpha[t, s] = best + lp[t, ext[s]]
            back[t, s] = arg

    # A valid path ends on the last token or the trailing blank.
    end_candidates = [S - 1] if S == 1 else [S - 1, S - 2]
    end_state = max(end_candidates, key=lambda s: alpha[T - 1, s])
    score = float(alpha[T - 1, end_state])
    if score == NEG_INF:
        raise ValueError("no valid alignment path exists")

    path = [0] * T
    path[T - 1] = end_state
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return Alignment(path=path, score=score, extended=ext,
                     targets=tuple(targets))


def token_timings(alignment: Alignment, mapper: Optional[FrameTimeMapper] = None,
                  log_probs=None) -> List[TokenTiming]:
    """Frame spans and times for each target token.

    Token ``j`` (1-based) occupies extended state ``2j - 1``; its span is the
    contiguous run of frames assigned to that state.
    """
    mapper = mapper or FrameTimeMapper()
    lp = _as_numpy(log_probs) if log_probs is not None else None
    out: List[TokenTiming] = []
    for j, tok in enumerate(alignment.targets, start=1):
        state = 2 * j - 1
        frames = [t for t, s in enumerate(alignment.path) if s == state]
        if not frames:
            # Unreachable for a valid Viterbi path, but never emit a silent
            # zero-length timing: that would misplace a sign in the schedule.
            raise ValueError(f"token {j} received no frames in the alignment")
        start_f, end_f = frames[0], frames[-1]
        score = float(np.mean([lp[t, tok] for t in frames])) if lp is not None else 0.0
        out.append(TokenTiming(
            token=tok, start_frame=start_f, end_frame=end_f,
            start_s=mapper.start_s(start_f), end_s=mapper.end_s(end_f),
            score=score))
    return out


def align_and_time(log_probs, targets: Sequence[int], blank: int = 0,
                   mapper: Optional[FrameTimeMapper] = None
                   ) -> Tuple[Alignment, List[TokenTiming]]:
    """Convenience: forced-align then extract timings in one call."""
    al = ctc_forced_alignment(log_probs, targets, blank)
    return al, token_timings(al, mapper, log_probs)
