"""Streaming decoding with a committed prefix and a revisable suffix.

The specification is explicit: *"The downstream planner must be able to revise
uncommitted speech rather than receive only a final string."* This module
supplies that contract.

At every update the decoder re-decodes all audio seen so far and splits its
hypothesis in two:

* **committed** -- already handed downstream; treated as immutable. Once a sign
  has been rendered from a word, retracting it is not free, so commitment must
  be conservative.
* **uncommitted** -- the current best guess, which the planner may see but must
  be prepared to have replaced.

Commitment requires two independent forms of evidence:

1. **Beam agreement.** The prefix must be common to the top-``k`` hypotheses.
   If plausible alternatives already disagree about a word, it is not settled.
2. **Temporal stability.** That agreement must persist for ``stability``
   consecutive updates. Agreement at a single instant can be an artefact of the
   audio ending mid-word.

Requiring both is deliberate: either alone commits too eagerly. Because
commitment is monotone by construction, later evidence can contradict it; that
event is *counted* (``commitment_errors``) rather than silently swallowed,
because it is the quantity that tells you whether the policy is too aggressive.

Metrics here (revision rate, commitment error rate) are among those the source
document requires be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .decoding import (
    ctc_prefix_beam_search, NBestList, Hypothesis, Lattice,
)

Tokens = Tuple[int, ...]


def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> Tokens:
    """Longest prefix shared by every sequence (empty if none, or if empty input)."""
    if not sequences:
        return ()
    shortest = min(len(s) for s in sequences)
    out: List[int] = []
    for i in range(shortest):
        first = sequences[0][i]
        if all(s[i] == first for s in sequences):
            out.append(int(first))
        else:
            break
    return tuple(out)


@dataclass
class StreamingHypothesis:
    """What the planner sees after an update."""

    committed: Tokens
    uncommitted: Tokens
    lattice: Optional[Lattice] = None
    retained_mass: float = 0.0

    @property
    def full(self) -> Tokens:
        return self.committed + self.uncommitted

    def __len__(self) -> int:
        return len(self.full)


@dataclass
class RevisionStats:
    """Revision behaviour over a stream.

    ``revision_rate`` is the fraction of *position emissions* that changed: each
    update emits ``len(full)`` positions, and a position counts as revised when
    its token differs from what the previous update showed there. Defining it
    per position-emission (rather than per final token) makes it comparable
    across streams of different length and update cadence.
    """

    position_emissions: int = 0
    revised_positions: int = 0
    updates: int = 0
    commitment_errors: int = 0
    committed_tokens: int = 0

    @property
    def revision_rate(self) -> float:
        if self.position_emissions == 0:
            return 0.0
        return self.revised_positions / self.position_emissions

    @property
    def commitment_error_rate(self) -> float:
        if self.committed_tokens == 0:
            return 0.0
        return self.commitment_errors / self.committed_tokens

    def report(self) -> str:
        return (f"updates={self.updates} | revision_rate="
                f"{self.revision_rate:.4f} | commitment_errors="
                f"{self.commitment_errors} | committed={self.committed_tokens}")


class StreamingDecoder:
    """Incremental CTC decoding with monotone commitment.

    Args:
        beam_width: prefixes retained by the beam search.
        agreement_k: how many top hypotheses must agree for a prefix to be a
            commitment candidate.
        stability: consecutive updates that agreement must persist.
        blank: CTC blank index.
    """

    def __init__(self, beam_width: int = 10, agreement_k: int = 3,
                 stability: int = 2, blank: int = 0) -> None:
        if agreement_k < 1:
            raise ValueError("agreement_k must be >= 1")
        if stability < 1:
            raise ValueError("stability must be >= 1")
        self.beam_width = beam_width
        self.agreement_k = agreement_k
        self.stability = stability
        self.blank = blank
        self.reset()

    def reset(self) -> None:
        self._frames: List[np.ndarray] = []
        self._committed: Tokens = ()
        self._candidates: List[Tokens] = []
        self._last_full: Optional[Tokens] = None
        self.stats = RevisionStats()
        self._last_nbest: Optional[NBestList] = None

    # -- internals ----------------------------------------------------------
    def _all_frames(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0)

    def _commit_candidate(self, nbest: NBestList) -> Tokens:
        top = [h.tokens for h in nbest.hypotheses[:self.agreement_k]]
        return longest_common_prefix(top)

    def _advance_commitment(self, candidate: Tokens) -> None:
        self._candidates.append(candidate)
        if len(self._candidates) < self.stability:
            return                              # not enough evidence yet
        stable = longest_common_prefix(self._candidates[-self.stability:])

        n = len(self._committed)
        if n and stable[:n] != self._committed:
            # New evidence contradicts text already handed downstream. We cannot
            # un-emit it, so keep it and record the error -- the honest signal
            # that the commitment policy is too aggressive.
            self.stats.commitment_errors += 1
            return
        if len(stable) > n:
            self.stats.committed_tokens += len(stable) - n
            self._committed = stable

    def _track_revisions(self, full: Tokens) -> None:
        self.stats.updates += 1
        self.stats.position_emissions += len(full)
        if self._last_full is not None:
            for i in range(min(len(full), len(self._last_full))):
                if full[i] != self._last_full[i]:
                    self.stats.revised_positions += 1
        self._last_full = full

    # -- public API ---------------------------------------------------------
    def update(self, log_probs) -> StreamingHypothesis:
        """Feed new frames ``(T_new, C)`` and return the current split."""
        arr = (log_probs.detach().cpu().double().numpy()
               if isinstance(log_probs, torch.Tensor)
               else np.asarray(log_probs, dtype=np.float64))
        if arr.ndim != 2:
            raise ValueError("log_probs must be (T, C)")
        if arr.shape[0]:
            self._frames.append(arr)
        if not self._frames:
            return StreamingHypothesis(committed=(), uncommitted=())

        nbest = ctc_prefix_beam_search(self._all_frames(),
                                       beam_width=self.beam_width,
                                       blank=self.blank)
        self._last_nbest = nbest
        self._advance_commitment(self._commit_candidate(nbest))

        best = nbest.best.tokens
        full = self._committed + best[len(self._committed):]
        self._track_revisions(full)
        return StreamingHypothesis(
            committed=self._committed,
            uncommitted=full[len(self._committed):],
            lattice=Lattice.from_nbest(nbest),
            retained_mass=nbest.retained_mass)

    def finalize(self) -> StreamingHypothesis:
        """End of stream: everything remaining becomes committed."""
        if self._last_nbest is None:
            return StreamingHypothesis(committed=self._committed, uncommitted=())
        best = self._last_nbest.best.tokens
        full = self._committed + best[len(self._committed):]
        if len(full) > len(self._committed):
            self.stats.committed_tokens += len(full) - len(self._committed)
        self._committed = full
        return StreamingHypothesis(
            committed=self._committed, uncommitted=(),
            lattice=Lattice.from_nbest(self._last_nbest),
            retained_mass=self._last_nbest.retained_mass)

    @property
    def committed(self) -> Tokens:
        return self._committed


def commitment_error_count(committed: Sequence[int],
                           reference: Sequence[int]) -> int:
    """Committed tokens that disagree with a reference transcript, positionally.

    Tokens committed beyond the reference's length count as errors: emitting a
    word that was never said is a failure, not a neutral extension.
    """
    errors = 0
    for i, tok in enumerate(committed):
        if i >= len(reference) or reference[i] != tok:
            errors += 1
    return errors
