"""Constrained decoding against the schema automaton.

Given decoder logits and the automaton's allowed set ``A_t`` at each step, the
masked distribution is the exact conditional ``p(v | v in A_t)``:

    p'(v) = p(v) / Z   for v in A_t,   0 otherwise,   Z = sum_{u in A_t} p(u).

Proved properties (see docs/SEMANTIC_PLANNER.md §3.3):

1. ``p'`` is a probability distribution whenever ``A_t`` is non-empty (liveness
   guarantees it is, before EOP).
2. ``p'`` preserves ratios among allowed tokens: ``p'(u)/p'(v) = p(u)/p(v)``.
   Constraining removes illegal mass; it never reorders legal preferences.
3. Greedy/sampled decoding under ``p'`` stays in ``A_t``, so by induction the
   produced sequence is always accepted by the automaton -- constrained decoding
   can only emit well-formed plan skeletons.

Masking is done in log space (``-inf`` on disallowed tokens) so that a
disallowed token contributes exactly 0 probability with no NaN -- a naive
``exp`` on a masked probability can produce ``0 * inf = NaN``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

import torch

from .automaton import SchemaAutomaton, S
from .schema import PlanVocabulary, DEFAULT_VOCAB

NEG_INF = float("-inf")


def allowed_mask(allowed: Set[int], vocab_size: int,
                 device=None) -> torch.Tensor:
    """Boolean vector, ``True`` on allowed token ids."""
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    if allowed:
        idx = torch.tensor(sorted(allowed), dtype=torch.long, device=device)
        mask[idx] = True
    return mask


def masked_log_softmax(logits: torch.Tensor, allowed: Set[int]) -> torch.Tensor:
    """Log of the conditional distribution ``p(v | v in allowed)``.

    Disallowed positions are set to ``-inf`` *before* the log-softmax, so they
    receive exactly ``0`` probability and never produce NaN.
    """
    if logits.dim() != 1:
        raise ValueError("logits must be 1-D (vocab,)")
    if not allowed:
        raise ValueError("empty allowed set: cannot form a distribution")
    mask = allowed_mask(allowed, logits.shape[0], logits.device)
    masked = logits.masked_fill(~mask, NEG_INF)
    return torch.log_softmax(masked, dim=-1)


def masked_distribution(logits: torch.Tensor, allowed: Set[int]) -> torch.Tensor:
    """The conditional probability vector ``p'`` (0 on disallowed tokens)."""
    return masked_log_softmax(logits, allowed).exp()


class ConstrainedDecoder:
    """Wraps an automaton to constrain a stream of decoder logits."""

    def __init__(self, automaton: Optional[SchemaAutomaton] = None,
                 vocab: PlanVocabulary = DEFAULT_VOCAB) -> None:
        self.automaton = automaton or SchemaAutomaton(vocab)
        self.vocab = vocab

    def next_distribution(self, logits: torch.Tensor, state: S) -> torch.Tensor:
        return masked_distribution(logits, self.automaton.allowed_tokens(state))

    def _decode_cap(self, max_length: Optional[int]) -> int:
        # Default to the automaton's own finite upper bound (+slack), so a bounded
        # run can always complete; a caller may still pass a smaller max_length.
        auto_cap = self.automaton.max_generated_length() + 4
        return auto_cap if max_length is None else min(max_length, auto_cap * 4)

    @torch.no_grad()
    def greedy_decode(self, logits_fn, max_length: Optional[int] = None) -> List[int]:
        """Greedy constrained decode over the **bounded** language.

        ``logits_fn(prefix) -> (vocab,) logits`` supplies the model's scores for
        the next token. Slot caps make the language finite, so decoding is
        guaranteed to reach ``ACCEPT`` -- unlike the unbounded language, where a
        model that always prefers a repeatable token would never stop.
        """
        cap = self._decode_cap(max_length)
        state = self.automaton.initial_state
        counts: dict = {}
        tokens: List[int] = []
        for _ in range(cap):
            if self.automaton.is_accepting(state):
                return tokens
            allowed = self.automaton.bounded_allowed(state, counts)
            log_p = masked_log_softmax(logits_fn(tokens), allowed)
            token = int(torch.argmax(log_p))
            assert token in allowed, "greedy picked a disallowed token"
            tokens.append(token)
            state, counts = self.automaton.bounded_step(state, token, counts)
            assert state is not None, "automaton rejected an allowed token"
        raise RuntimeError(f"bounded decode exceeded its cap of {cap} steps "
                           "(should be impossible)")

    @torch.no_grad()
    def sample_decode(self, logits_fn, max_length: Optional[int] = None,
                      temperature: float = 1.0,
                      generator: Optional[torch.Generator] = None) -> List[int]:
        """Ancestral constrained sampling over the bounded language."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        cap = self._decode_cap(max_length)
        state = self.automaton.initial_state
        counts: dict = {}
        tokens: List[int] = []
        for _ in range(cap):
            if self.automaton.is_accepting(state):
                return tokens
            allowed = self.automaton.bounded_allowed(state, counts)
            probs = masked_distribution(logits_fn(tokens) / temperature, allowed)
            token = int(torch.multinomial(probs, 1, generator=generator))
            assert token in allowed, "sampling drew a disallowed token"
            tokens.append(token)
            state, counts = self.automaton.bounded_step(state, token, counts)
        raise RuntimeError("bounded sampling exceeded its cap (should be impossible)")

    def allowed_at_each_step(self, tokens: Sequence[int]) -> List[Set[int]]:
        """The allowed set the decoder saw at each step of a produced sequence.

        Used by training to build a per-step constraint mask.
        """
        state = self.automaton.initial_state
        out: List[Set[int]] = []
        for tok in tokens:
            out.append(self.automaton.allowed_tokens(state))
            state = self.automaton.step(state, int(tok))
            if state is None:
                raise ValueError("token sequence is not accepted by the automaton")
        return out
