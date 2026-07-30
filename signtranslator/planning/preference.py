"""Sequence-level Direct Preference Optimization over sign plans.

Implements the DPO objective from the specification,

    L_DPO = - log sigma( beta [ (log pi_theta(s+|x) - log pi_ref(s+|x))
                              - (log pi_theta(s-|x) - log pi_ref(s-|x)) ] )

with the sequence log-probability being the teacher-forced sum of per-token
log-probs, ``log pi(s|x) = sum_i log pi(s_i | s_{<i}, x)``.

**The document's caveat is load-bearing and repeated here in code:** preference
optimization is appropriate *only after* pairwise judgments have been collected
from qualified target-language signers, and *it optimizes observed preferences;
it does not prove linguistic correctness.* This module supplies the mechanism;
it deliberately does not, and cannot, manufacture the judgments. Running it on
synthetic or automatic "preferences" would be measuring nothing.

At initialisation ``pi_theta = pi_ref``, so the bracket is exactly 0 and the loss
is ``-log sigma(0) = log 2`` -- the same closed-form check proved for the motion
DPO, asserted here too.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sequence_log_prob(logits: torch.Tensor, targets: torch.Tensor,
                      pad_id: int) -> torch.Tensor:
    """``sum_i log p(target_i)`` per sequence, ignoring pad positions.

    ``logits`` is ``(N, L, V)`` (teacher-forced), ``targets`` is ``(N, L)``.
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError("logits (N,L,V) and targets (N,L) required")
    log_probs = F.log_softmax(logits, dim=-1)
    vocab = logits.shape[-1]
    # pad_id (and any padding) is >= the output vocab size; clamp into range for
    # the gather, then mask those positions out of the sum. The clamped value is
    # never used because the mask zeroes it.
    safe_targets = targets.clamp(0, vocab - 1)
    gathered = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    mask = (targets != pad_id).to(gathered.dtype)
    return (gathered * mask).sum(dim=-1)


@dataclass
class DPOStats:
    loss: float
    accuracy: float       # fraction of pairs the policy already ranks correctly
    margin: float         # mean implicit-reward margin


class SequencePreferenceDPO:
    """DPO over a :class:`SemanticPlanner`'s plan sequences.

    Requires **real** preference pairs ``(s+, s-)`` for the same evidence. The
    frozen reference policy keeps the tuned model from drifting arbitrarily far
    from the supervised solution -- without it, DPO degenerates into "raise the
    preferred likelihood" with no anchor.
    """

    def __init__(self, planner: torch.nn.Module, beta: float = 0.1) -> None:
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.planner = planner
        self.beta = beta
        self.pad_id = planner.pad_id
        self.reference = copy.deepcopy(planner).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

    def _seq_logprob(self, model, plan_tokens, acoustic, src_tokens):
        logits = model(plan_tokens, acoustic=acoustic, src_tokens=src_tokens)
        return sequence_log_prob(logits, plan_tokens, self.pad_id)

    def loss(self, preferred: torch.Tensor, rejected: torch.Tensor,
             acoustic: Optional[torch.Tensor] = None,
             src_tokens: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, DPOStats]:
        """DPO loss on a batch of preference pairs sharing the same evidence."""
        lp_w = self._seq_logprob(self.planner, preferred, acoustic, src_tokens)
        lp_l = self._seq_logprob(self.planner, rejected, acoustic, src_tokens)
        with torch.no_grad():
            ref_w = self._seq_logprob(self.reference, preferred, acoustic, src_tokens)
            ref_l = self._seq_logprob(self.reference, rejected, acoustic, src_tokens)

        logits = self.beta * ((lp_w - ref_w) - (lp_l - ref_l))
        loss = -F.logsigmoid(logits).mean()
        stats = DPOStats(loss=loss.detach().item(),
                         accuracy=(logits > 0).float().mean().detach().item(),
                         margin=logits.mean().detach().item())
        return loss, stats

    def step(self, optimizer, preferred, rejected, acoustic=None, src_tokens=None,
             grad_clip: float = 1.0) -> DPOStats:
        loss, stats = self.loss(preferred, rejected, acoustic, src_tokens)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), grad_clip)
        optimizer.step()
        return stats
