"""Preference optimisation for motion naturalness (the RLHF-style stage).

After supervised training a generator is *correct* but not necessarily
*natural*: it may produce jerky trajectories or physically implausible joint
excursions that a human rater would reject. This module implements Direct
Preference Optimization (Rafailov et al., 2023) adapted to diffusion motion,
plus the automatic naturalness proxies used to build preference pairs when no
human labels are available.

DPO avoids training an explicit reward model. Given a preferred sample ``y_w``
and a rejected one ``y_l`` for the same conditioning, and a frozen reference
policy ``pi_ref``, the objective is

    L = -log sigmoid( beta * [ (log pi(y_w) - log pi_ref(y_w))
                             - (log pi(y_l) - log pi_ref(y_l)) ] ).

For diffusion we cannot evaluate exact likelihoods, so — following Diffusion-DPO
(Wallace et al., 2024) — we use the negative denoising error as a tractable
stand-in for ``log pi``: a sample the model reconstructs better under its own
denoiser is one it assigns higher likelihood to.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Automatic naturalness proxies (used to synthesise preference pairs)
# ---------------------------------------------------------------------------
def jerk(motion: torch.Tensor) -> torch.Tensor:
    """Mean squared third temporal derivative -- the standard smoothness proxy.

    Human motion minimises jerk; high jerk reads as robotic or twitchy.
    ``motion`` is ``(N, C, T, V)``; returns one scalar per sample.
    """
    if motion.shape[2] < 4:
        return torch.zeros(motion.shape[0], device=motion.device)
    d1 = motion[:, :, 1:] - motion[:, :, :-1]
    d2 = d1[:, :, 1:] - d1[:, :, :-1]
    d3 = d2[:, :, 1:] - d2[:, :, :-1]
    return d3.pow(2).mean(dim=(1, 2, 3))


def bone_length_variance(motion: torch.Tensor,
                         edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Temporal variance of bone lengths -- rigid bodies should keep them fixed.

    A generator that stretches the forearm frame to frame is physically wrong,
    and this penalises exactly that. Returns one scalar per sample.
    """
    if not edges:
        return torch.zeros(motion.shape[0], device=motion.device)
    idx_a = torch.tensor([a for a, _ in edges], device=motion.device)
    idx_b = torch.tensor([b for _, b in edges], device=motion.device)
    va = motion.index_select(3, idx_a)            # (N, C, T, E)
    vb = motion.index_select(3, idx_b)
    lengths = (va - vb).pow(2).sum(dim=1).sqrt()  # (N, T, E)
    return lengths.var(dim=1).mean(dim=1)


def naturalness_score(motion: torch.Tensor,
                      edges: Optional[List[Tuple[int, int]]] = None,
                      jerk_weight: float = 1.0,
                      bone_weight: float = 1.0) -> torch.Tensor:
    """Higher is more natural. Combines smoothness and skeletal consistency."""
    score = -jerk_weight * jerk(motion)
    if edges:
        score = score - bone_weight * bone_length_variance(motion, edges)
    return score


def build_preference_pairs(candidates: torch.Tensor,
                           score_fn: Callable[[torch.Tensor], torch.Tensor],
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``(N, P, C, T, V)`` candidates into (preferred, rejected).

    For each conditioning, the highest-scoring candidate is preferred and the
    lowest-scoring is rejected. Supplying a human-rating function instead of a
    proxy makes this genuine RLHF.
    """
    if candidates.dim() != 5:
        raise ValueError("candidates must be (N, P, C, T, V)")
    n, p = candidates.shape[:2]
    if p < 2:
        raise ValueError("need at least 2 candidates per conditioning")
    flat = candidates.reshape(n * p, *candidates.shape[2:])
    scores = score_fn(flat).reshape(n, p)
    best = scores.argmax(dim=1)
    worst = scores.argmin(dim=1)
    ar = torch.arange(n, device=candidates.device)
    return candidates[ar, best], candidates[ar, worst]


# ---------------------------------------------------------------------------
# Diffusion-DPO
# ---------------------------------------------------------------------------
@dataclass
class DPOStats:
    loss: float
    accuracy: float          # fraction of pairs ranked correctly
    margin: float            # mean implicit-reward margin


class DiffusionDPO:
    """Direct Preference Optimization over a diffusion motion generator."""

    def __init__(self, diffusion, beta: float = 0.1) -> None:
        self.diffusion = diffusion
        self.beta = beta
        # Frozen reference policy: the pre-DPO model. Without it, DPO degenerates
        # into "maximise preferred likelihood" and drifts arbitrarily far from
        # the supervised solution.
        self.reference = copy.deepcopy(diffusion).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _denoising_error(model, x0: torch.Tensor, t: torch.Tensor,
                         noise: torch.Tensor, cond) -> torch.Tensor:
        """Per-sample denoising error; the negative acts as a log-likelihood proxy."""
        x_t = model.q_sample(x0, t, noise=noise)
        out = model.denoiser(x_t, t, cond)
        target = x0 if model.parameterization == "x0" else noise
        return (out - target).pow(2).flatten(1).mean(dim=1)

    def loss(self, preferred: torch.Tensor, rejected: torch.Tensor,
             cond=None) -> Tuple[torch.Tensor, DPOStats]:
        """DPO loss on a batch of preference pairs.

        The same timestep and noise are used for both members of a pair so the
        comparison is paired rather than confounded by sampling variance.
        """
        n = preferred.shape[0]
        device = preferred.device
        t = self.diffusion.sample_timesteps(n, device)
        noise = torch.randn_like(preferred)

        err_w = self._denoising_error(self.diffusion, preferred, t, noise, cond)
        err_l = self._denoising_error(self.diffusion, rejected, t, noise, cond)
        with torch.no_grad():
            ref_w = self._denoising_error(self.reference, preferred, t, noise, cond)
            ref_l = self._denoising_error(self.reference, rejected, t, noise, cond)

        # log pi ~= -error, so (log pi - log pi_ref) ~= (ref_err - err).
        logits = self.beta * ((ref_w - err_w) - (ref_l - err_l))
        loss = -F.logsigmoid(logits).mean()
        stats = DPOStats(loss=loss.detach().item(),
                         accuracy=(logits > 0).float().mean().detach().item(),
                         margin=logits.mean().detach().item())
        return loss, stats

    def step(self, optimizer, preferred: torch.Tensor, rejected: torch.Tensor,
             cond=None, grad_clip: float = 1.0) -> DPOStats:
        loss, stats = self.loss(preferred, rejected, cond)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), grad_clip)
        optimizer.step()
        return stats
