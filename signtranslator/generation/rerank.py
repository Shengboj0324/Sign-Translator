"""Reranking a word's samples by the prior's own teacher-forced pointwise mutual information (round 15, word sample
quality; docs/NEXT_PHASE_PLAN.md after amendment 5).

For one word, ``rerank_word`` draws N samples with ``rerank_seeds`` (sample 0 is the word's existing seed, so N = 1 is
exactly ``compositional.generate_word``) and keeps the one whose sampled codes the prior itself finds most specific to
the word: ``word_pmi``, the teacher-forced log p(codes | word) - log p(codes | empty text), per code.

**The score uses the prior and nothing else.** ``word_pmi`` receives only the prior, the word, the codes and the
duration; this module imports no recogniser, back-translator or other evaluator (tested on its source). The
likelihood is ``motion_rvq.rvq_sequence_log_likelihood``'s computation (the prior's text memory, the codes shifted by
the start row, ``temporal_states``, ``depth_logits`` gathered at the codes) applied to the *sampled* codes with the
conditioning they were sampled under: the word's planned loci, the clock ``token_seconds`` and the requested
duration; the empty text with ``empty_loci`` for the unconditional side (classifier-free guidance's own pair). The
sampled codes are scored directly (re-encoding the decoded motion would score other codes). Every generated code is
part-valid (a generation is valid everywhere), so "per part-valid code" is per code: the sum over the S x J codes
divided by S x J. No guidance and no temperature enter the score.
"""
from __future__ import annotations

import torch

from .motion_rvq import generate_rvq_motion
from .motion_tokens import _empty_like, token_seconds
from .text_to_motion import PAD, empty_loci, encode_texts, generation_loci

FPS = 24.
RERANK_STRIDE = 100_000_000          # sample k of a word with seed s: s + k x RERANK_STRIDE (beyond every word-seed block)


def rerank_seeds(seed: int, n: int) -> list[int]:
    """The seeds of a word's ``n`` samples: the word's own seed first, then ``seed + k * RERANK_STRIDE``."""
    if n < 1:
        raise ValueError("rerank needs at least one sample")
    return [int(seed) + k * RERANK_STRIDE for k in range(n)]


@torch.no_grad()
def _log_likelihood(prior, memory, memory_mask, codes, clock, duration) -> torch.Tensor:
    start = torch.full((1, 1, prior.J), prior.K, dtype=torch.long, device=codes.device)
    previous = torch.cat((start, codes[None, :-1]), 1)                                  # (1, S, J)
    h = prior.temporal_states(memory, memory_mask, previous, clock, duration)[0]        # (S, d)
    logits = prior.depth_logits(h, codes[:, :-1])                                       # (S, J, K)
    return logits.log_softmax(-1).gather(-1, codes[..., None])[..., 0].double().sum()


@torch.no_grad()
def word_pmi(prior, word: str, codes: torch.Tensor, seconds: float, fps: float = FPS) -> dict:
    """The prior's teacher-forced PMI of a word's sampled ``codes`` (S, J) at ``seconds`` (module docstring):
    ``{"conditional", "unconditional", "codes", "pmi_per_code"}``."""
    prior.eval()
    config = prior.config
    device = next(prior.parameters()).device
    codes = codes.to(device).long()
    encoded, mask = encode_texts([word], max_bytes=config.max_text_bytes)
    encoded, mask = encoded.to(device), mask.to(device)
    hidden, _, mask = prior.encode(encoded, mask, generation_loci(config, [word], None, device))
    empty = _empty_like(encoded[:1, :2])
    e_hidden, _, e_mask = prior.encode(empty, empty != PAD, empty_loci(config, device))
    duration = torch.tensor([float(seconds)], device=device).clamp(1 / fps, 60.)
    clock = token_seconds(config, codes.shape[0], fps, device)
    conditional = float(_log_likelihood(prior, hidden, mask, codes, clock, duration))
    unconditional = float(_log_likelihood(prior, e_hidden, e_mask, codes, clock, duration))
    n = int(codes.numel())
    return dict(conditional=round(conditional, 4), unconditional=round(unconditional, 4), codes=n,
                pmi_per_code=round((conditional - unconditional) / n, 6))


def choose(scores) -> int:
    """The index of the highest score (the first on a tie)."""
    return max(range(len(scores)), key=lambda k: (scores[k], -k))


def rerank_word(prior, tokenizer, word: str, seconds: float, seed: int, n: int, *, guidance: float, temperature,
                post):
    """(state, info) of the best of ``n`` samples of ``word`` (module docstring): each sampled exactly as
    ``compositional.generate_word`` samples (``generate_rvq_motion`` at ``seconds``, then resampled to 24 fps and
    ``post``), scored by ``word_pmi`` on its codes; ``info``: the seeds, the scores and the chosen index."""
    from ..pose.resample import resample_motion
    seeds = rerank_seeds(seed, n)
    samples, scores = [], []
    for s in seeds:
        states, codes = generate_rvq_motion(prior, tokenizer, [word], fps=FPS, durations=[seconds], guidance=guidance,
                                            temperature=1. if temperature is None else temperature, seed=s,
                                            return_codes=True)
        samples.append(states[0])
        scores.append(word_pmi(prior, word, codes[0], seconds)["pmi_per_code"])
    k = choose(scores)
    return post(resample_motion(samples[k], FPS)), dict(seeds=seeds, scores=scores, chosen=k)


__all__ = ["RERANK_STRIDE", "rerank_seeds", "word_pmi", "choose", "rerank_word"]
