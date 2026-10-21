"""Text–motion retrieval: a meaning-level evaluator for generated signing.

Motion metrics (angle error, speed and variation ratios, Fréchet distance) measure
how signing-like the motion is, not whether it says the English. The standard
text-to-motion answer, at a data scale where a sign-to-text translator cannot be
trained (How2Sign subset: 3,968 training sentences), is a contrastive text–motion
model scored by **R-precision**. The model is trained on real pairs only and then
frozen. Each generated clip is ranked against its own sentence and the sentences of
``group − 1`` other clips; R@k is the fraction whose own sentence is among the top k
(Guo et al. 2022, HumanML3D protocol, groups of 32). The same score on the real
validation motion is the ceiling the evaluator can resolve.

**Motion encoder.**
- Input: the generators' unit representation (``text_to_motion.diffusion_targets``:
  rotations as 6D columns, blendshape coefficients, translations, gaze). Unsupported
  cells hold the neutral value, and padding is masked.
- Layers: a linear map to ``d``, a sinusoidal time code (seconds), a Transformer
  encoder under the frame mask, a masked mean over frames, and a linear projection
  to the unit sphere.

**Text encoder.** The generators' byte encoding (``encode_texts``), a Transformer
encoder, a masked mean and a projection.

**Loss.** Symmetric InfoNCE at a learned temperature τ = exp(−s), with s clamped to
τ ≥ 0.01 (CLIP). Because the logits are cosine similarities divided by τ, the ranking
used by R-precision does not depend on τ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.multichannel import MotionBatch
from .text_to_motion import (PAD, VOCABULARY, TextToMotionConfig, batch_seconds, diffusion_layout,
                             diffusion_targets, sinusoid)

MIN_TEMPERATURE = .01


@dataclass(frozen=True)
class TextMotionRetrievalConfig:
    motion: TextToMotionConfig                # channel bindings and trained channels (the generators' layout)
    d_model: int = 256
    heads: int = 4
    motion_layers: int = 4
    text_layers: int = 4
    feedforward: int = 1024
    dropout: float = .1
    embedding: int = 256
    initial_temperature: float = .07

    def __post_init__(self):
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not MIN_TEMPERATURE <= self.initial_temperature < 1:
            raise ValueError("initial_temperature must lie in [0.01, 1)")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["motion"] = self.motion.to_dict()
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "TextMotionRetrievalConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask[..., None].to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1.)


def _encoder(d: int, heads: int, feedforward: int, dropout: float, layers: int) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, feedforward, dropout, batch_first=True,
                                                            norm_first=True), layers, enable_nested_tensor=False)


class TextMotionRetrieval(nn.Module):
    def __init__(self, config: TextMotionRetrievalConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        width = sum(flat for _, _, _, flat in diffusion_layout(config.motion))      # = diffusion_targets width
        self.motion_in = nn.Linear(width, d)
        self.motion_encoder = _encoder(d, config.heads, config.feedforward, config.dropout, config.motion_layers)
        self.motion_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.embedding))
        self.embed = nn.Embedding(VOCABULARY, d, padding_idx=PAD)
        self.text_encoder = _encoder(d, config.heads, config.feedforward, config.dropout, config.text_layers)
        self.text_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.embedding))
        self.log_scale = nn.Parameter(torch.tensor(-math.log(config.initial_temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(-self.log_scale.clamp(max=-math.log(MIN_TEMPERATURE)))

    def encode_motion(self, motion: MotionBatch) -> torch.Tensor:
        """(B, embedding) unit vectors for a batch of (real or generated) motion."""
        unit = diffusion_targets(self.config.motion, motion)
        mask = motion.frame_valid
        seconds = batch_seconds(motion).float()
        h = self.motion_in(unit) + sinusoid(seconds, self.config.d_model, max_period=60.)
        h = self.motion_encoder(h, src_key_padding_mask=~mask)
        return F.normalize(self.motion_out(_masked_mean(h, mask)), dim=-1)

    def encode_text(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, embedding) unit vectors for byte-encoded texts (``encode_texts``)."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens)
        h = self.embed(tokens) + sinusoid(positions, self.config.d_model, max_period=1e4)
        h = self.text_encoder(h, src_key_padding_mask=~mask)
        return F.normalize(self.text_out(_masked_mean(h, mask)), dim=-1)


def retrieval_loss(model: TextMotionRetrieval, tokens, mask, motion: MotionBatch):
    """(symmetric InfoNCE, terms) on a batch of real pairs; duplicate sentences are not negatives of each other."""
    zm, zt = model.encode_motion(motion), model.encode_text(tokens, mask)
    tau = model.temperature
    logits = zm @ zt.t() / tau
    same = (tokens[:, None, :] == tokens[None, :, :]).all(-1) if tokens.shape[0] > 1 else None
    if same is not None:                       # identical texts in one batch would be contradictory negatives
        eye = torch.eye(len(tokens), dtype=torch.bool, device=tokens.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
    targets = torch.arange(len(tokens), device=tokens.device)
    loss = .5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    accuracy = (logits.argmax(1) == targets).float().mean()
    return loss, dict(retrieval=loss.detach(), batch_top1=accuracy.detach(), temperature=tau.detach())


def r_precision(motion_embeddings: torch.Tensor, text_embeddings: torch.Tensor, *, group: int = 32,
                ks=(1, 2, 3), seed: int = 0) -> dict:
    """HumanML3D-style R-precision: rows split into fixed random groups of ``group``.

    Row i (motion) is ranked against the texts of its group by cosine similarity; R@k
    is the fraction whose own text is in the top k. Also returns the matching score
    (mean cosine distance ‖m − t‖ between paired unit vectors, lower is better) and the
    chance level k / group. Groups are drawn once from ``seed``, so two generators
    scored on the same clips meet the same distractors.
    """
    n = len(motion_embeddings)
    if n != len(text_embeddings) or n < group:
        raise ValueError("need at least one full group of paired embeddings")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    hits = {k: 0 for k in ks}
    rows = 0
    for start in range(0, n - group + 1, group):
        idx = order[start:start + group]
        sim = motion_embeddings[idx] @ text_embeddings[idx].t()
        rank = (sim > sim.diagonal()[:, None]).sum(1)            # how many texts beat the own one
        for k in ks:
            hits[k] += int((rank < k).sum())
        rows += group
    matching = float(torch.linalg.vector_norm(motion_embeddings - text_embeddings, dim=-1).mean())
    return {**{f"r_at_{k}": hits[k] / rows for k in ks}, **{f"chance_at_{k}": k / group for k in ks},
            "matching_distance": matching, "clips": rows, "group": group}


__all__ = ["TextMotionRetrievalConfig", "TextMotionRetrieval", "retrieval_loss", "r_precision", "MIN_TEMPERATURE"]
