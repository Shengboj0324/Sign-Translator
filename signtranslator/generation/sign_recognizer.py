"""Isolated-sign recogniser on the canonical motion representation (plan item 4a, ASL Citizen).

It classifies a clip of canonical motion into one of the training partition's glosses, reading the same per-frame
features the generators emit (``diffusion_targets``), so generated citation-form signs can be scored at the sign
level beside the sentence-level back-translator. Encoder: per-frame features -> linear -> strided 1D convolution ->
transformer -> masked mean pool -> linear classifier; label-smoothed cross-entropy. ``GlossMotionDataset`` serves
one partition of a gloss-labelled manifest (ASL Citizen: the publisher's signer-disjoint splits).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data.multichannel import MotionBatch, collate_multichannel
from ..pose.resample import resample_motion
from ..data_engineering.corpus_manifest import ContentCache, load_sample, usable_channels
from .text_to_motion import TextToMotionConfig, diffusion_layout, diffusion_targets, sinusoid


class GlossMotionDataset(Dataset):
    """(label index, state) for one partition; records whose gloss is not in ``labels`` are left out."""

    def __init__(self, manifest: Mapping, *, partition: str, labels: Mapping[str, int], cache: ContentCache,
                 fetch: Callable[[str], bytes], min_hand_fraction: float = 0., fps: float | None = None,
                 augment: Callable | None = None) -> None:
        def keep(r):
            q = r.get("quality")
            # the hand-visibility filter needs the record's quality counts; a corpus without them (ASLLVD) is not
            # filtered here (ASLLVD sign clips: dominant hand tracked in a median 100%, p10 97% of frames)
            hands = (max(q.get("left_hand_frames") or 0, q.get("right_hand_frames") or 0) / max(q.get("frames") or 1, 1)
                     if q else 1.)
            return (r.get("partition") == partition and (r.get("gloss") or {}).get("label") in labels
                    and usable_channels(r, "continuous") and hands >= min_hand_fraction)
        self.records = [r for r in manifest["records"] if keep(r)]
        self.labels, self.cache, self.fetch = labels, cache, fetch
        self.fps, self.augment = fps, augment        # corpora differ in frame rate: resample before the model sees frames

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        state = load_sample(r, "continuous", cache=self.cache, fetch=self.fetch)
        if self.augment is not None:
            state = self.augment(state)
        if self.fps:
            state = resample_motion(state, self.fps)
        return self.labels[r["gloss"]["label"]], state


def augment_sign(state, rng, *, speed=(.8, 1.25), trim=.1, face_dropout=.2):
    """Training-time variation a recogniser must be invariant to: signing speed (the clock is rescaled, so a later
    resample sees the sign faster or slower), a little lead-in/lead-out trimmed (clip boundaries differ by corpus) and,
    sometimes, the face removed (so the hands carry the decision). Resample after this, never before."""
    from dataclasses import replace
    import numpy as np
    from ..data_engineering.corpus_manifest import mask_channels
    t = state.timestamps
    n = len(t)
    if n > 8 and trim > 0:
        a = int(rng.random() * trim * n); b = n - int(rng.random() * trim * n)
        if b - a >= 4:
            state = replace(state, timestamps=t[a:b], channels={k: replace(c, values=c.values[a:b], valid=c.valid[a:b],
                            observed=c.observed[a:b], confidence=c.confidence[a:b]) for k, c in state.channels.items()})
    factor = speed[0] + rng.random() * (speed[1] - speed[0])
    state = replace(state, timestamps=state.timestamps[0] + (state.timestamps - state.timestamps[0]) / factor)
    if rng.random() < face_dropout:
        state = mask_channels(state, [k for k in state.channels if k not in ("face", "blink")])
    return state


def collate_gloss_motion(items):
    labels, states = zip(*items)
    return torch.tensor(labels, dtype=torch.long), collate_multichannel(states)


@dataclass(frozen=True)
class SignRecognizerConfig:
    motion: TextToMotionConfig
    classes: int
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    feedforward: int = 1024
    dropout: float = .1
    downsample: int = 2
    label_smoothing: float = .1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["motion"] = self.motion.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "SignRecognizerConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)


class SignRecognizer(nn.Module):
    def __init__(self, config: SignRecognizerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        width = sum(flat for _, _, _, flat in diffusion_layout(config.motion))
        self.motion_in = nn.Linear(width, d)
        k = config.downsample
        self.downsample = nn.Conv1d(d, d, kernel_size=2 * k - 1, stride=k, padding=k - 1) if k > 1 else nn.Identity()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config.dropout), nn.Linear(d, config.classes))

    def forward(self, motion: MotionBatch) -> torch.Tensor:
        """Logits (B, classes)."""
        x = diffusion_targets(self.config.motion, motion)
        valid = motion.frame_valid.to(x.device)
        h = self.motion_in(x) * valid[..., None]
        if self.config.downsample > 1:
            h = self.downsample(h.transpose(1, 2)).transpose(1, 2)
            k = self.config.downsample
            valid = F.max_pool1d(valid.float()[:, None], k, k, ceil_mode=True)[:, 0, :h.shape[1]] > 0
        positions = torch.arange(h.shape[1], device=h.device)[None].expand(len(h), -1)
        h = self.encoder(h + sinusoid(positions, self.config.d_model, max_period=1e3), src_key_padding_mask=~valid)
        w = valid[..., None].to(h.dtype)
        return self.head((h * w).sum(1) / w.sum(1).clamp_min(1.))

    def loss(self, motion: MotionBatch, labels: torch.Tensor):
        logits = self(motion)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)
        return loss, dict(top1=(logits.argmax(1) == labels).float().mean().detach())


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> dict:
    ranked = logits.topk(min(max(ks), logits.shape[1]), dim=1).indices      # k beyond the class count = all classes
    return {f"top{k}": float((ranked[:, :k] == labels[:, None]).any(1).float().mean()) for k in ks}


__all__ = ["GlossMotionDataset", "augment_sign", "collate_gloss_motion", "SignRecognizerConfig", "SignRecognizer", "top_k_accuracy"]
