"""2D-keypoint back-translator (plan item 7 fallback): ``data.keypoints2d`` features -> encoder -> pretrained T5 decoder.

This is the second-opinion meaning instrument. It has the same encoder and decoder as back-translator v2
(``motion_to_text_t5.MotionToT5``), but reads the 48-point 2D layout (x, y, observed per point; 144 values a
frame) instead of canonical channels:
- **Pretraining:** on the YouTube-ASL Clip Keypoint Dataset (video landmarks).
- **Fine-tuning and judging:** How2Sign and generated motion, both through forward kinematics and front projection.

A batch is ``(features (B, T, 144), valid (B, T))``. ``loss`` and ``generate_text`` are inherited unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch.nn as nn

from ..data.keypoints2d import FEATURES
from .motion_to_text_t5 import MotionToT5

KIND = "keypoints_to_text_t5"


@dataclass(frozen=True)
class KeypointsToT5Config:
    features: int = FEATURES
    fps: float = 24.
    t5_name: str = "google/flan-t5-base"
    d_model: int = 384
    heads: int = 6
    encoder_layers: int = 4
    feedforward: int = 1536
    dropout: float = .1
    downsample: int = 2
    max_tokens: int = 64

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "KeypointsToT5Config":
        return cls(**data)


class KeypointsToT5(MotionToT5):
    def __init__(self, config: KeypointsToT5Config, t5, tokenizer) -> None:
        nn.Module.__init__(self)
        self.config, self.t5, self.tokenizer = config, t5, tokenizer
        d = config.d_model
        self.motion_in = nn.Linear(config.features, d)
        self.downsample = nn.Conv1d(d, d, kernel_size=config.downsample * 2 - 1, stride=config.downsample,
                                    padding=config.downsample - 1) if config.downsample > 1 else nn.Identity()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.encoder_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d)
        self.project = nn.Linear(d, t5.config.d_model)

    def encode(self, batch):
        features, valid = batch
        return self.encode_units(features, valid.to(features.device))


def build(config: KeypointsToT5Config, *, pretrained: bool = True):
    from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(config.t5_name)
    t5 = (T5ForConditionalGeneration.from_pretrained(config.t5_name) if pretrained else
          T5ForConditionalGeneration(AutoConfig.from_pretrained(config.t5_name)))
    return KeypointsToT5(config, t5, tokenizer)


__all__ = ["KIND", "KeypointsToT5Config", "KeypointsToT5", "build"]
