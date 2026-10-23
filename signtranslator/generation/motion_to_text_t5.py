"""Back-translation with a pretrained text decoder: motion encoder -> T5 decoder (plan item 3a, back-translator v2).

The first back-translator (``motion_to_text.MotionToText``) learns English from ~20k How2Sign sentences with a word
vocabulary and its own decoder; it reaches validation BLEU 2.1. Published pose-to-English systems on How2Sign gain
most from a pretrained language model on the text side. This model keeps the project's motion encoder (the same
layers as ``MotionToText``: unit representation -> linear -> strided conv -> Transformer, under the frame mask) and
feeds its states, projected to the language model's width, to a pretrained T5's decoder as ``encoder_outputs``: the
T5 encoder is not used; the decoder's cross-attention reads motion. T5 (Apache-2.0; default ``google/flan-t5-base``)
brings its tokenizer, so any English is representable and no corpus vocabulary is built.

Optimisation uses two learning rates (``param_groups``): the new motion encoder and projection train faster than the
pretrained decoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.multichannel import MotionBatch
from .text_to_motion import TextToMotionConfig, diffusion_layout, diffusion_targets, sinusoid

KIND = "motion_to_text_t5"


@dataclass(frozen=True)
class MotionToT5Config:
    motion: TextToMotionConfig
    t5_name: str = "google/flan-t5-base"
    d_model: int = 384
    heads: int = 6
    encoder_layers: int = 4
    feedforward: int = 1536
    dropout: float = .1
    downsample: int = 2
    max_tokens: int = 64

    def to_dict(self) -> dict:
        d = asdict(self)
        d["motion"] = self.motion.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "MotionToT5Config":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)


class MotionToT5(nn.Module):
    """``t5`` is a ``T5ForConditionalGeneration`` (pretrained, or built from a config in tests); ``tokenizer`` its
    tokenizer (``__call__`` -> input_ids, ``batch_decode``, ``pad_token_id``)."""

    def __init__(self, config: MotionToT5Config, t5, tokenizer) -> None:
        super().__init__()
        self.config, self.t5, self.tokenizer = config, t5, tokenizer
        d = config.d_model
        width = sum(flat for _, _, _, flat in diffusion_layout(config.motion))
        self.motion_in = nn.Linear(width, d)
        self.downsample = nn.Conv1d(d, d, kernel_size=config.downsample * 2 - 1, stride=config.downsample,
                                    padding=config.downsample - 1) if config.downsample > 1 else nn.Identity()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.encoder_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d)
        self.project = nn.Linear(d, t5.config.d_model)

    def encode(self, motion: MotionBatch):
        """(states (B, T', d_t5), mask (B, T') True = valid)."""
        x = diffusion_targets(self.config.motion, motion)
        valid = motion.frame_valid.to(x.device)
        h = self.motion_in(x) * valid[..., None]
        if self.config.downsample > 1:
            h = self.downsample(h.transpose(1, 2)).transpose(1, 2)
            k = self.config.downsample
            valid = F.max_pool1d(valid.float()[:, None], k, k, ceil_mode=True)[:, 0, :h.shape[1]] > 0
        positions = torch.arange(h.shape[1], device=h.device)[None].expand(len(h), -1)
        h = h + sinusoid(positions, self.config.d_model, max_period=1e3)
        h = self.encoder_norm(self.encoder(h, src_key_padding_mask=~valid))
        return self.project(h), valid

    def _labels(self, texts, device):
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.config.max_tokens,
                             return_tensors="pt")
        labels = enc["input_ids"].to(device)
        return labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

    def loss(self, motion: MotionBatch, texts):
        from transformers.modeling_outputs import BaseModelOutput
        states, valid = self.encode(motion)
        labels = self._labels(texts, states.device)
        out = self.t5(encoder_outputs=BaseModelOutput(last_hidden_state=states), attention_mask=valid.long(),
                      labels=labels)
        keep = labels != -100
        accuracy = ((out.logits.argmax(-1) == labels) & keep).sum() / keep.sum().clamp_min(1)
        return out.loss, dict(token_accuracy=accuracy.detach())

    @torch.no_grad()
    def generate_text(self, motion: MotionBatch, *, beam: int = 4, max_tokens: int | None = None) -> list[str]:
        from transformers.modeling_outputs import BaseModelOutput
        states, valid = self.encode(motion)
        ids = self.t5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=states), attention_mask=valid.long(),
                               num_beams=beam, max_new_tokens=max_tokens or self.config.max_tokens)
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def param_groups(self, *, motion_lr: float, decoder_lr: float, weight_decay: float) -> list[dict]:
        decoder = list(self.t5.parameters())
        ids = {id(p) for p in decoder}
        own = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=own, lr=motion_lr, base_lr=motion_lr, weight_decay=weight_decay),
                dict(params=decoder, lr=decoder_lr, base_lr=decoder_lr, weight_decay=0.)]


def build(config: MotionToT5Config, *, pretrained: bool = True):
    """The model with T5's pretrained weights (training start) or its architecture only (loading a checkpoint)."""
    from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(config.t5_name)
    t5 = (T5ForConditionalGeneration.from_pretrained(config.t5_name) if pretrained else
          T5ForConditionalGeneration(AutoConfig.from_pretrained(config.t5_name)))
    return MotionToT5(config, t5, tokenizer)


__all__ = ["KIND", "MotionToT5Config", "MotionToT5", "build"]
