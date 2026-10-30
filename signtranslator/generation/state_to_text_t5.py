"""Instrument v2: the state-to-text back-translator (run ``back-translator-state-t5-v1``, kind ``state_to_text_t5``;
docs/NEXT_PHASE_PLAN.md, "Instrument v2: design and declared gates", declared 2026-10-30 09:10 UTC).

The same encoder-under-T5 design as v1 (``keypoints_to_text_t5``: per-frame vector -> linear -> strided convolution
(x2 downsample) -> Transformer -> projection, read by a pretrained flan-t5-base decoder through cross-attention), with
the declared changes:
- **input:** the generator's own state, conformed to the ``how2sign-two-view-text-v1`` template and at 24 fps, read by
  ``data.state_features`` (3D points with z, native rotation, face, blink, head and gaze channels, root and head
  translation centred on the clip's median, per-element validity flags);
- **encoder:** d 512, 6 layers (``StateToT5Config`` defaults); not warm-started from v1, so the ranking check stays
  independent.

The configuration pins the feature version, the native channel layout (``motion``: the template's bindings and the
channels read) and the template it was taken from (``template``: manifest key and SHA-256, corpus, sign and state
SHA-256), and carries the licence tag (ASL STEM Wiki: instruments only). ``layout_template`` rebuilds a one-frame state
in exactly those layouts, so ``prepare`` can conform any state (``data.layout_adapter.conform``) without the stored
template. Checkpoints are plain tensors and JSON-like values (``load_checkpoint`` reads them with
``torch.load(weights_only=True)``: no pickled objects).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json

import numpy as np
import torch.nn as nn

from ..data.state_features import FEATURE_VERSION, NATIVE_CHANNELS, feature_width, state_features
from .motion_to_text_t5 import MotionToT5
from .text_to_motion import TextToMotionConfig

KIND = "state_to_text_t5"
RUN = "back-translator-state-t5-v1"
TEMPLATE_CORPUS = "how2sign-two-view-text-v1"
LICENCE = ("Trained on How2Sign (CC BY-NC 4.0), YouTube-ASL states and ASL STEM Wiki (Microsoft Research License: "
           "non-commercial research, no distribution). Instrument use only: never trains or reranks a generator, "
           "never released.")
TEMPLATE_FIELDS = ("manifest_key", "manifest_sha256", "corpus", "sign", "state_sha256")


@dataclass(frozen=True)
class StateToT5Config:
    motion: TextToMotionConfig
    template: tuple = ()                   # ((field, value), ...) in TEMPLATE_FIELDS order
    feature_version: str = FEATURE_VERSION
    fps: float = 24.
    licence: str = LICENCE
    t5_name: str = "google/flan-t5-base"
    d_model: int = 512
    heads: int = 8
    encoder_layers: int = 6
    feedforward: int = 2048
    dropout: float = .1
    downsample: int = 2
    max_tokens: int = 64

    def __post_init__(self):
        if self.feature_version != FEATURE_VERSION:
            raise ValueError(f"feature version {self.feature_version!r}: this code computes {FEATURE_VERSION!r}")
        if tuple(k for k, _ in self.template) not in ((), TEMPLATE_FIELDS):
            raise ValueError(f"template pin fields must be {TEMPLATE_FIELDS}")

    @property
    def features(self) -> int:
        return feature_width(self.motion)

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        out["motion"] = self.motion.to_dict()
        out["template"] = dict(self.template)
        return out

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "StateToT5Config":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        template = data.get("template") or {}
        data["template"] = tuple((k, template[k]) for k in TEMPLATE_FIELDS) if template else ()
        return cls(**data)


def motion_config(template_state, channels=NATIVE_CHANNELS) -> TextToMotionConfig:
    """The native-channel spec: the template's bindings, ``channels`` read."""
    from .canonical_state import channel_bindings_from_state
    return TextToMotionConfig(bindings=channel_bindings_from_state(template_state), trained_channels=tuple(channels))


def layout_template(config: StateToT5Config):
    """A one-frame state, every channel unavailable, in exactly the configuration's channel layouts: the template
    ``conform`` needs (it reads only the layouts)."""
    from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
    pin = dict(config.template)
    source = pin.get("state_sha256") or "0" * 64
    channels = {}
    for b in config.motion.bindings:
        width = CHANNELS[b.name][1]
        J = len(b.labels)
        channels[b.name] = MotionChannel(np.zeros((1, J, width), np.float32), np.zeros((1, J), bool),
                                         np.zeros((1, J), bool), np.zeros((1, J), np.float32), b.labels,
                                         b.coordinate_frame, b.units, "instrument-v2-layout-template", source,
                                         b.convention)
    return MultichannelMotion(np.zeros(1), "layout-template", "layout-template", channels)


def at_fps(state, fps: float):
    """``state`` resampled to ``fps`` when its median rate differs by more than 1% (``evaluate_text_to_motion.at_rate``'s
    rule)."""
    if len(state.timestamps) > 1:
        rate = 1. / float(np.median(np.diff(state.timestamps)))
        if abs(rate - fps) > .01 * fps:
            from ..pose.resample import resample_motion
            return resample_motion(state, fps)
    return state


def prepare(state, config: StateToT5Config, template=None) -> np.ndarray:
    """A state as the instrument reads it: conformed to the template layouts, at the configuration's rate,
    featurised. ``template``: ``layout_template(config)`` (built when not given)."""
    from ..data.layout_adapter import conform
    state = conform(state, template if template is not None else layout_template(config))
    return state_features(at_fps(state, config.fps), config.motion)


class StateToT5(MotionToT5):
    """A batch is ``(features (B, T, config.features), valid (B, T))``; ``loss`` and ``generate_text`` are
    ``MotionToT5``'s."""

    def __init__(self, config: StateToT5Config, t5, tokenizer) -> None:
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
        self._template = None

    def encode(self, batch):
        features, valid = batch
        return self.encode_units(features, valid.to(features.device))

    def featurise(self, states) -> list[np.ndarray]:
        if self._template is None:
            self._template = layout_template(self.config)
        return [prepare(s, self.config, self._template) for s in states]


def build(config: StateToT5Config, *, pretrained: bool = True):
    """The model with T5's pretrained weights (training start) or its architecture only (loading a checkpoint)."""
    from transformers import AutoConfig, AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(config.t5_name)
    t5 = (T5ForConditionalGeneration.from_pretrained(config.t5_name) if pretrained else
          T5ForConditionalGeneration(AutoConfig.from_pretrained(config.t5_name)))
    return StateToT5(config, t5, tokenizer)


def load_checkpoint(payload: bytes, *, expected_sha256: str | None = None) -> dict:
    """A ``state_to_text_t5`` checkpoint's contents, read without unpickling objects (``weights_only=True``), its
    SHA-256 checked when given."""
    import torch
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("checkpoint SHA-256 differs from the pin")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if saved.get("kind") != KIND:
        raise ValueError(f"not a {KIND} checkpoint: {saved.get('kind')!r}")
    return saved


def load_model(payload: bytes, *, expected_sha256: str | None = None, build_model=None):
    """(model with EMA weights in eval mode, saved checkpoint dict); ``build_model`` defaults to ``build``."""
    saved = load_checkpoint(payload, expected_sha256=expected_sha256)
    config = StateToT5Config.from_dict(saved["config"])
    if config.sha256() != saved["config_sha256"]:
        raise ValueError("the checkpoint's config does not hash to its config_sha256")
    model = (build_model or build)(config, pretrained=False)
    model.load_state_dict(saved.get("ema") or saved["model"])
    return model.eval(), saved


__all__ = ["KIND", "RUN", "LICENCE", "TEMPLATE_CORPUS", "TEMPLATE_FIELDS", "StateToT5Config", "StateToT5",
           "motion_config", "layout_template", "at_fps", "prepare", "build", "load_checkpoint", "load_model"]
