"""Local text-proposal and video-evidence models for pseudo-gloss lattices.

The text model never receives video. The video model has no transcript argument.
That separation is intentional and is tested by intervention, not inferred from
comments or a joint architecture.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_engineering.exporter import LandmarkTrack
from ..data_engineering.how2sign import openpose_holistic_graph
from ..models.stgcn import STGCNBlock
from .contracts import CandidateHypothesis, GlossLexicon
from .mathematics import ctc_minimum_frames
from .security import InputSecurityPolicy, validate_transcript


_SOURCE_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class SourceTokenizer:
    tokenizer_id: str
    tokens: tuple[str, ...]
    source_sha256: str

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __post_init__(self) -> None:
        if (not isinstance(self.tokenizer_id, str) or not self.tokenizer_id
                or not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.source_sha256)):
            raise ValueError("tokenizer ID and SHA-256 are required")
        if not isinstance(self.tokens, tuple) or not self.tokens \
                or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("source tokenizer tokens must be non-empty and unique")
        if any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("source tokenizer tokens must be non-empty strings")

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens) + 4

    def encode(self, transcript: str, policy: InputSecurityPolicy) -> tuple[int, ...]:
        validate_transcript(transcript, policy)
        mapping = {token: index + 4 for index, token in enumerate(self.tokens)}
        pieces = _SOURCE_TOKEN_RE.findall(transcript.casefold())
        if not pieces:
            raise ValueError("tokenizer produced an empty source sequence")
        if len(pieces) > policy.max_words:
            raise ValueError("source token sequence exceeds declared bound")
        return (self.BOS, *(mapping.get(piece, self.UNK) for piece in pieces), self.EOS)


@dataclass(frozen=True)
class TextProposalConfig:
    embedding_dim: int = 128
    feedforward_dim: int = 256
    layers: int = 3
    heads: int = 4
    dropout: float = 0.0
    beam_size: int = 16
    max_candidate_tokens: int = 64
    max_source_tokens: int = 1_024

    def __post_init__(self) -> None:
        integer_values = (
            self.embedding_dim, self.feedforward_dim, self.layers, self.heads,
            self.beam_size, self.max_candidate_tokens, self.max_source_tokens,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in integer_values):
            raise ValueError("text proposal dimensions must be positive")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) \
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.max_source_tokens < 2:
            raise ValueError("max_source_tokens must be at least two")


@dataclass(frozen=True)
class CandidateLatticeProposal:
    candidates: tuple[CandidateHypothesis, ...]
    retained_probability: float
    dropped_probability_mass: float


class NeuralTextProposalModel(nn.Module):
    """Checkpoint-loadable encoder-decoder with lexicon-constrained beam search.

    It must be trained or initialized from an approved external checkpoint; this
    class does not train on English-derived pseudo targets by itself.
    """

    def __init__(self, tokenizer: SourceTokenizer, lexicon: GlossLexicon,
                 config: TextProposalConfig = TextProposalConfig()) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.config = config
        # Decoder ids: 0=EOS, 1..K=lexicon tokens, K+1=BOS.
        self.target_bos = len(lexicon.tokens) + 1
        self.source_embedding = nn.Embedding(tokenizer.vocabulary_size, config.embedding_dim,
                                             padding_idx=tokenizer.PAD)
        self.target_embedding = nn.Embedding(len(lexicon.tokens) + 2, config.embedding_dim)
        self.transformer = nn.Transformer(
            d_model=config.embedding_dim, nhead=config.heads,
            num_encoder_layers=config.layers, num_decoder_layers=config.layers,
            dim_feedforward=config.feedforward_dim, dropout=config.dropout,
            batch_first=True, norm_first=False,
        )
        # The prototype nested-tensor inference path emits warnings under the
        # project's warnings-as-errors contract and is not needed for the
        # per-record generation/training path used here.
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False
        self.output = nn.Linear(config.embedding_dim, len(lexicon.tokens) + 1)

    @staticmethod
    def _position_encoding(length: int, width: int, *, device, dtype) -> torch.Tensor:
        """Deterministic sinusoidal positions; order must not be discarded."""
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        even_indices = torch.arange(0, width, 2, device=device, dtype=dtype)
        scale = torch.exp(-math.log(10_000.0) * even_indices / width)
        angles = position * scale.unsqueeze(0)
        encoding = torch.zeros((length, width), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(angles)
        if width > 1:
            encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
        return encoding

    def forward(self, source_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        if source_ids.ndim != 2 or decoder_ids.ndim != 2:
            raise ValueError("source_ids and decoder_ids must be rank-two")
        if source_ids.shape[0] != decoder_ids.shape[0] or source_ids.shape[1] < 2 \
                or decoder_ids.shape[1] < 1:
            raise ValueError("text model inputs have invalid batch or sequence dimensions")
        if torch.any((source_ids < 0) | (source_ids >= self.tokenizer.vocabulary_size)):
            raise ValueError("source token ID outside tokenizer vocabulary")
        if torch.any((decoder_ids < 0) | (decoder_ids > self.target_bos)):
            raise ValueError("decoder token ID outside target vocabulary")
        if source_ids.shape[1] > self.config.max_source_tokens \
                or decoder_ids.shape[1] > self.config.max_candidate_tokens + 1:
            raise ValueError("text model sequence exceeds configured positional bounds")
        source_padding = source_ids == self.tokenizer.PAD
        length = decoder_ids.shape[1]
        causal_mask = torch.triu(
            torch.full((length, length), float("-inf"), device=decoder_ids.device), diagonal=1)
        source_embedding = self.source_embedding(source_ids)
        target_embedding = self.target_embedding(decoder_ids)
        source_embedding = source_embedding + self._position_encoding(
            source_ids.shape[1], source_embedding.shape[-1],
            device=source_embedding.device, dtype=source_embedding.dtype)
        target_embedding = target_embedding + self._position_encoding(
            decoder_ids.shape[1], target_embedding.shape[-1],
            device=target_embedding.device, dtype=target_embedding.dtype)
        hidden = self.transformer(
            source_embedding, target_embedding,
            tgt_mask=causal_mask, src_key_padding_mask=source_padding,
        )
        return self.output(hidden)

    def freeze_for_generation(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _assert_frozen(self) -> None:
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("text proposal model must be frozen and in eval mode")

    @torch.no_grad()
    def propose(self, transcript: str, policy: InputSecurityPolicy,
                *, candidate_limit: int | None = None) -> CandidateLatticeProposal:
        """Deterministic schema-constrained beam search over the closed lexicon."""
        self._assert_frozen()
        limit = (min(policy.max_candidates, self.config.beam_size)
                 if candidate_limit is None else candidate_limit)
        if limit < 1 or limit > policy.max_candidates:
            raise ValueError("candidate limit exceeds the security policy")
        maximum = min(policy.max_candidate_tokens, self.config.max_candidate_tokens)
        encoded_source = self.tokenizer.encode(transcript, policy)
        if len(encoded_source) > self.config.max_source_tokens:
            raise ValueError("tokenized transcript exceeds model source-length bound")
        source = torch.tensor([encoded_source], dtype=torch.long,
                              device=next(self.parameters()).device)
        # Each live state is (generated decoder classes, cumulative log probability).
        live: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
        completed: list[tuple[tuple[int, ...], float]] = []
        for _ in range(maximum):
            expanded: list[tuple[tuple[int, ...], float]] = []
            for prefix, cumulative in live:
                decoder = torch.tensor(
                    [[self.target_bos, *prefix]], dtype=torch.long, device=source.device)
                log_probs = F.log_softmax(self(source, decoder)[0, -1], dim=-1)
                values, indices = torch.topk(log_probs, k=min(self.config.beam_size,
                                                               log_probs.numel()))
                for value, index in zip(values.tolist(), indices.tolist()):
                    score = cumulative + float(value)
                    if index == 0:
                        if prefix:
                            completed.append((prefix, score))
                    else:
                        expanded.append(((*prefix, index), score))
            expanded.sort(key=lambda item: (-item[1], item[0]))
            live = expanded[:self.config.beam_size]
            if not live:
                break
        completed.sort(key=lambda item: (-item[1], item[0]))
        completed = completed[:limit]
        if not completed:
            return CandidateLatticeProposal((), 0.0, 1.0)
        hypotheses = tuple(CandidateHypothesis(
            tokens=tuple(self.lexicon.tokens[index - 1] for index in prefix),
            text_log_probability=score, rank=rank,
        ) for rank, (prefix, score) in enumerate(completed, start=1))
        retained = math.fsum(math.exp(item.text_log_probability) for item in hypotheses)
        # Completed paths are mutually exclusive; roundoff may exceed one by a few ulps.
        retained = min(1.0, max(0.0, retained))
        return CandidateLatticeProposal(hypotheses, retained, 1.0 - retained)


@dataclass(frozen=True)
class VideoEvidenceConfig:
    hidden_channels: int = 64
    blocks: int = 3
    temporal_kernel: int = 5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (self.hidden_channels, self.blocks, self.temporal_kernel)):
            raise ValueError("video model dimensions must be exact integers")
        if self.hidden_channels < 4 or self.blocks < 1:
            raise ValueError("video model dimensions are too small")
        if self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("temporal kernel must be positive and odd")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) \
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


def prepare_openpose_features(track: LandmarkTrack) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert audited normalized 2D observations to explicit four-channel input."""
    values = np.asarray(track.values)
    confidence = np.asarray(track.confidence)
    validity = np.asarray(track.validity_mask)
    timestamps = np.asarray(track.timestamps)
    if values.ndim != 3 or values.shape[0] != 2 or values.shape[2] != 137:
        raise ValueError("pseudo-gloss video evidence requires 2D 137-node OpenPose tracks")
    if confidence.shape != values.shape[1:] or validity.shape != values.shape[1:]:
        raise ValueError("confidence and validity must align with the OpenPose track")
    if timestamps.shape != (values.shape[1],) or not np.isfinite(timestamps).all() \
            or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite, aligned, and strictly increasing")
    if validity.dtype != np.bool_:
        raise ValueError("validity must be boolean")
    if not np.isfinite(values[:, validity]).all() or not np.isfinite(confidence).all():
        raise ValueError("valid visual observations and confidence must be finite")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must lie in [0,1]")
    if np.any(confidence[~validity] != 0):
        raise ValueError("invalid observations must have zero confidence")
    if np.any((values[:, validity] < 0) | (values[:, validity] > 1)):
        raise ValueError("coordinates must be normalized image fractions in [0,1]")
    safe_xy = np.where(validity[None], values, 0.0).astype(np.float32, copy=False)
    channels = np.concatenate((
        safe_xy,
        confidence.astype(np.float32, copy=False)[None],
        validity.astype(np.float32, copy=False)[None],
    ), axis=0)
    frame_validity = validity.any(axis=1)
    if not frame_validity.any():
        raise ValueError("track contains no frame with a valid visual observation")
    return torch.from_numpy(channels), torch.from_numpy(frame_validity)


class VideoCTCEvidenceModel(nn.Module):
    """Transcript-independent graph-temporal CTC evidence model."""

    def __init__(self, lexicon: GlossLexicon,
                 config: VideoEvidenceConfig = VideoEvidenceConfig()) -> None:
        super().__init__()
        self.lexicon = lexicon
        self.config = config
        adjacency = openpose_holistic_graph().adjacency()
        self.input = nn.Conv2d(4, config.hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            STGCNBlock(config.hidden_channels, config.hidden_channels, adjacency,
                       temporal_kernel=config.temporal_kernel, dropout=config.dropout,
                       residual=True)
            for _ in range(config.blocks)
        ])
        self.output = nn.Linear(config.hidden_channels, len(lexicon.tokens) + 1)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 4 or visual_features.shape[1] != 4 \
                or visual_features.shape[-1] != 137:
            raise ValueError("visual_features must be (N,4,T,137)")
        if not torch.isfinite(visual_features).all():
            raise ValueError("visual features must be finite")
        hidden = self.input(visual_features)
        for block in self.blocks:
            hidden = block(hidden)
        sequence = hidden.mean(dim=3).transpose(1, 2).contiguous()
        return F.log_softmax(self.output(sequence), dim=-1)

    def loss(self, visual_features: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor, input_lengths: torch.Tensor,
             frame_validity: torch.Tensor | None = None) -> torch.Tensor:
        log_probs = self(visual_features)
        if targets.dtype not in {torch.int32, torch.int64} \
                or target_lengths.dtype not in {torch.int32, torch.int64} \
                or input_lengths.dtype not in {torch.int32, torch.int64}:
            raise TypeError("CTC targets and lengths must use integer tensor dtypes")
        if torch.any(target_lengths < 1):
            raise ValueError("CTC target sequences must be non-empty")
        flat_targets = targets.reshape(-1)
        if torch.any((flat_targets <= 0) | (flat_targets > len(self.lexicon.tokens))):
            raise ValueError("CTC targets must be valid non-blank lexicon IDs")
        if frame_validity is not None:
            if frame_validity.dtype != torch.bool \
                    or frame_validity.shape != log_probs.shape[:2]:
                raise ValueError("frame_validity must be a boolean (N,T) mask")
            if log_probs.shape[0] != 1:
                raise ValueError("masked CTC loss currently requires a single-example batch")
            selected_frames = int(frame_validity[0].sum())
            if selected_frames < 1:
                raise ValueError("masked CTC loss has no visually valid frame")
            if input_lengths.numel() != 1 or int(input_lengths[0]) != selected_frames:
                raise ValueError("input length must equal the number of visually valid frames")
            log_probs = log_probs[0, frame_validity[0]].unsqueeze(0)
        if target_lengths.shape != input_lengths.shape \
                or target_lengths.ndim != 1 or target_lengths.shape[0] != log_probs.shape[0]:
            raise ValueError("CTC lengths must be aligned batch vectors")
        if torch.any(input_lengths < 1) or torch.any(input_lengths > log_probs.shape[1]):
            raise ValueError("input lengths are outside the video sequence")
        offset = 0
        for input_length, target_length in zip(input_lengths.tolist(), target_lengths.tolist()):
            sequence = flat_targets[offset:offset + target_length].tolist()
            if input_length < ctc_minimum_frames(sequence):
                raise ValueError("CTC target is infeasible for an encoder output length")
            offset += target_length
        if offset != flat_targets.numel():
            raise ValueError("target lengths do not consume the supplied targets exactly")
        loss = F.ctc_loss(
            log_probs.transpose(0, 1), flat_targets, input_lengths, target_lengths,
            blank=0, reduction="mean", zero_infinity=False,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("video CTC loss is non-finite")
        return loss


def state_dict_sha256(model: nn.Module) -> str:
    """Stable hash over names, dtypes, shapes, and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()
