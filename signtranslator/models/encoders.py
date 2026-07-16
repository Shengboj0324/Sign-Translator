"""Language and speech encoders.

These map a *modality* (token ids for text/gloss, or a feature sequence for
speech) to a fixed-size embedding that will later be projected into the shared
contrastive manifold.

Design: ``TextEncoder`` / ``SpeechEncoder`` are thin abstract interfaces. The
default implementations (``StubTextEncoder`` / ``StubSpeechEncoder``) are small
self-contained Transformers so the whole system builds, trains, and is testable
without downloading multi-GB foundation models. Real backends (Whisper,
wav2vec2, an LLM planner, ...) can be dropped in by subclassing the interface and
returning an ``(N, embed_dim)`` tensor -- nothing else in the pipeline changes.
"""

from __future__ import annotations

import abc
import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal position encoding (Vaswani et al., 2017)."""

    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padded positions.

    x: (N, L, D); mask: (N, L) with 1 for valid tokens, 0 for padding.
    """
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(x.dtype).unsqueeze(-1)  # (N, L, 1)
    summed = (x * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


class TextEncoder(nn.Module, abc.ABC):
    """Interface: token ids -> (N, embed_dim) sentence/gloss embedding."""

    embed_dim: int

    @abc.abstractmethod
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ...


class SpeechEncoder(nn.Module, abc.ABC):
    """Interface: feature sequence (N, T, F) -> (N, embed_dim) embedding."""

    embed_dim: int

    @abc.abstractmethod
    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ...


class StubTextEncoder(TextEncoder):
    """Lightweight Transformer encoder over a token/gloss vocabulary."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1,
                 padding_idx: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos = _SinusoidalPositionalEncoding(embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = (tokens != self.padding_idx)
        key_padding = ~mask.bool()  # True where padded (nn convention)
        h = self.pos(self.token_emb(tokens))
        h = self.encoder(h, src_key_padding_mask=key_padding)
        return self.norm(_masked_mean(h, mask))


class StubSpeechEncoder(SpeechEncoder):
    """Lightweight Transformer over frame features (stands in for Whisper/wav2vec2)."""

    def __init__(self, input_dim: int, embed_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_in = nn.Linear(input_dim, embed_dim)
        self.pos = _SinusoidalPositionalEncoding(embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        key_padding = None if mask is None else ~mask.bool()
        h = self.pos(self.proj_in(features))
        h = self.encoder(h, src_key_padding_mask=key_padding)
        return self.norm(_masked_mean(h, mask))
