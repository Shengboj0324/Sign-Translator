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
        if (x.ndim != 3 or x.shape[1] < 1 or x.shape[1] > self.pe.shape[1]
                or x.shape[2] != self.pe.shape[2]):
            raise ValueError("sequence shape exceeds positional layout/capacity")
        return x + self.pe[:, : x.size(1)]


def _sequence_mask(shape, device, mask):
    """Require at least one available position in each nonempty sequence."""
    if len(shape) != 2 or any(d < 1 for d in shape):
        raise ValueError("sequence must have nonempty (N,L) support")
    if mask is None:
        mask = torch.ones(shape, dtype=torch.bool, device=device)
    if (not torch.is_tensor(mask) or mask.dtype != torch.bool
            or mask.shape != shape or mask.device != device):
        raise ValueError("sequence mask must be boolean (N,L) on the input device")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("sequence has no available evidence")
    return mask


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average available sequence entries; unavailable NaNs contribute nothing."""
    if x.ndim != 3 or not x.is_floating_point():
        raise ValueError("sequence features must be floating (N,L,D)")
    mask = _sequence_mask(x.shape[:2], x.device, mask)
    if not bool(torch.isfinite(x[mask]).all()):
        raise ValueError("available sequence features must be finite")
    safe = torch.where(mask.unsqueeze(-1), x, 0)
    return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True)


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
        h, mask = self.encode_sequence(tokens, mask)
        return self.norm(_masked_mean(h, mask))

    def encode_sequence(self, tokens: torch.Tensor,
                        mask: torch.Tensor | None = None):
        """Return per-token features ``(N, L, D)`` and the validity ``mask``.

        Used for cross-attention conditioning, where the generator attends to the
        whole gloss sequence rather than a single pooled vector.
        """
        if tokens.ndim != 2 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must be an integer (N,L) tensor")
        if mask is None:
            mask = tokens != self.padding_idx
        mask = _sequence_mask(tokens.shape, tokens.device, mask)
        available = tokens[mask]
        if bool(((available < 0) | (available >= self.token_emb.num_embeddings)
                 | (available == self.padding_idx)).any()):
            raise ValueError("available tokens must be nonpadding vocabulary IDs")
        safe_tokens = torch.where(mask, tokens, self.padding_idx)
        h = self.pos(self.token_emb(safe_tokens))
        h = self.encoder(h, src_key_padding_mask=~mask)
        return torch.where(mask.unsqueeze(-1), h, 0), mask


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
        if (features.ndim != 3 or not features.is_floating_point()
                or features.shape[-1] != self.proj_in.in_features):
            raise ValueError("speech features must be floating (N,T,F) matching input_dim")
        mask = _sequence_mask(features.shape[:2], features.device, mask)
        if not bool(torch.isfinite(features[mask]).all()):
            raise ValueError("available speech features must be finite")
        safe = torch.where(mask.unsqueeze(-1), features, 0)
        h = self.pos(self.proj_in(safe))
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.norm(_masked_mean(h, mask))
