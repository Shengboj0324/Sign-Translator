"""Multi-channel non-manual interval decoder (docs/FACIAL_NMM.md §3).

Conditioned on a hidden state ``h_t`` (from the semantic plan and the manual
motion), each concurrent channel predicts an **independent Bernoulli** per frame
(``p_{t,k} = σ(w_kᵀ h_t)``, NOT a softmax, because channels co-occur). Spans are
decoded with the Doc-03 ``spans_from_activations``. The BCE reuses the Doc-03
``multilabel_scope_bce``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..grammar.nonmanual import spans_from_activations, MarkerSpan


class MultiChannelDecoder(nn.Module):
    """Conditioned temporal decoder over concurrent non-manual channels."""

    def __init__(self, cond_dim: int, hidden_dim: int, num_channels: int,
                 num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.proj = nn.Linear(cond_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(hidden_dim, num_channels)     # rows w_k

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """``cond`` (N, T, cond_dim) -> per-channel logits (N, T, num_channels)."""
        h = self.encoder(self.proj(cond))
        return self.head(h)

    def probabilities(self, cond: torch.Tensor) -> torch.Tensor:
        """Independent per-channel Bernoulli probabilities (N, T, num_channels)."""
        return torch.sigmoid(self.forward(cond))

    def decode(self, probs: torch.Tensor, unit_starts, unit_ends,
               threshold: float = 0.5) -> List[MarkerSpan]:
        """Decode a single sequence's (T, num_channels) probs into scoped spans."""
        return spans_from_activations(probs, unit_starts, unit_ends, threshold)
