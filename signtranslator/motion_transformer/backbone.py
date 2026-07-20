"""Hierarchical temporal backbone (docs/MOTION_TRANSFORMER.md §6).

Four coupled modules at two rates:

1. ``ClausePlanner`` — a low-rate Transformer over clause/plan tokens.
2. ``DurationModel`` — predicts each event's duration (bucketed) and up-samples the
   low-rate plan to the high-rate motion timeline.
3. ``MotionDecoder`` — a high-rate Transformer over motion tokens with
   **cross-attention** to the plan events (optionally causal for streaming).
4. ``RecurrentMemory`` — a GRU state carrying spatial loci + prior pose across
   chunks, so discourse referents persist over long sequences.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) additive mask: 0 on/below the diagonal, -inf above (no future)."""
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)


class ClausePlanner(nn.Module):
    """Low-rate Transformer encoder over clause/plan tokens."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)

    def forward(self, clause_tokens: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(clause_tokens, src_key_padding_mask=key_padding_mask)


class DurationModel(nn.Module):
    """Predict per-event durations (buckets 1..max_duration) and expand the plan."""

    def __init__(self, dim: int, max_duration: int = 32) -> None:
        super().__init__()
        self.max_duration = max_duration
        self.head = nn.Linear(dim, max_duration)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """(N, L, d) -> (N, L, max_duration) duration logits (bucket k = k+1 frames)."""
        return self.head(events)

    @staticmethod
    def expand_by_duration(events: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Up-sample low-rate events to the high-rate timeline.

        ``events`` (L, d), ``durations`` (L,) positive ints -> (Σ durations, d),
        each event repeated for its duration. This is the rate bridge: a low-rate
        plan of L events becomes a high-rate sequence of Σ durations frames.
        """
        if events.dim() != 2:
            raise ValueError("expand_by_duration expects a single sequence (L, d)")
        return events.repeat_interleave(durations.to(torch.long), dim=0)


class MotionDecoder(nn.Module):
    """High-rate Transformer decoder: self-attention + cross-attention to the plan."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(dim, num_heads, dim * 4, batch_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers)

    def forward(self, motion_tokens: torch.Tensor, plan_memory: torch.Tensor,
                causal: bool = False) -> torch.Tensor:
        """``motion_tokens`` (N, T, d) attends to ``plan_memory`` (N, L, d)."""
        tgt_mask = causal_mask(motion_tokens.shape[1], motion_tokens.device) if causal else None
        return self.decoder(motion_tokens, plan_memory, tgt_mask=tgt_mask)


class RecurrentMemory(nn.Module):
    """GRU carrying spatial loci + prior pose across chunks."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.cell = nn.GRUCell(input_dim, state_dim)

    def init_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch, self.state_dim, device=device, dtype=dtype)

    def forward(self, chunk_summary: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """(N, input_dim), (N, state_dim) -> (N, state_dim)."""
        return self.cell(chunk_summary, state)
