"""Relation-biased graph attention and the SIR decoder.

The document's graph-aware attention adds a learned scalar bias per relation type
to the logits before the softmax:

    alpha_ij = softmax_j( (W_Q h_i)·(W_K h_j) / sqrt(d) + b_{r(i,j)} )

``b_r`` lets the graph structure (precedence, scope, coref, ...) shape attention.
When ``b == 0`` this is exactly standard scaled dot-product attention -- proved
in the tests, so the extension is a strict, auditable superset.

The SIR decoder consumes a semantic graph encoding and predicts, per event, its
time interval ``[t_start, t_end)`` (via a positive-width parameterisation that
keeps the interval valid by construction) and its non-manual marker activations.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def relation_biased_attention(query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor, relation_bias: torch.Tensor,
                              mask: Optional[torch.Tensor] = None):
    """Scaled dot-product attention with an additive per-pair relation bias.

    Shapes: ``query/key/value`` are ``(N, T, d)``; ``relation_bias`` is
    ``(N, T, T)`` (or broadcastable) added to the logits; ``mask`` is a boolean
    ``(N, T, T)`` with ``True`` where attention is DISALLOWED.

    Returns ``(context, weights)`` with ``weights`` summing to 1 over the last
    axis. Masking is done in log space so a disallowed pair gets exactly 0 with
    no NaN.
    """
    d = query.shape[-1]
    logits = query @ key.transpose(-2, -1) / math.sqrt(d)      # (N, T, T)
    logits = logits + relation_bias
    if mask is not None:
        logits = logits.masked_fill(mask, NEG_INF)
    weights = torch.softmax(logits, dim=-1)
    return weights @ value, weights


class RelationBiasedAttention(nn.Module):
    """Multi-head relation-biased self-attention over graph nodes.

    ``num_relations`` learned scalar biases (index 0 is the "no edge" default).
    An adjacency tensor of relation ids ``(N, T, T)`` selects the bias per pair.
    """

    def __init__(self, d_model: int, num_heads: int, num_relations: int) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        # one learned bias per (head, relation); relation 0 == "no edge".
        self.relation_bias = nn.Parameter(torch.zeros(num_heads, num_relations))
        self.num_relations = num_relations

    def _split_heads(self, x):
        n, t, _ = x.shape
        return x.view(n, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, relation_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x`` (N,T,d), ``relation_ids`` (N,T,T) in [0, num_relations)."""
        if relation_ids.max() >= self.num_relations or relation_ids.min() < 0:
            raise ValueError("relation id out of range")
        n, t, _ = x.shape
        q = self._split_heads(self.q_proj(x))              # (N, H, T, hd)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # (N,H,T,T)
        # gather the per-head bias for each pair's relation id
        bias = self.relation_bias[:, relation_ids]         # (H, N, T, T)
        bias = bias.permute(1, 0, 2, 3)                    # (N, H, T, T)
        logits = logits + bias
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1), NEG_INF)
        weights = torch.softmax(logits, dim=-1)
        context = weights @ v                              # (N, H, T, hd)
        context = context.transpose(1, 2).reshape(n, t, self.d_model)
        return self.o_proj(context)


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_relations: int,
                 ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = RelationBiasedAttention(d_model, num_heads, num_relations)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, relation_ids, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, relation_ids, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class SIRDecoder(nn.Module):
    """Predict event intervals and non-manual activations from node states.

    Interval parameterisation: the head emits ``(start_raw, log_width)`` and the
    interval is ``[start, start + softplus(log_width) + eps)``. Because the width
    is strictly positive, ``t_start < t_end`` holds **by construction** -- the
    validity constraint can never be violated, so no validity loss is needed at
    inference.
    """

    def __init__(self, d_model: int, num_relations: int, num_markers: int,
                 num_layers: int = 2, num_heads: int = 4,
                 min_width: float = 1e-2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, num_heads, num_relations)
             for _ in range(num_layers)])
        self.interval_head = nn.Linear(d_model, 2)         # (start_raw, log_width)
        self.marker_head = nn.Linear(d_model, num_markers)  # multilabel logits
        self.min_width = min_width

    def encode(self, x, relation_ids, mask=None):
        for layer in self.layers:
            x = layer(x, relation_ids, mask)
        return x

    def forward(self, x, relation_ids, mask=None):
        h = self.encode(x, relation_ids, mask)
        raw = self.interval_head(h)                         # (N, T, 2)
        start = raw[..., 0]
        width = F.softplus(raw[..., 1]) + self.min_width    # strictly positive
        end = start + width
        marker_logits = self.marker_head(h)                 # (N, T, M)
        return start, end, marker_logits
