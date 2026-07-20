"""Graphormer structural encodings (docs/HAND_GRAPH.md §5).

Three structural signals, following Ying et al. (arXiv:2106.05234):

* **Centrality** encoding -- a learnable per-degree vector added to node features
  before attention.
* **Spatial** encoding -- a learnable scalar bias ``b_{φ(i,j)}`` (per head) indexed
  by the shortest-path distance ``φ(i,j)``, added to the attention logits.
* **Edge** encoding -- a learnable per-edge-type bias added to the logits of
  adjacent node pairs.

Attention logit:  ``A_ij = (h_i W_Q)(h_j W_K)ᵀ/√d_h + b_{φ(i,j)} + c_ij``.

The biases are **zero-initialised**, so at initialisation the module is exactly
standard multi-head scaled dot-product attention; a flag / zeroed biases recover
vanilla attention exactly (proved in the tests). This is the strict-superset
property that lets Graphormer contain GNN variants as special cases.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn

from .hetero_graph import HandGraph, NUM_EDGE_TYPES, shortest_path_distances


def edge_type_matrix(edge_index: torch.Tensor, edge_type: torch.Tensor,
                     num_nodes: int) -> torch.Tensor:
    """(V, V) long: ``M[i,j] = edge_type+1`` if edge (i->j) exists, else 0.

    Index 0 is reserved for "no direct edge". If multiple edges share a pair, the
    last one written wins (deterministic given edge order).
    """
    M = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
    if edge_index.numel() > 0:
        M[edge_index[0], edge_index[1]] = edge_type + 1
    return M


def dense_bias_inputs(graph: HandGraph, max_spd: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (spd_clamped (V,V) long in [0,max_spd], edge_type_matrix (V,V))."""
    spd = shortest_path_distances(graph.edge_index, graph.num_nodes,
                                  max_distance=max_spd)
    spd = spd.clamp(max=max_spd)
    etm = edge_type_matrix(graph.edge_index, graph.edge_type, graph.num_nodes)
    return spd, etm


class CentralityEncoding(nn.Module):
    """Add a learnable per-degree vector to node features."""

    def __init__(self, dim: int, max_degree: int = 32) -> None:
        super().__init__()
        self.max_degree = max_degree
        self.emb = nn.Embedding(max_degree + 1, dim)
        nn.init.zeros_(self.emb.weight)                      # start as a no-op

    def forward(self, h: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        deg = degree.clamp(max=self.max_degree)
        return h + self.emb(deg)


class GraphormerAttention(nn.Module):
    """Multi-head attention with spatial (SPD) and edge structural biases."""

    def __init__(self, dim: int, num_heads: int = 4, max_spd: int = 16,
                 num_edge_types: int = NUM_EDGE_TYPES) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        # per-head scalar biases, zero-initialised -> vanilla attention at init
        self.spatial_bias = nn.Embedding(max_spd + 1, num_heads)
        self.edge_bias = nn.Embedding(num_edge_types + 1, num_heads)
        nn.init.zeros_(self.spatial_bias.weight)
        nn.init.zeros_(self.edge_bias.weight)

    def forward(self, h: torch.Tensor, spd: torch.Tensor,
                etm: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        """``h`` (V, dim); ``spd``/``etm`` (V, V) long. Returns (V, dim)."""
        V = h.shape[0]
        q = self.q(h).view(V, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k(h).view(V, self.num_heads, self.head_dim).transpose(0, 1)
        val = self.v(h).view(V, self.num_heads, self.head_dim).transpose(0, 1)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)    # (H, V, V)
        if use_bias:
            b_sp = self.spatial_bias(spd).permute(2, 0, 1)             # (H, V, V)
            b_ed = self.edge_bias(etm).permute(2, 0, 1)               # (H, V, V)
            logits = logits + b_sp + b_ed
        attn = torch.softmax(logits, dim=-1)
        ctx = (attn @ val).transpose(0, 1).reshape(V, -1)             # (V, dim)
        return self.out(ctx)
