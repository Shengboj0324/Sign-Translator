"""Relational message passing: R-GCN with basis decomposition + GAT attention.

Implements docs/HAND_GRAPH.md §2:

    h_i' = W_0 h_i + Σ_r Σ_{j∈N_r(i)} α^r_ij W_r h_j,
    α^r_ij = softmax_j( LeakyReLU( a_rᵀ [W_r h_i ‖ W_r h_j] ) ),
    W_r    = Σ_{b=1}^{B} a_{rb} V_b   (basis decomposition, Schlichtkrull et al.).

Edges are directed ``(src -> dst)``; a message flows from ``src`` to ``dst``, so
``N_r(i)`` are the sources of relation-``r`` edges whose destination is ``i``. The
attention softmax is computed **per (destination, relation)** group.

Confidence-aware masking (docs §7) is built in: an optional per-node confidence
reweights each neighbour by ``c_src`` and renormalises, so a fully-occluded
neighbour contributes nothing and an all-occluded neighbourhood falls back to the
self term ``W_0 h_i`` with no NaN.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hetero_graph import NUM_EDGE_TYPES

_EPS = 1e-16


def group_softmax(score: torch.Tensor, group: torch.Tensor,
                  num_groups: int) -> torch.Tensor:
    """Softmax of ``score`` within each ``group`` id. Numerically stable.

    ``score`` (E,), ``group`` (E,) in [0, num_groups). Groups with no members are
    irrelevant (never indexed). Returns (E,) weights summing to 1 within a group.
    """
    if score.numel() == 0:
        return score
    gmax = score.new_full((num_groups,), float("-inf"))
    gmax = gmax.scatter_reduce(0, group, score, reduce="amax", include_self=True)
    shifted = score - gmax[group]
    e = shifted.exp()
    gsum = torch.zeros(num_groups, dtype=score.dtype, device=score.device)
    gsum = gsum.index_add(0, group, e)
    return e / gsum[group].clamp_min(_EPS)


class RelationalGraphAttention(nn.Module):
    """One relational graph-attention layer over a typed edge set."""

    def __init__(self, in_dim: int, out_dim: int,
                 num_relations: int = NUM_EDGE_TYPES,
                 num_bases: Optional[int] = None,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases is not None else num_relations
        self.negative_slope = negative_slope

        self.self_transform = nn.Linear(in_dim, out_dim, bias=False)   # W_0
        self.bases = nn.Parameter(torch.empty(self.num_bases, out_dim, in_dim))  # V_b
        self.coeff = nn.Parameter(torch.empty(num_relations, self.num_bases))    # a_rb
        self.att = nn.Parameter(torch.empty(num_relations, 2 * out_dim))         # a_r
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coeff)
        nn.init.xavier_uniform_(self.att)

    def relation_weight(self, r: int) -> torch.Tensor:
        """W_r = Σ_b a_{rb} V_b  (out_dim, in_dim)."""
        return torch.einsum("b,boi->oi", self.coeff[r], self.bases)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                node_confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(V, in_dim) -> (V, out_dim)."""
        V = h.shape[0]
        R = self.num_relations
        out = self.self_transform(h)                          # W_0 h  (V, out)
        if edge_index.numel() == 0:
            return out

        # per-relation transformed node features: h_Wr[r, v] = W_r h_v
        Wr = torch.einsum("rb,boi->roi", self.coeff, self.bases)   # (R, out, in)
        h_Wr = torch.einsum("roi,vi->rvo", Wr, h)                  # (R, V, out)

        s, d = edge_index[0], edge_index[1]
        r = edge_type
        msg = h_Wr[r, s]                                      # (E, out) = W_r h_src
        tgt = h_Wr[r, d]                                      # (E, out) = W_r h_dst
        cat = torch.cat((tgt, msg), dim=-1)                  # (E, 2 out)
        raw = (cat * self.att[r]).sum(-1)                    # a_rᵀ [W_r h_i ‖ W_r h_j]
        score = F.leaky_relu(raw, self.negative_slope)

        group = d * R + r                                    # (E,) per (dst, relation)
        alpha = group_softmax(score, group, V * R)

        if node_confidence is not None:
            # reweight each neighbour by its confidence and renormalise per group
            w = alpha * node_confidence[s]
            wsum = torch.zeros(V * R, dtype=w.dtype, device=w.device)
            wsum = wsum.index_add(0, group, w)
            alpha = w / wsum[group].clamp_min(_EPS)          # all-occluded -> 0

        weighted = alpha.unsqueeze(-1) * msg                 # (E, out)
        out = out.index_add(0, d, weighted)
        return out


class RelationalGraphNetwork(nn.Module):
    """A stack of relational graph-attention layers with residual + norm."""

    def __init__(self, dim: int, num_layers: int = 2,
                 num_relations: int = NUM_EDGE_TYPES,
                 num_bases: Optional[int] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RelationalGraphAttention(dim, dim, num_relations, num_bases)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

    def forward(self, h, edge_index, edge_type, node_confidence=None):
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.gelu(layer(h, edge_index, edge_type, node_confidence)))
        return h
