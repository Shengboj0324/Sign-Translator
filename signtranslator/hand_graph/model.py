"""The hand-motion graph reasoner and its ablation baselines.

Combines the pieces of docs/HAND_GRAPH.md into one model:

    features = pos_proj(wrist_relative(x)) + type_emb(τ)  + centrality
    per layer:  h += RelationalGraphAttention(h)     (local kinematics)
                h += GraphormerAttention(h, SPD, edge) (global structural)

Because node features are built from **wrist-relative** coordinates (§3), the
whole model is **translation-invariant** by construction -- a property we prove,
not assume.

The three baselines the document requires ("compare against adaptive GCN, vanilla
Transformer, and kinematic-only") are obtained by flags on the same class:

* ``use_local=True,  use_global=False``  -> kinematic-only (graph conv only);
* ``use_local=False, use_global=True, use_structural_bias=False`` -> vanilla
  Transformer (attention, no graph structure);
* ``use_local=True,  use_global=True``   -> the full graph reasoner.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hetero_graph import HandGraph, NodeType, NUM_EDGE_TYPES
from .relational import RelationalGraphAttention
from .structural import GraphormerAttention, CentralityEncoding, dense_bias_inputs
from .geometry import wrist_relative


class HandGraphReasoner(nn.Module):
    def __init__(self, dim: int = 64, num_node_types: int = len(NodeType),
                 num_relations: int = NUM_EDGE_TYPES, num_heads: int = 4,
                 max_spd: int = 16, num_layers: int = 2, num_bases: Optional[int] = None,
                 use_local: bool = True, use_global: bool = True,
                 use_structural_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.max_spd = max_spd
        self.use_local = use_local
        self.use_global = use_global
        self.use_structural_bias = use_structural_bias

        self.pos_proj = nn.Linear(3, dim)
        self.type_emb = nn.Embedding(num_node_types, dim)
        self.centrality = CentralityEncoding(dim, max_degree=32)

        self.local_layers = nn.ModuleList(
            [RelationalGraphAttention(dim, dim, num_relations, num_bases)
             for _ in range(num_layers)] if use_local else [])
        self.global_layers = nn.ModuleList(
            [GraphormerAttention(dim, num_heads, max_spd, num_relations)
             for _ in range(num_layers)] if use_global else [])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, positions: torch.Tensor, node_type: torch.Tensor,
                wrist_of: torch.Tensor, graph: HandGraph,
                node_confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``positions`` (V,3), ``node_type`` (V,), ``wrist_of`` (V,). -> (V, dim)."""
        rel = wrist_relative(positions, wrist_of)            # translation-invariant
        h = self.pos_proj(rel) + self.type_emb(node_type)
        h = self.centrality(h, graph.in_degree().to(positions.device))

        spd = etm = None
        if self.use_global:
            spd, etm = dense_bias_inputs(graph, self.max_spd)
            spd, etm = spd.to(positions.device), etm.to(positions.device)

        for i in range(self.num_layers):
            if self.use_local:
                h = h + F.gelu(self.local_layers[i](
                    h, graph.edge_index, graph.edge_type, node_confidence))
            if self.use_global:
                h = h + self.global_layers[i](
                    h, spd, etm, use_bias=self.use_structural_bias)
            h = self.norms[i](h)
        return h

    @staticmethod
    def pool_by_type(h: torch.Tensor, node_type: torch.Tensor,
                     num_node_types: int = len(NodeType)) -> torch.Tensor:
        """Mean-pool node features within each node type -> (num_node_types, dim).

        Types with no nodes get a zero vector.
        """
        dim = h.shape[-1]
        out = torch.zeros(num_node_types, dim, dtype=h.dtype, device=h.device)
        count = torch.zeros(num_node_types, dtype=h.dtype, device=h.device)
        out.index_add_(0, node_type, h)
        count.index_add_(0, node_type, torch.ones_like(node_type, dtype=h.dtype))
        return out / count.clamp_min(1.0).unsqueeze(-1)


def hand_embeddings(h: torch.Tensor, node_type: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-hand pooled embeddings for the bridge back to the skeleton/SIR layers."""
    pooled = HandGraphReasoner.pool_by_type(h, node_type)
    return {
        "left_hand": pooled[int(NodeType.LEFT_HAND)],
        "right_hand": pooled[int(NodeType.RIGHT_HAND)],
        "body": pooled[int(NodeType.BODY)],
        "face": pooled[int(NodeType.FACE)],
    }
