"""Heterogeneous temporal graph for hand-motion reasoning.

See docs/HAND_GRAPH.md §1. Nodes are typed
``{BODY, LEFT_HAND, RIGHT_HAND, FACE, LOCUS}``; edges are typed
``{BONE, TEMPORAL, SYMMETRY, CONTACT, DISTANCE, SEMANTIC}``. The graph is pure
*structure* (node types, frame indices, typed edges); node features and 3D
positions are supplied separately to the message-passing / geometry layers, so a
single structural graph can be reused across a clip and across models.

Hands use the **MediaPipe Hands** 21-landmark topology (arXiv:2006.10214), which
is also the MANO keypoint convention:

    0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky.

Structural (undirected) relations BONE / SYMMETRY / DISTANCE are stored in BOTH
directions so neighbourhood aggregation is symmetric; TEMPORAL is stored forward
and backward too. Directedness is preserved in the representation (each edge is an
ordered ``(src, dst)``), matching the document's directed message passing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class NodeType(IntEnum):
    BODY = 0
    LEFT_HAND = 1
    RIGHT_HAND = 2
    FACE = 3
    LOCUS = 4


class EdgeType(IntEnum):
    BONE = 0
    TEMPORAL = 1
    SYMMETRY = 2
    CONTACT = 3
    DISTANCE = 4
    SEMANTIC = 5


NUM_EDGE_TYPES = len(EdgeType)

# --- MediaPipe / MANO hand topology (21 landmarks) -------------------------
HAND_LANDMARKS = 21
HAND_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
)
WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)
MIDDLE_MCP = 9                                # for hand-scale normalisation
FINGER_MCPS = (1, 5, 9, 13, 17)              # thumb CMC + index/middle/ring/pinky MCP


@dataclass
class HandGraph:
    """Structure only: node types/frames and typed directed edges.

    ``node_type`` (V,), ``node_frame`` (V,), ``edge_index`` (2, E),
    ``edge_type`` (E,). All long tensors.
    """

    node_type: torch.Tensor
    node_frame: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor

    def __post_init__(self) -> None:
        V = self.node_type.shape[0]
        if self.node_frame.shape != (V,):
            raise ValueError("node_frame must be (V,)")
        if self.edge_index.dim() != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be (2, E)")
        if self.edge_type.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_type must be (E,)")

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def edges_of_type(self, r: EdgeType) -> torch.Tensor:
        """(2, E_r) edges of relation ``r``."""
        mask = self.edge_type == int(r)
        return self.edge_index[:, mask]

    def neighbors(self, i: int, r: EdgeType) -> List[int]:
        """Target nodes j such that (i -> j) is an edge of relation r."""
        ei = self.edges_of_type(r)
        return ei[1, ei[0] == i].tolist()

    def in_degree(self) -> torch.Tensor:
        """(V,) number of incoming edges per node (over all relations)."""
        deg = torch.zeros(self.num_nodes, dtype=torch.long)
        deg.index_add_(0, self.edge_index[1], torch.ones(self.num_edges, dtype=torch.long))
        return deg


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------
class HandGraphBuilder:
    """Accumulate typed nodes and directed edges, then ``build()`` to tensors."""

    def __init__(self) -> None:
        self._types: List[int] = []
        self._frames: List[int] = []
        self._src: List[int] = []
        self._dst: List[int] = []
        self._etype: List[int] = []

    def add_node(self, node_type: NodeType, frame: int = 0) -> int:
        self._types.append(int(node_type))
        self._frames.append(int(frame))
        return len(self._types) - 1

    def add_nodes(self, node_type: NodeType, count: int, frame: int = 0) -> List[int]:
        return [self.add_node(node_type, frame) for _ in range(count)]

    def add_edge(self, src: int, dst: int, etype: EdgeType,
                 bidirectional: bool = False) -> None:
        self._src.append(int(src)); self._dst.append(int(dst))
        self._etype.append(int(etype))
        if bidirectional:
            self._src.append(int(dst)); self._dst.append(int(src))
            self._etype.append(int(etype))

    def build(self) -> HandGraph:
        if not self._src:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        else:
            edge_index = torch.tensor([self._src, self._dst], dtype=torch.long)
            edge_type = torch.tensor(self._etype, dtype=torch.long)
        return HandGraph(
            node_type=torch.tensor(self._types, dtype=torch.long),
            node_frame=torch.tensor(self._frames, dtype=torch.long),
            edge_index=edge_index, edge_type=edge_type,
        )


def build_two_hand_graph(frame: int = 0) -> Tuple[HandGraph, Dict[str, List[int]]]:
    """A single-frame graph of two 21-landmark hands with BONE + SYMMETRY edges.

    Returns the graph and an index map ``{"left": [...], "right": [...]}``.
    """
    b = HandGraphBuilder()
    left = b.add_nodes(NodeType.LEFT_HAND, HAND_LANDMARKS, frame)
    right = b.add_nodes(NodeType.RIGHT_HAND, HAND_LANDMARKS, frame)
    for a, c in HAND_BONES:                                   # bones within each hand
        b.add_edge(left[a], left[c], EdgeType.BONE, bidirectional=True)
        b.add_edge(right[a], right[c], EdgeType.BONE, bidirectional=True)
    for k in range(HAND_LANDMARKS):                           # mirror correspondence
        b.add_edge(left[k], right[k], EdgeType.SYMMETRY, bidirectional=True)
    return b.build(), {"left": left, "right": right}


def temporal_unroll(single: HandGraph, num_frames: int) -> HandGraph:
    """Replicate a single-frame graph over ``num_frames`` and add TEMPORAL edges.

    Node ``v`` in frame ``t`` gets global id ``t * V0 + v``. Intra-frame edges are
    copied per frame; TEMPORAL edges connect each node to itself in adjacent
    frames (forward and backward).
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    V0 = single.num_nodes
    b = HandGraphBuilder()
    for t in range(num_frames):
        for v in range(V0):
            b.add_node(NodeType(int(single.node_type[v])), frame=t)
    # copy intra-frame edges for every frame
    src0, dst0 = single.edge_index[0].tolist(), single.edge_index[1].tolist()
    et0 = single.edge_type.tolist()
    for t in range(num_frames):
        off = t * V0
        for s, d, e in zip(src0, dst0, et0):
            b.add_edge(off + s, off + d, EdgeType(e))
    # temporal identity edges between consecutive frames
    for t in range(num_frames - 1):
        for v in range(V0):
            b.add_edge(t * V0 + v, (t + 1) * V0 + v, EdgeType.TEMPORAL,
                       bidirectional=True)
    return b.build()


def knn_distance_edges(positions: torch.Tensor, k: int,
                       within_frame: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(2, E) DISTANCE edges to each node's ``k`` nearest neighbours in 3D.

    ``positions`` (V, 3). If ``within_frame`` (V,) is given, neighbours are
    restricted to the same frame (no cross-time distance edges). Self is excluded.
    """
    V = positions.shape[0]
    d = torch.cdist(positions, positions)                    # (V, V)
    eye = torch.eye(V, dtype=torch.bool, device=positions.device)
    d = d.masked_fill(eye, float("inf"))
    if within_frame is not None:
        diff_frame = within_frame[:, None] != within_frame[None, :]
        d = d.masked_fill(diff_frame, float("inf"))
    kk = min(k, V - 1)
    idx = torch.topk(d, kk, dim=1, largest=False).indices     # (V, kk)
    src = torch.arange(V, device=positions.device)[:, None].expand(-1, kk).reshape(-1)
    dst = idx.reshape(-1)
    # drop any edges that were masked to inf (e.g. lone node in its frame)
    valid = torch.isfinite(d[src, dst])
    return torch.stack((src[valid], dst[valid]), dim=0)


# ---------------------------------------------------------------------------
# graph analysis (shortest-path distance for Graphormer, §5)
# ---------------------------------------------------------------------------
def shortest_path_distances(edge_index: torch.Tensor, num_nodes: int,
                            max_distance: Optional[int] = None) -> torch.Tensor:
    """(V, V) unweighted shortest-path distances by BFS (over the edge set).

    Unreachable pairs get ``max_distance + 1`` (a dedicated "far" index) if
    ``max_distance`` is set, else the sentinel ``num_nodes`` (> any real path).
    Self-distance is 0.
    """
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    unreachable = (max_distance + 1) if max_distance is not None else num_nodes
    spd = torch.full((num_nodes, num_nodes), unreachable, dtype=torch.long)
    for start in range(num_nodes):
        spd[start, start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            du = int(spd[start, u])
            if max_distance is not None and du >= max_distance:
                continue
            for w in adj[u]:
                if spd[start, w] == unreachable:
                    spd[start, w] = du + 1
                    q.append(w)
    if max_distance is not None:
        spd = spd.clamp(max=max_distance + 1)
    return spd


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------
def validate_hand_graph(g: HandGraph) -> List[str]:
    """Return the list of violated structural rules (empty == valid)."""
    v: List[str] = []
    V = g.num_nodes
    n_types = len(NodeType)

    if g.num_nodes > 0 and (g.node_type.min() < 0 or g.node_type.max() >= n_types):
        v.append("node_type_out_of_range")
    if g.num_edges > 0:
        if g.edge_index.min() < 0 or g.edge_index.max() >= V:
            v.append("edge_endpoint_out_of_range")
        if g.edge_type.min() < 0 or g.edge_type.max() >= NUM_EDGE_TYPES:
            v.append("edge_type_out_of_range")

    ntype = g.node_type
    nframe = g.node_frame
    for e in range(g.num_edges):
        s, d = int(g.edge_index[0, e]), int(g.edge_index[1, e])
        if not (0 <= s < V and 0 <= d < V):
            continue                                          # already flagged
        et = int(g.edge_type[e])
        if et == EdgeType.BONE and ntype[s] != ntype[d]:
            v.append("bone_crosses_parts")
        if et == EdgeType.SYMMETRY:
            pair = {int(ntype[s]), int(ntype[d])}
            if pair != {int(NodeType.LEFT_HAND), int(NodeType.RIGHT_HAND)}:
                v.append("symmetry_not_cross_hand")
        if et == EdgeType.TEMPORAL:
            if abs(int(nframe[s]) - int(nframe[d])) != 1:
                v.append("temporal_not_consecutive_frames")
            if ntype[s] != ntype[d]:
                v.append("temporal_changes_node_type")
    return sorted(set(v))
