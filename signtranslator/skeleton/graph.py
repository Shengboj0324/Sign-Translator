"""Biomechanical skeleton graph and ST-GCN adjacency construction.

The human upper body + both hands are modelled as an undirected tree of ``V``
keypoints. For spatio-temporal graph convolution we build a *partitioned*
adjacency tensor ``A`` of shape ``(K, V, V)`` following the **spatial
configuration partitioning** of Yan et al., "Spatial Temporal Graph
Convolutional Networks for Skeleton-Based Action Recognition" (AAAI 2018).

Spatial configuration partitioning (K = 3):

    * subset 0 (root)        : the node itself.
    * subset 1 (centripetal) : neighbours strictly closer to the body centre
                               of gravity than the node.
    * subset 2 (centrifugal) : neighbours strictly farther from the centre.

"Closer/farther" is measured by graph hop-distance to the centre joint. Each
per-subset adjacency is normalised so that graph convolution
``sum_k Â_k X W_k`` is numerically stable.

We use **row (random-walk) normalisation** ``Â_k = D_k^{-1} A_k`` with
``D_k[i, i] = sum_j A_k[i, j]``. Unlike symmetric normalisation, this is valid
for the *directed* centripetal/centrifugal partitions (whose adjacency is not
symmetric): each row of ``Â_k`` is sub-stochastic (row sums <= 1), which by the
Gershgorin circle theorem bounds every eigenvalue by ``|lambda| <= 1`` and
therefore prevents activation blow-up across stacked graph-conv layers.
"""

from __future__ import annotations

from collections import deque
from numbers import Integral
from typing import List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Default 27-joint upper-body + two-hand skeleton (a connected tree, 26 edges).
# Indices:
#   0 head, 1 neck (centre), 2 chest,
#   3 r_shoulder, 4 r_elbow, 5 r_wrist,
#   6 l_shoulder, 7 l_elbow, 8 l_wrist,
#   right hand 9..17 rooted at r_wrist(5),
#   left  hand 18..26 rooted at l_wrist(8).
# ---------------------------------------------------------------------------
NUM_DEFAULT_JOINTS = 27
DEFAULT_CENTER = 1  # neck is the body centre of gravity reference

DEFAULT_EDGES: Tuple[Tuple[int, int], ...] = (
    # torso / arms
    (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 8),
    # right hand (root wrist = 5, palm = 9)
    (5, 9), (9, 10), (10, 11), (9, 12), (12, 13), (9, 14), (14, 15),
    (9, 16), (9, 17),
    # left hand (root wrist = 8, palm = 18)
    (8, 18), (18, 19), (19, 20), (18, 21), (21, 22), (18, 23), (23, 24),
    (18, 25), (18, 26),
)


def _bfs_hop_distances(num_nodes: int, edges: Sequence[Tuple[int, int]],
                       source: int) -> np.ndarray:
    """Unweighted shortest-path (hop) distance from ``source`` to every node."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full(num_nodes, -1, dtype=np.int64)
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if (dist < 0).any():
        raise ValueError("skeleton graph is not connected")
    return dist


def _normalize_adjacency(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Row (random-walk) normalisation Â = D^{-1} A for a single subset.

    Valid for directed partition matrices. Rows with zero degree (a node with no
    neighbours in this subset) are left as zero rather than producing NaNs.
    """
    degree = a.sum(axis=1)
    d_inv = np.zeros_like(degree)
    nonzero = degree > eps
    d_inv[nonzero] = 1.0 / degree[nonzero]
    return d_inv[:, None] * a


class SkeletonGraph:
    """Construct and hold the partitioned, normalised skeleton adjacency."""

    def __init__(self,
                 num_nodes: int = NUM_DEFAULT_JOINTS,
                 edges: Sequence[Tuple[int, int]] = DEFAULT_EDGES,
                 center: int = DEFAULT_CENTER,
                 joint_names: Sequence[str] | None = None) -> None:
        if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral) or num_nodes < 1:
            raise ValueError("num_nodes must be a positive integer")
        if isinstance(center, bool) or not isinstance(center, Integral):
            raise ValueError("center must be an integer")
        self.num_nodes = int(num_nodes)
        self.edges = tuple(tuple(edge) for edge in edges)
        self.center = int(center)
        self.joint_names = tuple(joint_names) if joint_names is not None else tuple(
            f"joint_{index}" for index in range(self.num_nodes))
        self._validate()
        self.hop = _bfs_hop_distances(num_nodes, self.edges, center)
        self.A = self._build_partitioned_adjacency()  # (K, V, V) float32

    # -- construction -------------------------------------------------------
    def _validate(self) -> None:
        if (len(self.joint_names) != self.num_nodes
                or any(not isinstance(name, str) or not name.strip() for name in self.joint_names)
                or len(set(self.joint_names)) != self.num_nodes):
            raise ValueError("joint_names must give one unique non-empty name per node")
        seen = set()
        for edge in self.edges:
            if len(edge) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) for i in edge):
                raise ValueError("edges must be pairs of integer node indices")
            i, j = edge
            key = tuple(sorted((i,j)))
            if key in seen:
                raise ValueError("duplicate undirected skeleton edge")
            seen.add(key)
            if not (0 <= i < self.num_nodes and 0 <= j < self.num_nodes):
                raise ValueError(f"edge ({i},{j}) out of range")
            if i == j:
                raise ValueError("self-loops are added implicitly; remove them")
        if not (0 <= self.center < self.num_nodes):
            raise ValueError("center index out of range")

    def _build_partitioned_adjacency(self) -> np.ndarray:
        v = self.num_nodes
        # Binary neighbour matrix (symmetric, no self-loops yet).
        neigh = np.zeros((v, v), dtype=np.float64)
        for i, j in self.edges:
            neigh[i, j] = 1.0
            neigh[j, i] = 1.0

        a_root = np.eye(v, dtype=np.float64)            # subset 0: self connection
        a_centripetal = np.zeros((v, v), dtype=np.float64)
        a_centrifugal = np.zeros((v, v), dtype=np.float64)

        for i in range(v):
            for j in range(v):
                if neigh[i, j] == 0:
                    continue
                # Assign neighbour j (feeding node i) by its distance to centre
                # relative to node i's distance to centre.
                if self.hop[j] < self.hop[i]:
                    a_centripetal[i, j] = 1.0
                elif self.hop[j] > self.hop[i]:
                    a_centrifugal[i, j] = 1.0
                else:
                    # equal hop distance -> treat as root-level (rare for a tree)
                    a_root[i, j] = 1.0

        partitions = [
            _normalize_adjacency(a_root),
            _normalize_adjacency(a_centripetal),
            _normalize_adjacency(a_centrifugal),
        ]
        return np.stack(partitions, axis=0).astype(np.float32)

    def to_dict(self) -> dict:
        """Bind topology AND column semantics, independent of tensor dimensions."""
        return {"schema_version": 1, "num_nodes": self.num_nodes,
                "edges": [[int(i), int(j)] for i,j in self.edges],
                "center": self.center, "joint_names": list(self.joint_names)}

    @classmethod
    def from_dict(cls, value: dict) -> "SkeletonGraph":
        fields = {"schema_version", "num_nodes", "edges", "center", "joint_names"}
        if (not isinstance(value, dict) or set(value) != fields
                or type(value["schema_version"]) is not int or value["schema_version"] != 1
                or not isinstance(value["edges"], list)
                or not isinstance(value["joint_names"], list)):
            raise ValueError("invalid skeleton schema")
        return cls(num_nodes=value["num_nodes"], edges=value["edges"],
                   center=value["center"], joint_names=value["joint_names"])

    # -- accessors ----------------------------------------------------------
    @property
    def num_partitions(self) -> int:
        return self.A.shape[0]

    def adjacency(self) -> np.ndarray:
        """Return a copy of the (K, V, V) normalised adjacency tensor."""
        return self.A.copy()

    def __repr__(self) -> str:
        return (f"SkeletonGraph(V={self.num_nodes}, E={len(self.edges)}, "
                f"K={self.num_partitions}, center={self.center})")
