"""Verification of the heterogeneous temporal graph structure.

The graph is pure structure; we check the MediaPipe hand topology, symmetry
correspondence, temporal unrolling, k-NN distance edges, BFS shortest-path
distances (needed for Graphormer), and rule-by-rule validation.
"""

import pytest
import torch

from signtranslator.hand_graph.hetero_graph import (
    NodeType, EdgeType, NUM_EDGE_TYPES, HAND_LANDMARKS, HAND_BONES,
    HandGraph, HandGraphBuilder, build_two_hand_graph, temporal_unroll,
    knn_distance_edges, shortest_path_distances, validate_hand_graph,
)


# ---------------------------------------------------------------------------
# hand topology
# ---------------------------------------------------------------------------
def test_hand_bones_form_a_spanning_tree_of_21_landmarks():
    # a tree over 21 nodes has exactly 20 edges and is connected & acyclic
    assert len(HAND_BONES) == 20
    nodes = set(range(HAND_LANDMARKS))
    seen = set()
    for a, b in HAND_BONES:
        assert a in nodes and b in nodes
        seen |= {a, b}
    assert seen == nodes                                     # every landmark covered
    # connected + acyclic (union-find)
    parent = list(range(HAND_LANDMARKS))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in HAND_BONES:
        ra, rb = find(a), find(b)
        assert ra != rb, "cycle in hand bones"
        parent[ra] = rb
    assert len({find(i) for i in range(HAND_LANDMARKS)}) == 1  # connected


def test_two_hand_graph_has_expected_nodes_and_edge_types():
    g, idx = build_two_hand_graph()
    assert g.num_nodes == 2 * HAND_LANDMARKS
    assert len(idx["left"]) == HAND_LANDMARKS and len(idx["right"]) == HAND_LANDMARKS
    # bones: 20 per hand x 2 hands x 2 directions = 80
    assert g.edges_of_type(EdgeType.BONE).shape[1] == 20 * 2 * 2
    # symmetry: 21 landmark pairs x 2 directions = 42
    assert g.edges_of_type(EdgeType.SYMMETRY).shape[1] == HAND_LANDMARKS * 2
    assert validate_hand_graph(g) == []


def test_symmetry_edges_are_strictly_cross_hand():
    g, idx = build_two_hand_graph()
    sym = g.edges_of_type(EdgeType.SYMMETRY)
    for e in range(sym.shape[1]):
        s, d = int(sym[0, e]), int(sym[1, e])
        types = {int(g.node_type[s]), int(g.node_type[d])}
        assert types == {int(NodeType.LEFT_HAND), int(NodeType.RIGHT_HAND)}


# ---------------------------------------------------------------------------
# temporal unrolling
# ---------------------------------------------------------------------------
def test_temporal_unroll_replicates_and_links_frames():
    g0, _ = build_two_hand_graph()
    T = 4
    g = temporal_unroll(g0, T)
    assert g.num_nodes == T * g0.num_nodes
    # frames are 0..T-1, each with V0 nodes
    for t in range(T):
        assert int((g.node_frame == t).sum()) == g0.num_nodes
    # temporal edges: V0 nodes x (T-1) gaps x 2 directions
    assert g.edges_of_type(EdgeType.TEMPORAL).shape[1] == g0.num_nodes * (T - 1) * 2
    assert validate_hand_graph(g) == []


def test_temporal_edges_only_link_consecutive_same_node():
    g0, _ = build_two_hand_graph()
    g = temporal_unroll(g0, 3)
    tmp = g.edges_of_type(EdgeType.TEMPORAL)
    for e in range(tmp.shape[1]):
        s, d = int(tmp[0, e]), int(tmp[1, e])
        assert abs(int(g.node_frame[s]) - int(g.node_frame[d])) == 1
        assert g.node_type[s] == g.node_type[d]


# ---------------------------------------------------------------------------
# k-NN distance edges
# ---------------------------------------------------------------------------
def test_knn_distance_edges_pick_nearest():
    # 4 collinear points; each node's nearest is its immediate neighbour
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    ei = knn_distance_edges(pos, k=1)
    # node 0 -> 1, node 1 -> 0 (tie 0/2 both dist 1 -> topk picks one), 3 -> 2
    nbr = {int(ei[0, e]): int(ei[1, e]) for e in range(ei.shape[1])}
    assert nbr[0] == 1
    assert nbr[3] == 2
    assert ei.shape[1] == 4                                   # one per node


def test_knn_distance_respects_frame_restriction():
    pos = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0, 0], [0.1, 0, 0]])
    frame = torch.tensor([0, 0, 1, 1])
    ei = knn_distance_edges(pos, k=3, within_frame=frame)
    for e in range(ei.shape[1]):
        assert frame[int(ei[0, e])] == frame[int(ei[1, e])]  # never cross frames


# ---------------------------------------------------------------------------
# shortest-path distances (Graphormer input)
# ---------------------------------------------------------------------------
def test_spd_matches_hand_chain_by_hand():
    g, idx = build_two_hand_graph()
    spd = shortest_path_distances(g.edges_of_type(EdgeType.BONE), g.num_nodes)
    L = idx["left"]
    assert spd[L[0], L[0]] == 0
    assert spd[L[0], L[4]] == 4                              # wrist->thumb tip: 4 bones
    assert spd[L[0], L[8]] == 4                              # wrist->index tip: 4 bones
    # the two hands are disconnected under BONE edges only -> unreachable sentinel
    assert spd[L[0], idx["right"][0]] == g.num_nodes


def test_spd_symmetric_edges_connect_hands():
    g, idx = build_two_hand_graph()
    spd = shortest_path_distances(g.edge_index, g.num_nodes)  # all relations
    # with symmetry edges, left wrist reaches right wrist in 1 hop
    assert spd[idx["left"][0], idx["right"][0]] == 1


def test_spd_max_distance_caps_and_flags_unreachable():
    g, idx = build_two_hand_graph()
    spd = shortest_path_distances(g.edges_of_type(EdgeType.BONE), g.num_nodes,
                                  max_distance=3)
    assert spd.max() <= 4                                     # capped at max_distance+1
    assert spd[idx["left"][0], idx["left"][4]] == 4          # dist 4 -> "far" bucket


# ---------------------------------------------------------------------------
# validation catches malformed graphs
# ---------------------------------------------------------------------------
def test_validation_flags_bad_edges():
    b = HandGraphBuilder()
    l = b.add_nodes(NodeType.LEFT_HAND, 2)
    r = b.add_nodes(NodeType.RIGHT_HAND, 1)
    b.add_edge(l[0], r[0], EdgeType.BONE)                    # bone crossing parts
    b.add_edge(l[0], l[1], EdgeType.SYMMETRY)                # symmetry within one hand
    g = b.build()
    viol = validate_hand_graph(g)
    assert "bone_crosses_parts" in viol
    assert "symmetry_not_cross_hand" in viol


def test_empty_graph_is_valid():
    g = HandGraphBuilder().build()
    assert g.num_nodes == 0 and g.num_edges == 0
    assert validate_hand_graph(g) == []
