"""Integration + whole-chain cycle stress for the hand-graph reasoner.

The decisive integration property: because node features are built from
wrist-relative coordinates, the ENTIRE model is translation-invariant (proved to
1e-10). Also checks the three ablation baselines, confidence propagation, the
per-hand bridge, and a 100-case determinism/finiteness stress loop.
"""

import pytest
import torch

from signtranslator.hand_graph.hetero_graph import (
    NodeType, HAND_LANDMARKS, build_two_hand_graph, validate_hand_graph,
)
from signtranslator.hand_graph.model import HandGraphReasoner, hand_embeddings


def _setup(seed=0):
    graph, idx = build_two_hand_graph()
    V = graph.num_nodes
    wrist_of = torch.zeros(V, dtype=torch.long)
    for i in idx["left"]:
        wrist_of[i] = idx["left"][0]
    for i in idx["right"]:
        wrist_of[i] = idx["right"][0]
    g = torch.Generator().manual_seed(seed)
    positions = torch.randn(V, 3, generator=g, dtype=torch.float64)
    return graph, idx, wrist_of, positions


def _model(**kw):
    torch.manual_seed(0)
    return HandGraphReasoner(dim=32, num_heads=4, max_spd=8, num_layers=2, **kw).double()


# ---------------------------------------------------------------------------
# runs + shape
# ---------------------------------------------------------------------------
def test_reasoner_runs_and_shapes():
    graph, idx, wrist_of, positions = _setup(1)
    model = _model()
    h = model(positions, graph.node_type, wrist_of, graph)
    assert h.shape == (graph.num_nodes, 32) and torch.isfinite(h).all()
    assert validate_hand_graph(graph) == []


# ---------------------------------------------------------------------------
# THE integration property: end-to-end translation invariance
# ---------------------------------------------------------------------------
def test_model_is_translation_invariant():
    graph, idx, wrist_of, positions = _setup(2)
    model = _model()
    h1 = model(positions, graph.node_type, wrist_of, graph)
    t = torch.tensor([7.0, -4.0, 3.0], dtype=torch.float64)
    h2 = model(positions + t, graph.node_type, wrist_of, graph)
    assert torch.allclose(h1, h2, atol=1e-10)


# ---------------------------------------------------------------------------
# the three required baselines (via flags)
# ---------------------------------------------------------------------------
def test_three_baseline_configurations_run_and_differ():
    graph, idx, wrist_of, positions = _setup(3)
    full = _model(use_local=True, use_global=True, use_structural_bias=True)
    kinematic = _model(use_local=True, use_global=False)
    vanilla = _model(use_local=False, use_global=True, use_structural_bias=False)

    hf = full(positions, graph.node_type, wrist_of, graph)
    hk = kinematic(positions, graph.node_type, wrist_of, graph)
    hv = vanilla(positions, graph.node_type, wrist_of, graph)
    for h in (hf, hk, hv):
        assert h.shape == (graph.num_nodes, 32) and torch.isfinite(h).all()
    # the configurations are genuinely different computations
    assert not torch.allclose(hf, hk, atol=1e-4)
    assert not torch.allclose(hf, hv, atol=1e-4)
    # kinematic-only must not use the global attention layers
    assert len(kinematic.global_layers) == 0
    # vanilla must not use the local relational layers
    assert len(vanilla.local_layers) == 0


def test_vanilla_transformer_ignores_structural_bias():
    """With structural bias off, changing the graph's edges must not change the
    vanilla-transformer output (it only attends over node features)."""
    graph, idx, wrist_of, positions = _setup(4)
    vanilla = _model(use_local=False, use_global=True, use_structural_bias=False)
    h1 = vanilla(positions, graph.node_type, wrist_of, graph)
    # a different edge set (drop all but one edge) -> SPD/edge bias differ but are unused
    from signtranslator.hand_graph.hetero_graph import HandGraph
    stripped = HandGraph(graph.node_type, graph.node_frame,
                         graph.edge_index[:, :1], graph.edge_type[:1])
    h2 = vanilla(positions, graph.node_type, wrist_of, stripped)
    assert torch.allclose(h1, h2, atol=1e-10)


# ---------------------------------------------------------------------------
# confidence propagation
# ---------------------------------------------------------------------------
def test_confidence_masking_changes_local_output():
    graph, idx, wrist_of, positions = _setup(5)
    model = _model(use_local=True, use_global=False)
    full_conf = torch.ones(graph.num_nodes, dtype=torch.float64)
    h_full = model(positions, graph.node_type, wrist_of, graph, full_conf)
    occl = full_conf.clone(); occl[idx["left"][4]] = 0.0     # occlude a fingertip
    h_occl = model(positions, graph.node_type, wrist_of, graph, occl)
    assert not torch.allclose(h_full, h_occl, atol=1e-6)
    assert torch.isfinite(h_occl).all()


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------
def test_hand_embeddings_bridge():
    graph, idx, wrist_of, positions = _setup(6)
    model = _model()
    h = model(positions, graph.node_type, wrist_of, graph)
    emb = hand_embeddings(h, graph.node_type)
    assert set(emb) == {"left_hand", "right_hand", "body", "face"}
    assert emb["left_hand"].shape == (32,)
    # left_hand pooled = mean of left-hand node features
    left_mean = h[idx["left"]].mean(0)
    assert torch.allclose(emb["left_hand"], left_mean, atol=1e-10)


# ---------------------------------------------------------------------------
# whole-chain cycle stress
# ---------------------------------------------------------------------------
def test_cycle_stress_determinism_and_finiteness():
    graph, idx, wrist_of, _ = _setup(7)
    model = _model()
    model.eval()
    for s in range(100):
        g = torch.Generator().manual_seed(1000 + s)
        pos = torch.randn(graph.num_nodes, 3, generator=g, dtype=torch.float64)
        conf = (torch.rand(graph.num_nodes, generator=g, dtype=torch.float64) > 0.1).double()
        with torch.no_grad():
            h1 = model(pos, graph.node_type, wrist_of, graph, conf)
            h2 = model(pos, graph.node_type, wrist_of, graph, conf)
        assert torch.equal(h1, h2)                            # deterministic
        assert torch.isfinite(h1).all()


def test_gradients_flow_through_full_model():
    graph, idx, wrist_of, positions = _setup(8)
    model = _model()
    positions.requires_grad_(True)
    out = model(positions, graph.node_type, wrist_of, graph)
    out.pow(2).sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    assert n_grad > 0
