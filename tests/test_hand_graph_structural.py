"""Verification of Graphormer structural encodings.

The decisive property (Graphormer as a strict superset of vanilla attention):
with the structural biases zeroed, the module reduces EXACTLY to standard
multi-head scaled dot-product attention. Also: centrality starts as a no-op,
biases actually change the distribution, and gradients flow.
"""

import math

import pytest
import torch

from signtranslator.hand_graph.hetero_graph import build_two_hand_graph
from signtranslator.hand_graph.structural import (
    edge_type_matrix, dense_bias_inputs, CentralityEncoding, GraphormerAttention,
)


def _vanilla_mha(mod, h):
    """Reference standard multi-head attention using mod's own Q/K/V/out."""
    V = h.shape[0]
    q = mod.q(h).view(V, mod.num_heads, mod.head_dim).transpose(0, 1)
    k = mod.k(h).view(V, mod.num_heads, mod.head_dim).transpose(0, 1)
    val = mod.v(h).view(V, mod.num_heads, mod.head_dim).transpose(0, 1)
    logits = q @ k.transpose(-2, -1) / math.sqrt(mod.head_dim)
    attn = torch.softmax(logits, dim=-1)
    ctx = (attn @ val).transpose(0, 1).reshape(V, -1)
    return mod.out(ctx)


def test_reduces_to_vanilla_attention_when_bias_disabled():
    torch.manual_seed(0)
    mod = GraphormerAttention(dim=16, num_heads=4, max_spd=8).double()
    h = torch.randn(10, 16, dtype=torch.float64)
    spd = torch.zeros(10, 10, dtype=torch.long)
    etm = torch.zeros(10, 10, dtype=torch.long)
    out = mod(h, spd, etm, use_bias=False)
    assert torch.allclose(out, _vanilla_mha(mod, h), atol=1e-12)


def test_zero_initialised_biases_are_vanilla_at_init():
    torch.manual_seed(1)
    mod = GraphormerAttention(dim=16, num_heads=4, max_spd=8).double()
    h = torch.randn(10, 16, dtype=torch.float64)
    spd = torch.randint(0, 9, (10, 10))
    etm = torch.randint(0, 7, (10, 10))
    # biases are zero-init -> even WITH bias enabled, equals vanilla at init
    assert torch.allclose(mod(h, spd, etm, use_bias=True), _vanilla_mha(mod, h),
                          atol=1e-12)


def test_nonzero_spatial_bias_changes_attention():
    torch.manual_seed(2)
    mod = GraphormerAttention(dim=8, num_heads=2, max_spd=8).double()
    with torch.no_grad():
        mod.spatial_bias.weight.normal_()                    # give the bias teeth
    h = torch.randn(6, 8, dtype=torch.float64)
    spd = torch.randint(0, 9, (6, 6))
    etm = torch.zeros(6, 6, dtype=torch.long)
    assert not torch.allclose(mod(h, spd, etm, use_bias=True), _vanilla_mha(mod, h),
                              atol=1e-6)


def test_attention_rows_are_distributions():
    torch.manual_seed(3)
    mod = GraphormerAttention(dim=8, num_heads=2, max_spd=8).double()
    with torch.no_grad():
        mod.spatial_bias.weight.normal_()
    h = torch.randn(5, 8, dtype=torch.float64)
    spd = torch.randint(0, 9, (5, 5))
    etm = torch.randint(0, 7, (5, 5))
    # recompute the internal attention weights and check they sum to 1 per query
    V = 5
    q = mod.q(h).view(V, mod.num_heads, mod.head_dim).transpose(0, 1)
    k = mod.k(h).view(V, mod.num_heads, mod.head_dim).transpose(0, 1)
    logits = q @ k.transpose(-2, -1) / math.sqrt(mod.head_dim)
    logits = logits + mod.spatial_bias(spd).permute(2, 0, 1) + mod.edge_bias(etm).permute(2, 0, 1)
    attn = torch.softmax(logits, dim=-1)
    assert torch.allclose(attn.sum(-1), torch.ones(mod.num_heads, V, dtype=torch.float64),
                          atol=1e-12)


def test_centrality_is_noop_at_init_and_adds_degree_signal():
    enc = CentralityEncoding(dim=8, max_degree=16).double()
    h = torch.randn(4, 8, dtype=torch.float64)
    deg = torch.tensor([1, 2, 3, 20])
    assert torch.allclose(enc(h, deg), h, atol=1e-12)        # zero-init -> no-op
    with torch.no_grad():
        enc.emb.weight.normal_()
    out = enc(h, deg)
    assert not torch.allclose(out, h)
    # degree clamped: node with deg 20 uses the max_degree(16) row, no index error
    assert torch.isfinite(out).all()


def test_edge_type_matrix_and_dense_inputs():
    g, idx = build_two_hand_graph()
    etm = edge_type_matrix(g.edge_index, g.edge_type, g.num_nodes)
    assert etm.shape == (g.num_nodes, g.num_nodes)
    assert etm.min() >= 0 and etm.max() <= 6                 # 0=no edge, else type+1
    spd, etm2 = dense_bias_inputs(g, max_spd=8)
    assert spd.shape == (g.num_nodes, g.num_nodes)
    assert spd.max() <= 8                                    # clamped
    assert int(spd[idx["left"][0], idx["left"][0]]) == 0     # self distance 0


def test_graphormer_gradients_flow():
    mod = GraphormerAttention(dim=8, num_heads=2, max_spd=8).double()
    with torch.no_grad():
        mod.spatial_bias.weight.normal_()
        mod.edge_bias.weight.normal_()
    h = torch.randn(5, 8, dtype=torch.float64, requires_grad=True)
    spd = torch.randint(0, 9, (5, 5))
    etm = torch.randint(0, 7, (5, 5))
    mod(h, spd, etm).pow(2).sum().backward()
    assert torch.isfinite(mod.spatial_bias.weight.grad).all()
    assert torch.isfinite(mod.edge_bias.weight.grad).all()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_dim_must_divide_heads():
    with pytest.raises(ValueError):
        GraphormerAttention(dim=10, num_heads=4)
