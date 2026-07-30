"""Verification of relational message passing (R-GCN + GAT, basis decomposition).

Proves the defining properties in float64: softmax normalisation, neighbour-
permutation equivariance, reduction to uniform mean at zero attention, basis
expressivity (B=R one-hot == independent W_r), the isolated-node self term,
confidence masking (occluded neighbours excluded, all-occluded fallback), and
gradient flow to every parameter group.
"""

import pytest
import torch

from signtranslator.hand_graph.relational import (
    group_softmax, RelationalGraphAttention, RelationalGraphNetwork,
)


def _layer(in_dim=4, out_dim=4, num_relations=3, num_bases=None, seed=0):
    torch.manual_seed(seed)
    m = RelationalGraphAttention(in_dim, out_dim, num_relations, num_bases).double()
    return m


def _small_graph():
    # 3 nodes; relation 0 edges: 1->0, 2->0 ; relation 1 edge: 0->1
    edge_index = torch.tensor([[1, 2, 0], [0, 0, 1]])
    edge_type = torch.tensor([0, 0, 1])
    return edge_index, edge_type


# ---------------------------------------------------------------------------
# group softmax
# ---------------------------------------------------------------------------
def test_group_softmax_normalises_within_groups():
    score = torch.tensor([1.0, 2.0, 0.5, 3.0], dtype=torch.float64)
    group = torch.tensor([0, 0, 1, 1])
    w = group_softmax(score, group, num_groups=2)
    assert abs(float(w[group == 0].sum()) - 1.0) < 1e-12
    assert abs(float(w[group == 1].sum()) - 1.0) < 1e-12
    # matches a manual softmax on group 0
    man = torch.softmax(torch.tensor([1.0, 2.0], dtype=torch.float64), 0)
    assert torch.allclose(w[:2], man, atol=1e-12)


def test_group_softmax_stable_with_large_scores():
    score = torch.tensor([1e6, 1e6 + 1.0], dtype=torch.float64)
    group = torch.tensor([0, 0])
    w = group_softmax(score, group, 1)
    assert torch.isfinite(w).all()
    assert abs(float(w.sum()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# attention normalisation, extracted via a probe
# ---------------------------------------------------------------------------
def test_attention_weights_sum_to_one_per_destination_relation():
    m = _layer(seed=1)
    h = torch.randn(3, 4, dtype=torch.float64)
    ei, et = _small_graph()
    # replicate the internal alpha computation to check normalisation
    Wr = torch.einsum("rb,boi->roi", m.coeff, m.bases)
    h_Wr = torch.einsum("roi,vi->rvo", Wr, h)
    s, d = ei[0], ei[1]
    cat = torch.cat((h_Wr[et, d], h_Wr[et, s]), -1)
    raw = (cat * m.att[et]).sum(-1)
    score = torch.nn.functional.leaky_relu(raw, m.negative_slope)
    alpha = group_softmax(score, d * m.num_relations + et, 3 * m.num_relations)
    # destination 0 relation 0 has two incoming edges -> they must sum to 1
    mask = (d == 0) & (et == 0)
    assert abs(alpha[mask].sum().detach().item() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# neighbour-permutation equivariance
# ---------------------------------------------------------------------------
def test_permutation_equivariance_over_edges():
    m = _layer(seed=2)
    h = torch.randn(3, 4, dtype=torch.float64)
    ei, et = _small_graph()
    out1 = m(h, ei, et)
    perm = torch.tensor([2, 0, 1])                            # shuffle edge order
    out2 = m(h, ei[:, perm], et[perm])
    assert torch.allclose(out1, out2, atol=1e-12)


# ---------------------------------------------------------------------------
# reduction to uniform mean at zero attention logits
# ---------------------------------------------------------------------------
def test_zero_attention_gives_uniform_mean_aggregation():
    m = _layer(seed=3)
    with torch.no_grad():
        m.att.zero_()                                         # score = leaky_relu(0) = 0 -> uniform
    h = torch.randn(3, 4, dtype=torch.float64)
    ei, et = _small_graph()
    out = m(h, ei, et)
    # manual: node 0 gets W0 h0 + mean over {W_r0 h1, W_r0 h2}
    W0 = m.self_transform.weight
    Wr0 = m.relation_weight(0)
    expected0 = W0 @ h[0] + 0.5 * (Wr0 @ h[1] + Wr0 @ h[2])
    assert torch.allclose(out[0], expected0, atol=1e-12)
    # node 2 has no incoming edges -> only the self term
    assert torch.allclose(out[2], W0 @ h[2], atol=1e-12)


# ---------------------------------------------------------------------------
# isolated node -> self term only
# ---------------------------------------------------------------------------
def test_isolated_node_returns_self_transform():
    m = _layer(seed=4)
    h = torch.randn(2, 4, dtype=torch.float64)
    empty = torch.zeros(2, 0, dtype=torch.long)
    out = m(h, empty, torch.zeros(0, dtype=torch.long))
    assert torch.allclose(out, m.self_transform(h), atol=1e-12)


# ---------------------------------------------------------------------------
# basis expressivity: B = R with one-hot coefficients == independent W_r
# ---------------------------------------------------------------------------
def test_basis_onehot_recovers_independent_relation_weights():
    m = _layer(num_relations=3, num_bases=3, seed=5)
    with torch.no_grad():
        m.coeff.copy_(torch.eye(3, dtype=torch.float64))
    for r in range(3):
        assert torch.allclose(m.relation_weight(r), m.bases[r], atol=1e-12)


# ---------------------------------------------------------------------------
# confidence masking
# ---------------------------------------------------------------------------
def test_occluded_neighbour_is_excluded():
    """A neighbour with confidence 0 must contribute nothing -- identical to
    removing its edges."""
    m = _layer(seed=6)
    h = torch.randn(3, 4, dtype=torch.float64)
    ei, et = _small_graph()                                   # node 0 <- {1,2} rel 0
    conf = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)  # node 2 occluded
    out_masked = m(h, ei, et, node_confidence=conf)
    # remove node 2's edge and recompute
    keep = torch.tensor([True, False, True])
    out_removed = m(h, ei[:, keep], et[keep])
    assert torch.allclose(out_masked[0], out_removed[0], atol=1e-12)


def test_all_occluded_neighbourhood_falls_back_to_self():
    m = _layer(seed=7)
    h = torch.randn(3, 4, dtype=torch.float64)
    ei, et = _small_graph()
    conf = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)  # both sources occluded
    out = m(h, ei, et, node_confidence=conf)
    # node 0's only neighbours (1,2) are occluded -> self term only, no NaN
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], m.self_transform(h)[0], atol=1e-12)


# ---------------------------------------------------------------------------
# gradient flow
# ---------------------------------------------------------------------------
def test_gradients_reach_all_parameter_groups():
    m = _layer(seed=8)
    h = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    ei, et = _small_graph()
    m(h, ei, et).pow(2).sum().backward()
    for name, p in [("W0", m.self_transform.weight), ("bases", m.bases),
                    ("coeff", m.coeff), ("att", m.att)]:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_network_stack_runs_and_is_finite():
    net = RelationalGraphNetwork(dim=8, num_layers=2).double()
    h = torch.randn(5, 8, dtype=torch.float64)
    ei = torch.tensor([[1, 2, 3, 4], [0, 0, 1, 1]])
    et = torch.tensor([0, 1, 2, 0])
    out = net(h, ei, et)
    assert out.shape == (5, 8) and torch.isfinite(out).all()
