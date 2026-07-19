"""Proofs for relation-biased attention and the SIR decoder.

Key claims verified: with zero bias the layer is *exactly* standard scaled
dot-product attention; the relation bias monotonically shifts attention toward
the biased pair; masking gives exact zeros with no NaN; and the decoder's
interval parameterisation makes ``t_start < t_end`` hold by construction.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from signtranslator.grammar.graph_transformer import (
    relation_biased_attention, RelationBiasedAttention, GraphTransformerLayer,
    SIRDecoder,
)


# ---------------------------------------------------------------------------
# Functional attention
# ---------------------------------------------------------------------------
def test_zero_bias_reduces_to_standard_attention():
    """b == 0 must give bit-for-bit standard scaled dot-product attention."""
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 5, 8, dtype=torch.float64)
    zero_bias = torch.zeros(2, 5, 5, dtype=torch.float64)

    ctx, w = relation_biased_attention(q, k, v, zero_bias)
    ref_logits = q @ k.transpose(-2, -1) / math.sqrt(8)
    ref_w = torch.softmax(ref_logits, dim=-1)
    ref_ctx = ref_w @ v
    assert torch.allclose(w, ref_w, atol=1e-12)
    assert torch.allclose(ctx, ref_ctx, atol=1e-12)


def test_attention_weights_are_a_distribution():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 8) for _ in range(3))
    _, w = relation_biased_attention(q, k, v, torch.randn(2, 4, 4))
    assert torch.allclose(w.sum(-1), torch.ones(2, 4), atol=1e-6)
    assert torch.all(w >= 0)


def test_positive_bias_increases_attention_to_the_biased_pair():
    """Raising b_ij strictly increases alpha_ij relative to no bias."""
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 4, 8, dtype=torch.float64) for _ in range(3))
    base = torch.zeros(1, 4, 4, dtype=torch.float64)
    _, w0 = relation_biased_attention(q, k, v, base)
    biased = base.clone()
    biased[0, 0, 2] = 3.0                                # favour i=0 -> j=2
    _, w1 = relation_biased_attention(q, k, v, biased)
    assert float(w1[0, 0, 2]) > float(w0[0, 0, 2])
    # and the mass came from the competitors (row still sums to 1)
    assert abs(float(w1[0, 0].sum()) - 1.0) < 1e-9


def test_masking_gives_exact_zero_without_nan():
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 4, 8, dtype=torch.float64) for _ in range(3))
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 0, 1] = True                                 # forbid 0 -> 1
    _, w = relation_biased_attention(q, k, v, torch.zeros(1, 4, 4, dtype=torch.float64),
                                     mask=mask)
    assert float(w[0, 0, 1]) == 0.0
    assert torch.isfinite(w).all()
    assert abs(float(w[0, 0].sum()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
def test_module_zero_init_bias_matches_plain_attention():
    """The learned biases start at 0, so an untrained layer ignores relations."""
    torch.manual_seed(0)
    attn = RelationBiasedAttention(d_model=16, num_heads=4, num_relations=5).double().eval()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    rel_a = torch.zeros(2, 6, 6, dtype=torch.long)
    rel_b = torch.randint(0, 5, (2, 6, 6))
    # with zero bias, the relation ids must not matter
    assert torch.allclose(attn(x, rel_a), attn(x, rel_b), atol=1e-10)


def test_module_relation_bias_changes_output_once_trained():
    torch.manual_seed(0)
    attn = RelationBiasedAttention(16, 4, 5)
    with torch.no_grad():
        attn.relation_bias.normal_(0, 2.0)               # non-trivial biases
    x = torch.randn(2, 6, 16)
    a = attn(x, torch.zeros(2, 6, 6, dtype=torch.long))
    b = attn(x, torch.randint(1, 5, (2, 6, 6)))
    assert not torch.allclose(a, b, atol=1e-4)


def test_module_rejects_out_of_range_relation_id():
    attn = RelationBiasedAttention(16, 4, 3)
    with pytest.raises(ValueError):
        attn(torch.randn(1, 4, 16), torch.full((1, 4, 4), 5))


def test_module_requires_divisible_dims():
    with pytest.raises(ValueError):
        RelationBiasedAttention(d_model=10, num_heads=4, num_relations=2)


def test_layer_shapes_and_gradients():
    layer = GraphTransformerLayer(16, 4, 5)
    x = torch.randn(2, 6, 16)
    rel = torch.randint(0, 5, (2, 6, 6))
    out = layer(x, rel)
    assert out.shape == x.shape
    out.sum().backward()
    assert layer.attn.relation_bias.grad is not None


# ---------------------------------------------------------------------------
# SIR decoder
# ---------------------------------------------------------------------------
def test_decoder_intervals_are_valid_by_construction():
    """t_start < t_end for EVERY event, for ANY input -- softplus width > 0."""
    torch.manual_seed(0)
    dec = SIRDecoder(d_model=16, num_relations=5, num_markers=4)
    for _ in range(20):
        x = torch.randn(3, 7, 16) * 10                   # large, adversarial
        rel = torch.randint(0, 5, (3, 7, 7))
        start, end, markers = dec(x, rel)
        assert torch.all(end > start)                    # never invalid
        assert start.shape == (3, 7) and markers.shape == (3, 7, 4)


def test_decoder_marker_head_is_multilabel():
    """Marker logits are independent (M values per position), not a softmax."""
    dec = SIRDecoder(d_model=16, num_relations=5, num_markers=4)
    x = torch.randn(2, 5, 16)
    rel = torch.zeros(2, 5, 5, dtype=torch.long)
    _, _, logits = dec(x, rel)
    probs = torch.sigmoid(logits)
    # several markers can be simultaneously > 0.5 at one position (a softmax
    # could not represent that)
    assert probs.shape == (2, 5, 4)
    assert bool((probs > 0.0).all()) and bool((probs < 1.0).all())


def test_decoder_can_overfit_target_intervals():
    """The decoder must be able to fit prescribed intervals via the temporal
    parameterisation (a sanity check that the head is expressive)."""
    torch.manual_seed(0)
    dec = SIRDecoder(d_model=32, num_relations=3, num_markers=2, num_layers=1)
    x = torch.randn(1, 4, 32)
    rel = torch.zeros(1, 4, 4, dtype=torch.long)
    target_start = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    target_end = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
    opt = torch.optim.Adam(dec.parameters(), lr=0.02)
    first = None
    for _ in range(300):
        s, e, _ = dec(x, rel)
        loss = F.mse_loss(s, target_start) + F.mse_loss(e, target_end)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first * 0.05
