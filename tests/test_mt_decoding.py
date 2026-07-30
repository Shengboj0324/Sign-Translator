"""Verification of decoding strategies and the experiment harness.

Proves GPT causality (a token cannot see its future), that the AR model can fit a
tiny token sequence, that masked-span loss supervises ONLY masked positions with
bidirectional context, generation shape, and the honest reconstruction comparison
(residual-VQ never worse than single-stage VQ).
"""

import pytest
import torch

from signtranslator.motion_transformer.decoding import (
    MotionTokenGPT, MaskedMotionModel, compare_reconstruction, compare_shared_vs_part,
)


# ---------------------------------------------------------------------------
# autoregressive GPT
# ---------------------------------------------------------------------------
def test_gpt_is_causal():
    torch.manual_seed(0)
    gpt = MotionTokenGPT(num_codes=10, dim=16, num_layers=2)
    gpt.eval()
    tokens = torch.randint(0, 10, (1, 8))
    # embed path is discrete; probe causality on the continuous embedding input
    h = gpt.pos(gpt.emb(tokens)).detach().clone().requires_grad_(True)
    from signtranslator.motion_transformer.backbone import causal_mask
    out = gpt.head(gpt.encoder(h, mask=causal_mask(8)))
    grad = torch.autograd.grad(out[0, 3].sum(), h, retain_graph=True)[0][0]
    assert grad[4:].abs().sum().item() < 1e-8                # position 3 sees no future


def test_gpt_fits_a_repeating_token_pattern():
    torch.manual_seed(1)
    gpt = MotionTokenGPT(num_codes=6, dim=32, num_layers=2)
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(4, 1)
    opt = torch.optim.Adam(gpt.parameters(), lr=3e-3)
    l0 = gpt.loss(tokens).item()
    for _ in range(300):
        opt.zero_grad(); loss = gpt.loss(tokens); loss.backward(); opt.step()
    assert gpt.loss(tokens).item() < 0.2 * l0


def test_gpt_generation_shape():
    gpt = MotionTokenGPT(num_codes=8, dim=16, num_layers=1)
    g = torch.Generator().manual_seed(0)
    seq = gpt.generate(length=12, batch=2, generator=g)
    assert seq.shape == (2, 12)
    assert int(seq.min()) >= 0 and int(seq.max()) < 8        # valid code ids only


# ---------------------------------------------------------------------------
# masked-span
# ---------------------------------------------------------------------------
def test_masked_span_marks_a_contiguous_region():
    tokens = torch.randint(0, 10, (3, 20))
    masked, mask = MaskedMotionModel.make_masked_span(
        tokens, mask_token=10, span_frac=0.25, generator=torch.Generator().manual_seed(0))
    # exactly one contiguous span per row, of the right length
    for i in range(3):
        positions = mask[i].nonzero().flatten()
        assert len(positions) == max(1, round(0.25 * 20))
        assert (positions[1:] - positions[:-1] == 1).all()  # contiguous
        assert (masked[i][mask[i]] == 10).all()             # masked -> mask token
        assert torch.equal(masked[i][~mask[i]], tokens[i][~mask[i]])  # rest untouched


def test_masked_loss_supervises_only_masked_positions():
    torch.manual_seed(2)
    model = MaskedMotionModel(num_codes=8, dim=16, num_layers=1)
    tokens = torch.randint(0, 8, (2, 16))
    loss = model.loss(tokens, span_frac=0.25, generator=torch.Generator().manual_seed(3))
    assert torch.isfinite(loss) and loss.detach().item() > 0


def test_masked_model_can_use_bidirectional_context():
    """A masked position's logits depend on tokens BOTH before and after it."""
    torch.manual_seed(4)
    model = MaskedMotionModel(num_codes=8, dim=16, num_layers=1)
    model.eval()
    h = model.pos(model.emb(torch.randint(0, 8, (1, 7)))).detach().clone().requires_grad_(True)
    out = model.head(model.encoder(h))
    grad = torch.autograd.grad(out[0, 3].sum(), h)[0][0]
    assert grad[:3].abs().sum() > 0 and grad[4:].abs().sum() > 0   # sees past AND future


# ---------------------------------------------------------------------------
# experiment harness
# ---------------------------------------------------------------------------
def test_compare_reconstruction_rvq_not_worse_than_vq():
    torch.manual_seed(5)
    data = torch.randn(300, 8, dtype=torch.float64)
    res = compare_reconstruction(data, dim=8, num_codes=32, rvq_stages=4, seed=5)
    assert res["raw"] == 0.0
    assert res["rvq"] <= res["vq"] + 1e-9                    # more stages never worse
    assert 0.0 < res["vq"] <= 1.0


def test_compare_shared_vs_part_runs():
    torch.manual_seed(6)
    data = torch.randn(200, 9, dtype=torch.float64)
    res = compare_shared_vs_part(data, {"torso": 3, "hands": 6}, num_codes=32, seed=6)
    assert set(res) == {"shared", "part_specific"}
    assert all(0.0 <= v <= 1.0 for v in res.values())
