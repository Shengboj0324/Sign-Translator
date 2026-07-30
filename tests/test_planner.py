"""Tests for the English->gloss semantic planner."""

import torch

from signtranslator.models.planner import GlossPlanner, causal_mask, BOS, EOS, PAD


def test_causal_mask_is_strictly_upper_triangular_boolean():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    assert bool(m[0, 1]) is True    # future disallowed
    assert bool(m[1, 0]) is False   # past allowed
    assert bool(m[2, 2]) is False   # self allowed
    # True strictly above the diagonal; False on/below.
    for i in range(4):
        for j in range(4):
            assert bool(m[i, j]) == (j > i)


def test_forward_shape():
    p = GlossPlanner(src_vocab=20, tgt_vocab=15, d_model=32, nhead=2, num_layers=2)
    src = torch.randint(1, 20, (3, 6))
    tgt_in = torch.randint(1, 15, (3, 5))
    logits = p.forward(src, tgt_in)
    assert logits.shape == (3, 5, 15)


def test_decoder_is_causal():
    """Changing a future target token must not change earlier-position logits."""
    torch.manual_seed(0)
    p = GlossPlanner(src_vocab=20, tgt_vocab=15, d_model=32, nhead=2, num_layers=2).eval()
    src = torch.randint(1, 20, (1, 6))
    tgt = torch.tensor([[BOS, 5, 7, 9]])
    base = p.forward(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, 3] = 12  # change last position
    alt = p.forward(src, tgt2)
    # Logits at positions 0..2 must be unchanged (no leakage from position 3).
    assert torch.allclose(base[:, :3], alt[:, :3], atol=1e-5)


def test_planner_overfits_reversal_mapping():
    """Learn a deterministic gloss reordering: gloss = reverse(src content)."""
    torch.manual_seed(0)
    Vsrc, Vtgt = 12, 12
    p = GlossPlanner(src_vocab=Vsrc, tgt_vocab=Vtgt, d_model=64, nhead=4, num_layers=2)

    # Build a fixed dataset: src content tokens 3..8, target = reversed + BOS/EOS.
    def make_batch():
        content = torch.randint(3, 9, (16, 4))
        src = content
        rev = torch.flip(content, dims=[1])
        bos = torch.full((16, 1), BOS)
        eos = torch.full((16, 1), EOS)
        tgt = torch.cat([bos, rev, eos], dim=1)
        return src, tgt

    src, tgt = make_batch()
    opt = torch.optim.Adam(p.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        loss = p.loss(src, tgt)
        if first is None:
            first = loss.detach().item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.detach().item() < first * 0.2

    # Greedy decode should reproduce the reversal on the training input.
    decoded = p.greedy_decode(src, max_len=8)
    expected = torch.flip(src, dims=[1]).tolist()
    correct = sum(d == e for d, e in zip(decoded, expected))
    assert correct >= 12  # the large majority exactly right
