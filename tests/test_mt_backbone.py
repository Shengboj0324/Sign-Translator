"""Verification of the hierarchical temporal backbone.

Proves: the duration model bridges the low-rate plan to the high-rate timeline
(Σ durations frames, correct frame->event map); the motion decoder's output truly
depends on the plan (cross-attention conditioning); causal masking blocks future
attention; and the recurrent memory carries information across chunks.
"""

import pytest
import torch

from signtranslator.motion_transformer.backbone import (
    causal_mask, ClausePlanner, DurationModel, MotionDecoder, RecurrentMemory,
)


# ---------------------------------------------------------------------------
# clause planner
# ---------------------------------------------------------------------------
def test_clause_planner_shape_and_gradient():
    planner = ClausePlanner(dim=16, num_layers=2, num_heads=4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = planner(x)
    assert out.shape == (2, 5, 16)
    out.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# duration model: the rate bridge
# ---------------------------------------------------------------------------
def test_duration_logits_shape():
    dm = DurationModel(dim=8, max_duration=16)
    logits = dm(torch.randn(2, 4, 8))
    assert logits.shape == (2, 4, 16)


def test_expand_by_duration_bridges_rates():
    events = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])   # (L=3, d=2)
    durations = torch.tensor([2, 1, 3])
    frames = DurationModel.expand_by_duration(events, durations)
    assert frames.shape[0] == int(durations.sum())                # 6 frames total
    # frame->event assignment is exactly the durations
    expected = torch.tensor([[1., 1.], [1., 1.], [2., 2.], [3., 3.], [3., 3.], [3., 3.]])
    assert torch.allclose(frames, expected)


def test_expand_rejects_batched_input():
    with pytest.raises(ValueError):
        DurationModel.expand_by_duration(torch.randn(2, 3, 4), torch.tensor([1, 1]))


# ---------------------------------------------------------------------------
# motion decoder: cross-attention conditioning
# ---------------------------------------------------------------------------
def test_motion_decoder_depends_on_plan():
    torch.manual_seed(0)
    dec = MotionDecoder(dim=16, num_layers=2, num_heads=4)
    dec.eval()
    motion = torch.randn(1, 6, 16)
    plan_a = torch.randn(1, 3, 16)
    plan_b = torch.randn(1, 3, 16)
    out_a = dec(motion, plan_a)
    out_b = dec(motion, plan_b)
    # changing the plan must change the output (the decoder actually conditions)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_causal_mask_blocks_future_attention():
    torch.manual_seed(1)
    dec = MotionDecoder(dim=16, num_layers=1, num_heads=4)
    dec.eval()
    T = 6
    motion = torch.randn(1, T, 16, requires_grad=True)
    plan = torch.randn(1, 2, 16)
    out = dec(motion, plan, causal=True)
    # output at t=2 must not depend on motion tokens t>2
    grad = torch.autograd.grad(out[0, 2].sum(), motion, retain_graph=True)[0][0]
    future = grad[3:].abs().sum()
    assert float(future) < 1e-8                               # no leakage from the future


def test_causal_mask_values():
    m = causal_mask(3)
    assert m[0, 0] == 0.0 and m[0, 1] == float("-inf")
    assert m[2, 0] == 0.0 and m[1, 2] == float("-inf")


# ---------------------------------------------------------------------------
# recurrent memory
# ---------------------------------------------------------------------------
def test_memory_carries_information_across_chunks():
    mem = RecurrentMemory(input_dim=8, state_dim=12)
    state0 = mem.init_state(2)
    assert state0.shape == (2, 12)
    # two different histories must produce different states (memory is not amnesiac)
    a1, a2 = torch.randn(2, 8), torch.randn(2, 8)
    s = mem(a1, state0); s = mem(a2, s)
    s_alt = mem(a2, state0); s_alt = mem(a1, s_alt)          # swapped order
    assert not torch.allclose(s, s_alt, atol=1e-5)           # order/history matters


def test_memory_state_depends_on_early_input():
    """The state after 3 chunks must depend on the FIRST chunk (long-range memory)."""
    mem = RecurrentMemory(input_dim=6, state_dim=10)
    first = torch.randn(1, 6, requires_grad=True)
    s = mem(first, mem.init_state(1))
    for _ in range(2):
        s = mem(torch.randn(1, 6), s)
    grad = torch.autograd.grad(s.sum(), first)[0]
    assert grad.abs().sum() > 0                               # first chunk still influences state
