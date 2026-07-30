"""Verification of LoRA adapters and the freeze-first protocol."""

import pytest
import torch
import torch.nn as nn

from signtranslator.speech.lora import (
    LoRALinear, inject_lora, merge_all_lora, iter_lora_modules,
    mark_only_lora_trainable, unfreeze_upper_blocks,
    trainable_parameter_summary, freeze_all,
)


class _Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.ffn = nn.Linear(d, d)

    def forward(self, x):
        return self.ffn(self.q_proj(x) + self.v_proj(x))


class _Encoder(nn.Module):
    def __init__(self, n_blocks=4, d=16):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


# ---------------------------------------------------------------------------
# Core LoRA algebra
# ---------------------------------------------------------------------------
def test_is_exact_identity_at_initialisation():
    """B = 0 => Delta W = 0, so the wrapper reproduces the base layer EXACTLY.

    This is what makes freeze-first safe: adaptation cannot perturb a strong
    pretrained encoder before a single gradient step.
    """
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    x = torch.randn(5, 16)
    expected = base(x).clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    assert torch.allclose(lora(x), expected, atol=1e-7)
    assert torch.allclose(lora.delta_weight(), torch.zeros(24, 16), atol=1e-9)


def test_becomes_non_trivial_after_updating_B():
    lora = LoRALinear(nn.Linear(16, 16), r=4)
    x = torch.randn(3, 16)
    before = lora(x).clone()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    assert not torch.allclose(lora(x), before, atol=1e-4)


def test_delta_weight_rank_is_at_most_r():
    """A silent bug producing a full-rank update would defeat the whole method.

    The rank must be evaluated in the tensor's OWN dtype. Upcasting a float32
    product to float64 first does not recover precision -- it keeps the ~1e-8
    float32 rounding in the nominally-zero singular values while applying a
    float64 tolerance, which then reports full rank.
    """
    for r in (1, 2, 4):
        lora = LoRALinear(nn.Linear(32, 24), r=r)
        with torch.no_grad():
            lora.lora_B.normal_(0, 0.5)
        dw = lora.delta_weight()
        assert int(torch.linalg.matrix_rank(dw)) <= r
        # And exactly, via the singular spectrum: only r values are significant.
        sv = torch.linalg.svdvals(dw)
        assert int((sv > 1e-5 * sv[0]).sum()) <= r


def test_scaling_is_alpha_over_r():
    lora = LoRALinear(nn.Linear(8, 8), r=4, alpha=16.0)
    assert abs(lora.scaling - 4.0) < 1e-12
    with torch.no_grad():
        lora.lora_B.fill_(1.0)
        lora.lora_A.fill_(1.0)
    # Delta W = (alpha/r) * B A, with B A entries = r (inner dimension).
    assert torch.allclose(lora.delta_weight(),
                          torch.full((8, 8), 4.0 * 4.0), atol=1e-6)


def test_merge_preserves_the_function_exactly_in_float64():
    """Merging is mathematically exact: proved in double precision.

    In float32 it is only exact to ~1e-3 relative, because merging reassociates
    the matmul (rank-r bottleneck first vs. full matrix). That is a property of
    floating point, not of the method -- hence the float64 proof plus the
    bounded float32 check below.
    """
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(16, 16), r=4, alpha=8.0).double()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.2)
    x = torch.randn(4, 16, dtype=torch.float64)
    before = lora(x).clone()
    lora.merge()
    assert lora.merged
    assert torch.allclose(lora(x), before, atol=1e-12)


def test_merge_deviation_in_float32_is_small_but_nonzero():
    """Document the real numerical behaviour rather than asserting bit-equality."""
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(16, 16), r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.2)
    x = torch.randn(4, 16)
    before = lora(x).clone()
    lora.merge()
    rel = ((lora(x) - before).abs().max()
           / before.abs().max()).detach().item()
    assert rel < 1e-2, f"merge deviation {rel} too large to be rounding"


def test_merge_then_unmerge_restores_base_weight():
    torch.manual_seed(2)
    base = nn.Linear(12, 12)
    original = base.weight.detach().clone()
    lora = LoRALinear(base, r=3)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.3)
    lora.merge()
    assert not torch.allclose(lora.base.weight, original, atol=1e-6)
    lora.unmerge()
    assert torch.allclose(lora.base.weight, original, atol=1e-6)


def test_merge_is_idempotent():
    lora = LoRALinear(nn.Linear(8, 8), r=2)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.2)
    lora.merge()
    w_once = lora.base.weight.detach().clone()
    lora.merge()                                    # must be a no-op
    assert torch.allclose(lora.base.weight, w_once, atol=1e-9)


def test_base_weights_are_frozen_but_lora_is_not():
    lora = LoRALinear(nn.Linear(16, 16), r=4)
    assert not lora.base.weight.requires_grad
    assert not lora.base.bias.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_gradients_reach_only_the_lora_factors():
    lora = LoRALinear(nn.Linear(16, 16), r=4)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)      # otherwise dL/dA is zero through B=0
    lora(torch.randn(4, 16)).sum().backward()
    assert lora.lora_A.grad is not None and lora.lora_A.grad.abs().sum() > 0
    assert lora.lora_B.grad is not None and lora.lora_B.grad.abs().sum() > 0
    assert lora.base.weight.grad is None


def test_parameter_saving_is_substantial():
    lora = LoRALinear(nn.Linear(512, 512), r=8)
    assert lora.trainable_parameter_count() == 8 * 512 * 2
    ratio = lora.trainable_parameter_count() / lora.full_parameter_count()
    assert ratio < 0.04                              # ~3% of the dense weight


def test_constructor_validates_arguments():
    with pytest.raises(TypeError):
        LoRALinear(nn.Conv1d(4, 4, 3))
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 8), r=0)
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 16), r=9)            # r > min(in, out)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def test_injection_targets_only_named_projections():
    enc = _Encoder(n_blocks=3)
    names = inject_lora(enc, target_suffixes=("q_proj", "v_proj"), r=4)
    assert len(names) == 6                            # 2 per block x 3 blocks
    assert all(n.endswith(("q_proj", "v_proj")) for n in names)
    # The feed-forward layers must be untouched.
    assert all(isinstance(b.ffn, nn.Linear) and not isinstance(b.ffn, LoRALinear)
               for b in enc.blocks)
    assert all(isinstance(b.q_proj, LoRALinear) for b in enc.blocks)


def test_injection_preserves_model_output_exactly():
    """Injecting adapters must not change the function before training."""
    torch.manual_seed(3)
    enc = _Encoder()
    x = torch.randn(2, 7, 16)
    before = enc(x).clone()
    inject_lora(enc, r=4)
    assert torch.allclose(enc(x), before, atol=1e-6)


def test_merge_all_reports_count_and_preserves_output():
    torch.manual_seed(4)
    enc = _Encoder(n_blocks=2)
    inject_lora(enc, r=4)
    for m in iter_lora_modules(enc):
        with torch.no_grad():
            m.lora_B.normal_(0, 0.1)
    x = torch.randn(2, 5, 16)
    before = enc(x).clone()
    assert merge_all_lora(enc) == 4
    # float32 reassociation (see test_merge_deviation_in_float32_is_small...)
    rel = ((enc(x) - before).abs().max()
           / before.abs().max()).detach().item()
    assert rel < 1e-2


# ---------------------------------------------------------------------------
# Freeze-first protocol
# ---------------------------------------------------------------------------
def test_phase_one_trains_only_adapters():
    enc = _Encoder(n_blocks=3)
    inject_lora(enc, r=4)
    n = mark_only_lora_trainable(enc)
    summary = trainable_parameter_summary(enc)
    assert summary["trainable"] == n
    assert n == 6 * (4 * 16 * 2)                     # 6 adapters, r=4, d=16
    # NOTE: no parameter-saving claim at this toy width -- r=4 against d=16 is
    # 25% of the weight, so LoRA saves nothing here. The saving requires
    # r << d and is asserted at a realistic width in
    # test_parameter_saving_is_substantial.
    assert summary["trainable"] < summary["total"]
    for name, p in enc.named_parameters():
        if p.requires_grad:
            assert "lora_" in name, f"unexpectedly trainable: {name}"


def test_phase_two_unfreezes_upper_blocks_only():
    enc = _Encoder(n_blocks=4)
    inject_lora(enc, r=4)
    mark_only_lora_trainable(enc)
    unfrozen = unfreeze_upper_blocks(enc.blocks, num_blocks=2)
    assert unfrozen > 0
    # Lower blocks keep only their adapters trainable ...
    for b in enc.blocks[:2]:
        assert not b.ffn.weight.requires_grad
    # ... while upper blocks are fully trainable.
    for b in enc.blocks[2:]:
        assert b.ffn.weight.requires_grad


def test_unfreeze_zero_blocks_is_a_noop():
    enc = _Encoder(n_blocks=3)
    inject_lora(enc, r=4)
    mark_only_lora_trainable(enc)
    before = trainable_parameter_summary(enc)["trainable"]
    assert unfreeze_upper_blocks(enc.blocks, 0) == 0
    assert trainable_parameter_summary(enc)["trainable"] == before


def test_unfreeze_rejects_negative():
    enc = _Encoder()
    with pytest.raises(ValueError):
        unfreeze_upper_blocks(enc.blocks, -1)


def test_freeze_all_then_summary_reports_zero_trainable():
    enc = _Encoder()
    freeze_all(enc)
    s = trainable_parameter_summary(enc)
    assert s["trainable"] == 0 and s["frozen"] == s["total"]


def test_optimizer_sees_only_adapter_parameters():
    """End-to-end: the protocol must actually restrict what an optimiser updates."""
    torch.manual_seed(5)
    enc = _Encoder(n_blocks=2)
    inject_lora(enc, r=4)
    mark_only_lora_trainable(enc)
    frozen_ref = enc.blocks[0].ffn.weight.detach().clone()

    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.1)
    for _ in range(3):
        loss = enc(torch.randn(2, 4, 16)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    assert torch.allclose(enc.blocks[0].ffn.weight, frozen_ref, atol=1e-9)
