"""Attention support must affect nontrivial predictions and gradients."""
from copy import deepcopy

import pytest
import torch

from signtranslator.models.denoiser import MotionDenoiser, CrossModalDenoiser
from signtranslator.models.guided_diffusion import GuidedMotionDiffusion


def make(kind):
    torch.manual_seed(9)
    cls = MotionDenoiser if kind == 'pooled' else CrossModalDenoiser
    model = cls(3, 2, 4, hidden_dim=8, num_layers=2, num_heads=2,
                dropout=0, max_frames=20)
    # Initial zero projection would trivially pass every invariance test.
    torch.nn.init.normal_(model.output_proj.weight, std=.2)
    return model


@pytest.mark.parametrize('kind', ['pooled', 'cross'])
@pytest.mark.parametrize('training', [True, False])
def test_attention_padding_preserves_predictions_and_gradients(kind, training):
    first = make(kind).train(training)
    second = deepcopy(first)
    raw = torch.randn(2, 2, 5, 3, requires_grad=True)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    valid[:, 2, 1] = False
    pad = torch.cat([raw.detach(), torch.full((2, 2, 3, 3), float('nan'))], 2)
    pad.requires_grad_()
    padded_valid = torch.cat([valid, torch.zeros(2, 3, 3, dtype=torch.bool)], 1)
    cond = torch.randn(2, 4) if kind == 'pooled' else (torch.randn(2, 3, 4), None)
    time = torch.tensor([2, 3])
    a = first(raw, time, cond, motion_mask=valid)
    b = second(pad, time, cond, motion_mask=padded_valid)
    assert a.abs().max() > .01
    torch.testing.assert_close(a, b[:, :, :5], rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(b[:, :, 5:]) == 0
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(raw.grad, pad.grad[:, :, :5], rtol=2e-4, atol=2e-5)
    assert torch.count_nonzero(pad.grad[:, :, 5:]) == 0
    assert torch.count_nonzero(raw.grad[:, :, 2, 1]) == 0
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5)


def test_masked_and_dropped_memory_cannot_poison_attention_or_gradients():
    model = make('cross')
    x = torch.randn(2, 2, 5, 3)
    t = torch.tensor([1, 2])
    memory = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, False, False], [False, False, False]])
    clean = model(x, t, (memory, valid))
    memory[~valid] = float('nan')
    memory.requires_grad_()
    out = model(x, t, (memory, valid))
    torch.testing.assert_close(out, clean, rtol=0, atol=0)
    out.square().sum().backward()
    assert torch.isfinite(memory.grad).all()
    assert torch.count_nonzero(memory.grad[~valid]) == 0
    dropped = model(x, t, (torch.full_like(memory, float('nan')), None),
                    drop=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(dropped, model(x, t, None), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('parameterization', ['eps', 'x0'])
def test_diffusion_forwards_support_and_has_padding_invariant_objective(parameterization):
    first = GuidedMotionDiffusion(make('cross'), num_timesteps=10, cond_drop_prob=0,
                                 parameterization=parameterization, velocity_weight=.1)
    second = deepcopy(first)
    x = torch.randn(2, 2, 5, 3)
    noise = torch.randn_like(x)
    cond = (torch.randn(2, 3, 4), None)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    t = torch.tensor([2, 4])
    padded = torch.cat([x, torch.full((2, 2, 3, 3), float('nan'))], 2)
    pn = torch.cat([noise, torch.full((2, 2, 3, 3), float('nan'))], 2)
    pv = torch.cat([valid, torch.zeros(2, 3, 3, dtype=torch.bool)], 1)
    a = first.p_losses(x, t, cond, noise=noise, validity_mask=valid)
    b = second.p_losses(padded, t, cond, noise=pn, validity_mask=pv)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.backward()
    b.backward()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5)
    with pytest.raises(ValueError, match='derived from objective support'):
        first.p_losses(x, t, cond, noise=noise, motion_mask=valid)


@pytest.mark.parametrize('kind', ['pooled', 'cross'])
def test_support_domains_and_capacity_are_explicit(kind):
    model = make(kind)
    x = torch.ones(1, 2, 5, 3)
    t = torch.ones(1, dtype=torch.long)
    for mask in (torch.zeros(1, 5, 3, dtype=torch.bool), torch.ones(1, 5, 3),
                 torch.ones(1, 4, 3, dtype=torch.bool)):
        with pytest.raises(ValueError):
            model(x, t, motion_mask=mask)
    with pytest.raises(ValueError, match='capacity'):
        model(torch.ones(1, 2, 21, 3), t)
    with pytest.raises(ValueError, match='finite'):
        model(x * float('nan'), t)


def test_conditioning_rejects_wrong_mask_and_observed_nan():
    model = make('cross')
    x = torch.ones(1, 2, 5, 3)
    t = torch.ones(1, dtype=torch.long)
    for memory, mask in ((torch.ones(1, 2, 4), torch.ones(1, 2)),
                         (torch.ones(1, 2, 4), torch.ones(1, 3, dtype=torch.bool)),
                         (torch.full((1, 2, 4), float('nan')), None)):
        with pytest.raises(ValueError):
            model(x, t, (memory, mask))
