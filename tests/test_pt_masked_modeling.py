"""Adversarial tests for masked motion modeling (Doc-11, stage 11b)."""

import math

import pytest
import torch

from signtranslator.pretraining.masked_modeling import (
    masked_token_nll, masked_rotation_geodesic, codebook_diversity_loss,
    copy_through_logits, MaskedMotionModel,
)
from signtranslator.pose.rotations import (
    matrix_to_rotation_6d, axis_angle_to_matrix,
)

torch.manual_seed(0)


def _rand_6d(n):
    R = axis_angle_to_matrix(torch.randn(n, 3, dtype=torch.float64))
    return matrix_to_rotation_6d(R)


def test_masked_nll_averages_over_masked_only():
    K, N = 5, 6
    logits = torch.zeros(N, K)
    target = torch.tensor([0, 1, 2, 3, 4, 0])
    mask = torch.tensor([True, True, False, False, False, False])
    # uniform logits => nll = log K on every position; masked mean also log K.
    assert torch.isclose(masked_token_nll(logits, target, mask),
                         torch.tensor(math.log(K)), atol=1e-6)


def test_masked_nll_ignores_visible_positions():
    K, N = 4, 4
    logits = torch.full((N, K), -10.0)
    target = torch.tensor([0, 1, 2, 3])
    # make the VISIBLE position badly wrong, masked positions correct.
    logits[0, target[0]] = 10.0    # masked correct
    logits[1, target[1]] = 10.0    # masked correct
    logits[2, 0] = 10.0            # visible: wrong (target 2) -> must be IGNORED
    mask = torch.tensor([True, True, False, False])
    assert masked_token_nll(logits, target, mask) < 1e-3


def test_masked_nll_requires_masked_positions():
    with pytest.raises(ValueError):
        masked_token_nll(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long),
                         torch.zeros(3, dtype=torch.bool))


def test_copy_through_is_zero_on_visible_chance_on_masked():
    K, N = 8, 10
    tokens = torch.randint(0, K, (N,))
    mask = torch.zeros(N, dtype=torch.bool); mask[:4] = True
    logits = copy_through_logits(tokens, mask, K)
    vis_loss = masked_token_nll(logits, tokens, ~mask)     # score on visible
    msk_loss = masked_token_nll(logits, tokens, mask)      # score on masked
    assert vis_loss < 1e-6                                  # copier perfect visible
    assert torch.isclose(msk_loss, torch.tensor(math.log(K)), atol=1e-6)  # chance


def test_diversity_loss_minimised_at_uniform_usage():
    K = 6
    uniform = torch.arange(K).repeat(4)                    # each code 4x
    collapsed = torch.zeros(24, dtype=torch.long)          # all one code
    assert codebook_diversity_loss(uniform, K) < codebook_diversity_loss(collapsed, K)
    assert torch.isclose(codebook_diversity_loss(uniform, K),
                         torch.tensor(-math.log(K)), atol=1e-6)


def test_masked_rotation_geodesic_zero_when_equal():
    d6 = _rand_6d(5)
    mask = torch.tensor([True, False, True, False, True])
    assert float(masked_rotation_geodesic(d6, d6.clone(), mask)) < 1e-9


def test_encoder_is_invariant_to_masked_token_identity():
    # MAE asymmetry: the encoder never reads masked tokens, so changing their
    # identity cannot change the model output.
    torch.manual_seed(1)
    K, T, P = 12, 6, 2
    model = MaskedMotionModel(K, dim=16, max_frames=T, num_parts=P,
                              heads=2, enc_layers=1, dec_layers=1).eval()
    positions = torch.tensor([[t, p] for t in range(T) for p in range(P)])
    tokens = torch.randint(0, K, (T * P,))
    mask = torch.zeros(T * P, dtype=torch.bool); mask[::3] = True
    with torch.no_grad():
        out_a = model(tokens, positions, mask)
        tampered = tokens.clone()
        tampered[mask] = (tampered[mask] + 5) % K          # change masked tokens
        out_b = model(tampered, positions, mask)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_output_depends_on_visible_tokens():
    torch.manual_seed(2)
    K, T, P = 12, 6, 2
    model = MaskedMotionModel(K, dim=16, max_frames=T, num_parts=P,
                              heads=2, enc_layers=1, dec_layers=1).eval()
    positions = torch.tensor([[t, p] for t in range(T) for p in range(P)])
    tokens = torch.randint(0, K, (T * P,))
    mask = torch.zeros(T * P, dtype=torch.bool); mask[::3] = True
    with torch.no_grad():
        out_a = model(tokens, positions, mask)
        tampered = tokens.clone()
        vis = ~mask
        tampered[vis] = (tampered[vis] + 3) % K            # change VISIBLE tokens
        out_b = model(tampered, positions, mask)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_masked_loss_has_gradient():
    torch.manual_seed(3)
    K, T, P = 10, 5, 2
    model = MaskedMotionModel(K, dim=16, max_frames=T, num_parts=P,
                              heads=2, enc_layers=1, dec_layers=1)
    positions = torch.tensor([[t, p] for t in range(T) for p in range(P)])
    tokens = torch.randint(0, K, (T * P,))
    mask = torch.zeros(T * P, dtype=torch.bool); mask[::2] = True
    logits = model(tokens, positions, mask)
    loss = masked_token_nll(logits, tokens, mask)
    loss.backward()
    assert model.mask_token.grad is not None
    assert model.head.weight.grad.abs().sum() > 0
