"""Shared sequence encoder support: absence is not a zero-valued observation."""
from copy import deepcopy

import pytest
import torch

from signtranslator.models.encoders import StubSpeechEncoder, StubTextEncoder, _masked_mean


def test_masked_mean_matches_hand_values_and_excludes_nan_gradients():
    x = torch.tensor([[[1., 5.], [float('nan'), float('inf')], [3., 9.]]],
                     requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    mean = _masked_mean(x, mask)
    torch.testing.assert_close(mean, torch.tensor([[2., 7.]]), rtol=0, atol=0)
    mean.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[[.5, .5], [0., 0.], [.5, .5]]]))
    with pytest.raises(ValueError, match='no available evidence'):
        _masked_mean(x, torch.zeros_like(mask))


@pytest.mark.parametrize('training', [True, False])
def test_speech_embedding_padding_values_and_gradients(training):
    torch.manual_seed(12)
    first = StubSpeechEncoder(3, 8, num_layers=2, num_heads=2, dropout=0).double().train(training)
    second = deepcopy(first)
    x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[1, 2] = False
    padded = torch.cat([x.detach(), torch.full((2, 3, 3), float('nan'), dtype=torch.float64)], 1)
    padded[1, 2] = float('inf')
    padded.requires_grad_()
    mask = torch.cat([valid, torch.zeros(2, 3, dtype=torch.bool)], 1)
    a, b = first(x, valid), second(padded, mask)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    # A nonconstant linear objective avoids LayerNorm's nearly constant squared norm.
    weights = torch.arange(8, dtype=torch.float64)
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    torch.testing.assert_close(x.grad, padded.grad[:, :5], rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(padded.grad[:, 5:]) == 0
    assert torch.count_nonzero(x.grad[1, 2]) == 0
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('training', [True, False])
def test_text_encoder_excludes_masked_ids_before_embedding(training):
    torch.manual_seed(13)
    first = StubTextEncoder(12, 8, num_layers=2, num_heads=2, dropout=0).double().train(training)
    second = deepcopy(first)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    masked_tokens = torch.cat([tokens, torch.full((2, 3), -999)], 1)
    valid = torch.tensor([[True, True, True, False, False, False]]).expand(2, -1)
    a, b = first(tokens), second(masked_tokens, valid)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    weights = torch.arange(8, dtype=torch.float64)
    (a * weights).sum().backward()
    (b * weights).sum().backward()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)
    memory, _ = second.encode_sequence(masked_tokens, valid)
    assert torch.count_nonzero(memory[:, 3:]) == 0


def test_sequence_domains_reject_empty_evidence_and_invalid_masks():
    text = StubTextEncoder(12, 8, num_layers=1, num_heads=2)
    speech = StubSpeechEncoder(3, 8, num_layers=1, num_heads=2)
    for encoder, values in ((text, torch.ones(2, 3, dtype=torch.long)),
                            (speech, torch.ones(2, 3, 3))):
        for mask in (torch.zeros(2, 3, dtype=torch.bool), torch.ones(2, 3),
                     torch.ones(2, 4, dtype=torch.bool)):
            with pytest.raises(ValueError):
                encoder(values, mask)
    with pytest.raises(ValueError, match='no available evidence'):
        text(torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match='vocabulary'):
        text(torch.ones(2, 3, dtype=torch.long) * 12)
    with pytest.raises(ValueError, match='nonpadding'):
        text(torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='finite'):
        speech(torch.full((2, 3, 3), float('nan')))
    with pytest.raises(ValueError, match='capacity'):
        text(torch.ones(1, 2049, dtype=torch.long))


def test_legacy_joint_api_routes_support_to_encoder_and_diffusion(monkeypatch):
    from test_pipeline import _small_model
    model, cfg = _small_model()
    pose = torch.randn(2, cfg.in_channels, 6, cfg.num_joints)
    pose[:, :, 4:] = float('nan')
    valid = torch.ones(2, 6, cfg.num_joints, dtype=torch.bool)
    valid[:, 4:] = False
    seen = []
    original = model.diffusion.denoiser.forward
    def forward(x, t, cond=None, **kwargs):
        seen.append(kwargs['motion_mask'].clone())
        return original(x, t, cond, **kwargs)
    monkeypatch.setattr(model.diffusion.denoiser, 'forward', forward)
    out = model(pose, torch.ones(2, 3, dtype=torch.long), validity_mask=valid)
    assert torch.isfinite(out['loss'])
    assert len(seen) == 1 and torch.equal(seen[0], valid)
    with pytest.raises(ValueError, match='no observed support'):
        model(pose, torch.ones(2, 3, dtype=torch.long), validity_mask=torch.zeros_like(valid))
