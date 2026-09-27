"""Training and inference invariants for unavailable ST-GCN observations."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from signtranslator.models.masked_normalization import masked_batch_norm
from signtranslator.models.stgcn import STGCNEncoder


def encoder():
    torch.manual_seed(5)
    return STGCNEncoder(2, np.ones((1, 3, 3), dtype=np.float32) / 3,
                       channels=(4, 6), temporal_kernel=3).double()


def test_masked_batch_norm_matches_hand_moments_and_buffer_updates():
    bn = nn.BatchNorm1d(2, momentum=0.25).double()
    x = torch.tensor([[[1., 3., float('nan')], [5., float('nan'), float('nan')]]],
                     dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[[True, True, False], [True, False, False]]])
    out = masked_batch_norm(x, valid, bn)
    expected = torch.tensor([[[-1., 1., 0.], [0., 0., 0.]]], dtype=torch.float64)
    expected[:, 0] /= (1 + bn.eps) ** .5
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(bn.running_mean, torch.tensor([.5, 1.25], dtype=torch.float64))
    torch.testing.assert_close(bn.running_var, torch.tensor([1.25, 1.], dtype=torch.float64))
    out.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~valid]) == 0


def test_unsupported_normalization_channel_does_not_update_statistics():
    bn = nn.BatchNorm1d(2).double()
    x = torch.tensor([[[1., 3.], [float('nan'), float('inf')]]], dtype=torch.float64)
    valid = torch.tensor([[[True, True], [False, False]]])
    out = masked_batch_norm(x, valid, bn)
    assert bn.running_mean[1] == 0 and bn.running_var[1] == 1
    assert torch.count_nonzero(out[:, 1]) == 0


@pytest.mark.parametrize('training', [True, False])
def test_padding_cannot_change_features_gradients_or_running_statistics(training):
    first = encoder().train(training)
    padded_model = deepcopy(first)
    raw = torch.randn(2, 2, 5, 3, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    valid[:, 2, 1] = False
    padded = torch.cat([raw.detach(), torch.full((2, 2, 4, 3), float('nan'),
                                              dtype=torch.float64)], dim=2).requires_grad_()
    padded_valid = torch.cat([valid, torch.zeros(2, 4, 3, dtype=torch.bool)], dim=1)
    a = first(raw, validity_mask=valid)
    b = padded_model(padded, validity_mask=padded_valid)
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(raw.grad, padded.grad[:, :, :5], rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(padded.grad[:, :, 5:]) == 0
    assert torch.count_nonzero(raw.grad[:, :, 2, 1]) == 0
    for name, buffer in first.named_buffers():
        torch.testing.assert_close(buffer, dict(padded_model.named_buffers())[name],
                                   rtol=1e-10, atol=1e-10)
    for left, right in zip(first.parameters(), padded_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)


def test_all_true_support_matches_legacy_encoder():
    a = encoder().train()
    b = deepcopy(a)
    x = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    torch.testing.assert_close(a(x), b(x, validity_mask=torch.ones(2, 5, 3, dtype=torch.bool)))


def test_occluded_values_and_zero_confidence_are_excluded():
    a = encoder().eval()
    x = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    confidence = torch.ones(2, 5, 3, dtype=torch.float64)
    confidence[:, 1:3, 1] = 0
    altered = x.clone()
    altered[:, :, 1:3, 1] = float('nan')
    torch.testing.assert_close(a(x, confidence=confidence), a(altered, confidence=confidence),
                               rtol=0, atol=0)


def test_support_pooling_matches_frame_mean_and_rejects_empty_sample():
    model = encoder().eval()
    x = torch.randn(2, 2, 5, 3, dtype=torch.float64)
    valid = torch.ones(2, 5, 3, dtype=torch.bool)
    valid[0, 3:] = False
    seq = model(x, return_sequence=True, validity_mask=valid)
    pooled = model(x, validity_mask=valid)
    torch.testing.assert_close(pooled[0], seq[0, :3].mean(0))
    assert torch.count_nonzero(seq[0, 3:]) == 0
    valid[0] = False
    with pytest.raises(ValueError, match='no observed support'):
        model(x, validity_mask=valid)


def test_frame_mask_must_be_prefix():
    with pytest.raises(ValueError, match='prefix'):
        encoder()(torch.ones(1, 2, 3, 3, dtype=torch.float64),
                  frame_mask=torch.tensor([[True, False, True]]))


def test_active_recognition_training_is_padding_invariant(tmp_path):
    from test_trainer import _tiny_setup
    first, train, _ = _tiny_setup(tmp_path)
    second = deepcopy(first)
    original = next(iter(train))
    keys = ('pose', 'ctc_targets', 'ctc_lengths', 'ctc_input_lengths',
            'validity_mask', 'confidence', 'frame_mask')
    batch = {k: original[k] for k in keys}
    padded = {k: v.clone() for k, v in batch.items()}
    n, c, _, v = batch['pose'].shape
    padded['pose'] = torch.cat([batch['pose'], torch.full((n, c, 4, v), float('nan'))], 2)
    for key in ('validity_mask', 'confidence', 'frame_mask'):
        value = batch[key]
        padding_shape = (n, 4) + value.shape[2:]
        padded[key] = torch.cat([value, torch.zeros(padding_shape, dtype=value.dtype)], 1)
    a = first.training_step(batch)['recognition']
    b = second.training_step(padded)['recognition']
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    a.backward()
    b.backward()
    for left, right in zip(first.recognizer.parameters(), second.recognizer.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-5)
    padded['ctc_input_lengths'] += 4
    with pytest.raises(ValueError, match='agree with frame_mask'):
        second.training_step(padded)


def test_decode_excludes_padded_classifier_outputs(monkeypatch):
    from signtranslator.models.recognition import SignRecognizer
    recognizer = SignRecognizer(encoder(), num_glosses=2)
    log_probs = torch.tensor([[[0., 8., 0.], [0., 8., 0.], [0., 0., 8.]]])
    monkeypatch.setattr(recognizer, 'forward', lambda pose, **support: log_probs)
    pose = torch.ones(1, 2, 3, 3)
    assert recognizer.decode(pose) == [[1, 2]]
    assert recognizer.decode(pose, frame_mask=torch.tensor([[True, True, False]])) == [[1]]
