"""Acoustic prefix support through subsampling, attention, CTC and decoding."""
from copy import deepcopy

import pytest
import torch

from signtranslator.models.speech import SpeechRecognizer


def make(subsample):
    torch.manual_seed(3)
    return SpeechRecognizer(4, 3, hidden_dim=8, num_layers=2, num_heads=2,
                            dropout=0, subsample=subsample).double()


@pytest.mark.parametrize('subsample', [1, 2, 4])
@pytest.mark.parametrize('training', [True, False])
def test_padding_preserves_features_loss_and_gradients(subsample, training):
    model = make(subsample).train(training)
    padded_model = deepcopy(model)
    raw = torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True)
    lengths = torch.tensor([13, 9])
    padded = torch.cat([raw.detach(), torch.full((2, 5, 4), float('nan'),
                                               dtype=torch.float64)], 1).requires_grad_()
    # Alter padding inside the original batch too, not only the appended suffix.
    with torch.no_grad():
        padded[1, 9:] = float('nan')
    a = model.encode(raw, lengths)
    b = padded_model.encode(padded, lengths)
    torch.testing.assert_close(a, b[:, :a.shape[1]], rtol=1e-10, atol=1e-10)
    out_lengths = model.output_lengths(lengths)
    for i, length in enumerate(out_lengths.tolist()):
        assert torch.count_nonzero(b[i, length:]) == 0
        single = model.encode(raw[i:i+1, :lengths[i]])
        torch.testing.assert_close(a[i, :length], single[0], rtol=1e-10, atol=1e-10)
    targets = torch.tensor([[1, 2], [2, 3]])
    target_lengths = torch.tensor([2, 2])
    la = model.loss(raw, targets, target_lengths, lengths)
    lb = padded_model.loss(padded, targets, target_lengths, lengths)
    torch.testing.assert_close(la, lb, rtol=1e-10, atol=1e-10)
    la.backward()
    lb.backward()
    torch.testing.assert_close(raw.grad, padded.grad[:, :13], rtol=1e-9, atol=1e-9)
    assert torch.count_nonzero(raw.grad[1, 9:]) == 0
    assert torch.count_nonzero(padded.grad[:, 13:]) == 0
    for left, right in zip(model.parameters(), padded_model.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('subsample', [1, 2, 4])
def test_subsampling_odd_lengths_and_integer_extremes(subsample):
    model = make(subsample).eval()
    for length in range(1, 18):
        expected = (length + subsample - 1) // subsample
        assert model.output_lengths(torch.tensor([length])).item() == expected
        assert model(torch.ones(1, length, 4, dtype=torch.float64)).shape[1] == expected
    maximum = torch.iinfo(torch.int64).max
    assert model.output_lengths(torch.tensor([maximum])).item() == (maximum + subsample - 1) // subsample


@pytest.mark.parametrize('lengths', [torch.tensor([0]), torch.tensor([-1]),
                                    torch.tensor([6]), torch.tensor([3.]),
                                    torch.tensor([True]), torch.tensor([[3]]),
                                    torch.tensor([3, 3])])
def test_invalid_lengths_rejected_before_processing(lengths):
    model = make(2)
    with pytest.raises(ValueError):
        model(torch.ones(1, 5, 4, dtype=torch.float64), lengths)


def test_unmasked_nan_is_rejected_and_decode_discards_padded_tokens(monkeypatch):
    model = make(1)
    x = torch.ones(1, 5, 4, dtype=torch.float64)
    x[:, 3:] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(x)
    model(x, torch.tensor([3]))
    logits = torch.tensor([[[0., 8., 0., 0.]] * 3 + [[0., 0., 8., 0.]] * 2])
    monkeypatch.setattr(model, 'forward', lambda features, input_lengths=None: logits)
    assert model.decode(x, torch.tensor([3])) == [[1]]


def test_active_speech_training_passes_true_lengths(tmp_path):
    from test_trainer import _tiny_setup
    model, _, _ = _tiny_setup(tmp_path)
    # Deterministic acoustic model isolates propagation from stochastic dropout.
    model.speech_recognizer = make(2)
    speech = torch.randn(2, 9, 4, dtype=torch.float64)
    batch = dict(speech=speech, speech_input_lengths=torch.tensor([9, 7]),
                 speech_ctc_targets=torch.tensor([1, 2, 2, 3]),
                 speech_ctc_lengths=torch.tensor([2, 2]))
    before = model.training_step(batch)['speech']
    batch['speech'] = torch.cat([speech, torch.full((2, 4, 4), float('nan'),
                                                   dtype=torch.float64)], 1)
    after = model.training_step(batch)['speech']
    torch.testing.assert_close(before, after, rtol=1e-10, atol=1e-10)
