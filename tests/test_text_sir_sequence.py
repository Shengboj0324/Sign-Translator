"""Autoregression engineering, never empirical ASL validation."""
import pytest
import torch
from torch import nn

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from signtranslator.planning.label_sequence import label_sequence_targets
from signtranslator.planning.text_sequence import AutoregressiveTextSIRLabelModel
from signtranslator.training import Trainer
from test_text_sir_labels import setup
from test_governed_trainer import loader


def model(vocab, max_events=4):
    return AutoregressiveTextSIRLabelModel(
        vocab, declared_encoding=PLAINTEXT_ENCODING, max_bytes=64,
        max_events=max_events, embedding_dim=8, hidden_dim=16)


class ScriptedScores(nn.Module):
    """Control only decoder decisions for termination-boundary unit tests."""
    def __init__(self, steps, classes):
        super().__init__()
        self.steps, self.classes, self.index = steps, classes, 0

    def forward(self, hidden):
        selected = self.steps[self.index]
        self.index += 1
        scores = hidden.new_full((hidden.shape[0], self.classes), -10.)
        for row, value in enumerate(selected):
            scores[row, value] = 10.
        return scores


def test_stop_empty_and_capacity_are_distinct_with_no_supplied_event_counts(tmp_path):
    vocab, _ = setup(tmp_path)
    net = model(vocab, max_events=2)
    net.classifier = ScriptedScores([[1, 0, 2], [2, 0, 3], [0, 0, 1]], 4)
    before = torch.get_rng_state().clone()
    result = net.generate(torch.tensor([[66], [67], [68]]), torch.tensor([1, 1, 1]))
    assert [r.status for r in result] == ['terminated', 'empty_prediction', 'capacity_exceeded']
    assert result[0].labels == (vocab.entries[0], vocab.entries[1])
    assert result[1].labels == result[1].diagnostic_prefix == ()
    assert result[2].labels == () and len(result[2].diagnostic_prefix) == 2
    assert net.training and torch.equal(before, torch.get_rng_state())


def test_causal_teacher_forcing_cannot_see_future_labels(tmp_path):
    vocab, _ = setup(tmp_path)
    torch.manual_seed(4)
    net = model(vocab)
    source, lengths = torch.tensor([[66, 67]]), torch.tensor([2])
    a = net(source, lengths, torch.tensor([[1, 2, 3, 4]]), torch.tensor([4]))
    b = net(source, lengths, torch.tensor([[1, 2, 4, 2]]), torch.tensor([4]))
    torch.testing.assert_close(a[:, :2], b[:, :2], rtol=0, atol=0)
    assert not torch.equal(a[:, 2:], b[:, 2:])
    padded = net(torch.tensor([[66, 67, 0]]), lengths,
                 torch.tensor([[1, 2, 3, 0]]), torch.tensor([3]))
    unpadded = net(source, lengths, torch.tensor([[1, 2, 3]]), torch.tensor([3]))
    torch.testing.assert_close(padded[:, :3], unpadded, rtol=0, atol=0)
    assert not padded[:, 3].any()


def test_actual_trainer_learns_fixture_sequence_and_stop_then_reloads(tmp_path):
    vocab, corpus = setup(tmp_path)
    torch.manual_seed(9)
    cfg = TrainerConfig(epochs=60, lr=.05, loss_weights={'sir_sequence': 1.},
                        selection_metric='sir_sequence', seed=9)
    train, val = loader(corpus, 'train'), loader(corpus, 'val')
    net = model(vocab)
    trainer = Trainer(net, cfg, train, val)
    initial = trainer.validate()['sir_sequence']
    trainer.fit()
    assert trainer.validate()['sir_sequence'] < initial * .1
    batch = next(iter(train))
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=PLAINTEXT_ENCODING, max_bytes=64)
    result = net.generate(text.token_ids, text.lengths)
    assert result[0].status == 'terminated'
    assert result[0].labels == vocab.entries
    path = trainer.save(tmp_path / 'sequence.pt')
    restored = Trainer(model(vocab), cfg, train, val)
    restored.load(path)
    assert restored.model.generate(text.token_ids, text.lengths) == result
    assert restored.validate() == trainer.validate()


@pytest.mark.parametrize('inputs,lengths', [([[2, 2]], [2]), ([[1, 1]], [2]),
                                           ([[1, 0]], [2]), ([[1, 5]], [2]),
                                           ([[1, 2, 3]], [2]), ([[1, 2]], [1])])
def test_bad_teacher_shift_or_padding_fails(tmp_path, inputs, lengths):
    vocab, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        model(vocab)(torch.tensor([[66]]), torch.tensor([1]),
                     torch.tensor(inputs), torch.tensor(lengths))


def test_over_capacity_training_is_not_truncated(tmp_path):
    vocab, corpus = setup(tmp_path)
    with pytest.raises(ValueError, match='capacity'):
        model(vocab, max_events=2).training_step(next(iter(loader(corpus, 'train'))),
                                                weights={'sir_sequence': 1.})


def test_nonfinite_decode_restores_mode_and_returns_no_candidate(tmp_path):
    vocab, _ = setup(tmp_path)
    net = model(vocab)
    with torch.no_grad():
        net.classifier.bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        net.generate(torch.tensor([[66]]), torch.tensor([1]))
    assert net.training
