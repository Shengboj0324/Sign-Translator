"""Joint temporal baseline checks, never real ASL acceptance."""
import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from signtranslator.planning.text_timing import TemporalTextSIRModel
from signtranslator.training import Trainer
from test_text_sir_labels import setup
from test_governed_trainer import loader
from test_text_sir_sequence import ScriptedScores


def model(vocab):
    return TemporalTextSIRModel(vocab, declared_encoding=PLAINTEXT_ENCODING,
                                max_bytes=64, max_events=4, embedding_dim=8, hidden_dim=16,
                                timing_scale_seconds=.5)


def test_joint_training_reduces_both_losses_and_reload_preserves_temporal_candidates(tmp_path):
    vocab, corpus = setup(tmp_path)
    torch.manual_seed(3)
    net = model(vocab)
    cfg = TrainerConfig(epochs=60, lr=.03, loss_weights={'sir_sequence': 1., 'event_timing': 1.},
                        selection_metric='sir_sequence')
    train, val = loader(corpus, 'train'), loader(corpus, 'val')
    trainer = Trainer(net, cfg, train, val)
    initial = trainer.validate()
    trainer.fit()
    final = trainer.validate()
    assert final['sir_sequence'] < initial['sir_sequence'] * .2
    assert final['event_timing'] < initial['event_timing'] * .2
    batch = next(iter(train))
    assert batch.annotation_extents == ((0., 1.),)
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=PLAINTEXT_ENCODING, max_bytes=64)
    result = net.generate_temporal(text.token_ids, text.lengths,
                                   origins_seconds=torch.tensor([0.], dtype=torch.float64))
    assert result[0].status == 'terminated'
    assert [event.label for event in result[0].events] == list(vocab.entries)
    assert all(event.start_seconds >= 0 and event.end_seconds > event.start_seconds
               for event in result[0].events)
    path = trainer.save(tmp_path / 'timed.pt')
    restored = Trainer(model(vocab), cfg, train, val)
    restored.load(path)
    assert restored.model.generate_temporal(text.token_ids, text.lengths,
        origins_seconds=torch.tensor([0.], dtype=torch.float64)) == result
    assert restored.validate() == final


def test_generated_timing_uses_declared_clock_and_own_predicted_labels(tmp_path):
    vocab, _ = setup(tmp_path)
    net = model(vocab)
    # No annotation or reference event count is supplied to generation.
    def generate(origin):
        net.classifier = ScriptedScores([[1], [2], [0]], 4)
        return net.generate_temporal(torch.tensor([[66]]), torch.tensor([1]),
                                     origins_seconds=torch.tensor([origin], dtype=torch.float64))[0]
    first, shifted = generate(-10.), generate(20.)
    assert first.status == shifted.status == 'terminated'
    assert net.training
    for a, b in zip(first.events, shifted.events):
        assert b.start_seconds - a.start_seconds == pytest.approx(30.)
        assert b.end_seconds - a.end_seconds == pytest.approx(30.)
        assert a.label == b.label


def test_empty_label_prediction_does_not_fabricate_a_timed_event(tmp_path):
    vocab, _ = setup(tmp_path)
    net = model(vocab)
    net.classifier = ScriptedScores([[0]], 4)
    result = net.generate_temporal(torch.tensor([[66]]), torch.tensor([1]),
                                   origins_seconds=torch.tensor([0.], dtype=torch.float64))
    assert result[0].status == 'empty_prediction' and result[0].events == ()


def test_generated_clock_must_be_explicit_float64(tmp_path):
    vocab, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='float64 origins'):
        model(vocab).generate_temporal(torch.tensor([[66]]), torch.tensor([1]),
                                       origins_seconds=torch.tensor([0.]))
