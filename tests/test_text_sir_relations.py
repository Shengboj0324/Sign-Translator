"""Trainable relation branch with fictional, partially observed graph evidence."""
import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from signtranslator.planning.text_relations import RelationalTextSIRModel
from signtranslator.planning.tensors import EDGE_TYPES
from signtranslator.grammar.sir import EdgeType
from signtranslator.training import Trainer
from test_text_sir_labels import setup
from test_governed_trainer import loader
from test_text_sir_sequence import ScriptedScores


def model(vocab):
    return RelationalTextSIRModel(vocab, declared_encoding=PLAINTEXT_ENCODING,
                                  max_bytes=64, max_events=4, embedding_dim=8, hidden_dim=16,
                                  timing_scale_seconds=.5)


def test_joint_relation_optimization_and_raw_candidate_reload(tmp_path):
    vocab, corpus = setup(tmp_path)
    torch.manual_seed(8)
    net = model(vocab)
    weights = {'sir_sequence': 1., 'event_timing': 1., 'sir_relations': 1.}
    cfg = TrainerConfig(epochs=60, lr=.03, loss_weights=weights, selection_metric='sir_relations')
    train, val = loader(corpus, 'train'), loader(corpus, 'val')
    trainer = Trainer(net, cfg, train, val)
    initial = trainer.validate()
    trainer.fit()
    final = trainer.validate()
    assert final['sir_relations'] < initial['sir_relations'] * .25
    assert final['sir_sequence'] < initial['sir_sequence'] * .25
    assert final['event_timing'] < initial['event_timing'] * .25
    batch = next(iter(train))
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=PLAINTEXT_ENCODING, max_bytes=64)
    def generate(candidate):
        return candidate.generate_relational(text.token_ids, text.lengths,
                                             origins_seconds=torch.tensor([0.], dtype=torch.float64))[0]
    result = generate(net)
    assert result.temporal.status == 'terminated'
    assert result.relation_logits.shape == (3, 3, 5)
    assert not result.relation_valid.diagonal().any()
    # The fixture has a recorded nonmanual -> manual scope edge at 1 -> 0.
    assert result.relation_logits[1, 0, EDGE_TYPES.index(EdgeType.SCOPE)].sigmoid() > .7
    path = trainer.save(tmp_path / 'relation.pt')
    restored = Trainer(model(vocab), cfg, train, val)
    restored.load(path)
    actual = generate(restored.model)
    assert actual.temporal == result.temporal
    assert torch.equal(actual.relation_logits, result.relation_logits)
    assert restored.validate() == final


def test_cached_edges_and_references_do_not_enter_model_inputs(tmp_path):
    vocab, corpus = setup(tmp_path)
    net = model(vocab)
    batch = next(iter(loader(corpus, 'train')))
    weights = {'sir_sequence': 1., 'event_timing': 1., 'sir_relations': 1.}
    expected = net.training_step(batch, weights=weights)['sir_relations']
    batch.sir_targets.edge_indices.fill_(9999)
    batch.sir_targets.edge_types.fill_(9999)
    batch.sir_targets.referent_ids.fill_(9999)
    assert torch.equal(expected, net.training_step(batch, weights=weights)['sir_relations'])


def test_empty_generation_has_no_fabricated_relation_scores(tmp_path):
    vocab, _ = setup(tmp_path)
    net = model(vocab)
    net.classifier = ScriptedScores([[0]], 4)
    result = net.generate_relational(torch.tensor([[66]]), torch.tensor([1]),
                                     origins_seconds=torch.tensor([0.], dtype=torch.float64))[0]
    assert result.temporal.status == 'empty_prediction'
    assert result.relation_logits is None and result.relation_valid is None
    assert net.training


def test_relation_objective_cannot_be_implicitly_disabled(tmp_path):
    vocab, corpus = setup(tmp_path)
    with pytest.raises(ValueError, match='objective weights'):
        model(vocab).training_step(next(iter(loader(corpus, 'train'))),
                                   weights={'sir_sequence': 1., 'event_timing': 1.})
