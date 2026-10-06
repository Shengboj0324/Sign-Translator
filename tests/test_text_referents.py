import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from signtranslator.planning.text_referents import ReferentTextSIRModel
from signtranslator.planning.referent_partition import decode_referent_partition
from signtranslator.training import Trainer
from test_text_sir_labels import setup
from test_governed_trainer import loader


WEIGHTS = dict(sir_sequence=1., event_timing=1., sir_relations=1., referent_equality=1.)


def model(vocab):
    return ReferentTextSIRModel(vocab, declared_encoding=PLAINTEXT_ENCODING, max_bytes=64,
                                max_events=4, embedding_dim=8, hidden_dim=16, timing_scale_seconds=.5)


def test_joint_referent_learning_symmetry_inference_and_checkpoint(tmp_path):
    v, corpus = setup(tmp_path, referents=(0, 0, 17))
    torch.manual_seed(8)
    net = model(v)
    cfg = TrainerConfig(epochs=65, lr=.03, loss_weights=WEIGHTS, selection_metric='referent_equality')
    trainer = Trainer(net, cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    initial = trainer.validate()['referent_equality']
    trainer.fit()
    assert trainer.validate()['referent_equality'] < initial * .15
    batch = next(iter(loader(corpus, 'train')))
    objective = net.training_step(batch, weights=WEIGHTS)
    assert objective.terms['referent_equality'].support_mask == (True,)
    expected = objective['referent_equality']
    batch.sir_targets.referent_ids.fill_(888)
    assert torch.equal(expected, net.training_step(batch, weights=WEIGHTS)['referent_equality'])
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=PLAINTEXT_ENCODING, max_bytes=64)
    def generate(n):
        return n.generate_referents(text.token_ids, text.lengths,
                                    origins_seconds=torch.tensor([0.], dtype=torch.float64))[0]
    candidate = generate(net)
    scores = candidate.equality_logits
    assert candidate.relational.temporal.status == 'terminated'
    assert scores.shape == (3, 3) and torch.equal(scores, scores.T)
    assert scores[0, 1].sigmoid() > .8 and scores[0, 2].sigmoid() < .2
    assert not candidate.pair_valid.diagonal().any()
    partition = decode_referent_partition(candidate, max_events=4, max_search_nodes=100)
    assert partition.referents == (0, 0, 1)
    assert partition.status == 'unique_optimum_candidate'
    path = trainer.save(tmp_path / 'references.pt')
    restored = Trainer(model(v), cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    restored.load(path)
    assert torch.equal(generate(restored.model).equality_logits, scores)
    assert restored.validate() == trainer.validate()


def test_missing_references_do_not_train_exclusive_head_or_fabricate_metric(tmp_path):
    v, corpus = setup(tmp_path)
    net = model(v)
    trainer = Trainer(net, TrainerConfig(epochs=1, loss_weights=WEIGHTS, selection_metric='sir_sequence'),
                      loader(corpus, 'train'), loader(corpus, 'val'))
    before = [p.detach().clone() for p in net.referent_head.parameters()]
    trainer.fit()
    assert 'referent_equality' not in trainer.validate()
    assert trainer.last_validation_support['referent_equality'] == 0
    for old, p in zip(before, net.referent_head.parameters()):
        assert torch.equal(old, p) and p.grad is None
