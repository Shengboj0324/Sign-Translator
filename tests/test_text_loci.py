import torch

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_text import PLAINTEXT_ENCODING, encode_plaintext_transcripts
from signtranslator.planning.text_loci import LocusTextSIRModel
from signtranslator.planning.locus_assignment import decode_locus_assignment
from signtranslator.training import Trainer
from test_loci import fixture
from test_governed_trainer import loader


WEIGHTS = dict(sir_sequence=1., event_timing=1., sir_relations=1., referent_equality=1., locus_assignment=1.)


def model(v, alphabet):
    return LocusTextSIRModel(v, locus_alphabet=alphabet, declared_encoding=PLAINTEXT_ENCODING,
                            max_bytes=64, max_events=4, embedding_dim=8, hidden_dim=16, timing_scale_seconds=.5)


def test_joint_locus_learning_source_inference_and_checkpoint(tmp_path):
    v, corpus, alphabet, _ = fixture(tmp_path)
    torch.manual_seed(8)
    net = model(v, alphabet)
    cfg = TrainerConfig(epochs=65, lr=.03, loss_weights=WEIGHTS, selection_metric='locus_assignment')
    trainer = Trainer(net, cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    initial = trainer.validate()['locus_assignment']
    trainer.fit()
    assert trainer.validate()['locus_assignment'] < initial * .1
    batch = next(iter(loader(corpus, 'train')))
    objective = net.training_step(batch, weights=WEIGHTS)
    assert {name: term.support_mask for name, term in objective.terms.items()} == {
        'sir_sequence': (True,), 'event_timing': (True,), 'sir_relations': (True,),
        'referent_equality': (False,), 'locus_assignment': (True,)}
    expected = objective['locus_assignment']
    batch.sir_targets.locus_ids.fill_(999)
    assert torch.equal(expected, net.training_step(batch, weights=WEIGHTS)['locus_assignment'])
    text = encode_plaintext_transcripts(batch.transcript_payloads, batch.annotations,
                                       declared_encoding=PLAINTEXT_ENCODING, max_bytes=64)
    def generate(n):
        return n.generate_loci(text.token_ids, text.lengths,
                               origins_seconds=torch.tensor([0.], dtype=torch.float64))[0]
    result = generate(net)
    assert result.locus_logits.shape == (3, 2)
    assert result.locus_logits.argmax(-1)[[0, 2]].tolist() == [0, 1]
    assert result.convention_sha256 == alphabet.convention.sha256
    assert result.locus_identities == alphabet.identities
    assigned = decode_locus_assignment(result, alphabet, referents=(0, None, 1),
                                        place=(True, False, True), max_work=100)
    assert assigned.loci == (0, None, 1)
    assert assigned.status == 'unique_optimum_candidate'
    path = trainer.save(tmp_path / 'loci.pt')
    restored = Trainer(model(v, alphabet), cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    restored.load(path)
    assert torch.equal(generate(restored.model).locus_logits, result.locus_logits)
    assert restored.validate() == trainer.validate()


def test_absent_loci_leave_exclusive_head_unchanged(tmp_path):
    v, corpus, alphabet, _ = fixture(tmp_path, loci=(None, None, None))
    net = model(v, alphabet)
    trainer = Trainer(net, TrainerConfig(epochs=1, loss_weights=WEIGHTS, selection_metric='sir_sequence'),
                      loader(corpus, 'train'), loader(corpus, 'val'))
    before = [p.detach().clone() for p in net.locus_head.parameters()]
    trainer.fit()
    assert 'locus_assignment' not in trainer.validate()
    assert trainer.last_validation_support['locus_assignment'] == 0
    assert trainer.optimizer_exposure[0]['support_membership']['locus_assignment'] == [False]
    for old, p in zip(before, net.locus_head.parameters()):
        assert torch.equal(old, p) and p.grad is None
