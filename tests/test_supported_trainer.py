"""Real governed mixed-support training; reviewer evidence remains fictional."""
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.config import TrainerConfig
from signtranslator.data.governed_corpus import collate_governed_motion
from signtranslator.training import Trainer
from test_text_sir_labels import setup
from test_text_sir_relations import model


def mixed(tmp_path):
    return setup(tmp_path, configurations=[{}, {}, {}, {'split': 'val'}, {'split': 'val'}, {'split': 'val'}],
                 single_event_indices=(0, 2, 3, 5))


def loader(corpus, split, size=2):
    return DataLoader(corpus.split(split), batch_size=size, collate_fn=collate_governed_motion)


def config(epochs=2, metric='sir_sequence'):
    return TrainerConfig(epochs=epochs, lr=.03, weight_decay=.1, selection_metric=metric,
                         loss_weights={'sir_sequence': 1., 'event_timing': 1., 'sir_relations': 1.})


def test_validation_uses_branch_support_not_batch_size(tmp_path):
    vocab, corpus = mixed(tmp_path)
    net = model(vocab)
    a = Trainer(net, config(), loader(corpus, 'train'), loader(corpus, 'val', 1))
    b = Trainer(net, config(), loader(corpus, 'train'), loader(corpus, 'val', 2))
    first, second = a.validate(), b.validate()
    assert first == pytest.approx(second, abs=1e-6)
    assert first['total'] == pytest.approx(first['sir_sequence'] + first['event_timing']
                                           + first['sir_relations']/3, abs=1e-6)
    assert a.last_validation_support == {'sir_sequence': 3, 'event_timing': 3,
                                          'sir_relations': 1, 'total': 3}


def test_unsupported_last_batch_does_not_apply_relation_momentum_or_decay(tmp_path):
    vocab, corpus = mixed(tmp_path)
    net = model(vocab)
    trainer = Trainer(net, config(), loader(corpus, 'train'))
    original = net.training_step
    captured = {}
    def record(batch, *, weights):
        output = original(batch, weights=weights)
        if output.terms['sir_relations'].supported_examples == 0:
            captured['parameters'] = [p.detach().clone() for p in net.relation_head.parameters()]
            captured['state'] = [deepcopy(trainer.opt.state[p]) for p in net.relation_head.parameters()]
        return output
    net.training_step = record
    trainer.train_epoch()
    assert trainer.global_step == 2 and trainer.last_train_support['total'] == 3
    for parameter, expected, state in zip(net.relation_head.parameters(), captured['parameters'], captured['state']):
        assert torch.equal(parameter, expected) and parameter.grad is None
        assert state  # momentum was established by the earlier supported batch
        for key, value in state.items():
            assert torch.equal(trainer.opt.state[parameter][key], value)


def test_unavailable_validation_metric_is_omitted_and_cannot_select_checkpoint(tmp_path):
    vocab, corpus = setup(tmp_path, single_event_indices=(0, 1))
    trainer = Trainer(model(vocab), config(metric='sir_relations'), loader(corpus, 'train'), loader(corpus, 'val'))
    assert 'sir_relations' not in trainer.validate()
    assert trainer.last_validation_support['sir_relations'] == 0
    with pytest.raises(ValueError, match='selection metric.*unavailable'):
        trainer.fit()
    assert trainer.best_model_state is None
    assert trainer.history['val_support_sir_relations'] == [0]


def test_mixed_support_resume_preserves_parameters_and_denominator_history(tmp_path):
    vocab, corpus = mixed(tmp_path)
    torch.manual_seed(19)
    initial = model(vocab)
    def build():
        return Trainer(deepcopy(initial), config(epochs=3), loader(corpus, 'train'), loader(corpus, 'val'))
    full = build(); full.fit()
    partial = build(); partial.fit(max_epochs=1)
    path = partial.save(tmp_path / 'support.pt')
    resumed = build(); resumed.load(path); resumed.fit()
    assert resumed.history == full.history
    assert resumed.history['val_support_sir_relations'] == [1, 1, 1]
    assert resumed.history['val_sir_relations_epoch'] == [1, 2, 3]
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])


def test_exposure_records_order_support_and_survive_resume(tmp_path):
    vocab, corpus = mixed(tmp_path)
    initial = model(vocab)
    def build():
        return Trainer(deepcopy(initial), config(epochs=2), loader(corpus, 'train'))
    full = build(); full.fit()
    partial = build(); partial.fit(max_epochs=1)
    records = partial.optimizer_exposure
    batches = list(loader(corpus, 'train'))
    assert [r['sample_ids'] for r in records] == [list(b.motion.sample_ids) for b in batches]
    assert [r['annotation_sha256'] for r in records] == [
        [a.content_sha256() for a in b.annotations] for b in batches]
    assert [r['support']['sir_relations'] for r in records] == [1, 0]
    records[0]['support']['sir_relations'] = 999
    assert partial.optimizer_exposure[0]['support']['sir_relations'] == 1
    path = partial.save(tmp_path / 'exposure.pt')
    resumed = build(); resumed.load(path); resumed.fit()
    assert resumed.optimizer_exposure == full.optimizer_exposure
    assert [r['step'] for r in resumed.optimizer_exposure] == [1, 2, 3, 4]


def test_failed_optimizer_call_is_not_recorded(tmp_path, monkeypatch):
    vocab, corpus = mixed(tmp_path)
    trainer = Trainer(model(vocab), config(), loader(corpus, 'train'))
    def fail(*args, **kwargs):
        raise RuntimeError('before optimizer mutation')
    monkeypatch.setattr(trainer.opt, 'step', fail)
    with pytest.raises(RuntimeError, match='before optimizer mutation'):
        trainer.train_epoch()
    assert trainer.global_step == 0 and trainer.optimizer_exposure == []


def test_returned_optimizer_step_recorded_when_scheduler_fails(tmp_path, monkeypatch):
    vocab, corpus = mixed(tmp_path)
    trainer = Trainer(model(vocab), config(), loader(corpus, 'train'))
    def fail(self):
        raise RuntimeError('scheduler failure')
    # Inject the call failure without adding a function to the scheduler's
    # serialized instance state, which is now checked before optimization.
    monkeypatch.setattr(type(trainer.sched), 'step', fail)
    with pytest.raises(RuntimeError, match='scheduler failure'):
        trainer.train_epoch()
    assert trainer.global_step == 1 and len(trainer.optimizer_exposure) == 1
