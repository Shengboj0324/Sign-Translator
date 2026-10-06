"""The recorded governed objective must remain the one actually executed."""
from dataclasses import replace

import pytest
import torch

from test_epoch_commit import make
from test_supported_trainer import mixed


@pytest.mark.parametrize('operation', ['fit', 'validate', 'save', 'resume', 'weights', 'exposure_report'])
def test_changed_timing_scale_refused_before_effects(tmp_path, operation):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    checkpoint = trainer.save(tmp_path / 'original.pt')
    trainer.model.model_cfg = replace(trainer.model.model_cfg, timing_scale_seconds=1.)
    parameters = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    rng = torch.get_rng_state().clone()
    destination = tmp_path / 'must-not-exist' / 'changed.pt'
    with pytest.raises(ValueError, match='model contract changed'):
        if operation == 'save':
            trainer.save(destination)
        elif operation in ('resume', 'weights'):
            trainer.load(checkpoint, mode=operation)
        else:
            getattr(trainer, operation)()
    assert all(torch.equal(parameters[name], p) for name, p in trainer.model.named_parameters())
    assert torch.equal(rng, torch.get_rng_state())
    assert trainer.global_step == 0 and not trainer.opt.state
    assert not destination.parent.exists()


@pytest.mark.parametrize('field,value', [('lr', .01), ('epochs', 3), ('grad_clip', 2.),
                                         ('seed', 123), ('val_every', 2)])
def test_changed_trainer_configuration_is_refused(tmp_path, field, value):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    setattr(trainer.cfg, field, value)
    with pytest.raises(ValueError, match='trainer configuration changed'):
        trainer.fit()
    assert trainer.global_step == 0 and not trainer.opt.state


def test_mutable_weight_dictionary_cannot_relabel_training(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.cfg.loss_weights['event_timing'] = 2.
    with pytest.raises(ValueError, match='trainer configuration changed'):
        trainer.save(tmp_path / 'false-config.pt')
    assert not (tmp_path / 'false-config.pt').exists()


def test_public_contract_mutation_cannot_override_binding(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.model_contract['configs'].clear()
    with pytest.raises(ValueError, match='model contract changed'):
        trainer.fit()


@pytest.mark.parametrize('field', ['governed_batch_schema_version', 'governed_objective_schema_version'])
def test_boolean_schema_is_not_integer_one(tmp_path, field):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    setattr(trainer.model, field, True)
    with pytest.raises(ValueError, match='model contract changed'):
        trainer.save(tmp_path / 'boolean.pt')


def test_checkpoint_path_can_change_without_changing_training(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.cfg.ckpt_path = str(tmp_path / 'moved' / 'model.pt')
    trainer.fit(max_epochs=1)
    assert (tmp_path / 'moved' / 'model.last.pt').exists()
    assert trainer.completed_epochs == 1


def test_forward_configuration_mutation_cannot_reach_optimizer(tmp_path, monkeypatch):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    original = trainer.model.training_step
    calls = []
    def changed(batch, *, weights):
        objective = original(batch, weights=weights)
        trainer.model.model_cfg = replace(trainer.model.model_cfg, timing_scale_seconds=1.)
        return objective
    monkeypatch.setattr(trainer.model, 'training_step', changed)
    monkeypatch.setattr(trainer.opt, 'step', lambda: calls.append(True))
    with pytest.raises(ValueError, match='model contract changed'):
        trainer.fit()
    assert calls == [] and trainer.global_step == 0 and not trainer._epoch_committed
