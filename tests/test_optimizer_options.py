"""Optimizer group settings must agree with governed training metadata."""
import hashlib
import json

import pytest
import torch

from test_epoch_commit import make
from test_supported_trainer import mixed


@pytest.mark.parametrize('field,value', [('weight_decay', .9), ('eps', .1),
                                         ('betas', (.8, .9)), ('maximize', True),
                                         ('amsgrad', True), ('initial_lr', .7)])
def test_live_optimizer_options_cannot_drift(tmp_path, field, value):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.opt.param_groups[0][field] = value
    with pytest.raises(ValueError, match='optimizer options changed'):
        trainer.fit()
    with pytest.raises(ValueError, match='optimizer options changed'):
        trainer.save(tmp_path / 'invalid.pt')
    assert trainer.global_step == 0 and not trainer.opt.state
    assert not (tmp_path / 'invalid.pt').exists()


def test_live_learning_rate_must_agree_with_scheduler(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.opt.param_groups[0]['lr'] = .9
    with pytest.raises(ValueError, match='learning rates disagree'):
        trainer.fit()
    assert trainer.global_step == 0 and not trainer.opt.state


@pytest.mark.parametrize('field,value', [('weight_decay', .9), ('betas', (.1, .2)),
                                         ('lr', .9), ('maximize', 0)])
def test_checkpoint_option_mismatch_refused_before_tensor_load(tmp_path, field, value):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'source.pt')
    data = torch.load(path, weights_only=False)
    data['optimizer']['param_groups'][0][field] = value
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest))
    receiver = make(corpus, vocabulary)
    before = {name: value.clone() for name, value in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='optimizer options|learning rates disagree'):
        receiver.load(path)
    assert all(torch.equal(before[name], value) for name, value in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and receiver._epoch_committed
    assert receiver.global_step == 0 and not receiver.opt.state
    # A weights-only start deliberately does not import optimizer settings.
    receiver.load(path, mode='weights')
    assert receiver.opt.param_groups[0]['weight_decay'] == receiver.cfg.weight_decay
    assert not receiver.opt.state and receiver.global_step == 0


def test_backward_option_mutation_never_reaches_optimizer(tmp_path, monkeypatch):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    calls = []
    def mutate(gradient):
        trainer.opt.param_groups[0]['weight_decay'] = .9
        return gradient
    next(trainer.model.parameters()).register_hook(mutate)
    monkeypatch.setattr(trainer.opt, 'step', lambda: calls.append(True))
    with pytest.raises(ValueError, match='optimizer options changed'):
        trainer.fit()
    assert calls == [] and trainer.global_step == 0 and not trainer._epoch_committed


def test_normal_schedule_and_resume_preserve_rate_consistency(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    initial = source.opt.param_groups[0]['lr']
    source.fit(max_epochs=1)
    assert source.opt.param_groups[0]['lr'] != initial
    path = source.save(tmp_path / 'normal.pt')
    receiver = make(corpus, vocabulary)
    receiver.load(path)
    source.fit()
    receiver.fit()
    assert source.history == receiver.history
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, receiver.model.state_dict()[name])


def test_full_resume_recovers_live_rate_mismatch(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    path = trainer.save(tmp_path / 'committed.pt')
    trainer.opt.param_groups[0]['lr'] = .9
    trainer.load(path)
    assert [group['lr'] for group in trainer.opt.param_groups] == trainer.sched.get_last_lr()
    trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 1
