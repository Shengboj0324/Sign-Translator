"""Finite loss/gradients are insufficient to certify a finite updated state."""
import hashlib
import json

import pytest
import torch

from signtranslator.config import TrainerConfig
from signtranslator.training import Trainer
from test_epoch_commit import make
from test_supported_trainer import mixed


def test_real_adamw_overflow_cannot_commit_or_save(tmp_path):
    from test_governed_trainer import MechanicalProbe, setup, loader

    corpus, _, _ = setup(tmp_path)
    model = MechanicalProbe().float()
    with torch.no_grad():
        model.offset.fill_(100.)
    trainer = Trainer(model, TrainerConfig(epochs=1, lr=1e37, weight_decay=1.), loader(corpus, 'train'))
    initial = trainer.save(tmp_path / 'initial.pt')
    with pytest.raises(FloatingPointError, match='nonfinite governed state: model.offset'):
        trainer.fit()
    assert not torch.isfinite(model.offset)
    assert trainer.global_step == 1 and trainer.completed_epochs == 0 and not trainer._epoch_committed
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        trainer.save(tmp_path / 'bad.pt')
    assert not (tmp_path / 'bad.pt').exists()
    trainer.load(initial)
    assert model.offset.item() == 100. and trainer.global_step == 0 and trainer._epoch_committed


@pytest.mark.parametrize('target', ['parameter', 'moment'])
def test_returned_step_with_corrupted_state_retains_exposure_without_commit(tmp_path, target):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    initial = trainer.save(tmp_path / 'initial.pt')
    def corrupt(optimizer, args, kwargs):
        parameter = next(trainer.model.parameters())
        with torch.no_grad():
            value = parameter if target == 'parameter' else optimizer.state[parameter]['exp_avg']
            value.fill_(float('nan'))
    handle = trainer.opt.register_step_post_hook(corrupt)
    with pytest.raises(FloatingPointError, match='nonfinite governed state'):
        trainer.fit()
    handle.remove()
    assert trainer.global_step == len(trainer.optimizer_exposure) == 1
    assert trainer.completed_epochs == 0 and not trainer._epoch_committed
    report = trainer.exposure_report().to_dict()
    assert report['recorded_optimizer_steps'] == 1 and not report['committed_epoch_boundary']
    trainer.load(initial)
    trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 1


@pytest.mark.parametrize('target', ['model', 'optimizer', 'best_model'])
def test_nonfinite_checkpoint_rejected_before_receiver_mutation(tmp_path, target):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'source.pt')
    data = torch.load(path, weights_only=False)
    if target == 'model':
        next(iter(data['model'].values())).fill_(float('nan'))
    elif target == 'best_model':
        next(iter(data['best_model_state'].values())).fill_(float('inf'))
    else:
        next(iter(data['optimizer']['state'].values()))['exp_avg'].fill_(float('inf'))
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest))
    receiver = make(corpus, vocabulary)
    before = {name: value.clone() for name, value in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(FloatingPointError, match='nonfinite governed state: checkpoint'):
        receiver.load(path)
    assert all(torch.equal(before[name], value) for name, value in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and receiver._epoch_committed
    assert not receiver.opt.state and receiver.global_step == 0
    if target != 'model':
        receiver.load(path, mode='weights')
        assert receiver._epoch_committed and not receiver.opt.state


def test_nonfinite_live_state_refuses_save_before_directory_creation(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    with torch.no_grad():
        next(trainer.model.parameters()).fill_(float('inf'))
    destination = tmp_path / 'not-created' / 'invalid.pt'
    with pytest.raises(FloatingPointError, match='nonfinite governed state'):
        trainer.save(destination)
    assert not destination.parent.exists()


def test_validation_side_effect_cannot_commit_nonfinite_weights(tmp_path, monkeypatch):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    validate = trainer.validate
    def corrupt_after_validation():
        result = validate()
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float('nan'))
        return result
    monkeypatch.setattr(trainer, 'validate', corrupt_after_validation)
    with pytest.raises(FloatingPointError, match='nonfinite governed state'):
        trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 0 and not trainer._epoch_committed


@pytest.mark.parametrize('mode', ['resume', 'weights'])
def test_load_hook_nonfinite_result_never_commits(tmp_path, mode):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    path = source.save(tmp_path / 'initial.pt')
    receiver = make(corpus, vocabulary)
    def corrupt(module, keys):
        with torch.no_grad():
            next(module.parameters()).fill_(float('inf'))
    receiver.model.register_load_state_dict_post_hook(corrupt)
    with pytest.raises(FloatingPointError, match='nonfinite governed state'):
        receiver.load(path, mode=mode)
    assert not receiver._epoch_committed
