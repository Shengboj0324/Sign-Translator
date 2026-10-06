"""Interrupted epochs must not masquerade as exactly resumable boundaries."""
import hashlib
import json

import pytest
import torch

from signtranslator.training import Trainer
from test_supported_trainer import mixed, config, model, loader


def make(corpus, vocab):
    return Trainer(model(vocab), config(), loader(corpus, 'train'), loader(corpus, 'val'))


@pytest.mark.parametrize('failure', ['optimizer', 'scheduler', 'validation', 'selection'])
def test_failed_epoch_refuses_save_and_retry_then_recovers(tmp_path, monkeypatch, failure):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    committed = trainer.save(tmp_path / 'initial.pt')
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    with monkeypatch.context() as patch:
        if failure == 'optimizer':
            patch.setattr(trainer.opt, 'step', fail)
        elif failure == 'scheduler':
            patch.setattr(type(trainer.sched), 'step', fail)
        elif failure == 'validation':
            patch.setattr(trainer, 'validate', fail)
        else:
            original = trainer.validate
            def unavailable():
                original()
                return {}
            patch.setattr(trainer, 'validate', unavailable)
        with pytest.raises((RuntimeError, ValueError)):
            trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 0
    forbidden = tmp_path / 'incomplete.pt'
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        trainer.save(forbidden)
    assert not forbidden.exists() and not forbidden.with_suffix('.pt.json').exists()
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        trainer.fit()
    assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
    trainer.load(committed)
    trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 1 and trainer.global_step == len(trainer.train_loader)
    trainer.save(tmp_path / 'recovered.pt')


def test_standalone_epoch_does_not_commit_fit_state(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    trainer.train_epoch()
    assert trainer.global_step == 2 and trainer.completed_epochs == 0
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        trainer.save(tmp_path / 'standalone.pt')
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        trainer.fit()


def test_inconsistent_epoch_cursor_refused_before_model_load(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trained = make(corpus, vocab); trained.fit(max_epochs=1)
    path = trained.save(tmp_path / 'cursor.pt')
    data = torch.load(path, weights_only=False)
    data['training_state']['completed_epochs'] = 0
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest.update(training_state=data['training_state'],
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_size=path.stat().st_size)
    sidecar.write_text(json.dumps(manifest))
    fresh = make(corpus, vocab)
    before = {k: v.clone() for k, v in fresh.model.state_dict().items()}
    with pytest.raises(ValueError, match='committed governed epoch'):
        fresh.load(path)
    assert all(torch.equal(v, before[k]) for k, v in fresh.model.state_dict().items())
