"""Weight replacement and partial deserialization cannot certify stale state."""
from copy import deepcopy

import pytest
import torch

from test_epoch_commit import make
from test_supported_trainer import mixed


def test_weights_only_rejects_trained_receiver_without_mutation(tmp_path):
    vocab, corpus = mixed(tmp_path)
    source = make(corpus, vocab)
    path = source.save(tmp_path / 'initial.pt')
    receiver = make(corpus, vocab); receiver.fit(max_epochs=1)
    before = {k: v.clone() for k, v in receiver.model.state_dict().items()}
    history, exposure = deepcopy(receiver.history), receiver.optimizer_exposure
    with pytest.raises(ValueError, match='fresh trainer'):
        receiver.load(path, mode='weights')
    assert all(torch.equal(v, before[k]) for k, v in receiver.model.state_dict().items())
    assert receiver.history == history and receiver.optimizer_exposure == exposure
    # A rejected request did not invalidate the previously committed receiver.
    receiver.save(tmp_path / 'unchanged.pt')


def test_fresh_warm_start_keeps_empty_training_state_and_can_train(tmp_path):
    vocab, corpus = mixed(tmp_path)
    source = make(corpus, vocab); source.fit(max_epochs=1)
    path = source.save(tmp_path / 'trained.pt')
    receiver = make(corpus, vocab); receiver.load(path, mode='weights')
    assert receiver.completed_epochs == receiver.global_step == 0
    assert receiver.history == {} and receiver.optimizer_exposure == []
    assert not receiver.opt.state and receiver.best_model_state is None
    assert all(torch.equal(v, source.model.state_dict()[k])
               for k, v in receiver.model.state_dict().items())
    receiver.save(tmp_path / 'warm.pt')
    receiver.fit(max_epochs=1)
    assert receiver.completed_epochs == 1 and receiver.global_step == 2


@pytest.mark.parametrize('stage', ['weights', 'model', 'optimizer', 'scheduler', 'rng'])
def test_partial_load_refuses_save_and_fit_until_complete_resume(tmp_path, monkeypatch, stage):
    vocab, corpus = mixed(tmp_path)
    source = make(corpus, vocab); source.fit(max_epochs=1)
    path = source.save(tmp_path / 'valid.pt')
    receiver = make(corpus, vocab)
    def fail(*args, **kwargs):
        # Simulates a custom loader that changes state and then raises.
        with torch.no_grad():
            next(receiver.model.parameters()).fill_(123.)
        raise RuntimeError('partial load')
    with monkeypatch.context() as patch:
        if stage in ('weights', 'model'):
            patch.setattr(receiver.model, 'load_state_dict', fail)
        elif stage == 'optimizer':
            patch.setattr(receiver.opt, 'load_state_dict', fail)
        elif stage == 'scheduler':
            patch.setattr(type(receiver.sched), 'load_state_dict', fail)
        else:
            patch.setattr('signtranslator.training.trainer.restore_rng_state', fail)
        with pytest.raises(RuntimeError, match='partial load'):
            receiver.load(path, mode='weights' if stage == 'weights' else 'resume')
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        receiver.save(tmp_path / 'invalid.pt')
    with pytest.raises(RuntimeError, match='unfinished epoch'):
        receiver.fit()
    with pytest.raises(ValueError, match='fresh trainer'):
        receiver.load(path, mode='weights')
    receiver.load(path)
    assert receiver.completed_epochs == source.completed_epochs
    assert receiver.optimizer_exposure == source.optimizer_exposure
    assert all(torch.equal(v, source.model.state_dict()[k])
               for k, v in receiver.model.state_dict().items())
    receiver.fit()
    assert receiver.completed_epochs == 2
