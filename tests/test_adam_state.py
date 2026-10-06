"""Finite optimizer state still has mathematical and ownership domains."""
import copy
import hashlib
import json

import pytest
import torch

from signtranslator.training.adam_state import validate_adam_state
from test_epoch_commit import make
from test_supported_trainer import mixed


def corrupt(data, defect):
    state = next(iter(data['state'].values()))
    if defect == 'negative':
        state['exp_avg_sq'].fill_(-1)
    elif defect == 'fractional':
        state['step'].fill_(1.5)
    elif defect == 'future':
        state['step'].fill_(1000)
    elif defect == 'zero':
        state['step'].zero_()
    elif defect == 'shape':
        state['exp_avg'] = torch.zeros(0)
    elif defect == 'dtype':
        state['exp_avg'] = state['exp_avg'].double()
    elif defect == 'missing':
        del state['exp_avg_sq']
    elif defect == 'orphan':
        data['state'][10000] = copy.deepcopy(state)
    elif defect == 'duplicate':
        ids = data['param_groups'][0]['params']
        ids[1] = ids[0]


@pytest.mark.parametrize('defect', ['negative', 'fractional', 'future', 'zero', 'shape', 'dtype', 'missing', 'orphan', 'duplicate'])
def test_checkpoint_domains_preflight_before_mutation(tmp_path, defect):
    vocab, corpus = mixed(tmp_path)
    source = make(corpus, vocab)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'state.pt')
    data = torch.load(path, weights_only=False)
    corrupt(data['optimizer'], defect)
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest.update(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_size=path.stat().st_size)
    sidecar.write_text(json.dumps(manifest))
    receiver = make(corpus, vocab)
    before = {k: v.clone() for k, v in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='invalid governed AdamW state'):
        receiver.load(path)
    assert all(torch.equal(before[k], v) for k, v in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    assert receiver._epoch_committed and not receiver.opt.state
    receiver.load(path, mode='weights')
    assert not receiver.opt.state


def test_live_invalid_moment_cannot_save_or_train(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    trainer.fit(max_epochs=1)
    next(iter(trainer.opt.state.values()))['exp_avg_sq'].fill_(-1)
    with pytest.raises(ValueError, match='must be nonnegative'):
        trainer.save(tmp_path / 'absent' / 'bad.pt')
    assert not (tmp_path / 'absent').exists()
    before = trainer.global_step
    with pytest.raises(ValueError, match='must be nonnegative'):
        trainer.train_epoch()
    assert trainer.global_step == before


@pytest.mark.parametrize('complex_parameter', [False, True])
def test_real_adamw_sparse_step_counts_and_amsgrad(complex_parameter):
    dtype = torch.complex64 if complex_parameter else torch.float32
    parameters = [torch.nn.Parameter(torch.ones(2, dtype=dtype)) for _ in range(3)]
    optimizer = torch.optim.AdamW(parameters, amsgrad=True)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameters[0].grad = torch.ones_like(parameters[0])
        if step == 0:
            parameters[1].grad = torch.ones_like(parameters[1])
        optimizer.step()
    state = optimizer.state_dict()
    validate_adam_state(state, [parameters], global_step=2)
    assert [s['step'].item() for s in state['state'].values()] == [2, 1]
    next(iter(state['state'].values()))['max_exp_avg_sq'].zero_()
    with pytest.raises(ValueError, match='maximum is below'):
        validate_adam_state(state, [parameters], global_step=2)


def test_returned_update_with_invalid_moment_retains_exposure(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    initial = trainer.save(tmp_path / 'initial.pt')
    def corrupt_after_step(optimizer, args, kwargs):
        next(iter(optimizer.state.values()))['exp_avg_sq'].fill_(-1)
    handle = trainer.opt.register_step_post_hook(corrupt_after_step)
    with pytest.raises(ValueError, match='must be nonnegative'):
        trainer.fit(max_epochs=1)
    handle.remove()
    assert trainer.global_step == len(trainer.optimizer_exposure) == 1
    assert trainer.completed_epochs == 0 and not trainer._epoch_committed
    assert trainer.exposure_report().to_dict()['recorded_optimizer_steps'] == 1
    trainer.load(initial)
    trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 1
