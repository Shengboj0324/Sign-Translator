"""Optimizer/scheduler agreement alone does not establish the scheduled rate."""
import hashlib
import json

import pytest
import torch

from test_epoch_commit import make
from test_supported_trainer import mixed


def rewrite(path, mutation):
    checkpoint = torch.load(path, weights_only=False)
    mutation(checkpoint)
    torch.save(checkpoint, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest))


def test_consistently_wrong_live_rates_are_rejected_and_resume_recovers(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    path = trainer.save(tmp_path / 'initial.pt')
    trainer.opt.param_groups[0]['lr'] = .9
    trainer.sched._last_lr = [.9]
    with pytest.raises(ValueError, match='scheduler state does not match'):
        trainer.fit()
    with pytest.raises(ValueError, match='scheduler state does not match'):
        trainer.save(tmp_path / 'wrong.pt')
    assert not (tmp_path / 'wrong.pt').exists() and trainer.global_step == 0
    trainer.load(path)
    trainer.fit(max_epochs=1)
    assert trainer.completed_epochs == 1


@pytest.mark.parametrize('mutation', ['rates', 'cursor', 'step_count', 'base', 'bool_cursor'])
def test_checkpoint_schedule_preflight_precedes_tensor_loading(tmp_path, mutation):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'checkpoint.pt')
    def change(checkpoint):
        scheduler = checkpoint['scheduler']
        if mutation == 'rates':
            scheduler['_last_lr'] = [.9]
            checkpoint['optimizer']['param_groups'][0]['lr'] = .9
        elif mutation == 'cursor':
            scheduler['last_epoch'] += 1
        elif mutation == 'step_count':
            scheduler['_step_count'] += 1
        elif mutation == 'base':
            scheduler['base_lrs'] = [.9]
        else:
            scheduler['last_epoch'] = True
    rewrite(path, change)
    receiver = make(corpus, vocabulary)
    before = {name: value.clone() for name, value in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='checkpoint scheduler state does not match'):
        receiver.load(path)
    assert all(torch.equal(before[name], value) for name, value in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and not receiver.opt.state
    assert receiver.global_step == 0 and receiver._epoch_committed


def test_replaced_scheduler_callable_is_refused(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary)
    trainer.sched.lr_lambdas[0] = lambda step: .5
    with pytest.raises(ValueError, match='scheduler callable changed'):
        trainer.fit()
    assert trainer.global_step == 0


def test_valid_warmup_and_cosine_rates_match_analytic_values_through_resume(tmp_path):
    import math
    from signtranslator.training import Trainer
    from test_supported_trainer import config, model, loader

    vocabulary, corpus = mixed(tmp_path)
    cfg = config(epochs=4)
    cfg.warmup_frac = .5
    cfg.min_lr_frac = .1
    def build():
        return Trainer(model(vocabulary), cfg, loader(corpus, 'train'), loader(corpus, 'val'))
    trainer = build()
    total, warmup = 8, 4
    for epoch in range(4):
        step = trainer.global_step
        expected = cfg.lr * ((step + 1)/warmup if step < warmup else
                             .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup)/(total - warmup))))
        assert trainer.opt.param_groups[0]['lr'] == pytest.approx(expected)
        trainer.fit(max_epochs=1)
        path = trainer.save(tmp_path / f'epoch-{epoch}.pt')
        trainer = build()
        trainer.load(path)
    assert trainer.opt.param_groups[0]['lr'] == pytest.approx(cfg.lr * .1)
    assert trainer.global_step == trainer.sched.last_epoch == 8
