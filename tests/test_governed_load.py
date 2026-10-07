"""Checkpoint inspection must not silently execute another training epoch."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from signtranslator.governed_run import load_governed_planner, diagnose_governed_planner
from signtranslator.planning.source_intervention import _state_sha256
from signtranslator.training import checkpoint_paths
from signtranslator.training.trainer import Trainer
from test_governed_run import fixture, invoke


def load(values, path, **kwargs):
    corpus, vocab, alphabet, model, config = values
    return load_governed_planner(corpus, vocab, alphabet, model_config=model,
        trainer_config=config, validation=True, shuffle=kwargs.pop('shuffle', True),
        checkpoint_path=path, **kwargs)


def snapshot_files(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


def forbid(*args, **kwargs):
    raise AssertionError('load-only execution attempted training or saving')


@pytest.mark.parametrize('model_state', ['current', 'best_validation'])
def test_partial_checkpoint_load_preserves_state_and_diagnostic_without_writes(tmp_path, monkeypatch, model_state):
    values = fixture(tmp_path)
    values = (*values[:-1], replace(values[-1], ckpt_path=str(tmp_path / 'run.pt')))
    original = invoke(values, max_epochs=1)
    checkpoint = checkpoint_paths(values[-1].ckpt_path)['last']
    options = dict(view='train', model_state=model_state, sample_indices=(2, 0),
                   permutation=(1, 0), seed=9, max_samples=2)
    expected_diagnostic = diagnose_governed_planner(original, **options)
    # The load-only API must not inspect the reserved test view after admission.
    values[0]._records[-1].transcript_path.write_bytes(b'reserved test bytes')
    before_files = snapshot_files(tmp_path)
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    monkeypatch.setattr(Trainer, 'fit', forbid)
    monkeypatch.setattr(Trainer, 'save', forbid)
    restored = load(values, checkpoint)
    assert restored.trainer.completed_epochs == original.trainer.completed_epochs == 1
    assert restored.trainer.global_step == original.trainer.global_step
    assert _state_sha256(restored.trainer.model) == _state_sha256(original.trainer.model)
    torch.testing.assert_close(restored.trainer.opt.state_dict(),
                               original.trainer.opt.state_dict(), rtol=0, atol=0)
    assert restored.trainer.sched.state_dict() == original.trainer.sched.state_dict()
    assert restored.trainer.history == original.trainer.history
    assert restored.trainer.exposure_report().payload == original.trainer.exposure_report().payload
    assert restored.exposure_audit.payload == original.exposure_audit.payload
    assert diagnose_governed_planner(restored, **options).payload == expected_diagnostic.payload
    assert not restored.phase_exit_approved
    assert snapshot_files(tmp_path) == before_files
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and before_numpy[2:] == after_numpy[2:]
    np.testing.assert_array_equal(before_numpy[1], after_numpy[1])


@pytest.mark.parametrize('fault', ['missing', 'source', 'shuffle', 'checkpoint'])
def test_load_refuses_missing_stale_or_mismatched_evidence_without_writes(tmp_path, monkeypatch, fault):
    values = fixture(tmp_path)
    values = (*values[:-1], replace(values[-1], ckpt_path=str(tmp_path / 'run.pt')))
    invoke(values, max_epochs=1)
    checkpoint = checkpoint_paths(values[-1].ckpt_path)['last']
    kwargs = {}
    if fault == 'missing':
        checkpoint = tmp_path / 'missing.pt'
    elif fault == 'source':
        values[0]._records[0].transcript_path.write_bytes(b'changed source')
    elif fault == 'shuffle':
        kwargs['shuffle'] = False
    else:
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed checkpoint')
    before = snapshot_files(tmp_path)
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(Trainer, 'fit', forbid)
    monkeypatch.setattr(Trainer, 'save', forbid)
    with pytest.raises((ValueError, PermissionError)):
        load(values, checkpoint, **kwargs)
    assert snapshot_files(tmp_path) == before
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('path', [None, '', True, 123])
def test_explicit_checkpoint_is_required_before_preflight(tmp_path, monkeypatch, path):
    values = fixture(tmp_path)
    monkeypatch.setattr('signtranslator.governed_run.audit_supervision_support', forbid)
    with pytest.raises(ValueError, match='explicit checkpoint path'):
        load(values, path)


def test_finished_train_only_checkpoint_does_not_open_validation_or_test(tmp_path, monkeypatch):
    from signtranslator.governed_run import run_governed_planner
    corpus, vocab, alphabet, model, config = fixture(tmp_path)
    config = replace(config, ckpt_path=str(tmp_path / 'run.pt'))
    original = run_governed_planner(corpus, vocab, alphabet, model_config=model,
        trainer_config=config, validation=False, shuffle=False)
    for record in corpus._records[-2:]:
        record.transcript_path.write_bytes(b'unopened held-out view')
    before = snapshot_files(tmp_path)
    monkeypatch.setattr(Trainer, 'fit', forbid)
    monkeypatch.setattr(Trainer, 'save', forbid)
    restored = load_governed_planner(corpus, vocab, alphabet, model_config=model,
        trainer_config=config, validation=False, shuffle=False,
        checkpoint_path=checkpoint_paths(config.ckpt_path)['last'])
    assert restored.trainer.completed_epochs == config.epochs
    assert restored.validation_support is None and restored.trainer.val_loader is None
    assert _state_sha256(restored.trainer.model) == _state_sha256(original.trainer.model)
    assert restored.trainer.history == original.trainer.history
    assert snapshot_files(tmp_path) == before
