"""Explicit development model/view selection with immutable diagnostic evidence."""
from copy import deepcopy
import hashlib

import pytest
import torch

from signtranslator.governed_run import diagnose_governed_planner
from signtranslator.planning.source_intervention import _state_sha256
from test_governed_run import fixture, invoke


def diagnose(run, **kwargs):
    options = dict(view='train', model_state='current', sample_indices=(2, 0),
                   permutation=(1, 0), seed=9, max_samples=2)
    options.update(kwargs)
    return diagnose_governed_planner(run, **options)


def test_reordered_subset_binds_ids_and_preserves_training_state_modes_rng(tmp_path):
    run = invoke(fixture(tmp_path))
    trainer = run.trainer
    trainer.model.encoder.eval()  # Deliberately mixed modes.
    modes = [module.training for module in trainer.model.modules()]
    weights = _state_sha256(trainer.model)
    exposure = trainer.exposure_report().payload
    rng = torch.get_rng_state().clone()
    report = diagnose(run)
    data = report.to_dict()
    assert data['sample_indices'] == [2, 0]
    assert data['full_view_contract']['record_indices'] == [0, 1, 2]
    assert data['intervention']['data_contract']['record_indices'] == [2, 0]
    rows = data['intervention']['rows']
    assert [(r['anchor_sample_id'], r['source_sample_id']) for r in rows] == [
        ('sample-2', 'sample-0'), ('sample-0', 'sample-2')]
    assert data['intervention']['model_state_sha256'] == weights
    assert [module.training for module in trainer.model.modules()] == modes
    assert _state_sha256(trainer.model) == weights
    assert trainer.exposure_report().payload == exposure
    assert torch.equal(rng, torch.get_rng_state())
    assert hashlib.sha256(report.payload).hexdigest() == report.sha256
    data['intervention']['rows'].clear()
    assert len(report.to_dict()['intervention']['rows']) == 2
    assert diagnose(run).payload == report.payload


def test_best_selection_evaluates_retained_weights_without_overwriting_current(tmp_path):
    run = invoke(fixture(tmp_path))
    selected = deepcopy(run.trainer.model)
    selected.load_state_dict(run.trainer.best_model_state)
    best_hash = _state_sha256(selected)
    with torch.no_grad():
        next(run.trainer.model.parameters()).add_(.25)
    current_hash = _state_sha256(run.trainer.model)
    assert current_hash != best_hash
    result = diagnose(run, view='validation', model_state='best_validation',
                      sample_indices=None, permutation=(0,), max_samples=1).to_dict()
    assert result['intervention']['model_state_sha256'] == best_hash
    assert result['model_state'] == 'best_validation'
    assert result['intervention']['data_contract']['split'] == 'val'
    assert result['best_validation_value'] == run.trainer.best_val
    assert _state_sha256(run.trainer.model) == current_hash
    assert not result['phase_exit_approved']


@pytest.mark.parametrize('options', [dict(view='test'), dict(model_state='last'),
    dict(sample_indices=(0, 0)), dict(sample_indices=(True,)), dict(sample_indices=(3,)),
    dict(permutation=(0, 0)), dict(max_samples=1), dict(seed=-1)])
def test_invalid_selection_fails_before_copy(tmp_path, monkeypatch, options):
    run = invoke(fixture(tmp_path))
    def forbid(*args, **kwargs):
        raise AssertionError('copied before selection validation')
    monkeypatch.setattr('signtranslator.governed_run.deepcopy', forbid)
    with pytest.raises(ValueError):
        diagnose(run, **options)


def test_failed_selected_source_read_does_not_mutate_original(tmp_path):
    values = fixture(tmp_path)
    run = invoke(values)
    values[0]._records[2].transcript_path.write_bytes(b'changed')
    before = _state_sha256(run.trainer.model)
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, PermissionError)):
        diagnose(run)
    assert _state_sha256(run.trainer.model) == before
    assert torch.equal(rng, torch.get_rng_state())


def test_incomplete_epoch_and_missing_best_are_not_implicit_selection(tmp_path):
    run = invoke(fixture(tmp_path))
    run.trainer._epoch_committed = False
    with pytest.raises(ValueError, match='committed'):
        diagnose(run)
    run.trainer._epoch_committed = True
    run.trainer.best_model_state = None
    with pytest.raises(ValueError, match='no retained'):
        diagnose(run, model_state='best_validation')


def test_replaced_validation_loader_cannot_expose_test_partition(tmp_path):
    from test_governed_trainer import loader
    values = fixture(tmp_path)
    run = invoke(values)
    run.trainer.val_loader = loader(values[0], 'test')
    values[0]._records[-1].transcript_path.write_bytes(b'test bytes must not be read')
    with pytest.raises(ValueError, match='differs from recorded'):
        diagnose(run, view='validation', sample_indices=None, permutation=(0,), max_samples=1)
