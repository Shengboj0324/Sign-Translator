"""Explicit checkpoint diagnostics, with no training or automatic file output."""
import hashlib
import json

import pytest
import torch

from signtranslator.governed_diagnose import main, diagnose_from_configuration
from signtranslator.governed_run import diagnose_governed_planner
from signtranslator.governed_train import run_from_configuration
from signtranslator.reproducibility import canonical_json_bytes, sha256_file
from signtranslator.training.trainer import Trainer
from test_governed_train_cli import configured, save
from test_governed_load import snapshot_files, forbid


def setup(tmp_path):
    _, document = configured(tmp_path)
    path, digest = save(tmp_path, document)
    run, _ = run_from_configuration(path, expected_sha256=digest, max_epochs=1)
    checkpoint = tmp_path / 'outputs/run.last.pt'
    options = dict(expected_sha256=digest, checkpoint='outputs/run.last.pt',
        checkpoint_sha256=sha256_file(checkpoint), view='train', model_state='current',
        sample_indices=(2, 0), permutation=(1, 0), seed=9, max_samples=2)
    return path, run, options


@pytest.mark.parametrize('state', ['current', 'best_validation'])
def test_cli_matches_direct_diagnostics_and_preserves_files_rng(tmp_path, monkeypatch, capsys, state):
    path, run, options = setup(tmp_path)
    expected = diagnose_governed_planner(run, view='train', model_state=state,
        sample_indices=(2, 0), permutation=(1, 0), seed=9, max_samples=2)
    before = snapshot_files(tmp_path)
    rng = torch.get_rng_state().clone()
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(Trainer, 'fit', forbid)
    monkeypatch.setattr(Trainer, 'save', forbid)
    assert main([str(path), '--sha256', options['expected_sha256'],
        '--checkpoint', options['checkpoint'], '--checkpoint-sha256', options['checkpoint_sha256'],
        '--view', 'train', '--model-state', state, '--sample-indices', '2', '0',
        '--permutation', '1', '0', '--seed', '9', '--max-samples', '2']) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['diagnostic'] == expected.to_dict()
    assert payload['diagnostic_sha256'] == expected.sha256
    assert payload['checkpoint_sha256'] == options['checkpoint_sha256']
    assert payload['configuration_sha256'] == options['expected_sha256']
    assert payload['declaration_audit_sha256'] == hashlib.sha256(
        canonical_json_bytes(payload['declaration_audit'])).hexdigest()
    assert payload['diagnostic']['completed_epochs'] == 1
    assert not payload['phase_exit_approved']
    assert snapshot_files(tmp_path) == before
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('change', [dict(view='test'), dict(sample_indices=(0, 0)),
    dict(permutation=(True, 0)), dict(seed=True), dict(max_samples=True),
    dict(checkpoint_sha256='0' * 64), dict(checkpoint='../outside.pt'),
    dict(expected_sha256='0' * 64)])
def test_invalid_request_refuses_before_loading_model(tmp_path, monkeypatch, change):
    path, _, options = setup(tmp_path)
    options.update(change)
    before = snapshot_files(tmp_path)
    monkeypatch.setattr('signtranslator.governed_diagnose.load_governed_planner', forbid)
    with pytest.raises(ValueError):
        diagnose_from_configuration(path, **options)
    assert snapshot_files(tmp_path) == before


def test_complete_input_population_is_readmitted_including_test(tmp_path, monkeypatch):
    path, run, options = setup(tmp_path)
    corpus = run.trainer.train_loader.dataset[0].corpus
    corpus._records[-1].transcript_path.write_bytes(b'changed held-out input')
    before = snapshot_files(tmp_path)
    monkeypatch.setattr('signtranslator.governed_diagnose.load_governed_planner', forbid)
    with pytest.raises((ValueError, PermissionError)):
        diagnose_from_configuration(path, **options)
    assert snapshot_files(tmp_path) == before


def test_changed_checkpoint_during_diagnostics_is_not_reported_as_bound(tmp_path, monkeypatch):
    path, _, options = setup(tmp_path)
    def mutate_after_diagnostic(*args, **kwargs):
        report = diagnose_governed_planner(*args, **kwargs)
        target = tmp_path / options['checkpoint']
        target.write_bytes(target.read_bytes() + b'changed')
        return report
    monkeypatch.setattr('signtranslator.governed_diagnose.diagnose_governed_planner', mutate_after_diagnostic)
    with pytest.raises(ValueError, match='checkpoint changed during diagnostics'):
        diagnose_from_configuration(path, **options)


def test_cli_failure_has_no_diagnostic_stdout(tmp_path, capsys):
    path, _, options = setup(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([str(path), '--sha256', options['expected_sha256'],
            '--checkpoint', options['checkpoint'], '--checkpoint-sha256', '0' * 64,
            '--view', 'train', '--model-state', 'current', '--sample-indices', '0',
            '--permutation', '0', '--seed', '9', '--max-samples', '1'])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == '' and 'checkpoint hash mismatch' in output.err
