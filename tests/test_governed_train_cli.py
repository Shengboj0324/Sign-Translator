"""Explicit governed CLI uses fictional evidence and preserves exact CPU continuation."""
from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from signtranslator.governed_train import main, run_from_configuration
from signtranslator.reproducibility import canonical_json_bytes
from test_governed_manifest import handoff, write
from test_governed_run import invoke


def configured(tmp_path):
    values, inputs = handoff(tmp_path)
    _, digest = write(tmp_path, inputs)
    trainer = values[4].to_dict()
    trainer['values']['ckpt_path'] = 'outputs/run.pt'
    document = dict(schema_version=1, input_manifest='inputs.json', input_manifest_sha256=digest,
                    model_config=asdict(values[3]), trainer_config=trainer, validation=True, shuffle=True)
    return values, document


def save(tmp_path, document, name='run.json'):
    path = tmp_path / name
    payload = canonical_json_bytes(document)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_command_and_exact_resume_match_direct_runner(tmp_path, capsys):
    values, document = configured(tmp_path)
    path, digest = save(tmp_path, document)
    expected = invoke(values)
    assert main([str(path), '--sha256', digest, '--max-epochs', '1']) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['completed_epochs'] == 1 and summary['global_step'] == 2
    assert summary['configuration_sha256'] == digest and not summary['phase_exit_approved']
    with pytest.raises(ValueError, match='replacement flag'):
        run_from_configuration(path, expected_sha256=digest, resume_from='outputs/run.last.pt')
    resumed, payload = run_from_configuration(path, expected_sha256=digest,
        resume_from='outputs/run.last.pt', allow_checkpoint_replacement=True)
    assert resumed.trainer.history == expected.trainer.history
    assert resumed.trainer.optimizer_exposure == expected.trainer.optimizer_exposure
    assert all(torch.equal(v, resumed.trainer.model.state_dict()[k])
               for k, v in expected.trainer.model.state_dict().items())
    assert json.loads(payload)['completed_epochs'] == 2


@pytest.mark.parametrize('fault', ['version', 'trainer_version', 'missing_model', 'missing_trainer',
    'unknown', 'seed_bool', 'shuffle', 'no_checkpoint', 'escape', 'bad_parent', 'hash'])
def test_configuration_failure_precedes_training_and_output(tmp_path, monkeypatch, fault):
    _, document = configured(tmp_path)
    if fault == 'version':
        document['schema_version'] = True
    elif fault == 'trainer_version':
        document['trainer_config']['schema_version'] = True
    elif fault == 'missing_model':
        del document['model_config']['max_events']
    elif fault == 'missing_trainer':
        del document['trainer_config']['values']['lr']
    elif fault == 'unknown':
        document['implicit_acceptance'] = True
    elif fault == 'seed_bool':
        document['trainer_config']['values']['seed'] = False
    elif fault == 'shuffle':
        document['shuffle'] = 1
    elif fault == 'no_checkpoint':
        document['trainer_config']['values']['ckpt_path'] = None
    elif fault == 'escape':
        document['trainer_config']['values']['ckpt_path'] = '../escape.pt'
    elif fault == 'bad_parent':
        (tmp_path / 'blocked').write_text('not a directory')
        document['trainer_config']['values']['ckpt_path'] = 'blocked/run.pt'
    path, digest = save(tmp_path, document)
    def refuse(*args, **kwargs):
        raise AssertionError('training called for invalid configuration')
    monkeypatch.setattr('signtranslator.governed_train.run_governed_planner', refuse)
    with pytest.raises(ValueError):
        run_from_configuration(path, expected_sha256='0' * 64 if fault == 'hash' else digest)
    assert not (tmp_path / 'outputs').exists()


def test_replacement_flag_cannot_overwrite_configuration_input(tmp_path):
    _, document = configured(tmp_path)
    document['trainer_config']['values']['ckpt_path'] = 'run.json'
    path, digest = save(tmp_path, document, name='run.last.json')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='overlaps a declared input'):
        run_from_configuration(path, expected_sha256=digest, allow_checkpoint_replacement=True)
    assert path.read_bytes() == before


def test_replacement_flag_cannot_overwrite_declared_artifact(tmp_path):
    _, document = configured(tmp_path)
    manifest = json.loads((tmp_path / 'inputs.json').read_bytes())
    artifact = tmp_path / 'artifact.last.json'
    (tmp_path / 'lexicon.json').rename(artifact)
    manifest['lexicon']['path'] = artifact.name
    _, input_digest = write(tmp_path, manifest)
    document['input_manifest_sha256'] = input_digest
    document['trainer_config']['values']['ckpt_path'] = 'artifact.json'
    path, digest = save(tmp_path, document)
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match='overlaps a declared input'):
        run_from_configuration(path, expected_sha256=digest, allow_checkpoint_replacement=True)
    assert artifact.read_bytes() == before
