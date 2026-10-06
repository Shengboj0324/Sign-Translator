"""Local JSON handoff on fictional evidence; never real source/review approval."""
from dataclasses import asdict, fields
from enum import Enum
import hashlib
import json
from pathlib import Path

import pytest
import torch

from signtranslator.data.governed_manifest import load_governed_planner_inputs
from signtranslator.governed_run import run_governed_planner
from signtranslator.reproducibility import canonical_json_bytes
from test_governed_run import fixture, invoke


def portable(value, root):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    if isinstance(value, dict):
        return {k: portable(v, root) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable(v, root) for v in value]
    return value


def handoff(tmp_path):
    values = fixture(tmp_path)
    corpus, vocabulary, alphabet, _, _ = values
    rows = []
    for record in corpus._records:
        sample = portable(asdict(record.sample), tmp_path)
        sample['authorization'] = record.sample.authorization.to_manifest()
        row = {f.name: portable(getattr(record, f.name), tmp_path)
               for f in fields(record) if f.name not in ('sample', 'annotation')}
        row.update(sample=sample, annotation=record.annotation.to_manifest())
        rows.append(row)
    authorizations = {key: dict(source_id=value.source_id, consent=value.consent.name,
        local_evidence=portable(value.local_evidence, tmp_path), authorization=value.authorization.to_manifest())
        for key, value in corpus._authorizations.items()}
    (tmp_path / 'lexicon.json').write_bytes(vocabulary.payload)
    (tmp_path / 'convention.json').write_bytes(alphabet.payload)
    manifest = dict(schema_version=1, scope='research', records=rows,
                    sources=[portable(asdict(s), tmp_path) for s in corpus._sources], authorizations=authorizations,
                    lexicon=dict(artifact=vocabulary.lexicon.to_dict(), path='lexicon.json'),
                    convention=dict(artifact=vocabulary.convention.to_dict(), path='convention.json'))
    return values, manifest


def write(tmp_path, manifest):
    path = tmp_path / 'inputs.json'
    payload = canonical_json_bytes(manifest)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_full_population_handoff_matches_direct_admission_and_training(tmp_path, monkeypatch):
    values, manifest = handoff(tmp_path)
    path, digest = write(tmp_path, manifest)
    monkeypatch.chdir(tmp_path.parent)
    before = set(tmp_path.rglob('*'))
    loaded = load_governed_planner_inputs(path, expected_sha256=digest)
    assert loaded.corpus.manifest_bytes == values[0].manifest_bytes
    assert loaded.vocabulary == values[1] and loaded.alphabet == values[2]
    assert loaded.manifest_sha256 == digest and not loaded.phase_exit_approved
    assert set(tmp_path.rglob('*')) == before
    expected = invoke(values, max_epochs=1)
    actual = run_governed_planner(loaded.corpus, loaded.vocabulary, loaded.alphabet,
        model_config=values[3], trainer_config=values[4], validation=True, shuffle=True, max_epochs=1)
    assert expected.trainer.history == actual.trainer.history
    assert expected.trainer.optimizer_exposure == actual.trainer.optimizer_exposure
    assert all(torch.equal(v, actual.trainer.model.state_dict()[k]) for k, v in expected.trainer.model.state_dict().items())


@pytest.mark.parametrize('fault', ['scope', 'version', 'extra', 'missing_sample', 'consent_bool',
    'duplicate_source', 'missing_grants', 'withdrawn', 'escape', 'absolute', 'symlink',
    'changed_test_bytes', 'artifact_binding', 'unknown_sample_field'])
def test_manifest_cannot_bypass_existing_admission(tmp_path, fault):
    values, manifest = handoff(tmp_path)
    if fault == 'scope':
        manifest['scope'] = 'commercial'
    elif fault == 'version':
        manifest['schema_version'] = True
    elif fault == 'extra':
        manifest['approved'] = True
    elif fault == 'missing_sample':
        del manifest['records'][0]['sample']['authorization']
    elif fault == 'consent_bool':
        manifest['records'][0]['sample']['consent'] = False
    elif fault == 'duplicate_source':
        manifest['sources'].append(manifest['sources'][0])
    elif fault == 'missing_grants':
        manifest['authorizations'] = {}
    elif fault == 'withdrawn':
        next(iter(manifest['authorizations'].values()))['consent'] = 'WITHDRAWN'
    elif fault == 'escape':
        manifest['lexicon']['path'] = '../lexicon.json'
    elif fault == 'absolute':
        manifest['lexicon']['path'] = str(tmp_path / 'lexicon.json')
    elif fault == 'symlink':
        (tmp_path / 'linked').symlink_to(tmp_path / 'lexicon.json')
        manifest['lexicon']['path'] = 'linked'
    elif fault == 'changed_test_bytes':
        values[0]._records[-1].transcript_path.write_bytes(b'changed reserved data')
    elif fault == 'artifact_binding':
        manifest['lexicon']['artifact']['artifact_id'] = 'other'
    else:
        manifest['records'][0]['sample']['implicit_approval'] = True
    path, digest = write(tmp_path, manifest)
    with pytest.raises((ValueError, PermissionError)):
        load_governed_planner_inputs(path, expected_sha256=digest)


def test_hash_duplicate_keys_and_nonfinite_numbers_refused(tmp_path):
    _, manifest = handoff(tmp_path)
    path, digest = write(tmp_path, manifest)
    with pytest.raises(ValueError, match='hash mismatch'):
        load_governed_planner_inputs(path, expected_sha256='0' * 64)
    for payload in (b'{"schema_version":1,"schema_version":1}', b'{"extra":1e999}'):
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            load_governed_planner_inputs(path, expected_sha256=hashlib.sha256(payload).hexdigest())
