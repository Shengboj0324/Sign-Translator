import hashlib
import json

import pytest

from signtranslator.governed_diagnose import diagnose_from_configuration, main
from signtranslator.governed_run import diagnose_governed_planner
from signtranslator.planning.graph_decode import DiagnosticRelationThresholds
from signtranslator.planning.tensors import EDGE_TYPES
from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.trainer import Trainer
from test_governed_diagnose_cli import setup
from test_governed_load import snapshot_files, forbid


def write_thresholds(tmp_path, change=None):
    value = dict(schema_version=1, relation_types=[kind.value for kind in EDGE_TYPES],
                 negative_below=[-1.] * 5, positive_above=[1.] * 5)
    if change:
        change(value)
    payload = canonical_json_bytes(value)
    (tmp_path / 'thresholds.json').write_bytes(payload)
    return dict(relation_thresholds_manifest='thresholds.json',
                relation_thresholds_sha256=hashlib.sha256(payload).hexdigest())


def test_command_threshold_manifest_matches_direct_api_without_writes(tmp_path, monkeypatch, capsys):
    path, run, options = setup(tmp_path)
    options.update(write_thresholds(tmp_path))
    direct = diagnose_governed_planner(run, view='train', model_state='current',
        sample_indices=(2, 0), permutation=(1, 0), seed=9, max_samples=2,
        relation_thresholds=DiagnosticRelationThresholds((-1.,) * 5, (1.,) * 5))
    before = snapshot_files(tmp_path)
    monkeypatch.setattr(Trainer, 'fit', forbid)
    monkeypatch.setattr(Trainer, 'save', forbid)
    assert main([str(path), '--sha256', options['expected_sha256'], '--checkpoint', options['checkpoint'],
        '--checkpoint-sha256', options['checkpoint_sha256'], '--view', 'train', '--model-state', 'current',
        '--sample-indices', '2', '0', '--permutation', '1', '0', '--seed', '9', '--max-samples', '2',
        '--relation-thresholds-manifest', options['relation_thresholds_manifest'],
        '--relation-thresholds-sha256', options['relation_thresholds_sha256']]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['schema_version'] == 2
    assert report['relation_thresholds_manifest_sha256'] == options['relation_thresholds_sha256']
    assert report['diagnostic'] == direct.to_dict()
    relation = report['diagnostic']['intervention']['original_relation_evaluation']
    assert relation['schema_version'] == 2
    assert relation['threshold_diagnostic']['negative_below'] == [-1.] * 5
    assert snapshot_files(tmp_path) == before
    assert not report['phase_exit_approved']


@pytest.mark.parametrize('fault', ['missing_hash', 'missing_path', 'hash', 'version', 'types',
                                   'integers', 'overlap', 'escape', 'oversize'])
def test_invalid_threshold_manifest_refuses_before_checkpoint_load(tmp_path, monkeypatch, fault):
    path, _, options = setup(tmp_path)
    changes = dict(version=lambda x: x.update(schema_version=True),
                   types=lambda x: x['relation_types'].reverse(),
                   integers=lambda x: x.update(negative_below=[-1] * 5),
                   overlap=lambda x: x.update(negative_below=[2.] * 5))
    options.update(write_thresholds(tmp_path, changes.get(fault)))
    if fault == 'missing_hash':
        options['relation_thresholds_sha256'] = None
    elif fault == 'missing_path':
        options['relation_thresholds_manifest'] = None
    elif fault == 'hash':
        options['relation_thresholds_sha256'] = '0' * 64
    elif fault == 'escape':
        options['relation_thresholds_manifest'] = '../thresholds.json'
    elif fault == 'oversize':
        payload = b' ' * 4097
        (tmp_path / 'thresholds.json').write_bytes(payload)
        options['relation_thresholds_sha256'] = hashlib.sha256(payload).hexdigest()
    before = snapshot_files(tmp_path)
    monkeypatch.setattr('signtranslator.governed_diagnose.load_governed_planner', forbid)
    with pytest.raises(ValueError):
        diagnose_from_configuration(path, **options)
    assert snapshot_files(tmp_path) == before


@pytest.mark.parametrize('malformed', ['wrong_type', 'mutated'])
def test_api_threshold_validation_precedes_model_copy(tmp_path, monkeypatch, malformed):
    _, run, _ = setup(tmp_path)
    thresholds = {'negative_below': [-1.] * 5}
    if malformed == 'mutated':
        thresholds = DiagnosticRelationThresholds((-1.,) * 5, (1.,) * 5)
        object.__setattr__(thresholds, 'positive_above', (-2.,) * 5)
    monkeypatch.setattr('signtranslator.governed_run.deepcopy', forbid)
    with pytest.raises(ValueError):
        diagnose_governed_planner(run, view='train', model_state='current',
            sample_indices=(2, 0), permutation=(1, 0), seed=9, max_samples=2,
            relation_thresholds=thresholds)
