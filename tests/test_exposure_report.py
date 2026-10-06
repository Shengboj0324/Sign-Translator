"""Exposure counts distinguish repeated presentations from corpus availability."""
import hashlib
from copy import deepcopy

import pytest

from test_epoch_commit import make
from test_supported_trainer import mixed


def test_empty_and_repeated_exposure_are_distinct_from_available_samples(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    empty = trainer.exposure_report().to_dict()
    assert empty['admitted_view_samples'] == 3
    assert empty['sample_presentations'] == empty['distinct_presented_samples'] == 0
    assert empty['recorded_optimizer_steps'] == 0 and empty['committed_epoch_boundary']
    assert all(s['presentations'] == 0 for s in empty['samples'])
    trainer.fit()
    report = trainer.exposure_report()
    result = report.to_dict()
    assert result['recorded_optimizer_steps'] == 4
    assert result['completed_epochs'] == 2 and result['committed_epoch_boundary']
    assert result['sample_presentations'] == 6 and result['distinct_presented_samples'] == 3
    assert all(s['presentations'] == 2 for s in result['samples'])
    assert result['branches']['sir_relations'] == {
        'supported_example_presentations': 2, 'steps_with_support': 2, 'steps_without_support': 2}
    assert result['branches']['sir_sequence']['supported_example_presentations'] == 6
    assert report.sha256 == hashlib.sha256(report.payload).hexdigest()
    result['samples'][0]['presentations'] = 999
    assert report.to_dict()['samples'][0]['presentations'] == 2
    assert not result['phase_exit_approved']


def test_exposure_report_preserved_by_resume_and_validation(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    before = trainer.exposure_report()
    trainer.validate()
    assert trainer.exposure_report().payload == before.payload
    path = trainer.save(tmp_path / 'summary.pt')
    restored = make(corpus, vocab); restored.load(path)
    assert restored.exposure_report().payload == before.payload


def test_interrupted_report_identifies_uncommitted_returned_calls(tmp_path, monkeypatch):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    def fail(*args):
        raise RuntimeError('scheduler failed')
    monkeypatch.setattr(type(trainer.sched), 'step', fail)
    with pytest.raises(RuntimeError, match='scheduler failed'):
        trainer.fit()
    result = trainer.exposure_report().to_dict()
    assert not result['committed_epoch_boundary'] and result['completed_epochs'] == 0
    assert result['recorded_optimizer_steps'] == 1 and result['sample_presentations'] == 2
    assert sorted(s['presentations'] for s in result['samples']) == [0, 1, 1]


def test_report_refuses_changed_view_and_does_not_mutate_ledger(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    before = deepcopy(trainer.optimizer_exposure)
    trainer.train_loader.dataset._indices = tuple(reversed(trainer.train_loader.dataset._indices))
    with pytest.raises(ValueError, match='training view changed'):
        trainer.exposure_report()
    assert trainer.optimizer_exposure == before


def test_example_membership_tracks_supported_and_unsupported_presentations(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit()
    records = trainer.optimizer_exposure
    assert records[0]['support_membership']['sir_relations'] == [False, True]
    assert records[1]['support_membership']['sir_relations'] == [False]
    result = trainer.exposure_report().to_dict()
    assert result['schema_version'] == 2
    assert result['unattributed_supported_example_presentations']['sir_relations'] == 0
    counts = [s['branch_presentations']['sir_relations'] for s in result['samples']]
    assert counts == [{'supported': 0, 'unsupported': 2, 'unattributed': 0},
                      {'supported': 2, 'unsupported': 0, 'unattributed': 0},
                      {'supported': 0, 'unsupported': 2, 'unattributed': 0}]


def test_historical_counts_remain_unattributed_without_masks(tmp_path):
    import json
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit()
    records = trainer.optimizer_exposure
    for record in records:
        del record['support_membership']
    # Simulate an older ledger; never infer membership from totals.
    trainer._optimizer_exposure = [json.dumps(r) for r in records]
    result = trainer.exposure_report().to_dict()
    assert result['unattributed_supported_example_presentations']['sir_relations'] == 2
    for sample in result['samples']:
        assert sample['branch_presentations']['sir_relations'] == {
            'supported': 0, 'unsupported': 0, 'unattributed': 2}
