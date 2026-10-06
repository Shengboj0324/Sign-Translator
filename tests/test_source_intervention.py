import random

import numpy as np
import pytest
import torch

from signtranslator.planning.loci import LocusAlphabet
from signtranslator.planning.source_intervention import compare_source_intervention
from test_text_sir_labels import setup
from test_text_loci import model
from test_loci import PAYLOAD


def fixture(tmp_path):
    v, corpus = setup(tmp_path, convention_payload=PAYLOAD, configurations=[{}, {}])
    return model(v, LocusAlphabet(v.convention, PAYLOAD)), corpus.split('train')


def test_identity_control_reproducibility_rng_and_model_mode(tmp_path):
    net, view = fixture(tmp_path)
    net.train()
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    report = compare_source_intervention(net, view, permutation=(0, 1), seed=123, max_samples=2)
    assert net.training and torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and np.array_equal(after[1], numpy_before[1]) and after[2:] == numpy_before[2:]
    assert compare_source_intervention(net, view, permutation=(0, 1), seed=123, max_samples=2).payload == report.payload
    data = report.to_dict()
    assert all(not r['source_bytes_changed'] and not r['raw_candidate_changed'] for r in data['rows'])
    assert data['phase_exit_approved'] is False
    data['rows'].clear()
    assert len(report.to_dict()['rows']) == 2


def test_source_permutation_reaches_model_without_rewriting_annotations(tmp_path, monkeypatch):
    net, view = fixture(tmp_path)
    calls = []
    actual = net.generate_loci
    def capture(ids, lengths, *, origins_seconds):
        calls.append((ids.clone(), lengths.clone(), origins_seconds.clone()))
        return actual(ids, lengths, origins_seconds=origins_seconds)
    monkeypatch.setattr(net, 'generate_loci', capture)
    data = compare_source_intervention(net, view, permutation=(1, 0), seed=7, max_samples=2).to_dict()
    assert len(calls) == 2
    assert torch.equal(calls[1][0], calls[0][0].flip(0))
    assert torch.equal(calls[1][1], calls[0][1].flip(0))
    assert torch.equal(calls[1][2], calls[0][2])
    assert all(r['source_bytes_changed'] for r in data['rows'])
    assert data['rows'][0]['intervened_transcript_sha256'] == data['rows'][1]['original_transcript_sha256']
    assert data['rows'][0]['source_annotation_sha256'] == data['rows'][1]['anchor_annotation_sha256']
    for r in data['rows']:
        if not r['both_terminated']:
            assert r['same_label_sequence'] is None


def test_invalid_permutation_changed_sources_and_failure_mode_restore(tmp_path, monkeypatch):
    net, view = fixture(tmp_path)
    for permutation in ((0, 0), (1,), (True, 0), (0, 2)):
        with pytest.raises(ValueError):
            compare_source_intervention(net, view, permutation=permutation, seed=0, max_samples=2)
    net.train()
    before = torch.get_rng_state().clone()
    def fail(*args, **kwargs):
        torch.rand(5)
        raise RuntimeError('fixture failure')
    monkeypatch.setattr(net, 'generate_loci', fail)
    with pytest.raises(RuntimeError, match='fixture failure'):
        compare_source_intervention(net, view, permutation=(1, 0), seed=0, max_samples=2)
    assert net.training and torch.equal(torch.get_rng_state(), before)
    view[0].corpus._records[0].transcript_path.write_bytes(b'changed')
    with pytest.raises((ValueError, PermissionError)):
        compare_source_intervention(net, view, permutation=(1, 0), seed=0, max_samples=2)


def test_mps_rng_wrapper_restores_state_even_when_intervention_fails(monkeypatch):
    from signtranslator.planning.source_intervention import _intervention_rng
    saved = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'get_rng_state', lambda: saved.clone())
    monkeypatch.setattr(torch.mps, 'set_rng_state', lambda state: restored.append(state.clone()))
    monkeypatch.setattr(torch.mps, 'manual_seed', lambda seed: None)
    with pytest.raises(RuntimeError, match='probe'):
        with _intervention_rng(5):
            raise RuntimeError('probe')
    assert len(restored) == 1 and torch.equal(restored[0], saved)


def test_original_reference_metrics_use_free_generated_labels(tmp_path):
    from test_text_sir_sequence import ScriptedScores
    net, view = fixture(tmp_path)
    net.classifier = ScriptedScores([[1, 1], [2, 2], [3, 3], [0, 0]] * 2, 4)
    report = compare_source_intervention(net, view, permutation=(1, 0), seed=7, max_samples=2).to_dict()
    evaluation = report['original_reference_evaluation']
    assert evaluation['overall_exact_match'] == dict(numerator=2, denominator=2)
    assert evaluation['conditional_edit_rate'] == dict(numerator=0, denominator=6)
    assert [r['annotation_sha256'] for r in evaluation['rows']] == [
        r['anchor_annotation_sha256'] for r in report['rows']]
    assert evaluation['phase_exit_approved'] is False
    assert report['original_temporal_evaluation']['coverage'] == dict(numerator=2, denominator=2)
    assert report['original_temporal_evaluation']['conditional_event_weighted']['events'] == 6


def test_original_relation_evaluation_snapshots_before_intervention_mutation(tmp_path, monkeypatch):
    from test_text_sir_sequence import ScriptedScores
    net, view = fixture(tmp_path)
    net.classifier = ScriptedScores([[1, 1], [2, 2], [3, 3], [0, 0]] * 2, 4)
    actual = net.generate_loci
    previous = []
    def generate(*args, **kwargs):
        if previous:
            for candidate in previous:
                candidate.referential.relational.relation_logits.fill_(float('nan'))
                candidate.referential.equality_logits.fill_(float('nan'))
                candidate.locus_logits.fill_(float('nan'))
        result = actual(*args, **kwargs)
        previous.extend(result)
        return result
    monkeypatch.setattr(net, 'generate_loci', generate)
    report = compare_source_intervention(net, view, permutation=(1, 0), seed=7, max_samples=2).to_dict()
    evaluation = report['original_relation_evaluation']
    assert evaluation['coverage'] == dict(numerator=2, denominator=2)
    assert any(row['known'] > 0 for row in evaluation['conditional_cell_weighted'].values())
    reference = report['original_referent_evaluation']
    assert reference['label_coverage'] == dict(numerator=2, denominator=2)
    assert reference['conditional_pair_weighted']['unknown'] == 6
    assert report['original_locus_evaluation']['label_coverage'] == dict(numerator=2, denominator=2)
    assert report['original_locus_evaluation']['conditional_event_weighted']['unknown'] == 6
