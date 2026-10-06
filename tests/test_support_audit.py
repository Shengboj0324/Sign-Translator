import pytest

from signtranslator.data.governed_corpus import GovernedMotionDataset
from signtranslator.planning.support_audit import audit_supervision_support
from signtranslator.planning.loci import LocusAlphabet
from test_loci import fixture, PAYLOAD
from test_text_sir_labels import setup


def test_actual_masks_count_known_unknown_polarities_and_classes(tmp_path):
    v, corpus, alphabet, _ = fixture(tmp_path)
    report = audit_supervision_support(corpus.split('train'), v, alphabet)
    data = report.to_dict()
    assert data['sample_count'] == 1 and data['event_count'] == 3
    assert data['sequence']['stop_targets'] == 1
    assert [r['target_events'] for r in data['sequence']['labels']] == [1, 1, 1]
    relations = data['relations']['types']
    assert relations['precedence']['negative'] == 6
    assert relations['precedence']['observed_target_polarities'] == 'negative_only'
    assert relations['scope']['positive'] == 1 and relations['scope']['negative'] == 4
    assert relations['scope']['unknown'] == 1 and relations['scope']['observed_target_polarities'] == 'both'
    assert relations['overlap']['unknown'] == 6 and relations['overlap']['positive'] == 0
    assert relations['locus']['observed_target_polarities'] == 'none'
    assert data['referent_equality']['unknown'] == 3
    assert data['locus_assignment']['known_events'] == 2 and data['locus_assignment']['unknown_events'] == 1
    assert data['phase_exit_approved'] is False
    original = report.sha256
    data['sample_count'] = 999
    assert report.sha256 == original and report.to_dict()['sample_count'] == 1
    assert audit_supervision_support(corpus.split('train'), v, alphabet).payload == report.payload


def test_polarity_counts_reference_equality_are_distinct_from_recorded_coref(tmp_path):
    v, corpus = setup(tmp_path, convention_payload=PAYLOAD, referents=(0, 0, 7), loci=(0, 0, 1))
    a = LocusAlphabet(v.convention, PAYLOAD)
    d = audit_supervision_support(corpus.split('train'), v, a).to_dict()
    assert d['referent_equality']['positive'] == 1 and d['referent_equality']['negative'] == 2
    assert d['referent_equality']['unknown'] == 0
    assert d['relations']['types']['coref']['positive'] == 0
    assert d['relations']['types']['coref']['negative'] == 4
    assert d['relations']['types']['coref']['unknown'] == 2
    assert [r['target_events'] for r in d['locus_assignment']['classes']] == [2, 1]
    assert [r['examples'] for r in d['locus_assignment']['classes']] == [1, 1]


def test_full_view_subset_and_split_identities_are_not_conflated(tmp_path):
    v, corpus = setup(tmp_path, convention_payload=PAYLOAD,
                      configurations=[{}, {}, {'split': 'val'}])
    a = LocusAlphabet(v.convention, PAYLOAD)
    full = audit_supervision_support(corpus.split('train'), v, a).to_dict()
    subset = audit_supervision_support(GovernedMotionDataset(corpus, (1,), 'train'), v, a).to_dict()
    val = audit_supervision_support(corpus.split('val'), v, a).to_dict()
    assert full['event_count'] == 2 * subset['event_count']
    assert full['data_contract']['record_indices'] == [0, 1]
    assert subset['data_contract']['record_indices'] == [1]
    assert val['data_contract']['split'] == 'val' and val['data_contract']['record_indices'] == [2]


def test_mutated_view_and_changed_source_bytes_fail(tmp_path):
    v, corpus, alphabet, _ = fixture(tmp_path)
    view = corpus.split('train')
    view._indices = (0, 0)
    with pytest.raises(ValueError, match='unique'):
        audit_supervision_support(view, v, alphabet)
    view = corpus.split('train')
    corpus._records[0].transcript_path.write_bytes(b'changed after admission')
    with pytest.raises((ValueError, PermissionError)):
        audit_supervision_support(view, v, alphabet)
