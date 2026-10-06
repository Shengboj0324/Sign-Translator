from dataclasses import replace
from functools import lru_cache
from itertools import product

import pytest

from signtranslator.planning.sequence_evaluation import _edit_distance, evaluate_label_sequences
from signtranslator.planning.text_sequence import LabelSequenceCandidate
from test_text_sir_labels import setup


def test_distance_matches_exhaustive_recursive_oracle():
    @lru_cache(None)
    def oracle(a, b):
        if not a or not b:
            return len(a) + len(b)
        return min(1 + oracle(a[1:], b), 1 + oracle(a, b[1:]),
                   (a[0] != b[0]) + oracle(a[1:], b[1:]))
    sequences = [s for n in range(5) for s in product(range(2), repeat=n)]
    for a in sequences:
        for b in sequences:
            assert _edit_distance(a, b) == oracle(a, b)


def inputs(tmp_path):
    vocab, corpus = setup(tmp_path, configurations=[{}, {}, {}], single_event_indices=(1,))
    annotations = tuple(r.annotation for r in corpus._records)
    def candidate(labels, status='terminated'):
        return LabelSequenceCandidate(labels if status == 'terminated' else (), status,
                                      labels, vocab.lexicon.sha256)
    candidates = (candidate(vocab.entries), candidate(vocab.entries),
                  candidate(vocab.entries[:1], 'capacity_exceeded'))
    return vocab, annotations, candidates


def score(vocab, annotations, candidates, **kwargs):
    return evaluate_label_sequences(candidates, annotations, vocab,
                                    annotation_sha256=tuple(a.content_sha256() for a in annotations),
                                    max_cells=kwargs.get('max_cells', 12))


def test_counts_failure_denominators_and_immutable_report(tmp_path):
    v, a, c = inputs(tmp_path)
    report = score(v, a, c)
    data = report.to_dict()
    assert data['termination'] == dict(numerator=2, denominator=3)
    assert data['overall_exact_match'] == dict(numerator=1, denominator=3)
    assert data['conditional_edit_rate'] == dict(numerator=2, denominator=4)
    assert data['rows'][2]['edit_distance'] is None
    assert data['rows'][2]['exact_match'] is False
    assert data['edit_cells'] == 12
    data['rows'].clear()
    assert len(report.to_dict()['rows']) == 3
    assert report.sha256 == score(v, a, c).sha256
    failed = tuple(replace(x, labels=(), status='empty_prediction', diagnostic_prefix=()) for x in c)
    data = score(v, a, failed).to_dict()
    assert data['conditional_edit_rate'] is None
    assert data['overall_exact_match'] == dict(numerator=0, denominator=3)


def test_rate_above_one_binding_budget_and_malformed_candidates(tmp_path):
    v, a, c = inputs(tmp_path)
    data = score(v, a[1:2], c[1:2]).to_dict()
    assert data['conditional_edit_rate'] == dict(numerator=2, denominator=1)
    with pytest.raises(ValueError, match='budget'):
        score(v, a, c, max_cells=11)
    with pytest.raises(ValueError, match='binding'):
        evaluate_label_sequences(c, a, v, annotation_sha256=tuple(x.content_sha256() for x in a)[::-1], max_cells=12)
    for changed in (replace(c[0], vocabulary_sha256='0' * 64),
                    replace(c[0], status='invented'), replace(c[0], labels=()),
                    replace(c[0], diagnostic_prefix=()),
                    replace(c[0], labels=(replace(v.entries[0], label_id=True),))):
        with pytest.raises(ValueError):
            score(v, a, (changed, *c[1:]))
    with pytest.raises(ValueError, match='duplicate'):
        score(v, (a[0], a[0]), (c[0], c[0]))
