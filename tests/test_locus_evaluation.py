from dataclasses import replace
from decimal import Decimal, localcontext
import math

import pytest
import torch

from signtranslator.planning.locus_evaluation import evaluate_locus_sequences, _categorical_scores
from signtranslator.planning.text_loci import LocusSequenceCandidate
from signtranslator.planning.text_referents import ReferentSequenceCandidate
from signtranslator.planning.text_relations import RelationalSequenceCandidate
from signtranslator.planning.text_timing import TemporalSequenceCandidate, TimedLabel
from test_loci import fixture as locus_fixture


def fixture(tmp_path, loci=(0, None, 1)):
    v, _, alphabet, annotations = locus_fixture(tmp_path, loci)
    a = tuple(annotations)
    temporal = TemporalSequenceCandidate(tuple(TimedLabel(label, e.t_start, e.t_end)
                                              for label, e in zip(v.entries, a[0].graph().events)),
                                         'terminated', v.entries, v.lexicon.sha256, 0.)
    referential = ReferentSequenceCandidate(RelationalSequenceCandidate(temporal, None, None), None, None)
    c = LocusSequenceCandidate(referential, torch.zeros(3, 2), alphabet.convention.sha256, alphabet.identities)
    return v, a, alphabet, (c,)


def evaluate(v, a, alphabet, c, budget=6):
    return evaluate_locus_sequences(c, a, v, alphabet, annotation_sha256=tuple(x.content_sha256() for x in a),
                                    max_cells=100, max_locus_cells=budget)


def test_uniform_known_unknown_and_class_support(tmp_path):
    v, a, alphabet, c = fixture(tmp_path)
    report = evaluate(v, a, alphabet, c)
    result = report.to_dict()
    assert result['conditional_event_weighted'] == dict(known=2, unknown=1, nll=math.log(2), brier=.5)
    assert [row['known'] for row in result['per_target_class']] == [1, 1]
    c[0].locus_logits[1] = torch.tensor([100., -100.])
    assert evaluate(v, a, alphabet, c).payload == report.payload
    result['rows'].clear()
    assert len(report.to_dict()['rows']) == 1


@pytest.mark.parametrize('logits,target', [([350., 0.], 0), ([40., 0., -1.], 0),
                                          ([0., 0., 0.], 1), ([0., 40.], 0), ([0.], 0)])
def test_categorical_scores_high_precision_oracle(logits, target):
    with localcontext() as context:
        context.prec = 400
        weights = [Decimal(x).exp() for x in logits]
        probabilities = [x / sum(weights) for x in weights]
        nll = float(-probabilities[target].ln())
        brier = float(sum((p - int(i == target))**2 for i, p in enumerate(probabilities)))
    observed = _categorical_scores(logits, target)
    assert math.isclose(observed[0], nll, rel_tol=3e-15, abs_tol=0.)
    assert math.isclose(observed[1], brier, rel_tol=3e-15, abs_tol=0.)
    assert _categorical_scores(logits[::-1], len(logits)-target-1) == pytest.approx(observed, rel=3e-15, abs=0.)


def test_unrepresentable_nll_refused():
    with pytest.raises(ValueError, match='representable'):
        _categorical_scores([1e308, -1e308], 1)


def test_unknown_targets_and_binding_budget_failures(tmp_path):
    v, a, alphabet, c = fixture(tmp_path, loci=(None, None, None))
    result = evaluate(v, a, alphabet, c).to_dict()
    assert result['supported_example_coverage'] == dict(numerator=0, denominator=1)
    assert result['conditional_event_weighted'] == dict(known=0, unknown=3, nll=None, brier=None)
    with pytest.raises(ValueError, match='budget'):
        evaluate(v, a, alphabet, c, budget=5)
    for changed in (replace(c[0], locus_identities=alphabet.identities[::-1]),
                    replace(c[0], convention_sha256='0'*64),
                    replace(c[0], locus_logits=torch.full((3, 2), float('nan')))):
        with pytest.raises(ValueError):
            evaluate(v, a, alphabet, (changed,))


def test_failed_and_mismatched_generations_remain_unavailable(tmp_path):
    v, a, alphabet, c = fixture(tmp_path)
    temporal = c[0].referential.relational.temporal
    failed = replace(temporal, events=(), diagnostic_prefix=(), status='empty_prediction')
    mismatch = replace(temporal, events=tuple(replace(e, label=v.entries[2]) for e in temporal.events),
                       diagnostic_prefix=(v.entries[2],)*3)
    for replacement in (failed, mismatch):
        candidate = replace(c[0], referential=replace(c[0].referential,
            relational=replace(c[0].referential.relational, temporal=replacement)),
            locus_logits=None if replacement.status == 'empty_prediction' else c[0].locus_logits)
        data = evaluate(v, a, alphabet, (candidate,)).to_dict()
        assert data['label_coverage'] == dict(numerator=0, denominator=1)
        assert data['conditional_event_weighted']['nll'] is None
    bad = replace(candidate, referential=replace(candidate.referential,
        relational=replace(candidate.referential.relational, temporal=failed)))
    with pytest.raises(ValueError, match='failed generation'):
        evaluate(v, a, alphabet, (bad,))
