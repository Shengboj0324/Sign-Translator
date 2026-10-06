"""Fictional lexicon coverage, identity and categorical-objective checks."""
from dataclasses import replace
import hashlib
import json
import math

import pytest
import torch

from signtranslator.grammar.sir import EventKind, SIREvent, SIRGraph
from signtranslator.planning.label_vocabulary import (
    GovernedLabelVocabulary, AlignedSIRLabelLogits, encode_governed_labels, aligned_sir_label_loss,
)
from test_phase3_governance import _annotation, _sample
from test_sir_tensors import annotation


ROWS = [dict(kind='manual', label_id=10, identity='fixture:manual-a'),
        dict(kind='manual', label_id=11, identity='fixture:manual-b'),
        dict(kind='nonmanual', label_id=10, identity='fixture:marker-a')]


def setup(rows=None, change=None):
    a = _annotation(0, _sample(0))
    b = annotation(SIRGraph([SIREvent(1, EventKind.MANUAL, 11, 0., .5),
                             SIREvent(2, EventKind.NONMANUAL, 10, 0., 1.)]))
    payload = dict(schema_version=1, artifact_id=a.lexicon.artifact_id, version=a.lexicon.version,
                   convention_sha256=a.convention.sha256, entries=ROWS if rows is None else rows)
    if change:
        payload.update(change)
    raw = json.dumps(payload).encode()
    lexicon = replace(a.lexicon, sha256=hashlib.sha256(raw).hexdigest())
    vocab = GovernedLabelVocabulary(lexicon, a.convention, raw)
    annotations = tuple(replace(item, lexicon=lexicon,
                               review=replace(item.review, reviewed_lexicon_sha256=lexicon.sha256))
                        for item in (a, b))
    return vocab, annotations


def prediction(scores, vocab, annotations):
    return AlignedSIRLabelLogits(scores, vocab.lexicon.sha256,
                                tuple(a.content_sha256() for a in annotations))


def test_namespaces_sparse_raw_ids_and_padding_are_distinct(tmp_path):
    vocab, annotations = setup()
    assert encode_governed_labels(annotations, vocab).tolist() == [[0, -1], [1, 2]]
    path = tmp_path / 'lexicon.json'; path.write_bytes(vocab.payload)
    restored = GovernedLabelVocabulary.load(path, lexicon=vocab.lexicon, convention=vocab.convention)
    assert restored.entries == vocab.entries
    path.write_bytes(vocab.payload + b' ')
    with pytest.raises(ValueError, match='bytes'):
        GovernedLabelVocabulary.load(path, lexicon=vocab.lexicon, convention=vocab.convention)


def test_unknown_marker_cannot_be_replaced_by_same_number_manual_label():
    vocab, annotations = setup(rows=ROWS[:2])
    with pytest.raises(ValueError, match='uncovered.*nonmanual:10'):
        encode_governed_labels(annotations, vocab)


@pytest.mark.parametrize('rows', [[], ROWS + [ROWS[0]],
    [dict(kind='manual', label_id=True, identity='fixture')],
    [dict(kind='manual', label_id=2**63, identity='fixture')],
    [dict(kind='manual', label_id=-1, identity='fixture')],
    [dict(kind='manual', label_id=10, identity='')],
    [dict(kind='english', label_id=10, identity='fixture')],
    [dict(kind='manual', label_id=10, identity='same'),
     dict(kind='manual', label_id=11, identity='same')],
])
def test_invalid_and_ambiguous_vocabularies_rejected(rows):
    with pytest.raises(ValueError):
        setup(rows=rows)


@pytest.mark.parametrize('change', [{'schema_version': True}, {'version': 'different'},
                                    {'convention_sha256': '0'*64}, {'extra': 0}])
def test_rehashed_bad_contract_still_rejected(change):
    with pytest.raises(ValueError):
        setup(change=change)


def test_duplicate_json_field_is_not_last_value_wins():
    vocab, _ = setup()
    raw = vocab.payload.replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1')
    lexicon = replace(vocab.lexicon, sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match='duplicate'):
        GovernedLabelVocabulary(lexicon, vocab.convention, raw)


def test_loss_matches_declared_per_example_average_and_ignores_padding():
    vocab, annotations = setup()
    scores = torch.log(torch.tensor([[[.5, .25, .25], [.2, .3, .5]],
                                      [[.2, .7, .1], [.2, .2, .6]]], dtype=torch.float64)).requires_grad_()
    loss = aligned_sir_label_loss(prediction(scores, vocab, annotations), annotations, vocab)
    expected = (-math.log(.5) + (-math.log(.7) - math.log(.6))/2)/2
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-12)
    loss.backward()
    assert torch.equal(scores.grad[0, 1], torch.zeros(3, dtype=torch.float64))
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0, 0, 0].item() == pytest.approx((.5 - 1)/2)
    assert scores.grad[1, 0, 1].item() == pytest.approx((.7 - 1)/4)


def test_float64_gradient_and_optimizer_step():
    vocab, annotations = setup()
    scores = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    objective = lambda values: aligned_sir_label_loss(prediction(values, vocab, annotations), annotations, vocab)
    assert torch.autograd.gradcheck(objective, (scores,), eps=1e-6, atol=1e-5)
    parameter = torch.nn.Parameter(torch.zeros(2, 2, 3, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=.5)
    initial = float(objective(parameter).detach())
    for _ in range(10):
        optimizer.zero_grad(); objective(parameter).backward(); optimizer.step()
    assert float(objective(parameter).detach()) < initial


def test_binding_or_numerical_corruption_cannot_produce_a_loss():
    vocab, annotations = setup()
    pred = prediction(torch.zeros(2, 2, 3), vocab, annotations)
    with pytest.raises(ValueError, match='binding'):
        aligned_sir_label_loss(replace(pred, vocabulary_sha256='0'*64), annotations, vocab)
    with pytest.raises(ValueError, match='binding'):
        aligned_sir_label_loss(replace(pred, annotation_sha256=tuple(reversed(pred.annotation_sha256))), annotations, vocab)
    with pytest.raises(ValueError, match='finite'):
        aligned_sir_label_loss(replace(pred, values=torch.full((2, 2, 3), float('nan'))), annotations, vocab)
    with pytest.raises(ValueError, match='shape'):
        aligned_sir_label_loss(replace(pred, values=torch.zeros(2, 2, 2)), annotations, vocab)


def test_finite_logits_with_unrepresentable_loss_fail_explicitly():
    vocab, annotations = setup()
    maximum = torch.finfo(torch.float32).max
    scores = torch.zeros(2, 2, 3)
    scores[0, 0] = torch.tensor([-maximum, maximum, maximum])
    with pytest.raises(ValueError, match='overflowed'):
        aligned_sir_label_loss(prediction(scores, vocab, annotations), annotations, vocab)
