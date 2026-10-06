"""Known/unknown relation support and analytical multilabel supervision."""
from dataclasses import replace
import math

import pytest
import torch

from signtranslator.grammar.sir import EdgeType, EventKind, SIREdge, SIREvent, SIRGraph, sir_sha256, validate_sir
from signtranslator.planning.supervision import GovernedSIRAnnotation
from signtranslator.planning.relations import relation_targets, relation_loss, SIRRelationLogits
from signtranslator.planning.tensors import EDGE_TYPES
from test_sir_label_vocabulary import setup


def fixture():
    vocab, originals = setup()
    events = [SIREvent(0, EventKind.MANUAL, 10, 0., 1., referent=0, locus=0),
              SIREvent(1, EventKind.MANUAL, 11, 1., 2., referent=1, locus=1),
              SIREvent(2, EventKind.NONMANUAL, 10, 0., 2.)]
    graphs = [SIRGraph(events, [SIREdge(0, 1, EdgeType.PRECEDENCE), SIREdge(0, 1, EdgeType.LOCUS),
                               SIREdge(2, 0, EdgeType.SCOPE), SIREdge(2, 0, EdgeType.OVERLAP)]),
              SIRGraph([events[0], events[2]], [SIREdge(2, 0, EdgeType.SCOPE)])]
    annotations = []
    for base, graph in zip(originals, graphs):
        annotations.append(GovernedSIRAnnotation.create(
            annotation_id=base.annotation_id, origin=base.origin, source=base.source,
            convention=base.convention, lexicon=base.lexicon,
            lexicon_convention_sha256=base.lexicon_convention_sha256, graph=graph,
            review=replace(base.review, reviewed_sir_sha256=sir_sha256(graph)), created_at=base.created_at))
    return vocab, annotations


def test_missing_edges_and_missing_referents_are_unknown_not_false():
    vocab, annotations = fixture()
    targets = relation_targets(annotations, vocab)
    index = EDGE_TYPES.index
    assert targets.positive[0, 2, 0, index(EdgeType.SCOPE)]
    assert targets.positive[0, 2, 0, index(EdgeType.OVERLAP)]  # simultaneous relation classes
    assert not targets.known[0, 0, 2, index(EdgeType.OVERLAP)]  # no automatic reverse closure
    assert not targets.known[0, 0, 2, index(EdgeType.COREF)]  # missing referent is unknown
    assert not targets.known[0, 2, 0, index(EdgeType.LOCUS)]  # no closed-world negatives
    assert targets.known[0, 0, 1, index(EdgeType.COREF)]  # two explicit different referents
    assert not targets.positive[0, 0, 1, index(EdgeType.COREF)]
    assert targets.known[0, 0, 1, index(EdgeType.OVERLAP)]  # half-open intervals touch only
    assert not targets.known[0, 0, 0].any()  # self edges excluded
    assert not targets.known[1, 2].any() and not targets.known[1, :, 2].any()  # padding


def test_analytical_loss_unequal_support_and_unknown_gradients():
    vocab, annotations = fixture()
    target = relation_targets(annotations, vocab)
    scores = torch.zeros(target.known.shape, dtype=torch.float64)
    scores[1] = math.log(3.)
    scores.requires_grad_()
    def objective(x):
        return relation_loss(SIRRelationLogits(x, target.annotation_sha256, target.vocabulary_sha256),
                              annotations, vocab)
    counts = target.known.flatten(1).sum(1)
    assert counts[0] != counts[1]
    positives = int(target.positive[1].sum())
    negatives = int(counts[1]) - positives
    expected = (math.log(2.) + (positives * math.log(4/3) + negatives * math.log(4.)) / int(counts[1])) / 2
    loss = objective(scores)
    assert loss.item() == pytest.approx(expected, abs=1e-14)
    loss.backward()
    assert not scores.grad[~target.known].any()
    assert torch.autograd.gradcheck(objective, (scores.detach().requires_grad_(),))


def test_unsupported_example_fails_instead_of_successful_zero_loss():
    vocab, annotations = setup()
    target = relation_targets(annotations[:1], vocab)
    assert not target.known.any()
    with pytest.raises(ValueError, match='unavailable'):
        relation_loss(SIRRelationLogits(torch.zeros(target.known.shape), target.annotation_sha256,
                                        target.vocabulary_sha256), annotations[:1], vocab)


def test_relation_identity_and_codebook_are_checked():
    vocab, annotations = fixture()
    target = relation_targets(annotations, vocab)
    good = SIRRelationLogits(torch.zeros(target.known.shape), target.annotation_sha256, target.vocabulary_sha256)
    for bad in (replace(good, relation_types=tuple(reversed(EDGE_TYPES))),
                replace(good, vocabulary_sha256='a'*64),
                replace(good, values=torch.full(target.known.shape, float('nan')))):
        with pytest.raises(ValueError):
            relation_loss(bad, annotations, vocab)


def test_locus_edge_requires_a_declared_target_locus_and_preserves_zero():
    graph = SIRGraph([SIREvent(0, EventKind.MANUAL, 10, 0., 1.),
                      SIREvent(1, EventKind.MANUAL, 11, 1., 2.)], [SIREdge(0, 1, EdgeType.LOCUS)])
    assert 'locus_target_missing' in validate_sir(graph)
    graph.events[1].locus = 0
    assert validate_sir(graph) == []
