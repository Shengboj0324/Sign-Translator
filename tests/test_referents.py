from dataclasses import replace
import math

import pytest
import torch

from signtranslator.grammar.sir import EventKind, SIREvent, SIRGraph, sir_sha256
from signtranslator.planning.supervision import GovernedSIRAnnotation
from signtranslator.planning.referents import referent_targets, referent_loss_per_example, ReferentLogits
from test_sir_label_vocabulary import setup


def annotation(base, refs):
    graph = SIRGraph([SIREvent(i, EventKind.MANUAL, 10, float(i), float(i+1), referent=r)
                      for i, r in enumerate(refs)])
    return GovernedSIRAnnotation.create(annotation_id=base.annotation_id, origin=base.origin,
        source=base.source, convention=base.convention, lexicon=base.lexicon,
        lexicon_convention_sha256=base.lexicon_convention_sha256, graph=graph,
        review=replace(base.review, reviewed_sir_sha256=sir_sha256(graph)), created_at=base.created_at)


def test_id_renaming_unknowns_and_unordered_pair_support():
    v, originals = setup()
    a = annotation(originals[0], [0, 0, 123, None])
    b = annotation(originals[0], [917, 917, 0, None])
    x, y = referent_targets([a], v), referent_targets([b], v)
    assert torch.equal(x.equal, y.equal) and torch.equal(x.known, y.known)
    assert x.known.sum() == 3 and x.equal.sum() == 1
    assert not x.known[0, 3].any() and not x.known[0, :, 3].any()
    assert not x.known[0].tril().any()


def test_loss_denominators_gradients_and_unsupported_example():
    v, base = setup()
    annotations = [annotation(base[0], [0, 0, 1]), annotation(base[1], [2, 2]), annotation(base[0], [None])]
    target = referent_targets(annotations, v)
    raw = torch.zeros(3, 3, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        raw[1].fill_(math.log(3.))
    def objective(x):
        symmetric = (x + x.transpose(1, 2)) / 2
        return referent_loss_per_example(ReferentLogits(symmetric, target.annotation_sha256,
                                        target.vocabulary_sha256), annotations, v)
    values, support = objective(raw)
    assert support.tolist() == [True, True, False]
    assert values.tolist() == pytest.approx([math.log(2), math.log(4/3), 0.])
    values.sum().backward()
    domain = target.known | target.known.transpose(1, 2)
    assert not raw.grad[~domain].any()
    assert torch.autograd.gradcheck(lambda x: objective(x)[0], (raw.detach().requires_grad_(),))


def test_identity_nonfinite_and_asymmetry_refused():
    v, base = setup()
    annotations = [annotation(base[0], [0, 1])]
    t = referent_targets(annotations, v)
    good = ReferentLogits(torch.zeros(1, 2, 2), t.annotation_sha256, t.vocabulary_sha256)
    for bad in (replace(good, vocabulary_sha256='a'*64),
                replace(good, values=torch.tensor([[[0., 1.], [0., 0.]]])),
                replace(good, values=torch.full((1, 2, 2), float('nan')))):
        with pytest.raises(ValueError):
            referent_loss_per_example(bad, annotations, v)
