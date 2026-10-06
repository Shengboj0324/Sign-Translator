from dataclasses import replace
import hashlib
import json
import math

import pytest
import torch

from signtranslator.planning.loci import LocusAlphabet, LocusLogits, locus_targets, locus_loss_per_example
from test_text_sir_labels import setup
from test_governed_trainer import loader


PAYLOAD = json.dumps({'locus_schema_version': 1, 'loci': [
    {'id': 0, 'identity': 'fictional-left'}, {'id': 1, 'identity': 'fictional-right'}]}).encode()


def fixture(tmp_path, loci=(0, None, 1)):
    v, corpus = setup(tmp_path, convention_payload=PAYLOAD, loci=loci)
    alphabet = LocusAlphabet(v.convention, PAYLOAD)
    annotations = next(iter(loader(corpus, 'train'))).annotations
    return v, corpus, alphabet, annotations


def test_known_locus_loss_unknown_gradient_and_numeric_derivative(tmp_path):
    v, _, alphabet, annotations = fixture(tmp_path)
    target = locus_targets(annotations, v, alphabet)
    assert target.classes.tolist() == [[0, -1, 1]]
    scores = torch.tensor([[[math.log(3), 0.], [4., -9.], [0., math.log(3)]]],
                          dtype=torch.float64, requires_grad=True)
    def loss(x):
        return locus_loss_per_example(LocusLogits(x, target.annotation_sha256,
            target.convention_sha256, target.vocabulary_sha256), annotations, v, alphabet)[0]
    assert loss(scores).item() == pytest.approx(math.log(4/3))
    loss(scores).sum().backward()
    assert not scores.grad[0, 1].any()
    assert torch.autograd.gradcheck(loss, (scores.detach().requires_grad_(),))


def test_alphabet_bytes_class_order_and_coverage_are_bound(tmp_path):
    v, _, alphabet, annotations = fixture(tmp_path, loci=(0, None, 2))
    with pytest.raises(ValueError, match='outside'):
        locus_targets(annotations, v, alphabet)
    with pytest.raises(ValueError, match='exact convention bytes'):
        LocusAlphabet(v.convention, PAYLOAD + b' ')
    for rows in ([{'id': True, 'identity': 'x'}], [{'id': 1, 'identity': 'x'}],
                 [{'id': 0, 'identity': 'x'}, {'id': 1, 'identity': 'x'}], []):
        payload = json.dumps({'locus_schema_version': 1, 'loci': rows}).encode()
        with pytest.raises(ValueError):
            LocusAlphabet(replace(v.convention, sha256=hashlib.sha256(payload).hexdigest()), payload)
    duplicate = b'{"locus_schema_version":1,"locus_schema_version":1,"loci":[]}'
    with pytest.raises(ValueError, match='duplicate'):
        LocusAlphabet(replace(v.convention, sha256=hashlib.sha256(duplicate).hexdigest()), duplicate)


def test_unsupported_and_prediction_identity_fail_closed(tmp_path):
    v, _, alphabet, annotations = fixture(tmp_path, loci=(None, None, None))
    t = locus_targets(annotations, v, alphabet)
    p = LocusLogits(torch.zeros(1, 3, 2), t.annotation_sha256, t.convention_sha256, t.vocabulary_sha256)
    values, support = locus_loss_per_example(p, annotations, v, alphabet)
    assert values.tolist() == [0.] and support.tolist() == [False]
    for bad in (replace(p, convention_sha256='f'*64), replace(p, values=torch.zeros(1, 3, 3)),
                replace(p, values=torch.full((1, 3, 2), float('nan')))):
        with pytest.raises(ValueError):
            locus_loss_per_example(bad, annotations, v, alphabet)
