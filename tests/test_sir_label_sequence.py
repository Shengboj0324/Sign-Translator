"""Analytical sequence/stop contracts, using fictional governed annotations."""
import math
from dataclasses import replace

import pytest
import torch

from signtranslator.planning.label_sequence import (
    SIRLabelSequenceLogits, label_sequence_targets, label_sequence_loss,
)
from test_sir_label_vocabulary import setup


def prediction(values, vocab, annotations):
    return SIRLabelSequenceLogits(values, vocab.lexicon.sha256,
                                  tuple(a.content_sha256() for a in annotations))


def test_shifted_teacher_inputs_and_single_stop_preserve_kind_namespaces():
    vocab, annotations = setup()
    target = label_sequence_targets(annotations, vocab)
    # a = class 0; b = classes 1,2. Input START=1; output STOP=0.
    assert target.inputs.tolist() == [[1, 2, 0], [1, 3, 4]]
    assert target.outputs.tolist() == [[1, 0, -1], [2, 3, 0]]
    assert target.lengths.tolist() == [2, 3]
    assert target.outputs.eq(0).sum(dim=1).tolist() == [1, 1]


def test_analytical_per_example_nll_includes_stop_and_excludes_padding():
    vocab, annotations = setup()
    probs = torch.tensor([[[.1, .6, .1, .2], [.7, .1, .1, .1], [.25]*4],
                          [[.1, .2, .5, .2], [.1, .1, .2, .6], [.4, .2, .2, .2]]],
                         dtype=torch.float64)
    scores = probs.log().requires_grad_()
    loss = label_sequence_loss(prediction(scores, vocab, annotations), annotations, vocab)
    expected = ((-math.log(.6)-math.log(.7))/2 + (-math.log(.5)-math.log(.6)-math.log(.4))/3)/2
    assert loss.item() == pytest.approx(expected, abs=1e-14)
    loss.backward()
    assert not scores.grad[0, 2].any()
    assert scores.grad[0, 1, 0] != 0  # stop target has a real gradient
    scores2 = probs.log().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: label_sequence_loss(prediction(x, vocab, annotations), annotations, vocab),
        (scores2,))


def test_sequence_binding_and_shape_cannot_be_reused_for_other_targets():
    vocab, annotations = setup()
    good = prediction(torch.zeros(2, 3, 4), vocab, annotations)
    for bad in (replace(good, annotation_sha256=tuple(reversed(good.annotation_sha256))),
                replace(good, vocabulary_sha256='a'*64),
                replace(good, values=torch.zeros(2, 2, 4)),
                replace(good, values=torch.full((2, 3, 4), float('nan')))):
        with pytest.raises(ValueError):
            label_sequence_loss(bad, annotations, vocab)


def test_finite_logits_with_overflowing_nll_are_rejected():
    vocab, annotations = setup()
    scores = torch.full((2, 3, 4), torch.finfo(torch.float32).max)
    scores[0, 0, 1] = -torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match='overflow'):
        label_sequence_loss(prediction(scores, vocab, annotations), annotations, vocab)
