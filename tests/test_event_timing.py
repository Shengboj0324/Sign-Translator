"""Continuous clock and timing-loss contracts on fictional annotations."""
import pytest
import torch

from signtranslator.planning.event_timing import (
    AlignedEventIntervals, aligned_timing_loss, event_intervals,
)
from test_sir_label_vocabulary import setup


def prediction(values, vocab, annotations):
    return AlignedEventIntervals(values, vocab.lexicon.sha256,
                                 tuple(a.content_sha256() for a in annotations))


def test_huber_averages_endpoints_events_and_examples_with_zero_padding_gradient():
    vocab, annotations = setup()
    values = torch.tensor([[[1., 3.], [0., 0.]], [[-1., .5], [2., 4.]]],
                          dtype=torch.float64, requires_grad=True)
    loss = aligned_timing_loss(prediction(values, vocab, annotations), annotations, vocab,
                               scale_seconds=1.)
    # a endpoint mean = 1; b event means = .25, 2; equal example weighting.
    assert loss.item() == pytest.approx(1.0625)
    loss.backward()
    assert not values.grad[0, 1].any()
    assert values.grad[0, 0].tolist() == pytest.approx([.25, .25])


def test_timing_parameterization_has_numerically_correct_gradients():
    vocab, annotations = setup()
    mask = torch.tensor([[True, False], [True, True]])
    origins = torch.tensor([-.3, -.2], dtype=torch.float64)
    raw = torch.tensor([[[.2, -.1], [5., 5.]], [[-.5, .3], [.1, .4]]],
                       dtype=torch.float64, requires_grad=True)
    def objective(parameters):
        intervals = event_intervals(parameters, origins, mask)
        return aligned_timing_loss(prediction(intervals, vocab, annotations), annotations, vocab,
                                    scale_seconds=.7)
    assert torch.autograd.gradcheck(objective, (raw,))
    objective(raw).backward()
    assert not raw.grad[0, 1].any()


def test_clock_shift_overlap_and_serialization_order_are_preserved():
    raw = torch.tensor([[[1., 2.], [-1., 3.]]], dtype=torch.float32)
    valid = torch.tensor([[True, True]])
    origin = torch.tensor([-4.], dtype=torch.float64)
    a = event_intervals(raw, origin, valid)
    b = event_intervals(raw, origin + 32., valid)
    assert a.dtype == torch.float64
    torch.testing.assert_close(b - a, torch.full_like(a, 32.))
    assert a[0, 0, 0] > a[0, 1, 0]  # deliberately not sorted by onset
    assert a[0, 0, 0] < a[0, 1, 1] and a[0, 1, 0] < a[0, 0, 1]


@pytest.mark.parametrize('raw,origin', [([0., -1000.], 0.), ([0., 0.], 2.**60),
                                      ([1e308, 1e308], 1e308), ([float('nan'), 0.], 0.)])
def test_nonrepresentable_intervals_are_rejected_without_epsilon_repair(raw, origin):
    with pytest.raises(ValueError):
        event_intervals(torch.tensor([[raw]], dtype=torch.float64),
                        torch.tensor([origin], dtype=torch.float64), torch.tensor([[True]]))


@pytest.mark.parametrize('scale', [0., -1., True, float('nan'), float('inf')])
def test_scale_is_explicit_and_finite(scale):
    vocab, annotations = setup()
    with pytest.raises(ValueError, match='scale_seconds'):
        aligned_timing_loss(prediction(torch.zeros(2, 2, 2, dtype=torch.float64), vocab, annotations),
                            annotations, vocab, scale_seconds=scale)


def test_float32_endpoints_are_not_silently_accepted():
    vocab, annotations = setup()
    with pytest.raises(ValueError, match='float64'):
        aligned_timing_loss(prediction(torch.ones(2, 2, 2), vocab, annotations),
                            annotations, vocab, scale_seconds=1.)


def test_bound_origin_is_source_extent_not_minimum_annotated_event(tmp_path):
    from test_governed_motion import inputs, bind_alignment
    from signtranslator.data.governed_motion import load_governed_motion_pair
    options, alignment = inputs(tmp_path)
    alignment['interval_start_seconds'] = -7.
    bind_alignment(options, alignment)
    pair = load_governed_motion_pair(**options)
    assert pair.annotation_extent == (-7., 1.)
    assert min(event.t_start for event in pair.annotation.graph().events) == 0.
