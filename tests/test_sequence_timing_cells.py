"""Categorical STOP and continuous clock targets retain their distinct domains."""
import json

import pytest
import torch

from signtranslator.training.target_cells import (
    ContinuousTargetCells, parse_target_cells, selected_continuous_target_cells,
)
from signtranslator.planning.label_sequence import (
    label_sequence_targets, label_sequence_loss, SIRLabelSequenceLogits,
)
from signtranslator.planning.event_timing import aligned_timing_loss
from signtranslator.planning.exposure_audit import audit_exposure_declarations
from test_event_timing import prediction
from test_sir_label_vocabulary import setup
from test_supported_trainer import mixed
from test_epoch_commit import make


def test_sequence_capture_includes_single_stop_excludes_padding_preserves_gradients():
    vocab, annotations = setup()
    target = label_sequence_targets(annotations, vocab)
    scores = torch.zeros(*target.outputs.shape, len(vocab.entries) + 1,
                         dtype=torch.float64, requires_grad=True)
    logits = SIRLabelSequenceLogits(scores, target.vocabulary_sha256, target.annotation_sha256)
    old = label_sequence_loss(logits, annotations, vocab)
    new, cells = label_sequence_loss(logits, annotations, vocab, with_target_cells=True)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, scores, retain_graph=True)[0],
                       torch.autograd.grad(new, scores)[0])
    for row, length in zip(cells.examples, target.lengths.tolist()):
        assert len(row) == length
        assert row[-1] == (length - 1, 0)
        assert sum(label == 0 for index, label in row) == 1
        assert all(label > 0 for index, label in row[:-1])


def test_timing_capture_preserves_float64_targets_loss_and_gradients():
    from signtranslator.planning.tensors import tensorize_sir_annotations
    vocab, annotations = setup()
    target = tensorize_sir_annotations(annotations, expected_lexicon=vocab.lexicon,
                                      expected_convention=vocab.convention)
    values = torch.tensor([[[1., 3.], [0., 0.]], [[-1., .5], [2., 4.]]],
                          dtype=torch.float64, requires_grad=True)
    pred = prediction(values, vocab, annotations)
    old = aligned_timing_loss(pred, annotations, vocab, scale_seconds=1.)
    new, cells = aligned_timing_loss(pred, annotations, vocab, scale_seconds=1., with_target_cells=True)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, values, retain_graph=True)[0],
                       torch.autograd.grad(new, values)[0])
    assert cells.unit == 'annotation_clock_seconds'
    assert [len(row) for row in cells.examples] == [2, 4]
    for batch, row in enumerate(cells.examples):
        for event, endpoint, value in row:
            assert value == target.intervals[batch, event, endpoint].item()
    assert parse_target_cells(json.loads(json.dumps(cells.to_dict()))) == cells


@pytest.mark.parametrize('value', [True, 1, float('nan'), float('inf'), -float('inf')])
def test_continuous_targets_do_not_coerce_invalid_values(value):
    with pytest.raises(ValueError, match='finite float'):
        ContinuousTargetCells(('event',), 'seconds', (((0, value),),))


def test_continuous_snapshot_retains_negative_and_subnormal_times_without_classes():
    values = torch.tensor([[-3., 5e-324, float('nan')]], dtype=torch.float64)
    cells = selected_continuous_target_cells(torch.tensor([[True, True, False]]), values,
                                             axes=('event',), unit='seconds')
    values.zero_()
    assert cells.examples == (((0, -3.), (1, 5e-324)),)
    assert parse_target_cells(json.loads(json.dumps(cells.to_dict()))) == cells


def test_all_current_branches_record_and_audit_cells_and_checkpoint(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    assert all(all(value is not None for value in record['target_cells'].values()) for record in records)
    summaries = trainer.exposure_report().to_dict()['target_cell_exposure']['branches']
    assert all(summary['unrecorded_example_presentations'] == 0 for summary in summaries.values())
    assert summaries['event_timing']['value_kind'] == 'continuous'
    assert summaries['event_timing']['class_count'] is None
    assert summaries['event_timing']['class_presentations'] == {}
    assert summaries['sir_sequence']['class_presentations']['0'] == 3
    audit = audit_exposure_declarations(trainer, vocab).to_dict()['target_cell_audit']
    assert all(row == dict(verified=3, unrecorded=0, contradictory=0)
               for row in audit['example_presentations'].values())
    path = trainer.save(tmp_path / 'all-cells.pt')
    receiver = make(corpus, vocab); receiver.load(path)
    assert receiver.optimizer_exposure == records


@pytest.mark.parametrize('branch', ['sir_sequence', 'event_timing'])
def test_valid_domain_target_changes_detected_by_fresh_audit(tmp_path, branch):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    cell = records[0]['target_cells'][branch]['examples'][0][0]
    cell[-1] = 0 if branch == 'sir_sequence' else cell[-1] + .25
    trainer._optimizer_exposure = [json.dumps(record) for record in records]
    trainer.exposure_report()
    result = audit_exposure_declarations(trainer, vocab).to_dict()['target_cell_audit']
    assert result['example_presentations'][branch]['contradictory'] == 1
    assert result['contradictions'] == [dict(step=1, sample_id='sample-0', branch=branch)]
