"""Loss-selected cells retain polarity, coordinates and missing historical scope."""
import json
from dataclasses import replace

import pytest
import torch

from signtranslator.training.target_cells import TargetCells, selected_target_cells
from signtranslator.training.objectives import SupportedObjective, SupportedTerm
from signtranslator.planning.exposure_audit import audit_exposure_declarations
from test_supported_trainer import mixed
from test_epoch_commit import make


def test_masked_cells_are_immutable_exact_and_exclude_unknown_padding():
    known = torch.tensor([[[False, True], [True, False]], [[False, False], [False, False]]])
    labels = torch.tensor([[[99, 0], [1, 99]], [[99, 99], [99, 99]]])
    cells = selected_target_cells(known, labels, axes=('a', 'b'), class_count=2)
    assert cells.examples == (((0, 1, 0), (1, 0, 1)), ())
    known.zero_(); labels.zero_()
    assert cells.examples[0][1][-1] == 1
    copy = cells.to_dict(); copy['examples'][0][0][-1] = 1
    assert cells.examples[0][0][-1] == 0
    assert TargetCells.from_dict(cells.to_dict()) == cells


@pytest.mark.parametrize('rows', [(((0, 1), (0, 0)),), (((1, 1), (0, 0)),),
                                  (((True, 1),),), (((0, 2),),), (((-1, 0),),)])
def test_malformed_or_duplicate_cells_refused(rows):
    with pytest.raises(ValueError, match='target cell'):
        TargetCells(('event',), 2, rows)


def test_cells_must_reconcile_objective_membership():
    cells = TargetCells(('event',), 2, (((0, 1),), ()))
    term = SupportedTerm(torch.tensor(1.), 1, (False, True), cells)
    with pytest.raises(ValueError, match='target cells'):
        SupportedObjective({'branch': term}, {'branch': 1.}, 2)


def test_training_records_repeated_cells_and_roundtrips_without_reconstruction(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit()
    records = trainer.optimizer_exposure
    cells = records[0]['target_cells']['sir_relations']
    assert cells['examples'][0] == [] and cells['examples'][1]
    assert {c[-1] for c in cells['examples'][1]} <= {0, 1}
    summary = trainer.exposure_report().to_dict()['target_cell_exposure']['branches']
    relation = summary['sir_relations']
    assert relation['target_presentations'] == 2 * len(cells['examples'][1])
    assert relation['recorded_example_presentations'] == 6
    assert relation['unrecorded_example_presentations'] == 0
    assert sum(relation['class_presentations'].values()) == relation['target_presentations']
    assert summary['sir_sequence']['unrecorded_example_presentations'] == 0
    path = trainer.save(tmp_path / 'cells.pt')
    receiver = make(corpus, vocab); receiver.load(path)
    assert receiver.optimizer_exposure == records
    audit = audit_exposure_declarations(receiver, vocab).to_dict()['target_cell_audit']
    assert audit['contradictions'] == []
    assert audit['example_presentations']['sir_relations'] == dict(verified=6, unrecorded=0, contradictory=0)
    for record in records:
        del record['target_cells']
    receiver._optimizer_exposure = [json.dumps(r) for r in records]
    old = receiver.exposure_report().to_dict()['target_cell_exposure']['branches']['sir_relations']
    assert old['unrecorded_example_presentations'] == 6 and old['target_presentations'] == 0
    assert audit_exposure_declarations(receiver, vocab).to_dict()['target_cell_audit']['example_presentations']['sir_relations']['unrecorded'] == 6


def test_polarity_swap_requires_fresh_semantic_audit(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    cell = records[0]['target_cells']['sir_relations']['examples'][1][0]
    cell[-1] = 1 - cell[-1]
    trainer._optimizer_exposure = [json.dumps(r) for r in records]
    trainer.exposure_report()  # Counts, identities and support still match.
    audit = audit_exposure_declarations(trainer, vocab).to_dict()['target_cell_audit']
    assert audit['example_presentations']['sir_relations']['contradictory'] == 1
    assert audit['contradictions'] == [dict(step=1, sample_id='sample-1', branch='sir_relations')]


def test_five_head_loss_captures_known_loci_and_unordered_referents(tmp_path):
    from signtranslator.config import TrainerConfig
    from signtranslator.training import Trainer
    from test_loci import fixture
    from test_text_loci import model, WEIGHTS
    from test_governed_trainer import loader
    vocab, corpus, alphabet, _ = fixture(tmp_path)
    trainer = Trainer(model(vocab, alphabet), TrainerConfig(epochs=1, loss_weights=WEIGHTS,
                      selection_metric='sir_sequence'), loader(corpus, 'train'))
    trainer.fit()
    record = trainer.optimizer_exposure[0]
    for cells in record['target_cells']['referent_equality']['examples']:
        assert all(first < second for first, second, label in cells)
    loci = record['target_cells']['locus_assignment']
    assert loci['class_count'] == len(alphabet.identities)
    assert any(loci['examples'])
    audit = audit_exposure_declarations(trainer, vocab, alphabet).to_dict()['target_cell_audit']
    assert not audit['contradictions']
    assert audit['example_presentations']['locus_assignment']['verified'] == 1


def test_relation_cell_capture_preserves_loss_and_gradient_exactly():
    from test_sir_relations import fixture
    from signtranslator.planning.relations import relation_targets, relation_loss_per_example, SIRRelationLogits
    vocab, annotations = fixture()
    target = relation_targets(annotations, vocab)
    scores = torch.linspace(-2., 2., target.known.numel(), dtype=torch.float64).reshape(target.known.shape).requires_grad_()
    prediction = SIRRelationLogits(scores, target.annotation_sha256, target.vocabulary_sha256)
    original, support = relation_loss_per_example(prediction, annotations, vocab)
    captured, new_support, cells = relation_loss_per_example(prediction, annotations, vocab, with_target_cells=True)
    assert torch.equal(original, captured) and torch.equal(support, new_support)
    left = torch.autograd.grad(original.sum(), scores, retain_graph=True)[0]
    right = torch.autograd.grad(captured.sum(), scores)[0]
    assert torch.equal(left, right)
    for row, selected in enumerate(cells.examples):
        assert len(selected) == int(target.known[row].sum())
        assert sum(cell[-1] for cell in selected) == int(target.positive[row].sum())
        for source, destination, kind, label in selected:
            assert bool(target.known[row, source, destination, kind])
            assert label == int(target.positive[row, source, destination, kind])
    assert not right[~target.known].any()


def test_new_cell_mask_catches_swapped_example_support_before_save(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    records[0]['support_membership']['sir_relations'] = [True, False]
    trainer._optimizer_exposure = [json.dumps(r) for r in records]
    with pytest.raises(ValueError, match='target cells disagree'):
        trainer.save(tmp_path / 'absent' / 'invalid.pt')
    assert not (tmp_path / 'absent').exists()
