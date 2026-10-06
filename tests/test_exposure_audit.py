"""Fresh target checks detect identity-valid but semantically false declarations."""
import json

import pytest

from signtranslator.planning.exposure_audit import audit_exposure_declarations
from signtranslator.planning.loci import LocusAlphabet
from test_loci import PAYLOAD
from test_text_sir_labels import setup
from test_epoch_commit import make


def fixture(tmp_path):
    v, corpus = setup(tmp_path, convention_payload=PAYLOAD,
                     configurations=[{}, {}, {}, {'split': 'val'}],
                     single_event_indices=(0, 2))
    return v, corpus, LocusAlphabet(v.convention, PAYLOAD)


def test_fresh_audit_verifies_mixed_declarations_without_mutating_training(tmp_path):
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v); trainer.fit()
    before = trainer.exposure_report().payload
    report = audit_exposure_declarations(trainer, v, alphabet)
    result = report.to_dict()
    assert result['status'] == 'consistent_declared_support'
    assert result['branch_presentations']['sir_relations'] == {
        'verified_supported': 2, 'verified_unsupported': 4,
        'unverified_membership': 0, 'contradictory': 0}
    assert result['contradictions'] == [] and not result['phase_exit_approved']
    assert trainer.exposure_report().payload == before
    result['samples'].clear()
    assert len(report.to_dict()['samples']) == 3


def test_same_count_swapped_membership_is_inconsistent(tmp_path):
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    records[0]['support_membership']['sir_relations'] = [True, False]
    # Exercise the historical membership-only schema. New cell declarations
    # would detect this disagreement already at ledger validation.
    for record in records:
        record.pop('target_cells')
    trainer._optimizer_exposure = [json.dumps(r) for r in records]
    # Ledger shape/count validation accepts it; target semantics must not.
    trainer.exposure_report()
    result = audit_exposure_declarations(trainer, v, alphabet).to_dict()
    assert result['status'] == 'inconsistent_declared_support'
    assert result['branch_presentations']['sir_relations']['contradictory'] == 2
    assert len(result['contradictions']) == 2


def test_missing_historical_membership_is_unverified_not_inferred(tmp_path):
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    for record in records:
        del record['support_membership']
    trainer._optimizer_exposure = [json.dumps(r) for r in records]
    result = audit_exposure_declarations(trainer, v, alphabet).to_dict()
    assert result['status'] == 'partially_attributed'
    assert result['branch_presentations']['sir_relations']['unverified_membership'] == 3
    assert result['branch_presentations']['sir_relations']['verified_supported'] == 0


def test_no_steps_is_not_positive_training_evidence(tmp_path):
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v)
    result = audit_exposure_declarations(trainer, v, alphabet).to_dict()
    assert result['status'] == 'no_recorded_exposure'
    assert all(sum(row.values()) == 0 for row in result['branch_presentations'].values())


def test_changed_source_bytes_refuse_fresh_audit(tmp_path):
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v); trainer.fit(max_epochs=1)
    corpus._records[0].transcript_path.write_bytes(b'changed after exposure')
    with pytest.raises((ValueError, PermissionError)):
        audit_exposure_declarations(trainer, v, alphabet)


def test_five_branch_audit_keeps_reference_and_locus_support_distinct(tmp_path):
    from signtranslator.config import TrainerConfig
    from signtranslator.training import Trainer
    from test_loci import fixture as locus_fixture
    from test_text_loci import model, WEIGHTS
    from test_governed_trainer import loader
    v, corpus, alphabet, _ = locus_fixture(tmp_path)
    trainer = Trainer(model(v, alphabet), TrainerConfig(epochs=1, loss_weights=WEIGHTS,
                      selection_metric='sir_sequence'), loader(corpus, 'train'))
    trainer.fit()
    result = audit_exposure_declarations(trainer, v, alphabet).to_dict()
    assert result['status'] == 'consistent_declared_support'
    assert result['branch_presentations']['referent_equality']['verified_unsupported'] == 1
    assert result['branch_presentations']['locus_assignment']['verified_supported'] == 1


def test_model_contract_change_refuses_audit(tmp_path):
    from dataclasses import replace
    v, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, v)
    trainer.model.model_cfg = replace(trainer.model.model_cfg, max_events=5)
    with pytest.raises(ValueError, match='model contract changed'):
        audit_exposure_declarations(trainer, v, alphabet)


def test_relation_only_audit_needs_no_locus_alphabet_and_exposes_target_counts(tmp_path):
    from test_supported_trainer import mixed
    v, corpus = mixed(tmp_path)
    trainer = make(corpus, v); trainer.fit(max_epochs=1)
    result = audit_exposure_declarations(trainer, v).to_dict()
    assert result['status'] == 'consistent_declared_support' and result['schema_version'] == 2
    single, multiple, _ = result['samples']
    assert single['available_targets']['sir_sequence']['event_labels'] == 1
    assert single['available_targets']['sir_sequence']['stop_targets'] == 1
    assert single['available_targets']['event_timing']['endpoint_targets'] == 2
    assert all(sum(counts.values()) == 0
               for counts in single['available_targets']['sir_relations'].values())
    labels = multiple['available_targets']['sir_sequence']['labels']
    assert [row['target_events'] for row in labels] == [1, 1, 1]
    assert [row['class_index'] for row in labels] == [0, 1, 2]
    relations = multiple['available_targets']['sir_relations']
    assert relations['precedence'] == {'positive': 0, 'negative': 6, 'unknown': 0}
    assert relations['scope'] == {'positive': 1, 'negative': 4, 'unknown': 1}
    assert relations['overlap'] == {'positive': 0, 'negative': 0, 'unknown': 6}
    assert 'locus_assignment' not in multiple['available_targets']


def test_reference_audit_needs_no_alphabet_and_counts_unordered_pairs(tmp_path):
    from signtranslator.config import TrainerConfig
    from signtranslator.training import Trainer
    from test_text_referents import model, WEIGHTS
    from test_governed_trainer import loader
    v, corpus = setup(tmp_path, referents=(0, 0, 7))
    trainer = Trainer(model(v), TrainerConfig(epochs=1, loss_weights=WEIGHTS,
                      selection_metric='sir_sequence'), loader(corpus, 'train'))
    trainer.fit()
    result = audit_exposure_declarations(trainer, v).to_dict()
    counts = result['samples'][0]['available_targets']['referent_equality']
    assert counts == {'positive': 1, 'negative': 2, 'unknown': 0}


def test_locus_audit_still_requires_bound_alphabet(tmp_path):
    from signtranslator.config import TrainerConfig
    from signtranslator.training import Trainer
    from test_loci import fixture as locus_fixture
    from test_text_loci import model, WEIGHTS
    from test_governed_trainer import loader
    v, corpus, alphabet, _ = locus_fixture(tmp_path)
    trainer = Trainer(model(v, alphabet), TrainerConfig(epochs=1, loss_weights=WEIGHTS,
                      selection_metric='sir_sequence'), loader(corpus, 'train'))
    with pytest.raises(ValueError, match='alphabet must match'):
        audit_exposure_declarations(trainer, v)
    result = audit_exposure_declarations(trainer, v, alphabet).to_dict()
    counts = result['samples'][0]['available_targets']['locus_assignment']
    assert counts['known'] == 2 and counts['unknown'] == 1
    assert [row['target_events'] for row in counts['classes']] == [1, 1]
