"""Resident pooling preserves logical history and returned-call boundaries."""
import json

import pytest

from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure_ledger import ExposureLedger
from test_exposure_codec import records
from test_supported_trainer import mixed
from test_epoch_commit import make


def encoded_records():
    return [canonical_json_bytes(record).decode('utf-8') for record in records()]


def test_incremental_pool_preserves_exact_independent_records_and_absence():
    original = encoded_records()
    ledger = ExposureLedger(original[:4])
    for record in original[4:]:
        ledger.append(record)
    assert len(ledger) == len(original)
    assert list(ledger) == original
    assert ledger.encoded_bytes < sum(len(s.encode('utf-8')) for s in original) / 2
    decoded = [json.loads(s) for s in ledger]
    decoded[0]['target_cells']['labels']['examples'][0].clear()
    assert list(ledger) == original
    assert 'target_cells' not in decoded[-1]


def test_prepare_does_not_record_before_returned_step_and_failure_does_not_mutate():
    ledger = ExposureLedger()
    prepared = ledger.prepare(encoded_records()[0])
    assert not ledger and ledger.encoded_bytes == 0
    ledger.append_prepared(prepared)
    before = list(ledger), ledger.encoded_bytes
    bad = records()[0]
    bad['target_cells']['labels']['class_count'] = True
    with pytest.raises(ValueError):
        ledger.append(json.dumps(bad))
    assert (list(ledger), ledger.encoded_bytes) == before


def test_iterator_expands_only_requested_record(monkeypatch):
    import signtranslator.training.exposure_ledger as module
    original = module.unpack_exposure
    calls = []
    def checked(envelope):
        calls.append(envelope)
        assert len(envelope['records']) == 1
        return original(envelope)
    ledger = ExposureLedger(encoded_records())
    monkeypatch.setattr(module, 'unpack_exposure', checked)
    iterator = iter(ledger)
    assert calls == []
    assert next(iterator) == encoded_records()[0]
    assert len(calls) == 1


def test_training_and_resume_retain_compact_resident_history(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    trainer.fit(max_epochs=1)
    assert isinstance(trainer._optimizer_exposure, ExposureLedger)
    original = trainer.optimizer_exposure
    report = trainer.exposure_report().payload
    path = trainer.save(tmp_path / 'resident.pt')
    receiver = make(corpus, vocab)
    receiver.load(path)
    assert isinstance(receiver._optimizer_exposure, ExposureLedger)
    assert receiver.optimizer_exposure == original
    assert receiver.exposure_report().payload == report
    receiver.fit()
    assert receiver.global_step == len(receiver._optimizer_exposure)
    assert receiver.optimizer_exposure[:len(original)] == original
