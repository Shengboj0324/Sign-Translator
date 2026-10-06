"""Save/history validation need not expand the complete public exposure ledger."""
from copy import deepcopy
import json

import pytest
import torch

from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure_codec import pack_exposure
from signtranslator.training.exposure_ledger import ExposureLedger
from signtranslator.training.history import validate_supported_history
from test_epoch_commit import make
from test_supported_trainer import mixed


def test_save_exports_identical_pool_without_public_expansion(tmp_path, monkeypatch):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    expected = pack_exposure(trainer.optimizer_exposure)
    report = trainer.exposure_report().payload
    def refuse(self):
        raise AssertionError('expanded public ledger requested')
    monkeypatch.setattr(type(trainer), 'optimizer_exposure', property(refuse))
    path = trainer.save(tmp_path / 'stream-save.pt')
    saved = torch.load(path, weights_only=False)
    assert canonical_json_bytes(saved['optimizer_exposure']) == canonical_json_bytes(expected)
    receiver = make(corpus, vocabulary); receiver.load(path)
    assert receiver.exposure_report().payload == report


def test_pool_export_is_an_independent_copy(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    ledger = trainer._optimizer_exposure
    expected = ledger.storage_envelope()
    exported = ledger.storage_envelope()
    exported['records'].clear()
    next(iter(exported['target_cells'].values()))['examples'].clear()
    assert ledger.storage_envelope() == expected


def test_late_invalid_record_refuses_save_before_creating_destination(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    records = trainer.optimizer_exposure
    records[-1]['annotation_sha256'][0] = 'b' * 64
    trainer._optimizer_exposure = ExposureLedger(json.dumps(r) for r in records)
    destination = tmp_path / 'not-created' / 'invalid.pt'
    with pytest.raises(ValueError, match='identities'):
        trainer.save(destination)
    assert not destination.parent.exists()


@pytest.mark.parametrize('change', ['none', 'short', 'long', 'duplicate', 'support'])
def test_one_pass_history_consumes_exact_epoch_population(change):
    rows = [dict(sample_ids=['a'], support={'x': 1}), dict(sample_ids=['b'], support={'x': 1})]
    history = dict(lr=[.1], train_support_x=[2], train_support_total=[2],
                   train_x=[1.], train_x_epoch=[1], train_total=[1.], train_total_epoch=[1])
    if change == 'short':
        rows.pop()
    elif change == 'long':
        rows.append(deepcopy(rows[0]))
    elif change == 'duplicate':
        rows[-1]['sample_ids'] = ['a']
    elif change == 'support':
        rows[-1]['support']['x'] = 0
    class Once:
        def __iter__(self):
            assert not getattr(self, 'used', False)
            self.used = True
            yield from rows
    options = dict(epochs=1, records=Once(), sample_ids=['a', 'b'], batch_sizes=[1, 1],
                   validation_epochs=[], validation_population=0, weights={'x': 1.}, expected_lrs=[.1])
    if change == 'none':
        validate_supported_history(history, **options)
    else:
        with pytest.raises(ValueError, match='support-aware history'):
            validate_supported_history(history, **options)
