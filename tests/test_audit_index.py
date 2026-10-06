"""Compact audit indexing preserves exact comparisons and presentation order."""
import json

from signtranslator.planning.exposure_audit import (
    _cell_signature, _index_declarations, audit_exposure_declarations,
)
from test_exposure_audit import fixture
from test_epoch_commit import make


def test_index_interns_values_without_losing_steps_rows_or_absence():
    declaration = dict(axes=['event'], class_count=2, examples=[[[0, 1]], []])
    records = [dict(step=1, sample_ids=['a', 'b'], target_cells={'labels': declaration}),
               dict(step=2, sample_ids=['b', 'a'], target_cells={'labels': dict(
                   axes=['event'], class_count=2, examples=[[], [[0, 1]]])}),
               dict(step=3, sample_ids=['a'])]
    indexed = _index_declarations((json.dumps(r) for r in records), ['a', 'b'], ['labels'])
    assert [step for step, _ in indexed['a']] == [1, 2, 3]
    assert [step for step, _ in indexed['b']] == [1, 2]
    assert indexed['a'][0][1][0] is indexed['a'][1][1][0]
    assert indexed['b'][0][1][0] is indexed['b'][1][1][0]
    assert indexed['a'][-1][1] == (None,)
    assert indexed['a'][0][1][0] != indexed['b'][0][1][0]
    declaration['examples'][0][0][-1] = 0
    assert indexed['a'][0][1][0][-1] == ((0, 1),)


def test_signature_preserves_numeric_equality_and_exact_large_values():
    def signature(value):
        return _cell_signature(dict(axes=['event'], unit='seconds', examples=[[[0, value]]]), 0)
    assert signature(-0.) == signature(0.)
    assert signature(5e-324) != signature(0.)
    assert signature(2**60) != signature(2**60 + 1)
    assert _cell_signature(None, 0) is None


def test_fresh_audit_does_not_expand_public_ledger(tmp_path, monkeypatch):
    vocabulary, corpus, alphabet = fixture(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit()
    before = audit_exposure_declarations(trainer, vocabulary, alphabet).payload
    def refuse(self):
        raise AssertionError('expanded public ledger requested')
    monkeypatch.setattr(type(trainer), 'optimizer_exposure', property(refuse))
    assert audit_exposure_declarations(trainer, vocabulary, alphabet).payload == before
