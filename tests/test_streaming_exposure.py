"""One-pass summaries retain canonical identity and refuse incomplete histories."""
from copy import deepcopy
import hashlib

import pytest

from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure import summarize_exposure, validate_exposure
from test_epoch_commit import make
from test_supported_trainer import mixed


def inputs():
    records = [dict(step=i, sample_ids=['样本'], annotation_sha256=['a' * 64],
                    support={'labels': 1}, weights={'labels': 1.},
                    support_membership={'labels': [True]},
                    target_cells={'labels': dict(axes=['event'], class_count=2,
                                                examples=[[[0, i % 2]]])}) for i in range(1, 4)]
    options = dict(global_step=3, completed_epochs=1, committed=True, weights={'labels': 1.},
                   identities={'样本': 'a' * 64}, data_contract={}, implementation_identity={}, model_contract={})
    return records, options


@pytest.mark.parametrize('historical', [False, True])
def test_single_pass_report_has_exact_array_hash_and_counts(historical):
    records, options = inputs()
    if historical:
        del records[-1]['target_cells']
        del records[-1]['support_membership']
    class Once:
        def __iter__(self):
            assert not getattr(self, 'used', False)
            self.used = True
            yield from records
    report = summarize_exposure(Once(), **options).to_dict()
    assert report['ledger_sha256'] == hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    assert report['sample_presentations'] == 3
    assert report['target_cell_exposure']['branches']['labels']['target_presentations'] == (2 if historical else 3)
    assert report['unattributed_supported_example_presentations']['labels'] == int(historical)
    empty = summarize_exposure(iter(()), **dict(options, global_step=0)).to_dict()
    assert empty['ledger_sha256'] == hashlib.sha256(b'[]').hexdigest()


@pytest.mark.parametrize('fault', ['short', 'long', 'step', 'contract', 'membership'])
def test_late_invalid_history_never_returns_a_partial_report(fault):
    records, options = inputs()
    if fault == 'short':
        records.pop()
    elif fault == 'long':
        records.append(dict(deepcopy(records[-1]), step=4))
    elif fault == 'step':
        records[-1]['step'] = 9
    elif fault == 'contract':
        records[-1]['target_cells']['labels']['class_count'] = 3
    else:
        records[-1]['support_membership']['labels'] = [False]
    with pytest.raises(ValueError):
        summarize_exposure(iter(records), **options)
    with pytest.raises(ValueError):
        validate_exposure(records, supported=True, global_step=3,
                          weights=options['weights'], identities=options['identities'])


def test_previous_input_mutation_cannot_rewrite_retained_contract():
    records, options = inputs()
    def stream():
        yield records[0]
        records[0]['target_cells']['labels']['axes'][0] = 'changed'
        for record in records[1:]:
            record['target_cells']['labels']['axes'][0] = 'changed'
            yield record
    with pytest.raises(ValueError, match='codebook changed'):
        summarize_exposure(stream(), **options)


def test_trainer_report_does_not_request_full_public_ledger(tmp_path, monkeypatch):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    expected = trainer.exposure_report().payload
    def refuse(self):
        raise AssertionError('full public exposure expansion requested')
    monkeypatch.setattr(type(trainer), 'optimizer_exposure', property(refuse))
    assert trainer.exposure_report().payload == expected
