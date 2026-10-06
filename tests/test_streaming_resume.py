"""Resume exhausts streaming integrity checks before mutating receiver state."""
import hashlib
import json

import pytest
import torch

from signtranslator.reproducibility import canonical_json_bytes
from signtranslator.training.exposure_codec import iter_unpack_exposure, pack_exposure
from test_exposure_codec import records, rewrite
from test_epoch_commit import make
from test_supported_trainer import mixed


def test_stream_decoder_preserves_independent_values_and_checks_end_of_pool():
    source = records()
    packed = pack_exposure(source)
    stream = iter_unpack_exposure(packed)
    first = next(stream)
    first['target_cells']['labels']['examples'][0].clear()
    assert next(stream) == source[1]
    assert canonical_json_bytes(list(iter_unpack_exposure(packed))) == canonical_json_bytes(source)
    extra = dict(axes=['unused'], class_count=1, examples=[[]])
    packed['target_cells'][hashlib.sha256(canonical_json_bytes(extra)).hexdigest()] = extra
    stream = iter_unpack_exposure(packed)
    for _ in source:
        next(stream)
    with pytest.raises(ValueError, match='unused'):
        next(stream)


@pytest.mark.parametrize('fault', ['unused', 'late_reference', 'short', 'legacy_tuple'])
def test_stream_failure_precedes_receiver_and_rng_mutation(tmp_path, fault):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'stream.pt')
    data = torch.load(path, weights_only=False)
    if fault == 'unused':
        declaration = dict(axes=['unused'], class_count=1, examples=[[]])
        data['optimizer_exposure']['target_cells'][hashlib.sha256(canonical_json_bytes(declaration)).hexdigest()] = declaration
    elif fault == 'late_reference':
        data['optimizer_exposure']['records'][-1]['target_cells']['sir_sequence'][0] = 'f' * 64
    elif fault == 'short':
        data['optimizer_exposure']['records'].pop()
    else:
        data['schema_version'] = 4
        data['optimizer_exposure'] = tuple(trainer.optimizer_exposure)
    rewrite(path, data)
    receiver = make(corpus, vocabulary)
    before = {k: v.clone() for k, v in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        receiver.load(path)
    assert all(torch.equal(v, receiver.model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert receiver._epoch_committed and receiver.global_step == 0
    assert not receiver.opt.state and not receiver.history and not receiver._optimizer_exposure


def test_resume_avoids_eager_full_ledger_validator(tmp_path, monkeypatch):
    import signtranslator.training.trainer as module
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary); trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'valid.pt')
    report = trainer.exposure_report().payload
    receiver = make(corpus, vocabulary)
    def refuse(*args, **kwargs):
        raise AssertionError('eager full-ledger validator invoked')
    monkeypatch.setattr(module, 'validate_exposure', refuse)
    receiver.load(path)
    assert receiver.exposure_report().payload == report
