"""Content-addressed storage preserves logical target declarations and absence."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from signtranslator.training.exposure_codec import pack_exposure, unpack_exposure
from signtranslator.reproducibility import canonical_json_bytes
from test_supported_trainer import mixed
from test_epoch_commit import make


def records():
    categorical = dict(axes=['event'], class_count=2, examples=[[[i, i % 2] for i in range(200)], []])
    continuous = dict(axes=['event', 'endpoint'], unit='seconds', examples=[[[0, 0, -0.], [0, 1, 5e-324]], []])
    return [dict(step=i, sample_ids=['a', 'b'], target_cells=dict(labels=deepcopy(categorical),
                 timing=deepcopy(continuous), absent=None)) for i in range(1, 21)] + [dict(step=21)]


def test_exact_roundtrip_deduplicates_repeated_rows_and_preserves_missing_history():
    source = records()
    packed = pack_exposure(source)
    assert len(packed['target_cells']) == 4
    assert canonical_json_bytes(unpack_exposure(packed)) == canonical_json_bytes(source)
    assert len(canonical_json_bytes(packed)) < len(canonical_json_bytes(source)) / 2
    restored = unpack_exposure(packed)
    restored[0]['target_cells']['labels']['examples'][0][0][-1] = 99
    assert restored[1]['target_cells']['labels']['examples'][0][0][-1] == 0
    assert unpack_exposure(packed) == source
    source[0]['target_cells']['timing']['examples'][0].clear()
    assert len(unpack_exposure(packed)[0]['target_cells']['timing']['examples'][0]) == 2
    assert 'target_cells' not in packed['records'][-1]


@pytest.mark.parametrize('fault', ['hash', 'reference', 'unused', 'contract', 'version', 'row_count'])
def test_corrupt_storage_is_refused(fault):
    packed = pack_exposure(records())
    first = next(iter(packed['target_cells']))
    if fault == 'hash':
        packed['target_cells'][first]['class_count'] = 3
    elif fault == 'reference':
        packed['records'][0]['target_cells']['labels'][0] = 'f' * 64
    elif fault == 'unused':
        extra = dict(axes=['new'], class_count=1, examples=[[]])
        packed['target_cells'][hashlib.sha256(canonical_json_bytes(extra)).hexdigest()] = extra
    elif fault == 'contract':
        packed['records'][0]['target_cells']['labels'][0] = packed['records'][0]['target_cells']['timing'][0]
    elif fault == 'version':
        packed['schema_version'] = True
    else:
        original = packed['target_cells'].pop(first)
        original['examples'].append([])
        key = hashlib.sha256(canonical_json_bytes(original)).hexdigest()
        packed['target_cells'][key] = original
    with pytest.raises(ValueError, match='exposure'):
        unpack_exposure(packed)


def rewrite(path, data):
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest.update(schema_version=data['schema_version'], optimizer_exposure=data['optimizer_exposure'],
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_size=path.stat().st_size)
    sidecar.write_text(json.dumps(manifest))


def test_checkpoint_schema_five_and_legacy_four_preserve_logical_ledger(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'compact.pt')
    data = torch.load(path, weights_only=False)
    assert data['schema_version'] == 5
    assert unpack_exposure(data['optimizer_exposure']) == trainer.optimizer_exposure
    receiver = make(corpus, vocab); receiver.load(path)
    assert receiver.exposure_report().payload == trainer.exposure_report().payload
    # Format compatibility does not waive implementation/runtime/data identities.
    data['schema_version'] = 4
    data['optimizer_exposure'] = trainer.optimizer_exposure
    rewrite(path, data)
    legacy = make(corpus, vocab); legacy.load(path)
    assert legacy.optimizer_exposure == trainer.optimizer_exposure
    legacy.fit(); receiver.fit()
    assert legacy.history == receiver.history
    assert all(torch.equal(v, receiver.model.state_dict()[k]) for k, v in legacy.model.state_dict().items())


def test_invalid_pool_rejected_before_model_or_rng_mutation(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab); trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'invalid.pt')
    data = torch.load(path, weights_only=False)
    data['optimizer_exposure']['records'][0]['target_cells']['sir_sequence'][0] = 'f' * 64
    rewrite(path, data)
    receiver = make(corpus, vocab)
    before = {k: v.clone() for k, v in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='exposure'):
        receiver.load(path)
    assert all(torch.equal(v, receiver.model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state()) and receiver._epoch_committed
    receiver.load(path, mode='weights')  # Unused optimizer evidence is not imported.
    assert receiver.optimizer_exposure == []
