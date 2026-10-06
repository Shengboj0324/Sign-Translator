"""Exposure preflight rejects consistently rehashed but malformed records."""
import hashlib
import json

import pytest
import torch

from signtranslator.training import Trainer
from test_supported_trainer import mixed, model, config, loader


@pytest.mark.parametrize('mutation', ['missing', 'step', 'support', 'identity', 'hash', 'binding', 'weights', 'membership_count', 'membership_type', 'membership_keys'])
def test_invalid_exposure_rejected_before_model_mutation(tmp_path, mutation):
    vocab, corpus = mixed(tmp_path)
    def build():
        return Trainer(model(vocab), config(), loader(corpus, 'train'))
    trainer = build(); trainer.fit(max_epochs=1)
    path = trainer.save(tmp_path / 'exposure.pt')
    data = torch.load(path, weights_only=False)
    records = data['optimizer_exposure']['records']
    if mutation == 'missing':
        records.pop()
    elif mutation == 'step':
        records[0]['step'] = True
    elif mutation == 'support':
        records[0]['support']['sir_sequence'] = 999
    elif mutation == 'identity':
        records[0]['sample_ids'][1] = records[0]['sample_ids'][0]
    elif mutation == 'hash':
        records[0]['annotation_sha256'][0] = 'invalid'
    elif mutation == 'binding':
        records[0]['annotation_sha256'][0] = '0' * 64
    elif mutation == 'membership_count':
        records[0]['support_membership']['sir_relations'] = [True, True]
    elif mutation == 'membership_type':
        records[0]['support_membership']['sir_relations'] = [0, 1]
    elif mutation == 'membership_keys':
        del records[0]['support_membership']['sir_relations']
    else:
        records[0]['weights']['sir_sequence'] = 3.
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest.update(optimizer_exposure=data['optimizer_exposure'],
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_size=path.stat().st_size)
    sidecar.write_text(json.dumps(manifest))
    fresh = build()
    before = {k: v.clone() for k, v in fresh.model.state_dict().items()}
    with pytest.raises(ValueError, match='exposure'):
        fresh.load(path)
    assert all(torch.equal(v, before[k]) for k, v in fresh.model.state_dict().items())
    assert fresh.optimizer_exposure == [] and fresh.global_step == 0
