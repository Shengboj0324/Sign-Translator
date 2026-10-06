"""Exact resume must preserve which parameters participate in autograd."""
import hashlib
import json

import pytest
import torch

from signtranslator.training import Trainer
from test_supported_trainer import mixed, model, config, loader


def make(corpus, vocabulary, *, frozen):
    net = model(vocabulary)
    net.byte_embedding.weight.requires_grad_(not frozen)
    return Trainer(net, config(), loader(corpus, 'train'), loader(corpus, 'val'))


def rewrite_contract(path, mutate):
    data = torch.load(path, weights_only=False)
    mutate(data['model_contract'])
    torch.save(data, path)
    sidecar = path.with_suffix('.pt.json')
    manifest = json.loads(sidecar.read_text())
    manifest['model_contract'] = data['model_contract']
    manifest['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['checkpoint_size'] = path.stat().st_size
    sidecar.write_text(json.dumps(manifest))


@pytest.mark.parametrize('source_frozen', [False, True])
def test_exact_resume_refuses_different_freeze_policy_without_mutation(tmp_path, source_frozen):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary, frozen=source_frozen)
    path = source.save(tmp_path / 'policy.pt')
    receiver = make(corpus, vocabulary, frozen=not source_frozen)
    before = {name: value.clone() for name, value in receiver.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='model contract does not match'):
        receiver.load(path)
    assert all(torch.equal(before[name], value) for name, value in receiver.model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and not receiver.opt.state
    assert receiver.global_step == 0 and receiver._epoch_committed


def test_matching_frozen_policy_resumes_exactly_and_remains_frozen(tmp_path):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary, frozen=True)
    original = source.model.byte_embedding.weight.detach().clone()
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'frozen.pt')
    receiver = make(corpus, vocabulary, frozen=True)
    receiver.load(path)
    assert not receiver.model.byte_embedding.weight.requires_grad
    source.fit()
    receiver.fit()
    assert source.history == receiver.history
    assert source.optimizer_exposure == receiver.optimizer_exposure
    for name, value in source.model.state_dict().items():
        assert torch.equal(value, receiver.model.state_dict()[name])
    assert torch.equal(original, receiver.model.byte_embedding.weight)
    assert receiver.model.byte_embedding.weight not in receiver.opt.state
    policy = receiver.exposure_report().to_dict()['model_contract']['parameter_trainability']
    assert policy[0] == {'name': 'byte_embedding.weight', 'requires_grad': False}


@pytest.mark.parametrize('operation', ['fit', 'save', 'exposure_report'])
def test_midrun_freeze_change_refused(tmp_path, operation):
    vocabulary, corpus = mixed(tmp_path)
    trainer = make(corpus, vocabulary, frozen=False)
    trainer.model.byte_embedding.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='model contract changed'):
        if operation == 'save':
            trainer.save(tmp_path / 'changed.pt')
        else:
            getattr(trainer, operation)()
    assert not trainer.opt.state and trainer.global_step == 0


@pytest.mark.parametrize('source_frozen', [False, True])
def test_explicit_warm_start_retains_new_receiver_policy(tmp_path, source_frozen):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary, frozen=source_frozen)
    source.fit(max_epochs=1)
    path = source.save(tmp_path / 'source.pt')
    receiver = make(corpus, vocabulary, frozen=not source_frozen)
    receiver.load(path, mode='weights')
    assert receiver.model.byte_embedding.weight.requires_grad is source_frozen
    assert not receiver.opt.state and receiver.global_step == 0
    assert receiver.optimizer_exposure == []
    assert all(torch.equal(value, receiver.model.state_dict()[name])
               for name, value in source.model.state_dict().items())
    receiver.fit(max_epochs=1)
    assert receiver.completed_epochs == 1


@pytest.mark.parametrize('mutation', ['missing', 'integer_flag'])
def test_incomplete_or_ill_typed_trainability_cannot_certify_resume(tmp_path, mutation):
    vocabulary, corpus = mixed(tmp_path)
    source = make(corpus, vocabulary, frozen=False)
    path = source.save(tmp_path / 'old.pt')
    def mutate(contract):
        if mutation == 'missing':
            del contract['parameter_trainability']
        else:
            contract['parameter_trainability'][0]['requires_grad'] = 1
    rewrite_contract(path, mutate)
    receiver = make(corpus, vocabulary, frozen=False)
    with pytest.raises(ValueError, match='model contract does not match'):
        receiver.load(path)
    assert receiver.global_step == 0 and not receiver.opt.state
    if mutation == 'missing':
        receiver.load(path, mode='weights')
        assert receiver.model.byte_embedding.weight.requires_grad
