"""Optimizer ownership must survive parameter replacement and load boundaries."""
import pytest
import torch

from test_epoch_commit import make
from test_supported_trainer import mixed


def replace_first_parameter(model):
    name, original = next(iter(model.named_parameters()))
    owner = model
    parts = name.split('.')
    for part in parts[:-1]:
        owner = getattr(owner, part)
    replacement = torch.nn.Parameter(original.detach().clone())
    setattr(owner, parts[-1], replacement)
    return original, replacement


@pytest.mark.parametrize('operation', ['fit', 'train_epoch', 'validate', 'save', 'resume', 'weights'])
def test_same_shape_parameter_replacement_refused_before_side_effects(tmp_path, operation):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    checkpoint = trainer.save(tmp_path / 'initial.pt')
    original, replacement = replace_first_parameter(trainer.model)
    before = replacement.detach().clone()
    rng = torch.get_rng_state().clone()
    destination = tmp_path / 'must-not-create' / 'invalid.pt'
    with pytest.raises(ValueError, match='model parameter binding changed'):
        if operation == 'save':
            trainer.save(destination)
        elif operation in ('resume', 'weights'):
            trainer.load(checkpoint, mode=operation)
        else:
            getattr(trainer, operation)()
    assert not destination.parent.exists()
    assert torch.equal(before, replacement) and torch.equal(rng, torch.get_rng_state())
    assert not trainer.opt.state and trainer.global_step == 0
    assert trainer.optimizer_exposure == [] and trainer._epoch_committed
    assert original.grad is None and replacement.grad is None


@pytest.mark.parametrize('mutation', ['reorder', 'duplicate', 'missing', 'extra_group', 'optimizer', 'scheduler'])
def test_optimizer_topology_drift_is_refused(tmp_path, mutation):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    parameters = trainer.opt.param_groups[0]['params']
    if mutation == 'reorder':
        parameters.reverse()
    elif mutation == 'duplicate':
        parameters[1] = parameters[0]
    elif mutation == 'missing':
        parameters.pop()
    elif mutation == 'extra_group':
        trainer.opt.add_param_group({'params': [torch.nn.Parameter(torch.ones(()))]})
    elif mutation == 'optimizer':
        trainer.opt = torch.optim.AdamW(trainer.model.parameters())
    else:
        trainer.sched.optimizer = torch.optim.AdamW(trainer.model.parameters())
    with pytest.raises(ValueError, match='optimizer parameter binding changed'):
        trainer.fit()
    assert trainer.global_step == 0 and trainer.optimizer_exposure == []


@pytest.mark.parametrize('stage', ['forward', 'backward'])
def test_mutation_inside_step_never_reaches_optimizer(tmp_path, monkeypatch, stage):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    called = []
    monkeypatch.setattr(trainer.opt, 'step', lambda: called.append(True))
    if stage == 'forward':
        original_step = trainer.model.training_step
        def mutate(batch, *, weights):
            objective = original_step(batch, weights=weights)
            replace_first_parameter(trainer.model)
            return objective
        monkeypatch.setattr(trainer.model, 'training_step', mutate)
    else:
        def mutate(gradient):
            replace_first_parameter(trainer.model)
            return gradient
        next(trainer.model.parameters()).register_hook(mutate)
    with pytest.raises(ValueError, match='model parameter binding changed'):
        trainer.fit()
    assert called == [] and trainer.global_step == 0 and trainer.optimizer_exposure == []
    assert not trainer._epoch_committed


@pytest.mark.parametrize('mode', ['resume', 'weights'])
def test_parameter_replacing_load_hook_cannot_commit(tmp_path, mode):
    vocab, corpus = mixed(tmp_path)
    source = make(corpus, vocab)
    checkpoint = source.save(tmp_path / 'initial.pt')
    receiver = make(corpus, vocab)
    def replace_after_load(module, keys):
        replace_first_parameter(module)
    receiver.model.register_load_state_dict_post_hook(replace_after_load)
    with pytest.raises(ValueError, match='model parameter binding changed'):
        receiver.load(checkpoint, mode=mode)
    assert not receiver._epoch_committed


def test_normal_resume_keeps_optimizer_ownership_and_exact_training(tmp_path):
    vocab, corpus = mixed(tmp_path)
    initial = make(corpus, vocab)
    initial.fit(max_epochs=1)
    checkpoint = initial.save(tmp_path / 'committed.pt')
    restored = make(corpus, vocab)
    identities = tuple(id(p) for p in restored.model.parameters())
    restored.load(checkpoint)
    assert identities == tuple(id(p) for p in restored.model.parameters())
    initial.fit()
    restored.fit()
    assert initial.history == restored.history
    assert initial.optimizer_exposure == restored.optimizer_exposure
    for name, value in initial.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[name])


def test_returned_optimizer_mutation_retains_exposure_but_not_commit(tmp_path):
    vocab, corpus = mixed(tmp_path)
    trainer = make(corpus, vocab)
    def mutate_after_step(optimizer, args, kwargs):
        replace_first_parameter(trainer.model)
    trainer.opt.register_step_post_hook(mutate_after_step)
    with pytest.raises(ValueError, match='model parameter binding changed'):
        trainer.fit()
    assert trainer.global_step == len(trainer.optimizer_exposure) == 1
    assert not trainer._epoch_committed and trainer.completed_epochs == 0


def test_governed_scalar_loss_route_also_checks_backward_mutation(tmp_path, monkeypatch):
    from signtranslator.config import TrainerConfig
    from signtranslator.training import Trainer
    from test_governed_trainer import MechanicalProbe, setup, loader

    corpus, _, _ = setup(tmp_path)
    trainer = Trainer(MechanicalProbe(), TrainerConfig(epochs=1), loader(corpus, 'train'))
    called = []
    monkeypatch.setattr(trainer.opt, 'step', lambda: called.append(True))
    def mutate(gradient):
        replace_first_parameter(trainer.model)
        return gradient
    trainer.model.offset.register_hook(mutate)
    with pytest.raises(ValueError, match='model parameter binding changed'):
        trainer.fit()
    assert called == [] and trainer.global_step == 0 and not trainer._epoch_committed
