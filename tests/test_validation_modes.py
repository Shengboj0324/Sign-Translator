"""Validation is observational with respect to heterogeneous module mode flags."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from signtranslator.config import TrainerConfig
from signtranslator.training import Trainer
from test_governed_trainer import MechanicalProbe, setup, loader
from test_supported_trainer import mixed, config, loader as supported_loader
from test_text_sir_relations import model as supported_model


def flags(net):
    return tuple(module.training for module in net.modules())


class LegacyProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 1)

    def training_step(self, batch, weights):
        return {'total': self.head(batch['x']).square().mean()}


@pytest.mark.parametrize('path', ['legacy', 'hook', 'governed', 'supported'])
@pytest.mark.parametrize('parent_mode', [True, False])
def test_validation_restores_every_flag_and_rng_on_success_and_failure(tmp_path, monkeypatch,
                                                                     path, parent_mode):
    if path == 'supported':
        vocabulary, corpus = mixed(tmp_path)
        net = supported_model(vocabulary)
        trainer = Trainer(net, config(), supported_loader(corpus, 'train'), supported_loader(corpus, 'val'))
    elif path == 'governed':
        corpus, _, _ = setup(tmp_path)
        net = MechanicalProbe()
        net.probe_child = nn.Dropout()
        trainer = Trainer(net, TrainerConfig(epochs=1), loader(corpus, 'train'), loader(corpus, 'val'))
    else:
        net = LegacyProbe()
        batches = DataLoader([{'x': torch.tensor([1.])}], batch_size=1)
        trainer = Trainer(net, TrainerConfig(epochs=1), batches, batches)
        if path == 'hook':
            monkeypatch.setattr(net, 'validation_metrics', lambda *args: {'total': 0.}, raising=False)
    method = 'validation_metrics' if path == 'hook' else 'training_step'
    net.train(parent_mode)
    next(iter(net.children())).train(not parent_mode)
    before = flags(net)
    rng_before = torch.get_rng_state().clone()
    actual = getattr(net, method)
    calls = []
    def observe(*args, **kwargs):
        calls.append(flags(net))
        assert not torch.is_grad_enabled()
        torch.rand(4)
        return actual(*args, **kwargs)
    monkeypatch.setattr(net, method, observe)
    trainer.validate()
    assert calls and all(not any(state) for state in calls)
    assert flags(net) == before
    assert torch.equal(torch.get_rng_state(), rng_before)
    def fail(*args, **kwargs):
        assert not any(flags(net))
        torch.rand(4)
        raise RuntimeError('validation failure')
    monkeypatch.setattr(net, method, fail)
    with pytest.raises(RuntimeError, match='validation failure'):
        trainer.validate()
    assert flags(net) == before
    assert torch.equal(torch.get_rng_state(), rng_before)
