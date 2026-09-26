"""Invalid training evidence must fail before optimizer/scheduler mutation."""
import copy
import inspect
import pytest
import torch
from torch.utils.data import DataLoader
from signtranslator import TrainerConfig
from signtranslator.training import Trainer
from signtranslator.run import make_loaders, run_pipeline
from test_trainer import _tiny_setup


class BadModel(torch.nn.Module):
    def __init__(self, kind):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.));self.kind=kind
        if kind=='gradient': self.weight.register_hook(lambda g: g*float('nan'))
    def training_step(self,batch,weights=None):
        loss=self.weight.square()
        if self.kind=='loss':loss=loss*float('nan')
        return {'total':loss}


@pytest.mark.parametrize('kind',['loss','gradient'])
def test_invalid_loss_or_gradient_preserves_optimizer_and_checkpoint(tmp_path,kind):
    model=BadModel(kind);loader=DataLoader([{'x':torch.tensor(1.)}],batch_size=1)
    trainer=Trainer(model,TrainerConfig(epochs=1),loader)
    path=tmp_path/'last.pt';trainer.save(path);before=path.read_bytes()
    schedule=copy.deepcopy(trainer.sched.state_dict())
    with pytest.raises(FloatingPointError,match='batch=0'):
        trainer.train_epoch()
    assert model.weight.item()==1.
    assert trainer.global_step==0 and not trainer.opt.state
    assert trainer.sched.state_dict()==schedule
    assert path.read_bytes()==before


def test_empty_training_loader_is_rejected():
    loader=DataLoader([{'x':1}],batch_size=2,drop_last=True)
    with pytest.raises(ValueError,match='zero batches'):
        Trainer(BadModel('valid'),TrainerConfig(),loader)


def test_oversized_batch_retains_tiny_corpus(tmp_path):
    model,_,_=_tiny_setup(tmp_path)
    train,val=make_loaders(str(tmp_path),10000)
    assert len(train)==1
    trainer=Trainer(model,TrainerConfig(epochs=1),train,val)
    trainer.fit()
    assert trainer.global_step==1


@pytest.mark.parametrize('kwargs',[{'lr':float('nan')},{'grad_clip':float('inf')},{'loss_weights':{'x':float('nan')}},{'loss_weights':{'x':0.}},{'epochs':True}])
def test_nonfinite_or_degenerate_training_configuration_rejected(kwargs):
    with pytest.raises(ValueError):TrainerConfig(**kwargs)


def test_api_defaults_are_bounded_and_checkpointable():
    defaults=inspect.signature(run_pipeline).parameters
    assert defaults['epochs'].default==1
    assert defaults['gen_finetune_epochs'].default==defaults['polish_epochs'].default==0
    assert defaults['lr'].default==3e-4


def test_cli_defaults_match_api_and_allow_checkpoint(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import signtranslator.run as entry
    captured = {}
    def record(*args, **kwargs):
        captured.update(kwargs)
        return {'report': SimpleNamespace(passed=True)}
    monkeypatch.setattr(entry, 'run_pipeline', record)
    monkeypatch.setattr(sys, 'argv', ['signtranslator.run', '--ckpt', str(tmp_path/'m.pt')])
    with pytest.raises(SystemExit) as result:
        entry.main()
    assert result.value.code == 0
    assert captured['epochs'] == 1
    assert captured['gen_finetune_epochs'] == captured['polish_epochs'] == 0
    assert captured['lr'] == 3e-4
