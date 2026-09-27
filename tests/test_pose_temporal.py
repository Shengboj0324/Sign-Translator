"""Analytic nonuniform-clock oracles and active objective integration."""
import pytest
import torch
from signtranslator.pose.temporal import cartesian_derivatives,time_intervals
from signtranslator.models.diffusion import GaussianMotionDiffusion


def test_quadratic_nonuniform_acceleration_and_linear_resampling():
    for times in ([0.,.1,.4,1.],[0.,.3,.7,1.],[100.,100.1,100.4,101.]):
        clock=torch.tensor([times],dtype=torch.float64)
        local=clock-clock[:, :1]
        valid=torch.ones((1,4,1),dtype=torch.bool)
        x=(3*local**2+2*local+5)[:,None,:,None]
        v,m,a,am=cartesian_derivatives(x,clock,valid)
        assert m.all() and am.all()
        assert torch.allclose(a,torch.full_like(a,6.),atol=1e-10)
        expected=3*(local[:,1:]+local[:,:-1])+2
        assert torch.allclose(v[:,0,:,0],expected,atol=1e-10)
        linear=(7*local+2)[:,None,:,None]
        v,_,a,_=cartesian_derivatives(linear,clock,valid)
        assert torch.allclose(v,torch.full_like(v,7.),atol=1e-10)
        assert torch.allclose(a,torch.zeros_like(a),atol=1e-10)


def test_derivatives_never_bridge_missing_samples_or_padding():
    x=torch.tensor([[[[0.],[1.],[float('nan')],[9.],[16.],[float('nan')]]]],requires_grad=True)
    clock=torch.tensor([[0.,1.,2.,3.,4.,float('nan')]],dtype=torch.float64)
    valid=torch.tensor([[[True],[True],[False],[True],[True],[False]]])
    frames=torch.tensor([[True,True,True,True,True,False]])
    v,m,a,am=cartesian_derivatives(x,clock,valid,frame_mask=frames)
    assert m.flatten().tolist()==[True,False,False,True,False]
    assert not am.any() and torch.isfinite(v).all() and torch.isfinite(a).all()
    v.sum().backward()
    assert x.grad[0,0,2,0]==0 and x.grad[0,0,5,0]==0


@pytest.mark.parametrize('times',[[0.,0.,1.],[0.,-1.,1.],[0.,float('nan'),1.],[0.,float('inf'),1.]])
def test_invalid_supported_clock_rejected(times):
    with pytest.raises(ValueError):
        time_intervals(torch.tensor([times]),torch.ones((1,3),dtype=torch.bool))


class ZeroDenoiser(torch.nn.Module):
    def forward(self,x,t,cond=None,**kwargs):return torch.zeros_like(x)


def test_diffusion_velocity_uses_seconds_not_frame_count():
    diffusion=GaussianMotionDiffusion(ZeroDenoiser(),num_timesteps=10,
                                       parameterization='x0',velocity_weight=1.)
    x=torch.tensor([[[[0.],[2.],[6.]]]],dtype=torch.float32)
    clock=torch.tensor([[0.,1.,3.]],dtype=torch.float64)
    actual=diffusion.p_losses(x,torch.tensor([2]),noise=torch.zeros_like(x),frame_timestamps=clock)
    # Coordinate MSE = 40/3; both physical velocities equal two -> velocity MSE=4.
    assert float(actual)==pytest.approx(40/3+4,rel=1e-7)
    index=diffusion.p_losses(x,torch.tensor([2]),noise=torch.zeros_like(x))
    assert float(index)==pytest.approx(40/3+10,rel=1e-7)


@pytest.mark.parametrize('route',['joint','generator_finetune','analysis','validation'])
def test_active_paths_cannot_drop_invalid_timestamp_evidence(tmp_path,route):
    from test_trainer import _tiny_setup
    from signtranslator.training import Trainer
    from signtranslator import TrainerConfig
    from signtranslator.analysis import analyze
    model,train,val=_tiny_setup(tmp_path)
    batch=next(iter(val))
    batch['frame_timestamps']=torch.zeros((batch['pose'].shape[0],batch['pose'].shape[2]),dtype=torch.float64)
    with pytest.raises(ValueError,match='strictly increasing'):
        if route=='joint':model.training_step(batch)
        elif route=='generator_finetune':Trainer.finetune_generation(model,[batch],epochs=1)
        elif route=='analysis':analyze(model,[batch],ddim_steps=2)
        else:
            trainer=Trainer(model,TrainerConfig(epochs=1),train,val)
            trainer.val_loader=[batch]
            trainer.validate()


def test_diffusion_clock_shape_cannot_broadcast_silently():
    diffusion=GaussianMotionDiffusion(ZeroDenoiser(),num_timesteps=10,
                                       parameterization='x0',velocity_weight=1.)
    with pytest.raises(ValueError,match='motion shape'):
        diffusion(torch.ones(1,1,2,1),frame_timestamps=torch.tensor([[0.,1.,2.]]))


def test_gap_policy_excludes_cross_gap_derivatives_and_gradients():
    x = torch.tensor([[[[0.], [1.], [100.], [104.], [110.]]]],
                     dtype=torch.float64, requires_grad=True)
    clock = torch.tensor([[0., 1., 10., 11., 12.]], dtype=torch.float64)
    v, valid_v, a, valid_a = cartesian_derivatives(
        x, clock, torch.ones(1, 5, 1, dtype=torch.bool), max_gap_seconds=1.)
    assert valid_v.flatten().tolist() == [True, False, True, True]
    assert valid_a.flatten().tolist() == [False, False, True]
    torch.testing.assert_close(v.flatten(), torch.tensor([1., 0., 4., 6.], dtype=torch.float64))
    torch.testing.assert_close(a.flatten(), torch.tensor([0., 0., 2.], dtype=torch.float64))
    v.sum().backward()
    torch.testing.assert_close(x.grad.flatten(), torch.tensor([-1., 1., -1., 0., 1.],
                                                             dtype=torch.float64))


def test_acceleration_midpoint_spacing_does_not_overflow():
    clock = torch.tensor([[-1.5e308, 0., 1.5e308]], dtype=torch.float64)
    x = torch.tensor([[[[-1.5e308], [0.], [0.]]]], dtype=torch.float64)
    _, _, acceleration, support = cartesian_derivatives(
        x, clock, torch.ones(1, 3, 1, dtype=torch.bool))
    assert support.all()
    assert acceleration.item() == pytest.approx(-1 / 1.5e308, rel=1e-12, abs=0)


@pytest.mark.parametrize('bad', [0., -1., float('inf'), float('nan'), True, '1'])
def test_invalid_gap_policy_rejected(bad):
    with pytest.raises(ValueError, match='max_gap_seconds'):
        cartesian_derivatives(torch.ones(1, 1, 2, 1), torch.tensor([[0., 1.]]),
                              torch.ones(1, 2, 1, dtype=torch.bool), max_gap_seconds=bad)


def test_gap_policy_requires_seconds_and_changes_only_supported_velocity():
    diffusion = GaussianMotionDiffusion(ZeroDenoiser(), num_timesteps=10,
                                        parameterization='x0', velocity_weight=1.)
    x = torch.tensor([[[[0.], [2.], [100.], [106.]]]])
    clock = torch.tensor([[0., 1., 10., 11.]], dtype=torch.float64)
    loss = diffusion.p_losses(x, torch.tensor([2]), noise=torch.zeros_like(x),
                              frame_timestamps=clock, max_gap_seconds=1.)
    assert loss.item() == pytest.approx(float(x.square().mean()) + (4 + 36) / 2)
    with pytest.raises(ValueError, match='requires frame_timestamps'):
        diffusion.p_losses(x, torch.tensor([2]), max_gap_seconds=1.)
    with pytest.raises(ValueError, match='no required support'):
        diffusion.p_losses(x, torch.tensor([2]), frame_timestamps=clock, max_gap_seconds=.5)


@pytest.mark.parametrize('route', ['joint', 'generator_finetune', 'analysis', 'validation'])
def test_active_paths_cannot_drop_declared_gap_policy(tmp_path, route):
    from test_trainer import _tiny_setup
    from signtranslator.training import Trainer
    from signtranslator import TrainerConfig
    from signtranslator.analysis import analyze
    model, train, val = _tiny_setup(tmp_path)
    batch = next(iter(val))
    batch['frame_timestamps'] = torch.arange(batch['pose'].shape[2], dtype=torch.float64).expand(
        batch['pose'].shape[0], -1)
    batch['max_gap_seconds'] = .5
    with pytest.raises(ValueError, match='no required support'):
        if route == 'joint':
            model.training_step(batch)
        elif route == 'generator_finetune':
            Trainer.finetune_generation(model, [batch], epochs=1)
        elif route == 'analysis':
            analyze(model, [batch], ddim_steps=2)
        else:
            trainer = Trainer(model, TrainerConfig(epochs=1), train, val)
            trainer.val_loader = [batch]
            trainer.validate()


def test_collation_preserves_one_consistent_gap_policy(tmp_path):
    from test_trainer import _tiny_setup
    from signtranslator.data.corpus import collate_corpus
    _, train, _ = _tiny_setup(tmp_path)
    samples = [train.dataset[i] for i in range(2)]
    for sample in samples:
        sample['max_gap_seconds'] = .2
    assert collate_corpus(samples)['max_gap_seconds'] == .2
    samples[1]['max_gap_seconds'] = .3
    with pytest.raises(ValueError, match='share one explicit'):
        collate_corpus(samples)
    del samples[1]['max_gap_seconds']
    with pytest.raises(ValueError, match='every sample'):
        collate_corpus(samples)
