"""Observation support must change the actual objective and its gradients."""
import pytest
import torch
from signtranslator.models.diffusion import GaussianMotionDiffusion
from signtranslator.models.guided_diffusion import GuidedMotionDiffusion


class MixingDenoiser(torch.nn.Module):
    def forward(self, x, t, cond=None, **kwargs):
        return x.mean(dim=2, keepdim=True).expand_as(x)


@pytest.mark.parametrize('cls', [GaussianMotionDiffusion, GuidedMotionDiffusion])
@pytest.mark.parametrize('parameterization', ['x0','eps'])
def test_invalid_values_cannot_change_loss_or_target_gradients(cls, parameterization):
    diffusion = cls(MixingDenoiser(),num_timesteps=10,parameterization=parameterization)
    x=torch.tensor([[[[1.],[2.],[3.],[4.]]]],requires_grad=True)
    mask=torch.tensor([[[True],[True],[False],[False]]])
    noise=torch.ones_like(x);t=torch.tensor([3])
    first=diffusion.p_losses(x,t,noise=noise,validity_mask=mask)
    first.backward()
    assert torch.equal(x.grad[:,:,2:],torch.zeros_like(x.grad[:,:,2:]))
    changed=x.detach().clone();changed[:,:,2:]=float('nan')
    second=diffusion.p_losses(changed,t,noise=noise,validity_mask=mask)
    assert torch.equal(first,second)


def test_weighted_mse_and_velocity_match_hand_computation():
    class Zero(torch.nn.Module):
        def forward(self,x,t,cond=None): return torch.zeros_like(x)
    d=GaussianMotionDiffusion(Zero(),num_timesteps=10,parameterization='x0',velocity_weight=2.)
    x=torch.tensor([[[[1.],[3.],[100.]]]])
    confidence=torch.tensor([[[1.],[.5],[0.]]])
    value=d.p_losses(x,torch.tensor([1]),noise=torch.zeros_like(x),confidence=confidence)
    # Coordinates (1 + .5*9)/1.5; only the first adjacent pair exists, slope 2.
    assert value.item()==pytest.approx(5.5/1.5+2*4)


@pytest.mark.parametrize('kind',['validity_mask','frame_mask','confidence'])
def test_all_missing_support_is_unavailable(kind):
    d=GaussianMotionDiffusion(MixingDenoiser(),num_timesteps=10)
    shape=(2,4) if kind=='frame_mask' else (2,4,1)
    mask=torch.zeros(shape,dtype=torch.float32 if kind=='confidence' else torch.bool)
    with pytest.raises(ValueError,match='no observed support'):
        d(torch.ones(2,1,4,1),**{kind:mask})


def test_velocity_cannot_bridge_missing_middle_frame():
    d=GaussianMotionDiffusion(MixingDenoiser(),num_timesteps=10,parameterization='x0',velocity_weight=1.)
    with pytest.raises(ValueError,match='no required support'):
        d(torch.ones(1,1,3,1),validity_mask=torch.tensor([[[True],[False],[True]]]))


def test_one_empty_sample_cannot_hide_inside_supported_batch():
    d=GaussianMotionDiffusion(MixingDenoiser(),num_timesteps=10)
    with pytest.raises(ValueError,match='no observed support'):
        d(torch.ones(2,1,3,1),frame_mask=torch.tensor([[True]*3,[False]*3]))


@pytest.mark.parametrize('support',[{'frame_mask':torch.ones(1,3)}, {'confidence':torch.full((1,3,1),float('nan'))}, {'confidence':torch.full((1,3,1),-1.)}])
def test_invalid_support_contract_rejected(support):
    d=GaussianMotionDiffusion(MixingDenoiser(),num_timesteps=10)
    with pytest.raises(ValueError): d(torch.ones(1,1,3,1),**support)


def test_active_joint_training_rejects_empty_support(tmp_path):
    from test_trainer import _tiny_setup
    model, train, _ = _tiny_setup(tmp_path)
    batch = next(iter(train))
    batch['validity_mask'].zero_()
    batch['confidence'].zero_()
    with pytest.raises(ValueError, match='no observed support'):
        model.training_step(batch)


def test_generator_finetune_cannot_drop_support_masks(tmp_path):
    from test_trainer import _tiny_setup
    from signtranslator.training import Trainer
    model, train, _ = _tiny_setup(tmp_path)
    batch = next(iter(train))
    batch['frame_mask'].zero_()
    with pytest.raises(ValueError, match='no observed support'):
        Trainer.finetune_generation(model, [batch], epochs=1)
