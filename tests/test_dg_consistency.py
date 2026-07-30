"""Verification of consistency + rectified-flow distillation.

Proves the consistency boundary condition f(x,t_min)=x, the self-consistency loss,
the rectified-flow interpolant endpoints, the constant straight-line velocity, x0
recovery, and that the straight path makes few-step (even one-step) sampling exact.
"""

import pytest
import torch
import torch.nn as nn

from signtranslator.diffusion_gen.consistency import (
    consistency_coeffs, ConsistencyModel, self_consistency_loss,
    rectified_flow_interpolant, rectified_flow_velocity,
    rectified_flow_x0_from_velocity, rectified_flow_sample,
)


class _ToyNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x, t, cond=None, cond_tokens=None):
        return self.fc(x) + t.reshape((-1,) + (1,) * (x.dim() - 1))


# ---------------------------------------------------------------------------
# consistency boundary
# ---------------------------------------------------------------------------
def test_consistency_coeffs_boundary():
    t = torch.tensor([0.002], dtype=torch.float64)
    c_skip, c_out = consistency_coeffs(t, t_min=0.002, sigma_data=0.5)
    assert abs(float(c_skip) - 1.0) < 1e-12
    assert abs(float(c_out) - 0.0) < 1e-12


def test_consistency_model_satisfies_boundary_condition():
    torch.manual_seed(0)
    model = ConsistencyModel(_ToyNet(4), t_min=0.002).double()
    x = torch.randn(5, 4, dtype=torch.float64)
    t = torch.full((5,), 0.002, dtype=torch.float64)
    out = model(x, t)
    assert torch.allclose(out, x, atol=1e-10)                # f(x, t_min) = x


def test_consistency_model_nontrivial_away_from_boundary():
    torch.manual_seed(1)
    model = ConsistencyModel(_ToyNet(4), t_min=0.002).double()
    x = torch.randn(5, 4, dtype=torch.float64)
    out = model(x, torch.full((5,), 1.0, dtype=torch.float64))
    assert not torch.allclose(out, x, atol=1e-3)             # transforms away from boundary


def test_self_consistency_loss_differentiable_and_zero_when_agree():
    torch.manual_seed(2)
    net = _ToyNet(4).double()
    model = ConsistencyModel(net, t_min=0.002)
    x_high = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    t_high = torch.full((3,), 0.9, dtype=torch.float64)
    t_low = torch.full((3,), 0.8, dtype=torch.float64)
    # same model, same input -> if x_low==x_high and t_low==t_high the loss is 0
    loss0 = self_consistency_loss(model, model, x_high, t_high, x_high, t_high)
    assert loss0.detach().item() < 1e-12
    loss = self_consistency_loss(model, model, x_high, t_high,
                                 torch.randn(3, 4, dtype=torch.float64), t_low)
    loss.backward()
    assert x_high.grad is not None and torch.isfinite(x_high.grad).all()


# ---------------------------------------------------------------------------
# rectified flow
# ---------------------------------------------------------------------------
def test_rectified_flow_interpolant_endpoints():
    x0 = torch.randn(4, 3, dtype=torch.float64)
    z = torch.randn(4, 3, dtype=torch.float64)
    assert torch.allclose(rectified_flow_interpolant(x0, z, torch.zeros(4, dtype=torch.float64)),
                          x0, atol=1e-12)                     # t=0 -> data
    assert torch.allclose(rectified_flow_interpolant(x0, z, torch.ones(4, dtype=torch.float64)),
                          z, atol=1e-12)                      # t=1 -> noise


def test_rectified_flow_velocity_is_constant_and_recovers_x0():
    x0 = torch.randn(4, 3, dtype=torch.float64)
    z = torch.randn(4, 3, dtype=torch.float64)
    v = rectified_flow_velocity(x0, z)
    for tv in (0.2, 0.5, 0.9):
        t = torch.full((4,), tv, dtype=torch.float64)
        x_t = rectified_flow_interpolant(x0, z, t)
        # x0 = x_t - t v exactly (straight path, constant velocity)
        assert torch.allclose(rectified_flow_x0_from_velocity(x_t, t, v), x0, atol=1e-12)


def test_straight_path_makes_few_step_sampling_exact():
    """With the true (constant) velocity, Euler integration recovers x0 exactly for
    ANY number of steps -- the property that justifies few-step distillation."""
    torch.manual_seed(3)
    x0 = torch.randn(6, 3, dtype=torch.float64)
    z = torch.randn(6, 3, dtype=torch.float64)
    v_true = rectified_flow_velocity(x0, z)                  # constant

    def velocity_fn(x, t):
        return v_true                                        # exact straight-line velocity

    for steps in (1, 2, 4):
        out = rectified_flow_sample(velocity_fn, z, num_steps=steps)
        assert torch.allclose(out, x0, atol=1e-10)           # exact even at 1 step
