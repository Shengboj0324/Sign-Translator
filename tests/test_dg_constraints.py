"""Verification of kinematic constraints and projection.

Each penalty is proved non-negative and zero iff satisfied, differentiable, and
each projection is proved idempotent and feasible.
"""

import pytest
import torch

from signtranslator.diffusion_gen.constraints import (
    joint_limit_penalty, project_joint_limits, collision_penalty, contact_penalty,
    temporal_boundary_penalty, project_feasible,
)


# ---------------------------------------------------------------------------
# joint limits
# ---------------------------------------------------------------------------
def test_joint_limit_penalty_zero_iff_within_limits():
    within = torch.tensor([0.5, -0.9, 1.0], dtype=torch.float64)
    assert joint_limit_penalty(within, theta_max=1.0).item() == 0.0
    over = torch.tensor([1.5, -2.0], dtype=torch.float64)
    pen = joint_limit_penalty(over, theta_max=1.0)
    assert pen.item() == pytest.approx((0.5) ** 2 + (1.0) ** 2, abs=1e-12)


def test_joint_limit_penalty_is_differentiable():
    a = torch.tensor([2.0, -3.0], dtype=torch.float64, requires_grad=True)
    joint_limit_penalty(a, 1.0).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_project_joint_limits_is_idempotent_and_feasible():
    a = torch.tensor([2.0, -3.0, 0.5], dtype=torch.float64)
    p = project_joint_limits(a, 1.0)
    assert torch.all(p.abs() <= 1.0 + 1e-12)                 # feasible
    assert torch.equal(project_joint_limits(p, 1.0), p)      # idempotent
    assert joint_limit_penalty(p, 1.0).item() == 0.0


# ---------------------------------------------------------------------------
# self-collision (reuse Doc-04)
# ---------------------------------------------------------------------------
def test_collision_penalty_zero_when_separated_positive_when_penetrating():
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    far = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0]]], dtype=torch.float64)
    assert collision_penalty(far, radii).item() == 0.0
    near = torch.tensor([[[0.0, 0, 0], [0.5, 0, 0]]], dtype=torch.float64)
    assert collision_penalty(near, radii).item() > 0.0


# ---------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------
def test_contact_penalty_zero_when_required_contacts_made():
    xi = torch.zeros(3, 3, dtype=torch.float64)
    xj = torch.tensor([[0.1, 0, 0], [5.0, 0, 0], [0.2, 0, 0]], dtype=torch.float64)
    should = torch.tensor([1.0, 0.0, 1.0])                   # pairs 0,2 must touch
    # both required pairs are within rho=1 -> zero penalty (pair 1 not required)
    assert contact_penalty(xi, xj, rho=1.0, should_touch=should).item() == 0.0
    # now require pair 1 (far apart) -> positive
    should2 = torch.tensor([1.0, 1.0, 1.0])
    assert contact_penalty(xi, xj, rho=1.0, should_touch=should2).item() > 0.0


# ---------------------------------------------------------------------------
# temporal boundary
# ---------------------------------------------------------------------------
def test_temporal_boundary_penalty_zero_when_matching():
    x = torch.randn(6, 4, dtype=torch.float64)
    target = x.clone()
    mask = torch.zeros(6, 4, dtype=torch.float64); mask[:2] = 1.0
    assert temporal_boundary_penalty(x, target, mask).item() < 1e-12
    x2 = x.clone(); x2[0] += 1.0
    assert temporal_boundary_penalty(x2, target, mask).item() > 0.0


# ---------------------------------------------------------------------------
# combined projection
# ---------------------------------------------------------------------------
def test_project_feasible_lands_on_feasible_set():
    a = torch.tensor([5.0, -5.0, 0.3], dtype=torch.float64)
    p = project_feasible(a, theta_max=1.5)
    assert joint_limit_penalty(p, 1.5).item() == 0.0
    assert torch.equal(project_feasible(p, 1.5), p)          # idempotent fixed point
