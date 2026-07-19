"""Verification of the fitting objective and its terms.

Unit properties (prior min at mean, smoothness zero iff constant, collision zero
iff no penetration) are proved exactly. The document's central caveat -- that
monocular fitting is underdetermined -- is demonstrated deterministically: two
different 3D configurations project identically under one camera, so 2D
re-projection alone cannot pin down 3D; a second view removes the ambiguity.
"""

import math

import pytest
import torch

from signtranslator.pose.camera import PerspectiveCamera, reprojection_loss
from signtranslator.pose.state import SMPLXLayout, MotionSequence
from signtranslator.pose.body_model import SMPLXBodyModel, make_toy_model, rest_pose_sequence
from signtranslator.pose.fitting import (
    GaussianPosePrior, GMMPosePrior, temporal_smoothness,
    self_collision_penalty, fitting_terms, FittingWeights,
)


# ---------------------------------------------------------------------------
# pose priors
# ---------------------------------------------------------------------------
def test_gaussian_prior_zero_at_mean_positive_elsewhere():
    D = 6
    mu = torch.randn(D, dtype=torch.float64)
    A = torch.randn(D, D, dtype=torch.float64)
    prec = A @ A.T + D * torch.eye(D, dtype=torch.float64)     # SPD
    prior = GaussianPosePrior(mean=mu, precision=prec)
    assert prior(mu).item() == pytest.approx(0.0, abs=1e-12)
    for _ in range(20):
        x = mu + torch.randn(D, dtype=torch.float64)
        assert prior(x).item() > 0.0


def test_gaussian_prior_equals_mahalanobis():
    D = 4
    mu = torch.zeros(D, dtype=torch.float64)
    prec = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    prior = GaussianPosePrior(mean=mu, precision=prec)
    x = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert prior(x).item() == pytest.approx(1.0 + 2.0 + 3.0 + 4.0, abs=1e-12)


def test_gmm_prior_lower_at_cluster_centre():
    means = torch.tensor([[5.0, 5.0], [-5.0, -5.0]], dtype=torch.float64)
    covs = torch.stack([torch.eye(2, dtype=torch.float64)] * 2)
    w = torch.tensor([0.5, 0.5], dtype=torch.float64)
    gmm = GMMPosePrior(w, means, covs)
    at_centre = gmm(means[0])
    far = gmm(torch.tensor([0.0, 0.0], dtype=torch.float64))
    assert at_centre.item() < far.item()                      # NLL lower at a mode


# ---------------------------------------------------------------------------
# temporal smoothness
# ---------------------------------------------------------------------------
def test_smoothness_zero_iff_constant():
    const = torch.ones(5, 3, dtype=torch.float64)
    assert temporal_smoothness(const).item() == 0.0
    varying = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    assert temporal_smoothness(varying).item() > 0.0


def test_smoothness_equals_l1_of_differences():
    q = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    assert temporal_smoothness(q).item() == pytest.approx(1.0 + 2.0, abs=1e-12)


def test_smoothness_single_frame_is_zero():
    assert temporal_smoothness(torch.randn(1, 4)).item() == 0.0


# ---------------------------------------------------------------------------
# self-collision
# ---------------------------------------------------------------------------
def test_collision_zero_when_spheres_separated():
    joints = torch.tensor([[[0.0, 0, 0], [10.0, 0, 0], [0.0, 10.0, 0]]],
                          dtype=torch.float64)
    radii = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    assert self_collision_penalty(joints, radii).sum().item() == 0.0


def test_collision_positive_and_correct_under_penetration():
    joints = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]]], dtype=torch.float64)
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)      # sum 2 > dist 1
    pen = self_collision_penalty(joints, radii).sum()
    assert pen.item() == pytest.approx((2.0 - 1.0) ** 2, abs=1e-12)


def test_collision_excludes_adjacent_pairs():
    joints = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]]], dtype=torch.float64)
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    adj = torch.tensor([[False, True], [True, False]])         # they are neighbours
    assert self_collision_penalty(joints, radii, adj).sum().item() == 0.0


# ---------------------------------------------------------------------------
# monocular is underdetermined (the document's point), demonstrated exactly
# ---------------------------------------------------------------------------
def test_monocular_reprojection_does_not_determine_3d():
    cam = PerspectiveCamera.look_at(500, 500, 320, 240,
                                    eye=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0))
    torch.manual_seed(0)
    X = torch.randn(12, 3, dtype=torch.float64) * 0.3
    X[:, 2] += 3.0                                             # in front of camera
    px, in_front = cam.project(X)
    assert in_front.all()

    # move each point along its ray from the camera centre by a random positive
    # factor: the projection is invariant, the 3D point is not
    eye = torch.tensor([0.0, 0.0, -3.0], dtype=torch.float64)
    rays = X - eye
    factors = 1.0 + 0.5 * torch.rand(12, 1, dtype=torch.float64)
    X2 = eye + factors * rays
    px2, _ = cam.project(X2)

    assert torch.allclose(px, px2, atol=1e-9)                 # identical 2D
    assert (X - X2).norm(dim=-1).mean() > 0.1                 # very different 3D
    # a SECOND camera breaks the tie: the two configs now differ in image
    cam2 = PerspectiveCamera.look_at(500, 500, 320, 240,
                                     eye=(3.0, 0.0, 0.0), target=(0.0, 0.0, 0.0))
    q1, _ = cam2.project(X)
    q2, _ = cam2.project(X2)
    assert (q1 - q2).norm(dim=-1).mean() > 1.0                # distinguishable now


# ---------------------------------------------------------------------------
# multi-view fit recovers 3D pose (gradient descent on the objective)
# ---------------------------------------------------------------------------
def test_multiview_fit_reduces_objective_and_recovers_pose():
    layout = SMPLXLayout(n_body=4, n_hand=2, n_expr=3, n_shape=3)
    model = SMPLXBodyModel(make_toy_model(layout, num_vertices=24, seed=2))

    # ground-truth motion: small random rotations
    torch.manual_seed(3)
    gt = rest_pose_sequence(layout, T=1)
    gt = MotionSequence(gamma=gt.gamma,
                        rot6d=gt.rot6d + 0.15 * torch.randn_like(gt.rot6d),
                        expr=gt.expr, beta=gt.beta, layout=layout)
    gt_joints = model(gt).joints.detach()

    cams = [
        PerspectiveCamera.look_at(500, 500, 256, 256, eye=(0.0, 0.0, -4.0), target=(0, 0, 0)),
        PerspectiveCamera.look_at(500, 500, 256, 256, eye=(4.0, 0.0, 0.0), target=(0, 0, 0)),
        PerspectiveCamera.look_at(500, 500, 256, 256, eye=(0.0, 4.0, 0.0), target=(0, 0, 0),
                                  up=(0.0, 0.0, 1.0)),
    ]
    targets = []
    for cam in cams:
        px, infront = cam.project(gt_joints)
        assert infront.all()
        targets.append(px.detach())

    # optimise rot6d from the rest pose
    rot = rest_pose_sequence(layout, T=1).rot6d.clone().requires_grad_(True)
    opt = torch.optim.Adam([rot], lr=0.05)

    def total():
        seq = MotionSequence(gamma=gt.gamma, rot6d=rot, expr=gt.expr, beta=gt.beta,
                             layout=layout)
        j = model(seq).joints
        loss = sum(reprojection_loss(cam.project(j)[0], tgt, sigma=200.0)
                   for cam, tgt in zip(cams, targets))
        return loss, j

    loss0, _ = total()
    for _ in range(400):
        opt.zero_grad()
        loss, _ = total()
        loss.backward()
        opt.step()
    lossN, jN = total()

    assert lossN.item() < 0.01 * loss0.item()                # objective collapsed
    mpjpe = (jN.detach() - gt_joints).norm(dim=-1).mean().item()
    assert mpjpe < 1e-2                                       # 3D recovered (multi-view)


def test_fitting_terms_assemble_and_weight():
    proj = torch.zeros(3, 2, dtype=torch.float64)
    kp = torch.ones(3, 2, dtype=torch.float64)
    motion = torch.zeros(2, 5, dtype=torch.float64)
    terms = fitting_terms(proj, kp, None, motion)
    w = FittingWeights(lam_2d=2.0)
    assert torch.isfinite(terms.total(w))
    assert terms.reproj.item() > 0.0
    assert terms.smooth.item() == 0.0                         # constant motion
