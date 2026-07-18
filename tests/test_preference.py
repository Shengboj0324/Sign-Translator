"""Tests for adaptive graph refinement and preference optimisation (DPO)."""

import torch

from signtranslator.skeleton import SkeletonGraph, DEFAULT_EDGES
from signtranslator.models.stgcn import GraphConvolution, STGCNEncoder
from signtranslator.models import CrossModalDenoiser
from signtranslator.models.guided_diffusion import GuidedMotionDiffusion
from signtranslator.training.preference import (
    DiffusionDPO, jerk, bone_length_variance, naturalness_score,
    build_preference_pairs,
)


# ---- adaptive (learnable) adjacency ---------------------------------------
def test_adaptive_adjacency_starts_at_anatomical_prior():
    g = SkeletonGraph()
    gc = GraphConvolution(3, 8, g.adjacency(), adaptive=True)
    assert torch.allclose(gc.effective_adjacency(), gc.A, atol=1e-8)


def test_adaptive_adjacency_is_learnable_and_changes_output():
    g = SkeletonGraph()
    gc = GraphConvolution(3, 8, g.adjacency(), adaptive=True)
    names = {n for n, _ in gc.named_parameters()}
    assert "A_refine" in names
    x = torch.randn(2, 3, 6, g.num_nodes)
    before = gc(x)
    with torch.no_grad():
        gc.A_refine.add_(0.1)
    assert not torch.allclose(before, gc(x), atol=1e-5)


def test_non_adaptive_has_no_refinement_parameter():
    g = SkeletonGraph()
    gc = GraphConvolution(3, 8, g.adjacency(), adaptive=False)
    assert "A_refine" not in {n for n, _ in gc.named_parameters()}


def test_adaptive_refinement_receives_gradient():
    g = SkeletonGraph()
    enc = STGCNEncoder(3, g.adjacency(), channels=(16, 16), adaptive=True)
    enc(torch.randn(2, 3, 12, g.num_nodes)).sum().backward()
    refine = enc.blocks[0].gcn.A_refine
    assert refine.grad is not None and refine.grad.abs().sum() > 0


# ---- naturalness proxies ---------------------------------------------------
def test_jerk_zero_for_constant_velocity_motion():
    """Constant-velocity motion has zero third derivative."""
    t = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1)
    motion = t.repeat(1, 3, 1, 4)
    assert float(jerk(motion)) < 1e-8


def test_jerk_larger_for_noisy_motion():
    smooth = torch.sin(torch.linspace(0, 6.28, 32)).view(1, 1, 32, 1).repeat(1, 3, 1, 4)
    noisy = smooth + 0.5 * torch.randn_like(smooth)
    assert float(jerk(noisy)) > float(jerk(smooth))


def test_bone_length_variance_zero_for_rigid_translation():
    """Translating a rigid skeleton keeps every bone length constant."""
    g = SkeletonGraph()
    base = torch.randn(1, 3, 1, g.num_nodes)
    shift = torch.linspace(0, 1, 12).view(1, 1, 12, 1)
    motion = base + shift                       # rigid translation over time
    v = bone_length_variance(motion, list(DEFAULT_EDGES))
    assert float(v) < 1e-8


def test_bone_length_variance_detects_stretching():
    g = SkeletonGraph()
    motion = torch.randn(1, 3, 12, g.num_nodes)  # unconstrained => bones vary
    assert float(bone_length_variance(motion, list(DEFAULT_EDGES))) > 0


def test_naturalness_prefers_smooth_motion():
    smooth = torch.sin(torch.linspace(0, 6.28, 32)).view(1, 1, 32, 1).repeat(1, 3, 1, 8)
    jerky = smooth + 0.4 * torch.randn_like(smooth)
    assert float(naturalness_score(smooth)) > float(naturalness_score(jerky))


def test_build_preference_pairs_picks_best_and_worst():
    torch.manual_seed(0)
    smooth = torch.sin(torch.linspace(0, 6.28, 24)).view(1, 1, 24, 1).repeat(1, 3, 1, 6)
    jerky = smooth + 0.6 * torch.randn_like(smooth)
    mid = smooth + 0.2 * torch.randn_like(smooth)
    cands = torch.stack([jerky, mid, smooth], dim=1)   # (1, 3, C, T, V)
    pref, rej = build_preference_pairs(cands, naturalness_score)
    assert torch.allclose(pref[0], smooth[0], atol=1e-6)
    assert torch.allclose(rej[0], jerky[0], atol=1e-6)


# ---- Diffusion-DPO ---------------------------------------------------------
def _diffusion():
    net = CrossModalDenoiser(num_joints=6, in_channels=3, context_dim=8,
                             hidden_dim=32, num_layers=2, num_heads=2)
    return GuidedMotionDiffusion(net, num_timesteps=40, parameterization="x0")


def test_reference_policy_is_frozen_and_separate():
    d = _diffusion()
    dpo = DiffusionDPO(d, beta=0.1)
    assert all(not p.requires_grad for p in dpo.reference.parameters())
    # Mutating the live policy must not change the reference.
    before = next(dpo.reference.parameters()).detach().clone()
    with torch.no_grad():
        for p in d.parameters():
            p.add_(0.5)
    assert torch.allclose(next(dpo.reference.parameters()), before, atol=1e-8)


def test_dpo_loss_is_finite_and_stats_valid():
    torch.manual_seed(0)
    d = _diffusion()
    dpo = DiffusionDPO(d, beta=0.1)
    pref = torch.randn(4, 3, 12, 6)
    rej = torch.randn(4, 3, 12, 6)
    loss, stats = dpo.loss(pref, rej, cond=None)
    assert torch.isfinite(loss)
    assert 0.0 <= stats.accuracy <= 1.0


def test_dpo_at_initialisation_is_log2():
    """Before any update the policy equals the reference, so the implicit
    reward margin is exactly 0 and the loss is -log sigmoid(0) = log 2."""
    torch.manual_seed(0)
    d = _diffusion().eval()
    dpo = DiffusionDPO(d, beta=0.1)
    pref = torch.randn(3, 3, 12, 6)
    rej = torch.randn(3, 3, 12, 6)
    loss, stats = dpo.loss(pref, rej, cond=None)
    assert abs(float(loss) - torch.log(torch.tensor(2.0))) < 1e-4
    assert abs(stats.margin) < 1e-5


def test_dpo_training_increases_preference_margin():
    """Optimising DPO must push the implicit reward margin above zero."""
    torch.manual_seed(0)
    d = _diffusion()
    dpo = DiffusionDPO(d, beta=1.0)
    pref = torch.randn(4, 3, 12, 6)
    rej = torch.randn(4, 3, 12, 6)
    opt = torch.optim.Adam(d.parameters(), lr=3e-3)
    first = dpo.loss(pref, rej)[1]
    for _ in range(25):
        stats = dpo.step(opt, pref, rej)
    assert stats.margin > first.margin
    assert stats.loss < first.loss
