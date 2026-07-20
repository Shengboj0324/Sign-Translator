"""Integration + evaluation for the diffusion motion generator.

Ties the DiT, schedule, CFG, and constraints together: training loss + gradient,
guided sampling shape, stochastic multimodality (vs a deterministic baseline),
semantic-preservation-verified multimodality, constraint-projected feasibility, a
real overfit, and a cycle-stress determinism/finiteness loop.
"""

import pytest
import torch

from signtranslator.diffusion_gen.dit import TemporalDiT
from signtranslator.diffusion_gen.generator import DiffusionMotionGenerator
from signtranslator.diffusion_gen.constraints import project_joint_limits, joint_limit_penalty
from signtranslator.diffusion_gen.evaluation import (
    multimodality, semantic_preservation_verified_multimodality, jerk,
    p95_generation_time, compare_generators,
)


def _gen(in_dim=8, T_steps=40, param="x0", seed=0):
    torch.manual_seed(seed)
    dit = TemporalDiT(in_dim=in_dim, dim=32, depth=2, num_heads=4)
    return DiffusionMotionGenerator(dit, in_dim=in_dim, num_timesteps=T_steps,
                                    param=param, p_uncond=0.1)


def _activate(model, scale=0.1, seed=0):
    """Give the zero-init adaLN modulation small random weights, so the denoiser is
    non-degenerate (as a trained model would be) and sampling is genuinely
    stochastic. Without this the adaLN-Zero DiT predicts 0 and the reverse process
    collapses -- a property of the *untrained* init, not the sampler."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.abs().sum() == 0:
                m.weight.copy_(scale * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(scale * torch.randn(m.bias.shape, generator=g))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def test_training_loss_finite_and_gradient_flows():
    gen = _gen()
    x0 = torch.randn(3, 10, 8)
    cond = torch.randn(3, 32)
    loss = gen.training_loss(x0, cond_vec=cond)
    assert torch.isfinite(loss)
    loss.backward()
    n = sum(1 for p in gen.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n > 0


def test_generator_overfits_a_tiny_clip():
    torch.manual_seed(1)
    gen = _gen(in_dim=8, T_steps=40, param="x0")
    x0 = torch.randn(1, 8, 8) * 0.5
    opt = torch.optim.Adam(gen.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    l0 = None
    for step in range(250):
        opt.zero_grad()
        loss = gen.training_loss(x0, generator=g)
        loss.backward(); opt.step()
        if step == 0:
            l0 = float(loss)
    # average recent loss well below the initial (denoiser learns the clip)
    final = float(torch.stack([gen.training_loss(x0, generator=g) for _ in range(8)]).mean())
    assert final < 0.6 * l0


# ---------------------------------------------------------------------------
# sampling + multimodality
# ---------------------------------------------------------------------------
def test_sampling_shape_and_finite():
    gen = _gen(T_steps=20)
    gen.eval()
    x = gen.sample((2, 6, 8), generator=torch.Generator().manual_seed(0))
    assert x.shape == (2, 6, 8) and torch.isfinite(x).all()


def test_diffusion_is_multimodal_deterministic_is_not():
    gen = _gen(T_steps=20)
    gen.eval()
    _activate(gen.denoiser)
    # stochastic diffusion: different noise seeds -> different samples
    samples = torch.stack([gen.sample((1, 6, 8), generator=torch.Generator().manual_seed(s))[0]
                           for s in range(5)])
    assert float(multimodality(samples)) > 1e-3
    # a deterministic "baseline": the same output repeated -> zero multimodality
    det = samples[0:1].expand(5, -1, -1)
    assert float(multimodality(det)) < 1e-9
    report = compare_generators({"diffusion": samples, "deterministic": det})
    assert report["diffusion"] > report["deterministic"]


def test_semantic_preservation_verified_multimodality():
    gen = _gen(T_steps=20)
    gen.eval()
    _activate(gen.denoiser)
    samples = torch.stack([gen.sample((1, 6, 8), generator=torch.Generator().manual_seed(s))[0]
                           for s in range(6)])
    preserved = torch.tensor([True, True, False, True, True, False])   # meaning-check
    div, frac = semantic_preservation_verified_multimodality(samples, preserved)
    assert float(div) > 0 and abs(frac - 4 / 6) < 1e-6       # float32 mean of a bool mask


# ---------------------------------------------------------------------------
# constraint-projected sampling
# ---------------------------------------------------------------------------
def test_constraint_projected_sampling_is_feasible():
    gen = _gen(T_steps=15)
    gen.eval()
    _activate(gen.denoiser)
    theta_max = 1.0
    out = gen.sample((1, 5, 8), project=lambda x0: project_joint_limits(x0, theta_max),
                     generator=torch.Generator().manual_seed(0))
    # the final sample equals the projected x0 at t=0 -> within joint limits
    assert joint_limit_penalty(out, theta_max).item() < 1e-6


def test_classifier_free_guidance_sampling_runs():
    gen = _gen(T_steps=15)
    gen.eval()
    _activate(gen.denoiser)
    x = gen.sample((1, 5, 8), cond_vec=torch.randn(1, 32), w=2.0,
                   generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_jerk_zero_for_constant_velocity_positive_for_jittery():
    T = 20
    ramp = torch.arange(T, dtype=torch.float64).reshape(1, T, 1).expand(1, T, 3)
    assert jerk(ramp).item() < 1e-9                          # constant velocity -> no jerk
    jittery = ramp + torch.randn(1, T, 3, dtype=torch.float64)
    assert jerk(jittery).item() > 0


def test_p95_generation_time():
    times = [0.1 * i for i in range(1, 101)]                 # 0.1 .. 10.0
    assert abs(p95_generation_time(times) - 9.6) < 0.11


# ---------------------------------------------------------------------------
# cycle stress
# ---------------------------------------------------------------------------
def test_cycle_stress_determinism_and_finiteness():
    gen = _gen(T_steps=15)
    gen.eval()
    _activate(gen.denoiser)
    for s in range(40):
        g1 = torch.Generator().manual_seed(100 + s)
        g2 = torch.Generator().manual_seed(100 + s)
        x1 = gen.sample((1, 5, 8), generator=g1)
        x2 = gen.sample((1, 5, 8), generator=g2)
        assert torch.equal(x1, x2)                           # deterministic given seed
        assert torch.isfinite(x1).all()
