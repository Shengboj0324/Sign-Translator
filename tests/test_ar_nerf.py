"""Verification of NeRF volume rendering.

Proves the transmittance recursion, weight normalisation (Σw = 1 − T_final), the
exact reduction to §4 alpha compositing, and the opaque/transparent limits.
"""

import math

import pytest
import torch

from signtranslator.avatar_render.nerf import (
    deltas_from_samples, alphas_from_density, transmittance, volume_render,
    expected_depth,
)
from signtranslator.avatar_render.gaussian import alpha_composite


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------
def test_alphas_from_density_formula():
    sigma = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float64)
    delta = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    a = alphas_from_density(sigma, delta)
    assert torch.allclose(a, 1.0 - torch.exp(-sigma * delta), atol=1e-12)
    assert float(a[0]) == 0.0                                # zero density -> zero alpha


def test_transmittance_recursion():
    alphas = torch.tensor([0.2, 0.5, 0.3, 0.9], dtype=torch.float64)
    T = transmittance(alphas)
    assert float(T[0]) == 1.0                                # nothing in front
    for i in range(len(alphas) - 1):
        assert abs(float(T[i + 1]) - float(T[i]) * (1 - float(alphas[i]))) < 1e-12


# ---------------------------------------------------------------------------
# weight normalisation
# ---------------------------------------------------------------------------
def test_weights_sum_to_one_minus_final_transmittance():
    torch.manual_seed(0)
    sigma = torch.rand(10, dtype=torch.float64) * 3
    colors = torch.rand(10, 3, dtype=torch.float64)
    t = torch.linspace(0, 1, 10, dtype=torch.float64)
    _, w, acc = volume_render(sigma, colors, t, far=1e10)
    one_minus = 1 - alphas_from_density(sigma, deltas_from_samples(t))
    T_final = float(torch.prod(one_minus))
    assert abs(float(w.sum()) - (1 - T_final)) < 1e-9
    assert abs(float(acc) - (1 - T_final)) < 1e-9
    assert float(w.sum()) <= 1.0 + 1e-9                      # energy conserving


# ---------------------------------------------------------------------------
# reduction to alpha compositing
# ---------------------------------------------------------------------------
def test_volume_render_equals_alpha_compositing():
    torch.manual_seed(1)
    sigma = torch.rand(6, dtype=torch.float64) * 2
    colors = torch.rand(6, 3, dtype=torch.float64)
    t = torch.linspace(0, 1, 6, dtype=torch.float64)
    color, _, _ = volume_render(sigma, colors, t)
    alphas = alphas_from_density(sigma, deltas_from_samples(t))
    C_ref, _ = alpha_composite(colors, alphas)               # §4 over operator
    assert torch.allclose(color, C_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------
def test_opaque_first_sample_dominates():
    sigma = torch.tensor([1e6, 1e6, 1e6], dtype=torch.float64)   # fully opaque
    colors = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=torch.float64)
    t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    color, w, acc = volume_render(sigma, colors, t)
    assert torch.allclose(color, colors[0], atol=1e-6)       # front sample seen
    assert abs(float(acc) - 1.0) < 1e-6


def test_transparent_medium_renders_nothing():
    sigma = torch.zeros(5, dtype=torch.float64)              # empty space
    colors = torch.rand(5, 3, dtype=torch.float64)
    t = torch.linspace(0, 1, 5, dtype=torch.float64)
    color, w, acc = volume_render(sigma, colors, t)
    assert torch.allclose(color, torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert float(acc) == 0.0 and float(w.sum()) == 0.0


def test_expected_depth_of_a_single_opaque_surface():
    sigma = torch.tensor([0.0, 0.0, 1e6, 0.0], dtype=torch.float64)  # surface at index 2
    colors = torch.zeros(4, 3, dtype=torch.float64)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float64)
    _, w, _ = volume_render(sigma, colors, t)
    assert abs(float(expected_depth(w, t)) - 0.5) < 1e-4     # depth = t of the surface
