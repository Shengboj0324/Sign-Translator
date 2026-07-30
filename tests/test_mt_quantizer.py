"""Verification of the VQ-VAE vector quantiser.

Proves the distance identity, nearest-neighbour assignment, the straight-through
gradient identity, the commitment/codebook loss forms, EMA convergence to cluster
means (with Laplace smoothing), Code Reset, and perplexity bounds.
"""

import math

import pytest
import torch

from signtranslator.motion_transformer.quantizer import VectorQuantizer


def _vq(num_codes=8, dim=4, ema=True, seed=0):
    torch.manual_seed(seed)
    return VectorQuantizer(num_codes, dim, ema=ema).double()


# ---------------------------------------------------------------------------
# distances / assignment
# ---------------------------------------------------------------------------
def test_distance_identity_matches_cdist():
    vq = _vq(seed=1)
    z = torch.randn(20, 4, dtype=torch.float64)
    d = vq.distances(z)
    ref = torch.cdist(z, vq.codebook) ** 2
    assert torch.allclose(d, ref, atol=1e-9)


def test_quantize_picks_nearest_code():
    vq = _vq(seed=2)
    z = torch.randn(15, 4, dtype=torch.float64)
    idx, zq = vq.quantize(z)
    brute = torch.cdist(z, vq.codebook).argmin(-1)
    assert torch.equal(idx, brute)
    assert torch.allclose(zq, vq.codebook[idx], atol=1e-12)


def test_ties_break_to_lowest_index():
    vq = VectorQuantizer(2, 2, ema=True).double()
    with torch.no_grad():
        vq.codebook.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64))
    z = torch.zeros(1, 2, dtype=torch.float64)               # equidistant
    idx, _ = vq.quantize(z)
    assert int(idx[0]) == 0


# ---------------------------------------------------------------------------
# straight-through estimator
# ---------------------------------------------------------------------------
def test_straight_through_gradient_is_identity():
    vq = _vq(seed=3)
    z_e = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    out = vq(z_e)
    out["z_q"].sum().backward()
    # d(sum z_q_ste)/d z_e = 1 everywhere (the quantisation is transparent to grad)
    assert torch.allclose(z_e.grad, torch.ones_like(z_e), atol=1e-12)


def test_z_q_forward_equals_hard_quantisation():
    vq = _vq(seed=4)
    vq.eval()                                                # no EMA mutation in eval
    z_e = torch.randn(6, 4, dtype=torch.float64)
    out = vq(z_e)
    idx, zq = vq.quantize(z_e)
    assert torch.allclose(out["z_q"], zq, atol=1e-12)        # forward value is exact
    assert torch.equal(out["indices"], idx)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_commitment_loss_only_moves_encoder():
    vq = _vq(ema=True, seed=5)
    vq.eval()                                                # stable codebook for the identity
    z_e = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    out = vq(z_e)
    # commitment loss value == mean ||z_e - z_q||^2
    idx, zq = vq.quantize(z_e)
    actual = out["commit_loss"].detach().item()
    expected = ((z_e - zq) ** 2).mean().detach().item()
    assert abs(actual - expected) < 1e-12
    # its gradient reaches the encoder (z_e), codebook is a buffer in EMA mode
    out["commit_loss"].backward()
    assert z_e.grad is not None and z_e.grad.abs().sum() > 0


def test_codebook_loss_trains_parameter_codebook():
    vq = _vq(ema=False, seed=6)
    assert isinstance(vq.codebook, torch.nn.Parameter)
    z_e = torch.randn(4, 4, dtype=torch.float64)
    out = vq(z_e)
    out["loss"].backward()
    assert vq.codebook.grad is not None and vq.codebook.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# EMA convergence
# ---------------------------------------------------------------------------
def test_ema_codebook_converges_to_cluster_means():
    torch.manual_seed(7)
    centers = torch.tensor([[5.0, 5.0], [-5.0, -5.0], [5.0, -5.0]], dtype=torch.float64)
    vq = VectorQuantizer(3, 2, ema=True, decay=0.9).double()
    with torch.no_grad():                                    # seed codes near centers
        vq.codebook.copy_(centers + 0.5 * torch.randn(3, 2, dtype=torch.float64))
        vq.ema_w.copy_(vq.codebook.clone())
    vq.train()
    for _ in range(300):
        z = centers[torch.randint(0, 3, (60,))] + 0.1 * torch.randn(60, 2, dtype=torch.float64)
        vq(z)
    # each code should sit near its cluster centre
    d = torch.cdist(vq.codebook, centers)
    assert torch.all(d.min(dim=1).values < 0.3)


def test_ema_laplace_smoothing_survives_empty_codes():
    vq = VectorQuantizer(6, 3, ema=True).double()
    vq.train()
    # feed data that only ever uses a couple of codes -> others stay empty
    z = torch.zeros(10, 3, dtype=torch.float64)
    for _ in range(20):
        vq(z + 0.01 * torch.randn(10, 3, dtype=torch.float64))
    assert torch.isfinite(vq.codebook).all()                # no divide-by-zero NaN


# ---------------------------------------------------------------------------
# Code Reset
# ---------------------------------------------------------------------------
def test_reset_dead_codes_reseeds_only_dead_rows():
    vq = VectorQuantizer(4, 3, ema=True, reset_threshold=1.0).double()
    with torch.no_grad():
        vq.cluster_size.copy_(torch.tensor([10.0, 0.0, 5.0, 0.0], dtype=torch.float64))
    before = vq.codebook.clone()
    pool = torch.randn(50, 3, dtype=torch.float64)
    n = vq.reset_dead_codes(pool)
    assert n == 2                                            # codes 1 and 3 were dead
    assert torch.equal(vq.codebook[0], before[0])           # live rows unchanged
    assert torch.equal(vq.codebook[2], before[2])
    assert not torch.equal(vq.codebook[1], before[1])       # dead rows reseeded
    assert not torch.equal(vq.codebook[3], before[3])


# ---------------------------------------------------------------------------
# perplexity
# ---------------------------------------------------------------------------
def test_perplexity_bounds():
    vq = _vq(num_codes=4, seed=8)
    # all inputs map to one code -> perplexity ~ 1
    with torch.no_grad():
        vq.codebook.copy_(torch.tensor([[0.0, 0, 0, 0], [9, 9, 9, 9],
                                        [8, 8, 8, 8], [7, 7, 7, 7]], dtype=torch.float64))
    out = vq(torch.zeros(20, 4, dtype=torch.float64))
    assert abs(float(out["perplexity"]) - 1.0) < 1e-6
    # uniform usage over K -> perplexity ~ K
    z = vq.codebook.repeat(5, 1) + 1e-6
    out2 = vq(z)
    assert abs(float(out2["perplexity"]) - 4.0) < 1e-3
