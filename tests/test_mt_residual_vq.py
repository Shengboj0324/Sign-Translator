"""Verification of residual VQ and part-specific codebooks.

Proves the residual algebra, monotone residual decrease across stages (codebooks
k-means-initialised on the data), the single straight-through gradient identity
over the whole cascade, single-stage==plain-VQ, index decode round-trip, and the
part-specific split/independence.
"""

import pytest
import torch

from signtranslator.motion_transformer.quantizer import VectorQuantizer
from signtranslator.motion_transformer.residual_vq import (
    kmeans, ResidualVQ, PartitionedVQ,
)


# ---------------------------------------------------------------------------
# residual algebra + monotone decrease
# ---------------------------------------------------------------------------
def test_residual_identity_holds_exactly():
    # ||r - c||^2 == ||r||^2 - 2 r.c + ||c||^2
    r = torch.randn(10, 4, dtype=torch.float64)
    c = torch.randn(10, 4, dtype=torch.float64)
    lhs = ((r - c) ** 2).sum(-1)
    rhs = (r * r).sum(-1) - 2 * (r * c).sum(-1) + (c * c).sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_residual_norm_is_monotone_non_increasing():
    torch.manual_seed(0)
    data = torch.randn(400, 8, dtype=torch.float64)
    rvq = ResidualVQ(num_stages=4, num_codes=32, dim=8, ema=True).double()
    rvq.init_from_data(data, seed=0)
    rvq.eval()
    out = rvq(data)
    norms = out["residual_norms"]                            # length num_stages+1
    for i in range(len(norms) - 1):
        assert norms[i + 1] <= norms[i] + 1e-9               # never increases
    # and the final residual is strictly smaller than the input residual
    assert norms[-1] < norms[0]


def test_more_stages_reduce_reconstruction_error():
    torch.manual_seed(1)
    data = torch.randn(300, 6, dtype=torch.float64)
    errs = []
    for stages in (1, 2, 4):
        rvq = ResidualVQ(num_stages=stages, num_codes=16, dim=6, ema=True).double()
        rvq.init_from_data(data, seed=1)
        rvq.eval()
        out = rvq(data)
        errs.append(float((out["z_q"] - data).norm()))
    assert errs[1] <= errs[0] + 1e-9 and errs[2] <= errs[1] + 1e-9


# ---------------------------------------------------------------------------
# single straight-through over the cascade
# ---------------------------------------------------------------------------
def test_cascade_straight_through_is_identity():
    torch.manual_seed(2)
    rvq = ResidualVQ(num_stages=3, num_codes=16, dim=4, ema=True).double()
    rvq.init_from_data(torch.randn(200, 4, dtype=torch.float64), seed=2)
    z_e = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    out = rvq(z_e)
    out["z_q"].sum().backward()
    # gradient passes through the ENTIRE cascade as identity (all stages, not just 1)
    assert torch.allclose(z_e.grad, torch.ones_like(z_e), atol=1e-12)


def test_z_q_equals_sum_of_codes():
    torch.manual_seed(3)
    rvq = ResidualVQ(num_stages=3, num_codes=16, dim=4, ema=True).double()
    rvq.init_from_data(torch.randn(200, 4, dtype=torch.float64), seed=3)
    rvq.eval()
    z_e = torch.randn(5, 4, dtype=torch.float64)
    out = rvq(z_e)
    summed = rvq.decode_indices(out["indices"])
    assert torch.allclose(out["z_q"], summed, atol=1e-12)    # forward = sum of codes


# ---------------------------------------------------------------------------
# single-stage RVQ == plain VQ
# ---------------------------------------------------------------------------
def test_single_stage_rvq_matches_plain_vq():
    torch.manual_seed(4)
    rvq = ResidualVQ(num_stages=1, num_codes=8, dim=4, ema=True).double()
    rvq.eval()
    z_e = torch.randn(6, 4, dtype=torch.float64)
    out = rvq(z_e)
    plain_idx, plain_zq = rvq.quantizers[0].quantize(z_e)
    assert torch.allclose(out["z_q"], plain_zq, atol=1e-12)
    assert torch.equal(out["indices"][..., 0], plain_idx)


# ---------------------------------------------------------------------------
# part-specific codebooks
# ---------------------------------------------------------------------------
def test_partitioned_vq_splits_and_concatenates():
    torch.manual_seed(5)
    pvq = PartitionedVQ({"torso": 3, "hands": 6, "face": 2}, num_stages=2,
                        num_codes=16).double()
    z_e = torch.randn(10, 11, dtype=torch.float64)
    pvq.init_from_data(z_e, seed=5)
    pvq.eval()
    out = pvq(z_e)
    assert out["z_q"].shape == (10, 11)                      # total dim preserved
    assert set(out["parts"]) == {"torso", "hands", "face"}


def test_partitioned_quantisation_is_independent_per_part():
    """Perturbing the torso channels must not change the hands' quantised output."""
    torch.manual_seed(6)
    pvq = PartitionedVQ({"torso": 3, "hands": 6}, num_stages=1, num_codes=16).double()
    z_e = torch.randn(8, 9, dtype=torch.float64)
    pvq.init_from_data(z_e, seed=6)
    pvq.eval()
    out1 = pvq(z_e)
    z2 = z_e.clone(); z2[:, :3] += 10.0                      # change torso only
    out2 = pvq(z2)
    hands_slice = pvq.offsets["hands"]
    assert torch.allclose(out1["z_q"][:, hands_slice], out2["z_q"][:, hands_slice],
                          atol=1e-12)
    assert not torch.allclose(out1["z_q"][:, :3], out2["z_q"][:, :3])   # torso changed


def test_kmeans_returns_k_centroids():
    data = torch.randn(50, 3, dtype=torch.float64)
    c = kmeans(data, k=5, iters=5, generator=torch.Generator().manual_seed(0))
    assert c.shape == (5, 3) and torch.isfinite(c).all()


def test_rvq_gradients_flow():
    rvq = ResidualVQ(num_stages=2, num_codes=8, dim=4, ema=False).double()  # param codebooks
    z_e = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    out = rvq(z_e)
    (out["z_q"].pow(2).sum() + out["loss"]).backward()
    assert z_e.grad is not None and torch.isfinite(z_e.grad).all()
    assert any(vq.codebook.grad is not None for vq in rvq.quantizers)
