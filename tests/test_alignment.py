"""Tests for contrastive alignment (InfoNCE) and projection heads."""

import math

import torch
import torch.nn.functional as F

from signtranslator.models import ProjectionHead, ContrastiveAligner, info_nce_loss


def test_projection_head_is_unit_norm():
    head = ProjectionHead(32, 16)
    z = head(torch.randn(8, 32))
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_info_nce_perfect_alignment_beats_shuffled():
    """Matched embeddings must yield lower loss than mismatched ones."""
    torch.manual_seed(0)
    z = F.normalize(torch.randn(16, 8), dim=-1)
    matched, _ = info_nce_loss(z, z.clone(), temperature=0.07)
    shuffled = z[torch.randperm(16)]
    mismatched, _ = info_nce_loss(z, shuffled, temperature=0.07)
    assert matched < mismatched


def test_info_nce_lower_bounded_and_symmetric():
    torch.manual_seed(1)
    a = F.normalize(torch.randn(12, 8), dim=-1)
    b = F.normalize(torch.randn(12, 8), dim=-1)
    loss_ab, logits = info_nce_loss(a, b, temperature=0.1)
    loss_ba, _ = info_nce_loss(b, a, temperature=0.1)
    # Symmetric InfoNCE is invariant to swapping the two modalities.
    assert torch.allclose(loss_ab, loss_ba, atol=1e-6)
    assert loss_ab >= 0.0
    assert logits.shape == (12, 12)


def test_info_nce_matches_manual_cross_entropy():
    torch.manual_seed(2)
    a = F.normalize(torch.randn(5, 4), dim=-1)
    b = F.normalize(torch.randn(5, 4), dim=-1)
    tau = 0.2
    loss, _ = info_nce_loss(a, b, temperature=tau)
    logits = (a @ b.t()) / tau
    tgt = torch.arange(5)
    manual = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
    assert torch.allclose(loss, manual, atol=1e-6)


def test_aligner_forward_and_gradients():
    aligner = ContrastiveAligner(motion_dim=24, language_dim=18, latent_dim=16)
    m = torch.randn(10, 24)
    l = torch.randn(10, 18)
    out = aligner(m, l)
    assert out["z_motion"].shape == (10, 16)
    assert out["logits"].shape == (10, 10)
    out["loss"].backward()
    assert aligner.log_scale.grad is not None  # temperature is learnable


def test_temperature_positive_by_construction():
    aligner = ContrastiveAligner(4, 4, 4, init_temperature=0.05)
    # log_scale = log(1/0.05) = log(20)
    assert math.isclose(float(aligner.log_scale.exp()), 20.0, rel_tol=1e-5)
