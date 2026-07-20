"""Integration + whole-chain cycle stress for the motion transformer.

Ties the tokenizer, backbone, memory, and streaming together over the Doc-04
pose-6D representation: reconstruction round-trip, plan-conditioned generation
(cross-attention), long-discourse locus persistence (memory carries early loci),
streaming SO(3) stitch validity, and a 100-case determinism/finiteness loop.
"""

import pytest
import torch

from signtranslator.pose.rotations import (
    axis_angle_to_matrix, matrix_to_rotation_6d, is_rotation_matrix,
)
from signtranslator.motion_transformer.chain import MotionGenerationChain
from signtranslator.motion_transformer.autoencoder import motion_loss, MotionLossWeights


def _chain(num_joints=3, seed=0, **kw):
    torch.manual_seed(seed)
    return MotionGenerationChain(num_joints=num_joints, dim=32, num_codes=64,
                                 num_downsamples=1, rvq_stages=2, **kw).double()


def _pose_motion(N=1, J=3, T=16, seed=0):
    """A batch of valid 6D-rotation motion sequences (N, C=J*6, T)."""
    g = torch.Generator().manual_seed(seed)
    aa = 0.5 * torch.randn(N, T, J, 3, generator=g, dtype=torch.float64)
    six = matrix_to_rotation_6d(axis_angle_to_matrix(aa))    # (N, T, J, 6)
    return six.reshape(N, T, J * 6).permute(0, 2, 1).contiguous()


# ---------------------------------------------------------------------------
# tokenizer round trip over the pose representation
# ---------------------------------------------------------------------------
def test_reconstruction_round_trip_shape_and_loss():
    chain = _chain(num_joints=3, seed=1)
    motion = _pose_motion(N=2, J=3, T=16, seed=1)
    chain.tokenizer.init_codebook(motion)
    recon, q = chain.reconstruct(motion)
    assert recon.shape == motion.shape and torch.isfinite(recon).all()
    terms = motion_loss(recon, motion, num_joints=3, commit_loss=q["loss"],
                        weights=MotionLossWeights())
    assert torch.isfinite(terms["total"])                   # geodesic+vel+acc+commit finite


# ---------------------------------------------------------------------------
# plan-conditioned generation (cross-attention integration)
# ---------------------------------------------------------------------------
def test_generation_depends_on_plan():
    chain = _chain(seed=2)
    chain.eval()
    plan_a = torch.randn(1, 4, 32, dtype=torch.float64)
    plan_b = torch.randn(1, 4, 32, dtype=torch.float64)
    out_a = chain.decode_from_plan(plan_a, motion_len=16)
    out_b = chain.decode_from_plan(plan_b, motion_len=16)
    assert out_a.shape == (1, 18, 16)                        # (N, C=J*6, T)
    assert not torch.allclose(out_a, out_b, atol=1e-4)       # plan actually conditions


# ---------------------------------------------------------------------------
# long-discourse locus persistence (recurrent memory)
# ---------------------------------------------------------------------------
def test_long_discourse_locus_persists():
    """A locus introduced in the FIRST chunk must still influence the memory state
    after many later chunks -- the discourse-memory requirement."""
    chain = _chain(seed=3)
    n_chunks = 10
    first = torch.randn(1, 32, dtype=torch.float64, requires_grad=True)
    summaries = [first] + [torch.randn(1, 32, dtype=torch.float64) for _ in range(n_chunks - 1)]
    final_state = chain.run_discourse(summaries)
    grad = torch.autograd.grad(final_state.sum(), first)[0]
    assert grad.abs().sum() > 0                              # early locus still present
    assert torch.isfinite(final_state).all()


# ---------------------------------------------------------------------------
# streaming SO(3) stitch validity
# ---------------------------------------------------------------------------
def test_streaming_stitch_produces_valid_continuous_rotations():
    chain = _chain(seed=4)
    chain.eval()
    # produce a few overlapping motion chunks and convert to rotations
    chunks_rot = []
    for c in range(3):
        motion = _pose_motion(N=1, J=3, T=8, seed=10 + c)
        R = chain.motion_to_rotations(motion)[0, :, 0]      # (T, 3, 3) joint 0
        chunks_rot.append(R)
    stitched = chain.stitch_rotation_chunks(chunks_rot, overlap=2)
    assert stitched.shape[0] == 8 * 3 - 2 * 2               # overlap-add length
    assert is_rotation_matrix(stitched, atol=1e-9).all()    # stays on SO(3)


# ---------------------------------------------------------------------------
# whole-chain cycle stress
# ---------------------------------------------------------------------------
def test_cycle_stress_determinism_and_finiteness():
    chain = _chain(seed=5)
    chain.eval()
    for s in range(100):
        motion = _pose_motion(N=1, J=3, T=16, seed=200 + s)
        with torch.no_grad():
            r1, _ = chain.reconstruct(motion)
            r2, _ = chain.reconstruct(motion)
        assert torch.equal(r1, r2)                           # deterministic
        assert torch.isfinite(r1).all()


def test_gradients_flow_through_chain():
    chain = _chain(seed=6)
    motion = _pose_motion(N=1, J=3, T=16, seed=7)
    chain.tokenizer.init_codebook(motion)
    recon, q = chain.reconstruct(motion)
    (torch.nn.functional.mse_loss(recon, motion) + q["loss"]).backward()
    n = sum(1 for p in chain.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n > 0
