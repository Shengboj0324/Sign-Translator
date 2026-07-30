"""Integration tests for the full bidirectional pipeline."""

import torch

from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import BidirectionalSignTranslator
from signtranslator.models.planner import BOS, EOS


def _model():
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=16)
    dcfg = DiffusionConfig(num_timesteps=40, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    return BidirectionalSignTranslator(mcfg, dcfg, src_vocab=64, gloss_vocab=48,
                                       num_glosses=20), mcfg


def test_builds_and_counts_params():
    m, _ = _model()
    assert m.num_parameters() > 0


def _batch(mcfg, n=4):
    pose = torch.randn(n, mcfg.in_channels, mcfg.num_frames, mcfg.num_joints)
    gloss_tokens = torch.randint(1, 48, (n, 6))
    src = torch.randint(3, 64, (n, 5))
    gloss_seq = torch.cat([torch.full((n, 1), BOS),
                           torch.randint(3, 48, (n, 4)),
                           torch.full((n, 1), EOS)], dim=1)
    ctc_targets = torch.randint(1, 21, (n, 3))
    ctc_lengths = torch.full((n,), 3, dtype=torch.long)
    return {"pose": pose, "gloss_tokens": gloss_tokens, "src": src,
            "gloss_seq": gloss_seq, "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths}


def test_all_branch_losses_finite():
    m, mcfg = _model()
    losses = m.training_step(_batch(mcfg))
    for key in ("generation", "planner", "recognition", "total"):
        assert torch.isfinite(losses[key]).all(), key


def test_combined_training_step_reduces_loss():
    torch.manual_seed(0)
    m, mcfg = _model()
    batch = _batch(mcfg)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    first = m.training_step(batch)["total"].detach().item()
    for _ in range(20):
        loss = m.training_step(batch)["total"]
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.detach().item() < first


def test_speech_to_sign_end_to_end_shapes():
    m, mcfg = _model()
    src = torch.randint(3, 64, (2, 5))
    out = m.translate_speech_to_sign(src, num_frames=12, ddim_steps=4)
    assert isinstance(out["gloss"], list) and len(out["gloss"]) == 2
    assert out["motion"].shape == (2, mcfg.in_channels, 12, mcfg.num_joints)
    assert torch.isfinite(out["motion"]).all()


def test_recognize_returns_gloss_lists():
    m, mcfg = _model()
    pose = torch.randn(3, mcfg.in_channels, mcfg.num_frames, mcfg.num_joints)
    decoded = m.recognize(pose)
    assert isinstance(decoded, list) and len(decoded) == 3


def test_generation_gradients_reach_denoiser_and_cond_encoder():
    m, mcfg = _model()
    batch = _batch(mcfg)
    # Warm up the zero-init denoiser output so upstream grads are non-zero.
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    m.generation_loss(batch["pose"], batch["gloss_tokens"]).backward()
    opt.step(); opt.zero_grad()
    m.generation_loss(batch["pose"], batch["gloss_tokens"]).backward()
    assert m.diffusion.denoiser.input_proj.weight.grad.abs().sum() > 0
    # Generation conditions on the generator-private encoder ...
    assert m.cond_encoder.token_emb.weight.grad.abs().sum() > 0
    # ... and must NOT touch the manifold encoder, or retrieval would collapse
    # during generator fine-tuning.
    assert (m.gloss_encoder.token_emb.weight.grad is None
            or m.gloss_encoder.token_emb.weight.grad.abs().sum() == 0)


def test_alignment_gradients_reach_manifold_encoder_only():
    m, mcfg = _model()
    batch = _batch(mcfg)
    m.alignment_loss(batch["pose"], batch["gloss_tokens"]).backward()
    assert m.gloss_encoder.token_emb.weight.grad.abs().sum() > 0
    assert (m.cond_encoder.token_emb.weight.grad is None
            or m.cond_encoder.token_emb.weight.grad.abs().sum() == 0)
