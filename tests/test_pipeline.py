"""Build / forward / generation tests for the end-to-end SignTranslator."""

import torch
import pytest

from signtranslator import ModelConfig, DiffusionConfig
from signtranslator.models import SignTranslator


def _small_model():
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=16,
                       vocab_size=128)
    dcfg = DiffusionConfig(num_timesteps=50, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    return SignTranslator(mcfg, dcfg), mcfg


def test_model_builds_and_reports_parameters():
    model, _ = _small_model()
    assert model.num_parameters() > 0


def test_forward_returns_finite_losses():
    model, mcfg = _small_model()
    n = 6
    pose = torch.randn(n, mcfg.in_channels, mcfg.num_frames, mcfg.num_joints)
    tokens = torch.randint(1, mcfg.vocab_size, (n, 6))
    out = model(pose, tokens)
    for key in ("loss", "contrastive_loss", "diffusion_loss"):
        assert torch.isfinite(out[key]).all()
    assert out["logits"].shape == (n, n)


def test_backward_updates_all_submodules():
    model, mcfg = _small_model()
    n = 6
    pose = torch.randn(n, mcfg.in_channels, mcfg.num_frames, mcfg.num_joints)
    tokens = torch.randint(1, mcfg.vocab_size, (n, 6))

    # The denoiser's output projection is zero-initialised, so on the very first
    # backward pass every *upstream* denoiser gradient is exactly zero (dy/dh =
    # W_out = 0). Take one optimiser step to move W_out off zero, then verify
    # gradients flow through every subsystem on the next step.
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    opt.zero_grad()
    model(pose, tokens)["loss"].backward()
    opt.step()

    opt.zero_grad()
    out = model(pose, tokens)
    out["loss"].backward()

    checks = {
        "motion_encoder": model.motion_encoder.blocks[0].gcn.theta.weight,
        "text_encoder": model.text_encoder.token_emb.weight,
        "aligner": model.aligner.motion_head.net[0].weight,
        "denoiser_output": model.diffusion.denoiser.output_proj.weight,
        "denoiser_input": model.diffusion.denoiser.input_proj.weight,
    }
    for name, p in checks.items():
        assert p.grad is not None, f"{name} received no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all-zero"


def test_generate_produces_motion_clip():
    model, mcfg = _small_model()
    tokens = torch.randint(1, mcfg.vocab_size, (2, 6))
    clip = model.generate(tokens, num_frames=12, use_ddim=True, ddim_steps=5)
    assert clip.shape == (2, mcfg.in_channels, 12, mcfg.num_joints)
    assert torch.isfinite(clip).all()


def test_configuration_roundtrip_is_versioned_and_typed():
    model_cfg = ModelConfig(stgcn_channels=(16, 32), text_embed_dim=32,
                            text_heads=4)
    payload = model_cfg.to_dict()
    assert payload["schema_version"] == 1
    assert payload["config_type"] == "ModelConfig"
    assert ModelConfig.from_dict(payload) == model_cfg


def test_configuration_rejects_wrong_schema_and_unknown_fields():
    payload = ModelConfig().to_dict()
    payload["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        ModelConfig.from_dict(payload)

    payload = ModelConfig().to_dict()
    payload["values"]["silent_fallback"] = True
    with pytest.raises(ValueError, match="unknown"):
        ModelConfig.from_dict(payload)


def test_configuration_validation_survives_python_optimized_mode():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(text_embed_dim=30, text_heads=4)
