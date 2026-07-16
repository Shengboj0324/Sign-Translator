"""End-to-end learning test: the model must actually reduce its loss.

This is the strongest sanity check that the architecture, math, and gradient
plumbing are correct together -- a broken sign in the diffusion coefficients or
a detached graph would show up as a flat or rising loss curve.
"""

import torch

from signtranslator import ModelConfig, DiffusionConfig, TrainConfig
from signtranslator.train import train


def test_training_reduces_loss_on_synthetic_data():
    torch.manual_seed(0)
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=16,
                       vocab_size=128)
    dcfg = DiffusionConfig(num_timesteps=50, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    tcfg = TrainConfig(max_steps=60, batch_size=16, lr=3e-4, device="cpu")

    result = train(mcfg, dcfg, tcfg, verbose=False)
    hist = result["history"]["total"]
    k = 10
    first = sum(hist[:k]) / k
    last = sum(hist[-k:]) / k
    # Require a clear reduction, not a coincidental blip.
    assert last < first * 0.9, f"loss did not decrease: first={first:.3f} last={last:.3f}"


def test_contrastive_component_learns():
    torch.manual_seed(0)
    mcfg = ModelConfig(num_joints=27, num_frames=16, stgcn_channels=(16, 32),
                       text_embed_dim=32, text_layers=2, text_heads=2, latent_dim=16,
                       vocab_size=128)
    dcfg = DiffusionConfig(num_timesteps=50, denoiser_dim=32, denoiser_layers=2,
                           denoiser_heads=2)
    tcfg = TrainConfig(max_steps=60, batch_size=16, lr=3e-4, device="cpu")
    result = train(mcfg, dcfg, tcfg, verbose=False)
    hist = result["history"]["contrastive"]
    assert sum(hist[-10:]) / 10 < sum(hist[:10]) / 10
