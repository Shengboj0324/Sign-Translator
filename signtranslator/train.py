"""Training loop for the SignTranslator joint objective.

Usage:
    python -m signtranslator.train --steps 300 --batch-size 16

The default configuration trains on the synthetic dataset and prints the total,
contrastive, and diffusion losses. It is intentionally small so it runs on CPU;
swap in a real dataset and foundation-model encoders to scale up.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, DiffusionConfig, TrainConfig
from .data import SyntheticSignDataset, collate_batch
from .models import SignTranslator


def build_model(model_cfg: ModelConfig, diff_cfg: DiffusionConfig,
                device: str = "cpu") -> SignTranslator:
    model = SignTranslator(model_cfg, diff_cfg).to(device)
    return model


def train(model_cfg: ModelConfig, diff_cfg: DiffusionConfig,
          train_cfg: TrainConfig, log_every: int = 20, verbose: bool = True) -> dict:
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device

    dataset = SyntheticSignDataset(
        num_joints=model_cfg.num_joints, in_channels=model_cfg.in_channels,
        num_frames=model_cfg.num_frames, vocab_size=model_cfg.vocab_size,
        seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True,
                        collate_fn=collate_batch, drop_last=True)

    model = build_model(model_cfg, diff_cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay)

    model.train()
    history = {"total": [], "contrastive": [], "diffusion": []}
    step = 0
    done = False
    while not done:
        for batch in loader:
            pose = batch["pose"].to(device)
            tokens = batch["tokens"].to(device)
            text_mask = batch["text_mask"].to(device)

            out = model(pose, tokens, text_mask=text_mask,
                        w_contrastive=train_cfg.w_contrastive,
                        w_diffusion=train_cfg.w_diffusion)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()

            history["total"].append(loss.detach().item())
            history["contrastive"].append(out["contrastive_loss"].detach().item())
            history["diffusion"].append(out["diffusion_loss"].detach().item())

            if verbose and step % log_every == 0:
                print(f"step {step:4d} | total {loss:.4f} | "
                      f"contrastive {out['contrastive_loss']:.4f} | "
                      f"diffusion {out['diffusion_loss']:.4f}")
            step += 1
            if step >= train_cfg.max_steps:
                done = True
                break

    return {"model": model, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SignTranslator (synthetic data).")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    diff_cfg = DiffusionConfig(num_timesteps=args.timesteps)
    train_cfg = TrainConfig(max_steps=args.steps, batch_size=args.batch_size,
                            lr=args.lr, device=args.device)

    result = train(model_cfg, diff_cfg, train_cfg)
    hist = result["history"]
    n = max(1, len(hist["total"]) // 10)
    print(f"\nfirst-{n} mean total loss: {sum(hist['total'][:n]) / n:.4f}")
    print(f"last-{n} mean total loss:  {sum(hist['total'][-n:]) / n:.4f}")
    print(f"model parameters: {result['model'].num_parameters():,}")


if __name__ == "__main__":
    main()
