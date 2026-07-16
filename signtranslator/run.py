"""End-to-end pipeline: ingest -> train -> analyze.

    python -m signtranslator.run --corpus-dir ./corpus --epochs 15

Generates (or reuses) an on-disk corpus, builds the bidirectional model sized to
the corpus, trains every branch jointly, and prints an analysis report with
pass/fail checks.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, DiffusionConfig, TrainerConfig
from .data.corpus import (
    generate_corpus, validate_corpus, SignDataset, collate_corpus, CorpusSpec,
)
from .models import BidirectionalSignTranslator
from .training import Trainer
from .analysis import analyze


def build_model(spec: CorpusSpec, diff_timesteps: int = 100) -> BidirectionalSignTranslator:
    model_cfg = ModelConfig(
        num_joints=spec.num_joints, in_channels=spec.in_channels,
        num_frames=spec.num_frames, stgcn_channels=(32, 64),
        text_embed_dim=64, text_layers=2, text_heads=4, latent_dim=64,
    )
    diff_cfg = DiffusionConfig(num_timesteps=diff_timesteps, denoiser_dim=64,
                               denoiser_layers=2, denoiser_heads=4)
    return BidirectionalSignTranslator(
        model_cfg, diff_cfg, src_vocab=spec.src_vocab, gloss_vocab=spec.gloss_vocab,
        num_glosses=spec.num_glosses, cond_drop_prob=0.1, planner_layers=3)


# Loss weighting: up-weight the planner (a sequence-reordering task that needs
# more gradient signal than the pooled contrastive alignment) and down-weight
# alignment (which converges quickly and can otherwise dominate).
DEFAULT_LOSS_WEIGHTS = {
    "generation": 1.0, "alignment": 0.3, "planner": 2.0, "recognition": 1.0,
}


def make_loaders(corpus_dir: str, batch_size: int):
    train_loader = DataLoader(SignDataset(corpus_dir, "train"), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_corpus, drop_last=True)
    val_loader = DataLoader(SignDataset(corpus_dir, "val"), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_corpus)
    return train_loader, val_loader


def run_pipeline(corpus_dir: str, epochs: int = 30, batch_size: int = 32,
                 lr: float = 4e-3, diff_timesteps: int = 100, seed: int = 0,
                 regenerate: bool = True, ckpt_path: Optional[str] = None,
                 do_train: bool = True, do_analyze: bool = True,
                 resume: bool = False, verbose: bool = True) -> dict:
    # 1. Ingest -----------------------------------------------------------
    if regenerate:
        generate_corpus(corpus_dir, seed=seed)
    spec = validate_corpus(corpus_dir)
    if verbose:
        print(f"[ingest] corpus OK: K={spec.num_concepts} L={spec.seq_len} "
              f"joints={spec.num_joints} frames={spec.num_frames}")

    train_loader, val_loader = make_loaders(corpus_dir, batch_size)

    torch.manual_seed(seed)
    model = build_model(spec, diff_timesteps=diff_timesteps)
    if verbose:
        print(f"[model] params: {model.num_parameters():,}")
    cfg = TrainerConfig(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                        ckpt_path=ckpt_path, loss_weights=dict(DEFAULT_LOSS_WEIGHTS))
    trainer = Trainer(model, cfg, train_loader, val_loader)

    history = None
    if resume and ckpt_path:
        trainer.load(ckpt_path, load_optimizer=False)  # warm-restart from prior chunk
    if do_train:
        history = trainer.fit(verbose=verbose)
        if ckpt_path:
            trainer.save(ckpt_path)  # persist final model for a later analyze pass
    elif ckpt_path:
        trainer.load(ckpt_path, load_optimizer=False)

    report = None
    if do_analyze:
        report = analyze(model, val_loader)
        if verbose:
            print(report.summary())
    return {"model": model, "history": history, "report": report, "spec": spec,
            "trainer": trainer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest -> train -> analyze.")
    parser.add_argument("--corpus-dir", type=str, default="./corpus")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    result = run_pipeline(args.corpus_dir, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, diff_timesteps=args.timesteps, seed=args.seed,
                          ckpt_path=args.ckpt)
    raise SystemExit(0 if result["report"].passed else 1)


if __name__ == "__main__":
    main()
