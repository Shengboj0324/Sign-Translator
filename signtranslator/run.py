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
from .data.readiness import assess_corpus
from .models import BidirectionalSignTranslator
from .training import Trainer
from .analysis import analyze


def build_model(spec: CorpusSpec, diff_timesteps: int = 100) -> BidirectionalSignTranslator:
    model_cfg = ModelConfig(
        num_joints=spec.num_joints, in_channels=spec.in_channels,
        num_frames=spec.num_frames, stgcn_channels=(32, 64),
        text_embed_dim=64, text_layers=2, text_heads=4, latent_dim=64,
        # Driven by the corpus so the acoustic front-end can never drift out of
        # sync with the feature width actually stored on disk.
        speech_input_dim=spec.speech_dim,
    )
    # The generator carries the hardest job (synthesising 3D motion from gloss
    # alone), so it gets more capacity than the discriminative branches.
    diff_cfg = DiffusionConfig(num_timesteps=diff_timesteps, denoiser_dim=128,
                               denoiser_layers=3, denoiser_heads=4)
    return BidirectionalSignTranslator(
        model_cfg, diff_cfg, src_vocab=spec.src_vocab, gloss_vocab=spec.gloss_vocab,
        num_glosses=spec.num_glosses,
        num_spoken_tokens=spec.source_token_count,
        cond_drop_prob=0.1, planner_layers=3)


# Loss weighting: up-weight the planner (a sequence-reordering task that needs
# more gradient signal than the pooled contrastive alignment) and down-weight
# alignment (which converges quickly and can otherwise dominate).
DEFAULT_LOSS_WEIGHTS = {
    "generation": 1.0, "alignment": 0.3, "planner": 2.0, "recognition": 1.0,
    "speech": 1.0,
}


def make_loaders(corpus_dir: str, batch_size: int):
    train_loader = DataLoader(SignDataset(corpus_dir, "train"), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_corpus, drop_last=True)
    val_loader = DataLoader(SignDataset(corpus_dir, "val"), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_corpus)
    return train_loader, val_loader


def run_pipeline(corpus_dir: str, epochs: int = 30, batch_size: int = 32,
                 lr: float = 4e-3, diff_timesteps: int = 100, seed: int = 0,  # noqa: E501
                 regenerate: bool = False, overwrite_corpus: bool = False,
                 ckpt_path: Optional[str] = None,
                 do_train: bool = True, do_analyze: bool = True,
                 resume: bool = False, gen_finetune_epochs: int = 0,
                 gen_finetune_lr: float = 1e-3, polish_epochs: int = 0,
                 polish_lr: float = 1.2e-3, require_ready: bool = True,
                 verbose: bool = True) -> dict:
    # 1. Ingest -----------------------------------------------------------
    if regenerate:
        generate_corpus(corpus_dir, seed=seed, overwrite=overwrite_corpus)
    spec = validate_corpus(corpus_dir)
    if verbose:
        print(f"[ingest] corpus OK: K={spec.num_concepts} L={spec.seq_len} "
              f"joints={spec.num_joints} frames={spec.num_frames}")

    # Readiness gate: refuse to train on a corpus that is structurally unfit
    # (too few samples, missing classes, split leakage, corrupt poses, ...).
    readiness = assess_corpus(corpus_dir)
    if verbose:
        print(readiness.summary())
    if require_ready and not readiness.passed:
        failed = [c.name for c in readiness.checks if not c.passed]
        raise RuntimeError(
            f"corpus is not training-ready; failed checks: {failed}. "
            "Fix the data or pass require_ready=False to override.")

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
        if gen_finetune_epochs > 0:
            if verbose:
                print(f"[gen-ft] {gen_finetune_epochs} generator-only epochs")
            gen_hist = Trainer.finetune_generation(
                model, train_loader, val_loader, epochs=gen_finetune_epochs,
                lr=gen_finetune_lr, device=cfg.device, verbose=verbose)
            history = {**(history or {}), "genft_train": gen_hist["train_generation"],
                       "genft_val": gen_hist["val_generation"]}
        if polish_epochs > 0:
            # Phase 3 ("polish"): a short low-LR joint pass. Generator
            # fine-tuning advances only the diffusion branch, so the manifold
            # and recogniser are left slightly behind; this re-converges every
            # branch together without disturbing the generator.
            if verbose:
                print(f"[polish] {polish_epochs} low-LR joint epochs")
            polish_cfg = TrainerConfig(epochs=polish_epochs, batch_size=batch_size,
                                       lr=polish_lr, seed=seed + 2,
                                       loss_weights=dict(DEFAULT_LOSS_WEIGHTS))
            Trainer(model, polish_cfg, train_loader, val_loader).fit(verbose=verbose)
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
            "trainer": trainer, "readiness": readiness}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest -> train -> analyze.")
    parser.add_argument("--corpus-dir", type=str, default="./corpus")
    parser.add_argument("--generate-synthetic", action="store_true",
                        help="explicitly create a synthetic corpus")
    parser.add_argument("--overwrite-synthetic", action="store_true",
                        help="allow synthetic generation in a non-empty corpus directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="joint multi-branch epochs")
    parser.add_argument("--gen-finetune-epochs", type=int, default=175,
                        help="generator-only epochs after joint training "
                             "(the generator needs far more steps than the "
                             "discriminative branches)")
    parser.add_argument("--gen-finetune-lr", type=float, default=1.2e-3)
    parser.add_argument("--polish-epochs", type=int, default=16,
                        help="low-LR joint epochs after generator fine-tuning, "
                             "re-converging every branch together")
    parser.add_argument("--polish-lr", type=float, default=1.2e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    if args.overwrite_synthetic and not args.generate_synthetic:
        parser.error("--overwrite-synthetic requires --generate-synthetic")

    result = run_pipeline(args.corpus_dir, epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, diff_timesteps=args.timesteps, seed=args.seed,
                          regenerate=args.generate_synthetic,
                          overwrite_corpus=args.overwrite_synthetic,
                          ckpt_path=args.ckpt,
                          gen_finetune_epochs=args.gen_finetune_epochs,
                          gen_finetune_lr=args.gen_finetune_lr,
                          polish_epochs=args.polish_epochs,
                          polish_lr=args.polish_lr)
    raise SystemExit(0 if result["report"].passed else 1)


if __name__ == "__main__":
    main()
