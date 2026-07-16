"""Unified trainer for the bidirectional multi-branch model.

Trains the planner, cross-modal motion generator, CTC recogniser, and the
contrastive manifold *jointly* with a weighted sum of their losses, a cosine
learning-rate schedule with linear warmup, gradient clipping, per-epoch
validation, and best-checkpoint tracking.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..config import TrainerConfig


def cosine_warmup_lambda(total_steps: int, warmup_steps: int,
                         min_lr_frac: float = 0.05) -> Callable[[int], float]:
    """LR multiplier: linear warmup then cosine decay to ``min_lr_frac``.

    Returns a function ``step -> multiplier`` in ``[min_lr_frac, 1]`` suitable
    for ``torch.optim.lr_scheduler.LambdaLR``.
    """
    warmup_steps = max(1, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return fn


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        torch.manual_seed(cfg.seed)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        total_steps = cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(cfg.warmup_frac * total_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, cosine_warmup_lambda(total_steps, warmup_steps, cfg.min_lr_frac))

        self.history: Dict[str, List[float]] = {}
        self.best_val = math.inf
        self.global_step = 0

    # -- helpers ------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.cfg.device) if torch.is_tensor(v) else v
        return out

    def _record(self, prefix: str, losses: Dict[str, torch.Tensor]) -> None:
        for k, v in losses.items():
            self.history.setdefault(f"{prefix}_{k}", []).append(float(v))

    # -- loops --------------------------------------------------------------
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        agg: Dict[str, float] = {}
        count = 0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            losses = self.model.training_step(batch, weights=self.cfg.loss_weights)
            loss = losses["total"]

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            self.sched.step()
            self.global_step += 1

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            count += 1
        return {k: v / max(1, count) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        agg: Dict[str, float] = {}
        count = 0
        for batch in self.val_loader:
            batch = self._to_device(batch)
            losses = self.model.training_step(batch, weights=self.cfg.loss_weights)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            count += 1
        return {k: v / max(1, count) for k, v in agg.items()}

    def fit(self, verbose: bool = False) -> Dict[str, List[float]]:
        for epoch in range(self.cfg.epochs):
            train_losses = self.train_epoch()
            self.history.setdefault("lr", []).append(self.sched.get_last_lr()[0])
            for k, v in train_losses.items():
                self.history.setdefault(f"train_{k}", []).append(v)

            val_losses = {}
            if self.val_loader is not None and (epoch + 1) % self.cfg.val_every == 0:
                val_losses = self.validate()
                for k, v in val_losses.items():
                    self.history.setdefault(f"val_{k}", []).append(v)
                if val_losses.get("total", math.inf) < self.best_val:
                    self.best_val = val_losses["total"]
                    if self.cfg.ckpt_path:
                        self.save(self.cfg.ckpt_path)

            if verbose:
                msg = f"epoch {epoch + 1:3d} | lr {self.sched.get_last_lr()[0]:.2e} | " \
                      f"train {train_losses.get('total', 0):.4f}"
                if val_losses:
                    msg += f" | val {val_losses.get('total', 0):.4f}"
                print(msg)
        return self.history

    # -- checkpointing ------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "global_step": self.global_step,
                    "best_val": self.best_val}, path)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        # weights_only=False: we load our own trusted checkpoints (which contain
        # optimizer state and Python scalars, not just tensors).
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if load_optimizer and "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val = ckpt.get("best_val", math.inf)
