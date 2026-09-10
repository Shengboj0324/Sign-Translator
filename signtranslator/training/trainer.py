"""Unified trainer for the bidirectional multi-branch model.

Trains the planner, cross-modal motion generator, CTC recogniser, and the
contrastive manifold *jointly* with a weighted sum of their losses, a cosine
learning-rate schedule with linear warmup, gradient clipping, per-epoch
validation, and best-checkpoint tracking.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from ..config import TrainerConfig
from ..reproducibility import (
    canonical_json_bytes,
    capture_rng_state,
    isolated_deterministic_rng,
    package_implementation_identity,
    restore_rng_state,
    runtime_environment,
    seed_all,
    sha256_file,
)


CHECKPOINT_SCHEMA_VERSION = 2
VALIDATION_SEED_OFFSET = 1_000_003


def checkpoint_paths(path: str | os.PathLike[str]) -> dict[str, Path]:
    """Return distinct best/last artifact paths from one user-supplied prefix."""
    base = Path(path)
    suffix = base.suffix or ".pt"
    stem = base.stem if base.suffix else base.name
    return {
        "best": base.with_name(f"{stem}.best{suffix}"),
        "last": base.with_name(f"{stem}.last{suffix}"),
    }


def _model_contract(model: torch.nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    configs = {}
    for name in ("model_cfg", "diff_cfg"):
        config = getattr(model, name, None)
        if config is not None and hasattr(config, "to_dict"):
            configs[name] = config.to_dict()
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_signature": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in state.items()
        ],
        "configs": configs,
    }


def _loader_contract(loader: Optional[DataLoader]) -> Optional[dict[str, Any]]:
    if loader is None:
        return None
    return {
        "class": f"{type(loader).__module__}.{type(loader).__qualname__}",
        "dataset_class": (
            f"{type(loader.dataset).__module__}.{type(loader.dataset).__qualname__}"),
        "dataset_length": len(loader.dataset),
        "sampler_class": (
            f"{type(loader.sampler).__module__}.{type(loader.sampler).__qualname__}"),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "persistent_workers": loader.persistent_workers,
    }


def _json_domain(value: Mapping[str, Any]) -> dict[str, Any]:
    """Defensively copy metadata and reject non-JSON or non-finite values."""
    return json.loads(canonical_json_bytes(dict(value)))


def _strict_json_loads(data: bytes) -> dict[str, Any]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate checkpoint-manifest field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    result = json.loads(
        data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return result


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
                 val_loader: Optional[DataLoader] = None,
                 *, artifact_context: Optional[Mapping[str, Any]] = None) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        seed_all(cfg.seed)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        total_steps = cfg.epochs * max(1, len(train_loader))
        warmup_steps = int(cfg.warmup_frac * total_steps)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, cosine_warmup_lambda(total_steps, warmup_steps, cfg.min_lr_frac))

        self.history: Dict[str, List[float]] = {}
        self.best_val = math.inf
        self.global_step = 0
        self.completed_epochs = 0
        self.artifact_context = _json_domain(artifact_context or {})
        self.model_contract = _model_contract(model)
        self.implementation_identity = package_implementation_identity()
        self.runtime_environment = runtime_environment()
        self.data_contract = {
            "train": _loader_contract(train_loader),
            "validation": _loader_contract(val_loader),
        }

    # -- helpers ------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.cfg.device) if torch.is_tensor(v) else v
        return out

    def _record(self, prefix: str, losses: Dict[str, torch.Tensor]) -> None:
        for k, v in losses.items():
            self.history.setdefault(f"{prefix}_{k}", []).append(v.detach().item())

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
                agg[k] = agg.get(k, 0.0) + v.detach().item()
            count += 1
        return {k: v / max(1, count) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        agg: Dict[str, float] = {}
        count = 0
        # Diffusion validation samples timesteps/noise.  Fix that stream and restore
        # the training RNG afterward so validation is repeatable and observational.
        try:
            with isolated_deterministic_rng(self.cfg.seed + VALIDATION_SEED_OFFSET):
                for batch in self.val_loader:
                    batch = self._to_device(batch)
                    losses = self.model.training_step(batch, weights=self.cfg.loss_weights)
                    for k, v in losses.items():
                        agg[k] = agg.get(k, 0.0) + v.detach().item()
                    count += 1
        finally:
            self.model.train(was_training)
        return {k: v / max(1, count) for k, v in agg.items()}

    def fit(self, verbose: bool = False,
            max_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be positive when supplied")
        if self.completed_epochs > self.cfg.epochs:
            raise RuntimeError("checkpoint epoch exceeds configured training horizon")
        stop_epoch = self.cfg.epochs
        if max_epochs is not None:
            stop_epoch = min(stop_epoch, self.completed_epochs + max_epochs)
        for epoch in range(self.completed_epochs, stop_epoch):
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
                    improved = True
                else:
                    improved = False
            else:
                improved = False

            self.completed_epochs = epoch + 1
            if self.cfg.ckpt_path:
                paths = checkpoint_paths(self.cfg.ckpt_path)
                if improved:
                    self.save(paths["best"], kind="best")
                self.save(paths["last"], kind="last")

            if verbose:
                msg = f"epoch {epoch + 1:3d} | lr {self.sched.get_last_lr()[0]:.2e} | " \
                      f"train {train_losses.get('total', 0):.4f}"
                if val_losses:
                    msg += f" | val {val_losses.get('total', 0):.4f}"
                print(msg)
        return self.history

    # -- checkpointing ------------------------------------------------------
    def save(self, path: str | os.PathLike[str], *, kind: str = "last") -> Path:
        """Atomically persist a hash-verified, self-describing checkpoint."""
        if kind not in {"best", "last", "milestone"}:
            raise ValueError("checkpoint kind must be best, last, or milestone")
        if self.train_loader.num_workers != 0:
            raise RuntimeError(
                "exact checkpoint continuation currently requires num_workers=0; "
                "worker-local RNG state is not serializable by this trainer")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "model_contract": self.model_contract,
            "implementation_identity": self.implementation_identity,
            "runtime_environment": self.runtime_environment,
            "data_contract": self.data_contract,
            "trainer_config": self.cfg.to_dict(),
            "artifact_context": self.artifact_context,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "training_state": {
                "completed_epochs": self.completed_epochs,
                "global_step": self.global_step,
                "best_val": self.best_val,
                "history": self.history,
            },
            "rng_state": capture_rng_state(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(state, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        best_value = self.best_val if math.isfinite(self.best_val) else None
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "checkpoint_sha256": sha256_file(destination),
            "checkpoint_size": destination.stat().st_size,
            "model_contract": self.model_contract,
            "implementation_identity": self.implementation_identity,
            "runtime_environment": self.runtime_environment,
            "data_contract": self.data_contract,
            "trainer_config": self.cfg.to_dict(),
            "artifact_context": self.artifact_context,
            "training_state": {
                "completed_epochs": self.completed_epochs,
                "global_step": self.global_step,
                "best_val": best_value,
                "history": self.history,
            },
        }
        _atomic_write(Path(f"{destination}.json"), canonical_json_bytes(manifest) + b"\n")
        return destination

    @staticmethod
    def finetune_generation(model, train_loader, val_loader=None, epochs: int = 60,
                            lr: float = 1e-3, device: str = "cpu",
                            grad_clip: float = 1.0, verbose: bool = False) -> dict:
        """Curriculum stage: train **only** the conditional generator.

        The discriminative branches (recognition, planner, alignment) converge in
        a few hundred steps, whereas a diffusion generator needs far more. Once
        the former have converged, continuing to run them wastes most of the
        per-step cost. This stage optimises just the diffusion module, so many
        more generator updates fit in the same budget.

        Optimises the diffusion module and the generator-private
        ``cond_encoder``. The manifold's ``gloss_encoder`` is deliberately NOT
        touched: it is a separate encoder precisely so that generator
        fine-tuning cannot collapse motion<->language retrieval.
        """
        params = list(model.diffusion.parameters()) + list(model.cond_encoder.parameters())
        seen, unique = set(), []
        for p in params:                      # de-duplicate any shared tensors
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        opt = torch.optim.AdamW(unique, lr=lr, weight_decay=1e-4)
        total_steps = epochs * max(1, len(train_loader))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, cosine_warmup_lambda(total_steps, max(1, total_steps // 20), 0.05))

        history: Dict[str, List[float]] = {"train_generation": [], "val_generation": []}
        for epoch in range(epochs):
            model.train()
            agg, count = 0.0, 0
            for batch in train_loader:
                pose = batch["pose"].to(device)
                gloss = batch["gloss_tokens"].to(device)
                loss = model.generation_loss(pose, gloss)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unique, grad_clip)
                opt.step()
                sched.step()
                agg += loss.detach().item()
                count += 1
            history["train_generation"].append(agg / max(1, count))

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    v = [model.generation_loss(b["pose"].to(device),
                                               b["gloss_tokens"].to(device)).item()
                         for b in val_loader]
                history["val_generation"].append(sum(v) / max(1, len(v)))
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"  [gen-ft] epoch {epoch + 1:3d} train {history['train_generation'][-1]:.4f}"
                if history["val_generation"]:
                    msg += f" val {history['val_generation'][-1]:.4f}"
                print(msg)
        return history

    def load(self, path: str | os.PathLike[str], *, mode: str = "resume") -> None:
        """Load either an exact training continuation or model weights only.

        ``resume`` validates and restores optimizer, scheduler, epoch, history, and all
        RNG state.  ``weights`` is an explicit warm start and restores no training state.
        """
        if mode not in {"resume", "weights"}:
            raise ValueError("checkpoint load mode must be 'resume' or 'weights'")
        source = Path(path)
        manifest_path = Path(f"{source}.json")
        if source.is_symlink() or not source.is_file():
            raise ValueError("checkpoint must be a non-symlink regular file")
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("checkpoint manifest is missing or is a symlink")
        manifest = _strict_json_loads(manifest_path.read_bytes())
        manifest_fields = {
            "schema_version", "kind", "checkpoint_sha256", "checkpoint_size",
            "model_contract", "implementation_identity", "trainer_config",
            "runtime_environment", "data_contract", "artifact_context", "training_state",
        }
        if set(manifest) != manifest_fields:
            raise ValueError("checkpoint manifest fields do not match schema version 2")
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint manifest schema")
        if manifest.get("checkpoint_size") != source.stat().st_size:
            raise ValueError("checkpoint size does not match its manifest")

        # weights_only=False is necessary for optimizer/RNG state. Hash the opened file
        # descriptor before deserialization so path replacement cannot bypass integrity;
        # callers must still treat the checkpoint origin as trusted.
        with source.open("rb") as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if manifest.get("checkpoint_sha256") != digest.hexdigest():
                raise ValueError("checkpoint hash does not match its manifest")
            stream.seek(0)
            checkpoint = torch.load(
                stream, map_location=self.cfg.device, weights_only=False)
            after = os.fstat(stream.fileno())
        file_identity = lambda stat: (
            stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if file_identity(before) != file_identity(after):
            raise RuntimeError("checkpoint changed while being verified and loaded")
        required = {
            "schema_version", "kind", "model_contract", "implementation_identity",
            "runtime_environment", "data_contract", "trainer_config", "artifact_context",
            "model", "optimizer", "scheduler", "training_state", "rng_state",
        }
        if set(checkpoint) != required:
            raise ValueError("checkpoint fields do not match schema version 2")
        if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        for field in ("kind", "model_contract", "implementation_identity",
                      "runtime_environment", "data_contract", "trainer_config",
                      "artifact_context", "training_state"):
            expected = checkpoint[field]
            if field == "training_state" and not math.isfinite(expected["best_val"]):
                expected = dict(expected, best_val=None)
            if isinstance(expected, Mapping):
                expected = _json_domain(expected)
            if manifest.get(field) != expected:
                raise ValueError(f"checkpoint manifest disagrees on {field}")
        if checkpoint["model_contract"] != self.model_contract:
            raise ValueError("checkpoint model contract does not match this model")

        self.model.load_state_dict(checkpoint["model"], strict=True)
        if mode == "weights":
            return

        loaded_config = checkpoint["trainer_config"]
        current_config = self.cfg.to_dict()
        # Storage location has no mathematical effect and may change after moving an
        # artifact. Every optimization-relevant field must remain identical.
        loaded_config = _json_domain(loaded_config)
        current_config = _json_domain(current_config)
        loaded_config["values"]["ckpt_path"] = None
        current_config["values"]["ckpt_path"] = None
        if loaded_config != current_config:
            raise ValueError("resume trainer configuration does not match checkpoint")
        if checkpoint["artifact_context"] != self.artifact_context:
            raise ValueError("resume artifact context does not match checkpoint")
        if (checkpoint["implementation_identity"]["implementation_sha256"]
                != self.implementation_identity["implementation_sha256"]):
            raise ValueError("resume implementation bytes do not match checkpoint")
        if checkpoint["runtime_environment"] != self.runtime_environment:
            raise ValueError("resume numerical runtime does not match checkpoint")
        if checkpoint["data_contract"] != self.data_contract:
            raise ValueError("resume data-loader contract does not match checkpoint")

        self.opt.load_state_dict(checkpoint["optimizer"])
        self.sched.load_state_dict(checkpoint["scheduler"])
        training_state = checkpoint["training_state"]
        if set(training_state) != {
                "completed_epochs", "global_step", "best_val", "history"}:
            raise ValueError("checkpoint training state has unexpected fields")
        self.completed_epochs = int(training_state["completed_epochs"])
        self.global_step = int(training_state["global_step"])
        self.best_val = float(training_state["best_val"])
        self.history = {
            str(name): [float(value) for value in values]
            for name, values in training_state["history"].items()
        }
        if self.completed_epochs < 0 or self.global_step < 0:
            raise ValueError("checkpoint contains negative training progress")
        restore_rng_state(checkpoint["rng_state"])
