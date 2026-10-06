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
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from ..inference_context import preserving_eval_mode
from ..config import TrainerConfig
from .objectives import ObjectiveAccumulator, SupportedObjective
from .exposure import exposure_records, validate_exposure, summarize_exposure
from ..data.governed_corpus import (
    GovernedMotionBatch, GovernedMotionDataset, collate_governed_motion,
    move_governed_batch,
)
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


def _validated_training_state(state, *, epoch_limit):
    """Validate resumable progress without coercion or model/optimizer mutation."""
    fields = {"completed_epochs", "global_step", "best_val", "history"}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("checkpoint training state has unexpected fields")
    for name in ("completed_epochs", "global_step"):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError("checkpoint progress must contain nonnegative exact integers")
    if state["completed_epochs"] > epoch_limit:
        raise ValueError("checkpoint epoch exceeds configured training horizon")
    best = state["best_val"]
    if type(best) not in (int, float) or (isinstance(best, float) and (math.isnan(best) or best == -math.inf)):
        raise ValueError("checkpoint best metric must be finite or positive infinity")
    history = state["history"]
    if not isinstance(history, dict):
        raise ValueError("checkpoint history must be a dictionary")
    for name, values in history.items():
        if type(name) is not str or not name or type(values) is not list:
            raise ValueError("checkpoint history requires named lists")
        if any(type(value) not in (int, float)
               or (type(value) is float and not math.isfinite(value)) for value in values):
            raise ValueError("checkpoint history values must be finite numbers, excluding booleans")
    return {**state, "history": {name: list(values) for name, values in history.items()}}


CHECKPOINT_SCHEMA_VERSION = 4
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
        "skeleton": model.graph.to_dict() if hasattr(model, "graph") else None,
    }


def _loader_contract(loader: Optional[DataLoader]) -> Optional[dict[str, Any]]:
    if loader is None:
        return None
    contract = {
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
    if isinstance(loader.dataset, GovernedMotionDataset):
        contract["governed"] = loader.dataset.training_contract
    return contract


def _governed_loaders(model, train_loader, val_loader) -> bool:
    """Typed models opt into per-example-mean scalar losses and strict admission."""
    governed = isinstance(train_loader.dataset, GovernedMotionDataset)
    if val_loader is not None and isinstance(val_loader.dataset, GovernedMotionDataset) != governed:
        raise ValueError("cannot mix governed and legacy loaders")
    version = getattr(model, "governed_batch_schema_version", None)
    if not governed:
        if version is not None:
            raise ValueError("governed model requires governed loaders")
        return False
    if type(version) is not int or version != 1:
        raise ValueError("model must declare governed_batch_schema_version=1")
    for loader, split in ((train_loader, "train"), (val_loader, "val")):
        if loader is None:
            continue
        if loader.collate_fn is not collate_governed_motion:
            raise ValueError("governed training requires the revalidating canonical collator")
        # This initial route supports reproducible full-view sampling. Custom
        # weighting/distributed sampling needs its own declared estimand/state.
        if (type(loader.batch_sampler) is not BatchSampler
                or type(loader.sampler) not in (SequentialSampler, RandomSampler)
                or loader.generator is not None):
            raise ValueError("governed loaders require standard full-view sampling with global RNG")
        if isinstance(loader.sampler, RandomSampler) and (
                loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset)
                or loader.sampler.generator is not None):
            raise ValueError("governed random sampling must cover the view without replacement")
        if split == 'val' and type(loader.sampler) is not SequentialSampler:
            raise ValueError("governed validation requires sequential full-view sampling")
        if loader.dataset.training_contract["split"] != split:
            raise ValueError(f"governed {split} loader uses the wrong split")
        if (loader.dataset.training_contract["corpus_sha256"] !=
                train_loader.dataset.training_contract["corpus_sha256"]):
            raise ValueError("governed loaders must share the complete admitted corpus")
    if val_loader is not None and val_loader.drop_last:
        raise ValueError("governed validation must not drop the final batch")
    return True


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


def _require_batches(loader, name: str) -> None:
    if len(loader) == 0:
        raise ValueError(f"{name} loader contains zero batches; adjust batch size/drop_last or data")


def _finite_losses(losses: Mapping[str, torch.Tensor], context: str) -> None:
    if "total" not in losses:
        raise ValueError(f"missing total objective: {context}")
    for name, value in losses.items():
        if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
            raise FloatingPointError(f"invalid/non-finite {name} loss: {context}")


def _checked_step(loss: torch.Tensor, optimizer, parameters, grad_clip: float,
                  context: str) -> None:
    parameters = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        raise ValueError(f"objective has no gradient path: {context}")
    loss.backward()
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        optimizer.zero_grad()
        raise FloatingPointError(f"missing/non-finite gradients before optimizer step: {context}")
    try:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
    except RuntimeError as error:
        optimizer.zero_grad()
        raise FloatingPointError(f"invalid gradient norm before optimizer step: {context}") from error
    optimizer.step()


class Trainer:
    """Optimize legacy models or explicitly opted-in governed batch models.

    A model declaring ``governed_batch_schema_version = 1`` receives the intact
    typed batch in ``training_step`` and must return per-example-mean scalar
    losses, including ``total``. Epoch reporting weights those means by sample
    count. Additionally declaring ``governed_objective_schema_version = 1``
    requires a SupportedObjective with explicit branch numerators and support.
    Optimization uses full-population contributions; branch reports use their
    own support counts and omit unavailable metrics. These declarations are
    interface contracts, not ASL qualification.
    Governed validation always uses this path, never a model's legacy loader hook.
    """
    def __init__(self, model: torch.nn.Module, cfg: TrainerConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 *, artifact_context: Optional[Mapping[str, Any]] = None) -> None:
        _require_batches(train_loader, "training")
        if val_loader is not None:
            _require_batches(val_loader, "validation")
        self.governed = _governed_loaders(model, train_loader, val_loader)
        objective_schema = getattr(model, 'governed_objective_schema_version', None)
        self.supported_objectives = objective_schema is not None
        if self.supported_objectives and (not self.governed or type(objective_schema) is not int
                                          or objective_schema != 1):
            raise ValueError('supported objectives require governed batches and objective schema 1')
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
        self.best_model_state = None
        self.global_step = 0
        self._optimizer_exposure = []
        self.completed_epochs = 0
        self._epoch_committed = True
        self.artifact_context = _json_domain(artifact_context or {})
        self.model_contract = _model_contract(model)
        if self.governed:
            self.model_contract["governed_batch_schema_version"] = 1
        if self.supported_objectives:
            self.model_contract['governed_objective_schema_version'] = 1
        self.implementation_identity = package_implementation_identity()
        self.runtime_environment = runtime_environment()
        self.data_contract = {
            "train": _loader_contract(train_loader),
            "validation": _loader_contract(val_loader),
        }

    @property
    def optimizer_exposure(self):
        """Fresh batch records; corpus/view identity is in data_contract."""
        return exposure_records(self._optimizer_exposure)

    def exposure_report(self):
        """Historical returned-call summary, not renewed corpus authorization."""
        if not self.supported_objectives:
            raise ValueError('exposure report requires support-aware governed training')
        if _loader_contract(self.train_loader) != self.data_contract['train']:
            raise ValueError('training view changed since exposure was recorded')
        return summarize_exposure(
            self.optimizer_exposure, global_step=self.global_step,
            completed_epochs=self.completed_epochs, committed=self._epoch_committed,
            weights=self.cfg.loss_weights,
            identities=self.train_loader.dataset.supervision_identities,
            data_contract=self.data_contract['train'],
            implementation_identity=self.implementation_identity,
            model_contract=self.model_contract)

    # -- helpers ------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.cfg.device) if torch.is_tensor(v) else v
        return out

    def _prepare_batch(self, batch, loader, partition):
        if not self.governed:
            if not isinstance(batch, dict):
                raise TypeError("legacy trainer requires dictionary batches")
            return self._to_device(batch), 1, batch.get('sample_ids', 'unavailable')
        # Recheck configuration before any model or optimizer side effects.
        _governed_loaders(self.model, self.train_loader, self.val_loader)
        if _loader_contract(loader) != self.data_contract[partition]:
            raise ValueError("governed loader contract changed after trainer construction")
        expected = self.data_contract[partition]["governed"]
        if not isinstance(batch, GovernedMotionBatch):
            raise TypeError("governed loader must yield typed governed batches")
        if batch.corpus_sha256 != expected["corpus_sha256"] or batch.split != expected["split"]:
            raise ValueError("batch corpus/split does not match trainer admission")
        size = len(batch.motion.sample_ids)
        if size == 0 or size != len(batch.annotations):
            raise ValueError("invalid governed batch cardinality")
        return move_governed_batch(batch, self.cfg.device), size, batch.motion.sample_ids

    def _check_loss_keys(self, losses, aggregate):
        if self.governed and aggregate and losses.keys() != aggregate.keys():
            raise ValueError("governed loss fields must remain identical across batches")

    def _record(self, prefix: str, losses: Dict[str, torch.Tensor]) -> None:
        for k, v in losses.items():
            self.history.setdefault(f"{prefix}_{k}", []).append(v.detach().item())

    # -- loops --------------------------------------------------------------
    def _supported_epoch(self, loader, *, training: bool) -> Dict[str, float]:
        """Keep the full population while reporting branch-specific support."""
        _require_batches(loader, 'training' if training else 'validation')
        if training:
            self.model.train()
        mode = nullcontext() if training else preserving_eval_mode(self.model)
        accumulator = ObjectiveAccumulator()
        rng = nullcontext() if training else isolated_deterministic_rng(self.cfg.seed + VALIDATION_SEED_OFFSET)
        with mode:
            with rng:
                for index, batch in enumerate(loader):
                    partition = 'train' if training else 'validation'
                    batch, size, sample_ids = self._prepare_batch(batch, loader, partition)
                    objective = self.model.training_step(batch, weights=self.cfg.loss_weights)
                    if (not isinstance(objective, SupportedObjective)
                            or objective.population_size != size
                            or objective.weights != self.cfg.loss_weights):
                        raise ValueError('supported objective must bind batch size and configured weights')
                    schema = getattr(self.model, 'governed_objective_schema_version', None)
                    if type(schema) is not int or schema != 1:
                        raise ValueError('governed objective schema changed after construction')
                    total = objective.total()
                    accumulator.add(objective)
                    if training:
                        context = f'epoch={self.completed_epochs}, batch={index}, sample_ids={sample_ids!r}'
                        record = {
                            'step': self.global_step + 1,
                            'sample_ids': list(sample_ids),
                            'annotation_sha256': [a.content_sha256() for a in batch.annotations],
                            'support': {name: term.supported_examples for name, term in objective.terms.items()},
                            'support_membership': {name: list(term.support_mask)
                                                   if term.support_mask is not None else None
                                                   for name, term in objective.terms.items()},
                            'weights': dict(objective.weights),
                        }
                        validate_exposure([dict(record, step=1)], supported=True,
                                          global_step=1, weights=self.cfg.loss_weights,
                                          identities=loader.dataset.supervision_identities)
                        encoded = canonical_json_bytes(record).decode('utf-8')
                        _checked_step(total, self.opt, self.model.parameters(), self.cfg.grad_clip, context)
                        # Record once optimizer.step returns, even if scheduler.step fails.
                        self.global_step += 1
                        self._optimizer_exposure.append(encoded)
                        self.sched.step()
        result = accumulator.result()
        support = {**accumulator.support, 'total': accumulator.population}
        if training:
            self.last_train_support = support
        else:
            self.last_validation_support = support
        return result

    def train_epoch(self) -> Dict[str, float]:
        # Only fit() can commit metrics, selection and the epoch cursor together.
        # Exceptions may leave model/optimizer/RNG changes that cannot be undone.
        self._epoch_committed = False
        if self.supported_objectives:
            return self._supported_epoch(self.train_loader, training=True)
        _require_batches(self.train_loader, "training")
        self.model.train()
        agg: Dict[str, float] = {}
        count = 0
        weight = 0
        for batch in self.train_loader:
            batch, batch_weight, sample_ids = self._prepare_batch(batch, self.train_loader, "train")
            losses = self.model.training_step(batch, weights=self.cfg.loss_weights)
            context = (f"epoch={self.completed_epochs}, batch={count}, step={self.global_step}, "
                       f"sample_ids={sample_ids!r}")
            _finite_losses(losses, context)
            self._check_loss_keys(losses, agg)
            loss = losses["total"]
            _checked_step(loss, self.opt, self.model.parameters(), self.cfg.grad_clip, context)
            self.sched.step()
            self.global_step += 1

            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v.detach().item() * batch_weight
            count += 1
            weight += batch_weight
        if count == 0:
            raise RuntimeError("training iterator yielded zero batches")
        return {k: v / weight for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        if self.supported_objectives:
            return self._supported_epoch(self.val_loader, training=False)
        agg: Dict[str, float] = {}
        count = 0
        weight = 0
        # Diffusion validation samples timesteps/noise.  Fix that stream and restore
        # the training RNG afterward so validation is repeatable and observational.
        with preserving_eval_mode(self.model):
            with isolated_deterministic_rng(self.cfg.seed + VALIDATION_SEED_OFFSET):
                if not self.governed and hasattr(self.model, 'validation_metrics'):
                    return self.model.validation_metrics(self.val_loader, self.cfg.loss_weights)
                for batch in self.val_loader:
                    batch, batch_weight, sample_ids = self._prepare_batch(batch, self.val_loader, "validation")
                    losses = self.model.training_step(batch, weights=self.cfg.loss_weights)
                    _finite_losses(losses, f"validation batch={count}, sample_ids={sample_ids!r}")
                    self._check_loss_keys(losses, agg)
                    for k, v in losses.items():
                        agg[k] = agg.get(k, 0.0) + v.detach().item() * batch_weight
                    count += 1
                    weight += batch_weight
        if count == 0:
            raise RuntimeError("validation iterator yielded zero batches")
        return {k: v / weight for k, v in agg.items()}

    def fit(self, verbose: bool = False,
            max_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        if not self._epoch_committed:
            raise RuntimeError('unfinished epoch: restore a committed checkpoint before fit')
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
                if self.supported_objectives:
                    self.history.setdefault(f"train_{k}_epoch", []).append(epoch + 1)
            if self.supported_objectives:
                for name, support in self.last_train_support.items():
                    self.history.setdefault(f'train_support_{name}', []).append(support)

            val_losses = {}
            if self.val_loader is not None and (epoch + 1) % self.cfg.val_every == 0:
                val_losses = self.validate()
                for k, v in val_losses.items():
                    self.history.setdefault(f"val_{k}", []).append(v)
                    if self.supported_objectives:
                        self.history.setdefault(f"val_{k}_epoch", []).append(epoch + 1)
                if self.supported_objectives:
                    for name, support in self.last_validation_support.items():
                        self.history.setdefault(f'val_support_{name}', []).append(support)
                metric = self.cfg.selection_metric
                if metric not in val_losses or not math.isfinite(val_losses[metric]):
                    raise ValueError(f"selection metric {metric!r} is unavailable or nonfinite")
                if val_losses[metric] < self.best_val:
                    self.best_val = val_losses[metric]
                    self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    improved = True
                else:
                    improved = False
            else:
                improved = False

            self.completed_epochs = epoch + 1
            self._epoch_committed = True
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
        if not self._epoch_committed:
            raise RuntimeError('unfinished epoch cannot be saved as an exact checkpoint')
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
            "optimizer_exposure": self.optimizer_exposure,
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
            "best_model_state": self.best_model_state,
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
            "optimizer_exposure": self.optimizer_exposure,
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
        _require_batches(train_loader, "generator training")
        if val_loader is not None:
            _require_batches(val_loader, "generator validation")
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
                support = {key: (batch[key].to(device) if torch.is_tensor(batch[key]) else batch[key]) for key in
                           ("validity_mask", "confidence", "frame_mask", "frame_timestamps", "max_gap_seconds") if key in batch}
                loss = model.generation_loss(pose, gloss, **support)
                context = f"generator epoch={epoch}, batch={count}, sample_ids={batch.get('sample_ids')!r}"
                _finite_losses({"total": loss}, context)
                _checked_step(loss, opt, unique, grad_clip, context)
                sched.step()
                agg += loss.detach().item()
                count += 1
            if count == 0:
                raise RuntimeError("generator training iterator yielded zero batches")
            history["train_generation"].append(agg / count)

            if val_loader is not None:
                model.eval()
                from ..analysis.observations import observations
                with torch.no_grad(), isolated_deterministic_rng(VALIDATION_SEED_OFFSET):
                    v = [model.generation_loss(b["pose"].to(device),
                                               b["gloss_tokens"].to(device),
                                               **{key: (b[key].to(device) if torch.is_tensor(b[key]) else b[key]) for key in
                                                  ("validity_mask", "confidence", "frame_mask", "frame_timestamps", "max_gap_seconds")
                                                  if key in b}).item()
                         for batch in val_loader for b in observations(batch)]
                if not v or not all(math.isfinite(value) for value in v):
                    raise FloatingPointError("generator validation has empty/non-finite evidence")
                history["val_generation"].append(sum(v) / len(v))
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"  [gen-ft] epoch {epoch + 1:3d} train {history['train_generation'][-1]:.4f}"
                if history["val_generation"]:
                    msg += f" val {history['val_generation'][-1]:.4f}"
                print(msg)
        return history

    def load(self, path: str | os.PathLike[str], *, mode: str = "resume") -> None:
        """Load either an exact training continuation or model weights only.

        ``resume`` validates and restores optimizer, scheduler, epoch, history, and all
        RNG state.  ``weights`` requires a fresh trainer and restores no training state.
        """
        if mode not in {"resume", "weights"}:
            raise ValueError("checkpoint load mode must be 'resume' or 'weights'")
        if mode == "weights" and (
                not self._epoch_committed or self.completed_epochs != 0
                or self.global_step != 0 or self.history or self.opt.state
                or self.best_model_state is not None or self.best_val != math.inf
                or self._optimizer_exposure):
            raise ValueError('weights warm start requires a fresh trainer without training state')
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
        if manifest.get("schema_version") == 4:
            manifest_fields.add("optimizer_exposure")
        if set(manifest) != manifest_fields:
            raise ValueError("checkpoint manifest fields do not match the checkpoint schema")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] not in (2, 3, 4):
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
        if manifest["schema_version"] >= 3:
            required.add("best_model_state")
        if manifest["schema_version"] == 4:
            required.add("optimizer_exposure")
        if set(checkpoint) != required:
            raise ValueError("checkpoint fields do not match the checkpoint schema")
        if (type(checkpoint["schema_version"]) is not int
                or checkpoint["schema_version"] != manifest["schema_version"]):
            raise ValueError("unsupported checkpoint schema")
        if mode == "resume":
            training_state = _validated_training_state(
                checkpoint["training_state"], epoch_limit=self.cfg.epochs)
            if (self.governed and training_state['global_step'] !=
                    training_state['completed_epochs'] * len(self.train_loader)):
                raise ValueError('checkpoint steps do not describe a committed governed epoch')
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
        if checkpoint['schema_version'] == 4:
            if manifest['optimizer_exposure'] != checkpoint['optimizer_exposure']:
                raise ValueError('checkpoint manifest disagrees on optimizer exposure')
            if mode == 'resume':
                exposure = validate_exposure(
                    checkpoint['optimizer_exposure'], supported=self.supported_objectives,
                    global_step=training_state['global_step'],
                    weights=self.cfg.loss_weights,
                    identities=self.train_loader.dataset.supervision_identities
                    if self.supported_objectives else {})
        if checkpoint["model_contract"] != self.model_contract:
            raise ValueError("checkpoint model contract does not match this model")

        if mode == "weights":
            # Loading can partially copy tensors before a custom hook raises.
            self._epoch_committed = False
            self.model.load_state_dict(checkpoint["model"], strict=True)
            self._epoch_committed = True
            return

        if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("schema 2 lacks best-model state; schemas 2/3 lack optimizer exposure; use explicit weights warm start")

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

        best_state = checkpoint["best_model_state"]
        best_value = training_state["best_val"]
        if best_state is None:
            if best_value != math.inf:
                raise ValueError("finite best metric requires retained best-model weights")
        else:
            current_state = self.model.state_dict()
            if (not math.isfinite(best_value) or not isinstance(best_state, Mapping)
                    or best_state.keys() != current_state.keys()):
                raise ValueError("best-model state and metric are inconsistent")
            for name, value in best_state.items():
                expected = current_state[name]
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                        or value.dtype != expected.dtype):
                    raise ValueError("best-model tensor contract mismatch")
            best_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}

        # Preflight has completed. A later load failure can leave partial state;
        # prohibit exact continuation until a complete resume succeeds.
        self._epoch_committed = False
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.opt.load_state_dict(checkpoint["optimizer"])
        self.sched.load_state_dict(checkpoint["scheduler"])
        self.completed_epochs = training_state["completed_epochs"]
        self.global_step = training_state["global_step"]
        self.best_val = training_state["best_val"]
        self.history = training_state["history"]
        restore_rng_state(checkpoint["rng_state"])
        self.best_model_state = best_state
        self._optimizer_exposure = exposure
        self._epoch_committed = True
