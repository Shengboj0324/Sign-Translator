"""Freeze-first adaptation schedule.

The source specification: *"Freeze-first adapter training is the safe baseline.
Unfreeze only upper encoder blocks after the adapter converges, using LoRA or
low learning rates."*

The reasoning is asymmetric risk. A pretrained speech encoder is the most
valuable and least replaceable component; a large gradient through it early in
training -- while a randomly-initialised head is still emitting nonsense -- can
destroy representations that took enormous compute to learn. Adapters absorb
that early, high-variance signal. Only once the head has converged, when
gradients are small and informative, is it safe to let them reach the encoder,
and then only the upper blocks (which encode task-specific abstractions) and at
a reduced learning rate.

Two phases:

* **Phase 1 (adapt).** Everything frozen except LoRA factors and any new heads.
* **Phase 2 (refine).** Additionally unfreeze the top ``num_blocks`` encoder
  blocks at ``encoder_lr_scale x`` the base learning rate.

The schedule owns the optimiser so that parameter groups are rebuilt at the
transition. Rebuilding matters: a group added to an existing Adam optimiser
starts with zeroed moment estimates, and reusing a stale optimiser silently
applies phase-1 momentum to phase-2 parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .lora import iter_lora_modules, mark_only_lora_trainable, freeze_all


class Phase:
    ADAPT = "adapt"
    REFINE = "refine"


@dataclass
class FreezeFirstConfig:
    adapt_steps: int = 100          # length of phase 1
    base_lr: float = 1e-3           # for adapters and heads
    encoder_lr_scale: float = 0.1   # phase-2 encoder LR = base_lr * this
    weight_decay: float = 1e-4
    unfreeze_blocks: int = 2        # how many top blocks to release in phase 2

    def __post_init__(self) -> None:
        if self.adapt_steps < 0:
            raise ValueError("adapt_steps must be >= 0")
        if self.base_lr <= 0:
            raise ValueError("base_lr must be positive")
        if not 0.0 < self.encoder_lr_scale <= 1.0:
            raise ValueError(
                "encoder_lr_scale must be in (0, 1]; the pretrained encoder "
                "should never train faster than the adapters")
        if self.unfreeze_blocks < 0:
            raise ValueError("unfreeze_blocks must be >= 0")


class FreezeFirstSchedule:
    """Drives the two-phase adaptation and owns the optimiser.

    Args:
        model: the module being adapted.
        blocks: the encoder blocks, ordered bottom-to-top. The last
            ``unfreeze_blocks`` are released in phase 2.
        extra_trainable: modules that are new (heads, projections) and so should
            train from step 0 -- they carry no pretrained knowledge to protect.
    """

    def __init__(self, model: nn.Module, blocks: Sequence[nn.Module],
                 config: Optional[FreezeFirstConfig] = None,
                 extra_trainable: Sequence[nn.Module] = ()) -> None:
        self.model = model
        self.blocks = list(blocks)
        self.config = config or FreezeFirstConfig()
        self.extra_trainable = list(extra_trainable)
        self.step_count = 0
        self.phase = Phase.ADAPT
        self._enter_adapt_phase()

    # -- phases -------------------------------------------------------------
    def _adapter_and_head_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen = set()
        for m in iter_lora_modules(self.model):
            for p in (m.lora_A, m.lora_B):
                if id(p) not in seen:
                    seen.add(id(p)); params.append(p)
        for module in self.extra_trainable:
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); params.append(p)
        return params

    def _enter_adapt_phase(self) -> None:
        freeze_all(self.model)
        mark_only_lora_trainable(self.model)
        for module in self.extra_trainable:
            for p in module.parameters():
                p.requires_grad_(True)
        params = self._adapter_and_head_params()
        self.optimizer = torch.optim.AdamW(
            params, lr=self.config.base_lr,
            weight_decay=self.config.weight_decay) if params else None
        self.phase = Phase.ADAPT

    def _enter_refine_phase(self) -> None:
        encoder_params: List[nn.Parameter] = []
        if self.config.unfreeze_blocks > 0:
            for block in self.blocks[-self.config.unfreeze_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)
                    encoder_params.append(p)

        adapter_ids = {id(p) for p in self._adapter_and_head_params()}
        # A LoRA factor inside an unfrozen block would otherwise appear twice.
        encoder_params = [p for p in encoder_params if id(p) not in adapter_ids]

        groups = [{"params": self._adapter_and_head_params(),
                   "lr": self.config.base_lr}]
        if encoder_params:
            groups.append({"params": encoder_params,
                           "lr": self.config.base_lr * self.config.encoder_lr_scale})
        # Rebuild rather than add_param_group: a fresh optimiser gives the newly
        # released encoder parameters clean moment estimates.
        self.optimizer = torch.optim.AdamW(
            groups, weight_decay=self.config.weight_decay)
        self.phase = Phase.REFINE

    # -- driving ------------------------------------------------------------
    def step(self, loss: torch.Tensor, grad_clip: float = 1.0) -> None:
        """Backprop, clip, step, then advance the phase if it is time."""
        if self.optimizer is None:
            raise RuntimeError("no trainable parameters: nothing to optimise")
        self.optimizer.zero_grad()
        loss.backward()
        trainable = [p for g in self.optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        self.optimizer.step()
        self.step_count += 1
        if (self.phase == Phase.ADAPT
                and self.step_count >= self.config.adapt_steps):
            self._enter_refine_phase()

    # -- introspection ------------------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def summary(self) -> Dict[str, object]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {"phase": self.phase, "step": self.step_count,
                "trainable": trainable, "total": total,
                "learning_rates": [g["lr"] for g in self.optimizer.param_groups]
                if self.optimizer else []}
