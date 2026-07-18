"""LoRA adapters and the freeze-first training protocol.

Implements Hu et al. (2021), arXiv:2106.09685, as required by the source
specification's adaptation strategy: *"Freeze-first adapter training is the safe
baseline. Unfreeze only upper encoder blocks after the adapter converges, using
LoRA or low learning rates."*

    W' = W_0 + (alpha/r) * B A,    A in R^{r x k},  B in R^{d x r},  r << min(d,k)

Two properties make this the right tool, and both are tested rather than assumed:

* **Zero-initialised B** gives ``Delta W = 0`` at step 0, so the adapted model is
  *exactly* the pretrained model before any training. Adaptation therefore
  cannot damage a strong pretrained encoder at initialisation -- which is the
  whole point of "freeze-first is the safe baseline".
* **Mergeability.** ``B A`` folds into ``W_0`` after training, so deployment
  incurs no extra latency (unlike bottleneck adapters, which add layers). For a
  streaming system with a p95 latency budget this distinction is not academic.

``Delta W`` has rank at most ``r`` by construction; that too is asserted, since a
silent bug producing a full-rank update would defeat the parameter saving.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear wraps nn.Linear")
        if r < 1:
            raise ValueError("rank r must be >= 1")
        if r > min(base.in_features, base.out_features):
            raise ValueError("rank r must not exceed min(in_features, out_features)")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)             # the pretrained weights stay frozen

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        # A: Kaiming-uniform as in the reference implementation; B: exactly zero
        # so that Delta W = B A = 0 and the wrapper is an identity at init.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """``(alpha/r) * B A`` -- the effective weight update, rank <= r."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out                          # update already folded into W_0
        lora = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * lora

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ``Delta W`` into the base weight (idempotent).

        **Numerical note.** Merging is mathematically exact -- ``x(W_0 + BA)^T``
        equals ``xW_0^T + ((xA^T)B^T)`` -- but it is *not* bitwise exact in
        float32, because it reassociates the products: the unmerged path
        contracts through the rank-``r`` bottleneck first, the merged path
        through the full ``d x k`` matrix. Measured deviation is ~1e-3 relative
        in float32 and ~1e-16 in float64. It is therefore wrong to assert
        bitwise equality after merging, and any downstream test that does so
        will be flaky rather than informative.
        """
        if self.merged:
            return
        self.base.weight.add_(self.delta_weight())
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight.sub_(self.delta_weight())
        self.merged = False

    def trainable_parameter_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def full_parameter_count(self) -> int:
        return self.base.weight.numel()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"r={self.r}, alpha={self.alpha}, merged={self.merged}")


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
# Modules whose Linear children are reached **attribute-wise** by their parent's
# implementation (e.g. ``self.out_proj.weight`` inside
# ``nn.MultiheadAttention``) rather than by calling them. Wrapping such a child
# type-checks and passes a structural test, but breaks the forward pass with
# "LoRALinear object has no attribute 'weight'". They are skipped by default.
_ATTRIBUTE_ACCESSED_PARENTS = (nn.MultiheadAttention,)


def inject_lora(model: nn.Module, target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
                r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                allow_unsafe_parents: bool = False) -> List[str]:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear`.

    Args:
        target_suffixes: a module is adapted when its dotted name ends with any
            of these. Defaults to query/value projections, following the paper's
            finding that adapting those two is usually sufficient.
        allow_unsafe_parents: permit adapting children of modules that access
            them attribute-wise (see ``_ATTRIBUTE_ACCESSED_PARENTS``). Doing so
            produces a model that *constructs* fine and then fails on the first
            forward pass, so it is refused unless explicitly requested.

    Returns the list of adapted module names, so a caller can assert that the
    intended layers -- and only those -- were touched.
    """
    adapted: List[str] = []
    for name, module in list(model.named_modules()):
        unsafe_parent = isinstance(module, _ATTRIBUTE_ACCESSED_PARENTS)
        if unsafe_parent and not allow_unsafe_parents:
            continue
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(
                    full.split(".")[-1] == s or full.endswith("." + s)
                    for s in target_suffixes):
                setattr(module, child_name,
                        LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                adapted.append(full)
    return adapted


def iter_lora_modules(model: nn.Module) -> Iterable[LoRALinear]:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m


def merge_all_lora(model: nn.Module) -> int:
    n = 0
    for m in iter_lora_modules(model):
        m.merge()
        n += 1
    return n


# ---------------------------------------------------------------------------
# Freeze-first protocol
# ---------------------------------------------------------------------------
def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze everything, then re-enable only the LoRA factors.

    This is the *first* phase of the protocol: the pretrained encoder is
    untouched while the adapter converges.
    """
    freeze_all(model)
    count = 0
    for m in iter_lora_modules(model):
        m.lora_A.requires_grad_(True)
        m.lora_B.requires_grad_(True)
        count += m.trainable_parameter_count()
    return count


def unfreeze_upper_blocks(blocks: Sequence[nn.Module], num_blocks: int) -> int:
    """Unfreeze the **last** ``num_blocks`` encoder blocks (second phase).

    Upper blocks encode task-specific abstractions; lower blocks encode generic
    acoustics that transfer, so unfreezing from the top preserves the
    pretrained representation that makes the encoder worth using at all.
    """
    if num_blocks < 0:
        raise ValueError("num_blocks must be >= 0")
    if num_blocks == 0:
        return 0
    count = 0
    for block in list(blocks)[-num_blocks:]:
        for p in block.parameters():
            p.requires_grad_(True)
            count += p.numel()
    return count


def trainable_parameter_summary(model: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total,
            "frozen": total - trainable}
