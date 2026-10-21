"""Exponential moving average of model weights, for validation and export."""
from __future__ import annotations

import torch


class WeightEMA:
    """θ̄ ← d_t·θ̄ + (1 − d_t)·θ over trainable parameters; buffers stay the live model's.

    The decay warms up, d_t = min(decay, (1 + t)/(10 + t)) for update t, so the average
    is not dominated by the random initialisation early in training (with a constant
    0.999, step 500 would still weigh the initial weights by 0.999⁵⁰⁰ ≈ 0.61).
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay, self.updates = decay, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    def current_decay(self) -> float:
        return min(self.decay, (1 + self.updates) / (10 + self.updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.current_decay()
        self.updates += 1
        names, params = zip(*((n, p.detach()) for n, p in model.named_parameters()))
        shadow = [self.shadow[n] for n in names]
        torch._foreach_mul_(shadow, d)                     # one fused kernel per step, not two per tensor
        torch._foreach_add_(shadow, list(params), alpha=1 - d)

    def state_dict(self, model: torch.nn.Module) -> dict:
        """The model's state dict with EMA parameters (buffers from the live model)."""
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state.update({n: v.clone() for n, v in self.shadow.items()})
        return state

    def swap(self, model: torch.nn.Module) -> dict:
        """Load EMA parameters into ``model``; returns the live ones for :meth:`restore`."""
        live = {n: p.detach().clone() for n, p in model.named_parameters()}
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(self.shadow[n])
        return live

    @staticmethod
    def restore(model: torch.nn.Module, live: dict) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(live[n])


__all__ = ["WeightEMA"]
