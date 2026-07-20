"""Class imbalance, focal loss, and uncertainty (docs/FACIAL_NMM.md §5).

* **Focal loss** ``FL = −(1−p_t)^γ log p_t`` (Lin et al.) for rare markers, which
  down-weights easy examples: for a well-classified example ``(1−p_t)^γ → 0``.
* **Class-balanced** weights ``(1−β)/(1−β^{n_c})`` (Cui et al.), larger for rarer
  classes.
* **Innovation — annotation-agreement-weighted heteroscedastic uncertainty.** The
  model predicts a per-marker log-variance ``s = log σ²``; the Gaussian NLL
  ``½ e^{−s}(y−p)² + ½ s`` is minimised at ``σ² = (y−p)²``, and the target spread is
  tied to the Doc-03 inter-annotator ``κ``: low-agreement markers get a wider
  predictive interval (predicted σ² increases as κ decreases).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# focal loss
# ---------------------------------------------------------------------------
def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.25, reduction: str = "mean") -> torch.Tensor:
    """Binary focal loss. ``p_t`` is the probability of the true class; the
    modulating factor ``(1−p_t)^γ`` shrinks the loss for easy (confident-correct)
    examples so rare/hard markers dominate.
    """
    p = torch.sigmoid(logits)
    y = targets.to(logits.dtype)
    p_t = p * y + (1 - p) * (1 - y)                         # prob of the true class
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = (1 - p_t) ** gamma
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    loss = alpha_t * w * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def focal_modulation(p_t: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """The focal modulating factor ``(1−p_t)^γ`` (for inspection/proofs)."""
    return (1 - p_t) ** gamma


# ---------------------------------------------------------------------------
# class balancing (Cui et al. effective number)
# ---------------------------------------------------------------------------
def class_balanced_weights(counts: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """(K,) weights ``(1−β)/(1−β^{n_c})``, normalised to mean 1. Rarer -> larger."""
    counts = counts.to(torch.float64).clamp_min(1.0)
    eff = (1.0 - beta ** counts)
    w = (1.0 - beta) / eff
    return w / w.mean()


# ---------------------------------------------------------------------------
# heteroscedastic uncertainty
# ---------------------------------------------------------------------------
def heteroscedastic_nll(pred: torch.Tensor, target: torch.Tensor,
                        log_var: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL ½ e^{−s}(y−p)² + ½ s, minimised at σ² = (y−p)²."""
    return (0.5 * torch.exp(-log_var) * (target - pred) ** 2 + 0.5 * log_var).mean()


def agreement_to_target_logvar(kappa: torch.Tensor, base: float = 0.0,
                               slope: float = 2.0) -> torch.Tensor:
    """Target log-variance from inter-annotator agreement ``κ ∈ [0,1]``.

    Low agreement -> high target variance. ``s* = base + slope·(1−κ)`` is decreasing
    in κ, so a marker annotators disagree on gets a wider predictive interval.
    """
    return base + slope * (1.0 - kappa.clamp(0.0, 1.0))
