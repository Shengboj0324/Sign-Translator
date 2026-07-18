"""Confidence calibration: Brier score, ECE, reliability, temperature scaling.

The source specification lists **calibrated confidence** as part of the layer's
output interface, puts a Brier term in the training objective
(``lambda_cal * L_Brier``), and requires expected calibration error among the
reported metrics. The reason is the fail-closed rule: a system that abstains
below a confidence threshold is only as trustworthy as that confidence. An
uncalibrated 0.9 that is right 60% of the time makes the threshold meaningless.

**Why Brier and not accuracy.** The Brier score is a *strictly proper* scoring
rule: for a true conditional distribution ``q``, the expected score

    E_{y~q} BS(p, y) = sum_k p_k^2 - 2 sum_c q_c p_c + 1

is uniquely minimised at ``p = q``. Optimising it therefore drives the model
toward reporting its true uncertainty rather than merely ranking classes
correctly. Accuracy is not proper -- it is unchanged by any monotone distortion
of the probabilities, so it cannot detect miscalibration at all. The propriety
is proved numerically in the tests.

**Murphy's decomposition.** For binary outcomes, grouping predictions by unique
value gives the exact identity

    BS = REL - RES + UNC
    REL = (1/N) sum_k n_k (p_k - o_k)^2      (miscalibration; lower better)
    RES = (1/N) sum_k n_k (o_k - o)^2        (discrimination; higher better)
    UNC = o (1 - o)                          (irreducible base-rate variance)

This separates *being wrong* from *being uninformative*, which matters here: a
model that always predicts the base rate is perfectly calibrated (REL = 0) and
completely useless (RES = 0). Reporting ECE alone would hide that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_1d_float(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    t = t.detach().double().flatten()
    return t


def _as_1d_bool(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    return t.detach().flatten().bool()


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass Brier score ``mean_i sum_c (p_ic - y_ic)^2``.

    ``probs`` is ``(N, C)`` and must be a proper distribution per row; ``labels``
    is ``(N,)`` of class indices. Range ``[0, 2]``; 0 is perfect.
    """
    if probs.dim() != 2:
        raise ValueError("probs must be (N, C)")
    if labels.dim() != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must be (N,) matching probs")
    p = probs.detach().double()
    onehot = F.one_hot(labels.detach().long(), num_classes=p.shape[1]).double()
    return float((p - onehot).pow(2).sum(dim=1).mean())


def binary_brier_score(confidences, correct) -> float:
    """Binary Brier ``mean (p_i - y_i)^2`` over confidence/correctness pairs."""
    p = _as_1d_float(confidences)
    y = _as_1d_bool(correct).double()
    if p.shape != y.shape:
        raise ValueError("confidences and correct must have equal length")
    if p.numel() == 0:
        raise ValueError("empty input")
    return float((p - y).pow(2).mean())


@dataclass
class BrierDecomposition:
    reliability: float      # lower is better (calibration error)
    resolution: float       # higher is better (discrimination)
    uncertainty: float      # irreducible, depends only on the base rate
    brier: float

    @property
    def reconstructed(self) -> float:
        return self.reliability - self.resolution + self.uncertainty


def brier_decomposition(confidences, correct) -> BrierDecomposition:
    """Exact Murphy decomposition, grouping by unique confidence value.

    Grouping by *unique value* (rather than by histogram bins) is what makes the
    identity exact: within a group every prediction is identical, so no
    within-bin variance is lost. Binned versions are only approximate.
    """
    p = _as_1d_float(confidences)
    y = _as_1d_bool(correct).double()
    if p.shape != y.shape:
        raise ValueError("confidences and correct must have equal length")
    n = p.numel()
    if n == 0:
        raise ValueError("empty input")

    base = float(y.mean())
    rel = res = 0.0
    for value in torch.unique(p):
        mask = p == value
        n_k = int(mask.sum())
        o_k = float(y[mask].mean())
        rel += n_k * (float(value) - o_k) ** 2
        res += n_k * (o_k - base) ** 2
    rel /= n
    res /= n
    unc = base * (1.0 - base)
    return BrierDecomposition(reliability=rel, resolution=res, uncertainty=unc,
                              brier=binary_brier_score(p, y))


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
@dataclass
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_confidence)


def reliability_diagram(confidences, correct, n_bins: int = 10,
                        strategy: Literal["uniform", "quantile"] = "uniform"
                        ) -> List[CalibrationBin]:
    """Bin statistics for a reliability diagram (empty bins are dropped).

    ``uniform`` splits [0,1] into equal-width bins (the standard ECE).
    ``quantile`` splits into equal-*mass* bins, which is more stable when
    confidences pile up near 1.0 -- as they do for a converged model, where
    uniform binning leaves most bins empty and ECE is dominated by one bin.
    """
    p = _as_1d_float(confidences)
    y = _as_1d_bool(correct)
    if p.shape != y.shape:
        raise ValueError("confidences and correct must have equal length")
    if p.numel() == 0:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    if strategy == "uniform":
        edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    elif strategy == "quantile":
        qs = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
        edges = torch.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = torch.unique(edges)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bins: List[CalibrationBin] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Half-open bins, with the last one closed so p == 1.0 is included.
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(CalibrationBin(
            lower=lo, upper=hi, count=count,
            mean_confidence=float(p[mask].mean()),
            accuracy=float(y[mask].double().mean())))
    return bins


def expected_calibration_error(confidences, correct, n_bins: int = 10,
                               strategy: Literal["uniform", "quantile"] = "uniform"
                               ) -> float:
    """ECE ``= sum_b (n_b/N) |acc(b) - conf(b)|``; 0 is perfectly calibrated."""
    bins = reliability_diagram(confidences, correct, n_bins, strategy)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    return sum(b.count / total * b.gap for b in bins)


def maximum_calibration_error(confidences, correct, n_bins: int = 10,
                              strategy: Literal["uniform", "quantile"] = "uniform"
                              ) -> float:
    """The worst per-bin gap -- the guarantee a threshold can actually rely on."""
    bins = reliability_diagram(confidences, correct, n_bins, strategy)
    return max((b.gap for b in bins), default=0.0)


def negative_log_likelihood(log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean NLL of the true class; the objective temperature scaling minimises."""
    if log_probs.dim() != 2:
        raise ValueError("log_probs must be (N, C)")
    return float(F.nll_loss(log_probs.detach().double(),
                            labels.detach().long(), reduction="mean"))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Post-hoc calibration by a single scalar: ``p = softmax(z / T)``.

    One parameter, fitted on held-out data by minimising NLL. Because ``T > 0``
    is a strictly monotone rescaling, it **cannot change the argmax** -- accuracy
    is preserved exactly and only the confidences move. That is precisely what
    makes it safe to apply after training, and it is asserted in the tests.

    ``T`` is stored as ``log_temperature`` so the positivity constraint holds by
    construction rather than by clipping.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature),
                                                         dtype=torch.float64))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled **log**-probabilities."""
        z = logits.double() / self.log_temperature.exp()
        return F.log_softmax(z, dim=-1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 200, lr: float = 0.05) -> float:
        """Fit ``T`` by minimising validation NLL. Returns the fitted value."""
        if logits.dim() != 2:
            raise ValueError("logits must be (N, C)")
        if labels.dim() != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("labels must be (N,) matching logits")
        z = logits.detach().double()
        y = labels.detach().long()
        opt = torch.optim.Adam([self.log_temperature], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad()
            loss = F.nll_loss(F.log_softmax(z / self.log_temperature.exp(), dim=-1), y)
            loss.backward()
            opt.step()
        return self.temperature


class BrierLoss(nn.Module):
    """Differentiable multiclass Brier score, for the ``lambda_cal`` term.

    Unlike cross-entropy, Brier is bounded and penalises confident errors
    quadratically rather than logarithmically, so a single catastrophic
    over-confidence cannot dominate the gradient. Both are proper; combining
    them is the usual way to keep calibration without losing sharpness.
    """

    def __init__(self, from_logits: bool = True) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 2:
            raise ValueError("inputs must be (N, C)")
        probs = F.softmax(inputs, dim=-1) if self.from_logits else inputs
        onehot = F.one_hot(labels.long(), num_classes=probs.shape[1]).to(probs.dtype)
        return (probs - onehot).pow(2).sum(dim=1).mean()
