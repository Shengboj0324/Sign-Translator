"""Residual masked-motion reconstruction over canonical rotation channels.

The model reconstructs hidden spans of joint rotations and starts exactly at the
geodesic-interpolation baseline: its 6D output is a correction added to the
interpolated 6D value, and the correction layer is zero-initialised. It can
therefore only beat that baseline by learning structure interpolation lacks, and
any comparison against interpolation is immediate and fair.

Leakage control. Hidden cells are zeroed in the input and flagged by an explicit
mask channel; the interpolation input uses only visible reference neighbours. A
test perturbs hidden targets and asserts unchanged predictions.

Loss. For predicted ``R_hat`` and target ``R`` on hidden, valid cells,
``||R - R_hat||_F^2 / 8 = sin^2(theta/2)``, averaged over those cells only.

This is representation learning on solved skeletons, with no linguistic labels.
Its results say nothing about translation quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from ..pose.multichannel import MultichannelMotion
from ..pose.rotations import rotation_6d_to_matrix
from .representation_comparison import interpolation_baseline

ROTATION_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")
_IDENTITY_6D = np.array([1., 0., 0., 0., 1., 0.])
TIE_DEGREES = 1e-3


@dataclass(frozen=True)
class MaskedExample:
    """One sign, flattened over rotation elements: arrays of shape (T, E, ...)."""

    target: np.ndarray        # (T, E, 6) reference values; zero where invalid
    valid: np.ndarray         # (T, E) reference availability
    hidden: np.ndarray        # (T, E) cells withheld from the model
    interpolated: np.ndarray  # (T, E, 6) baseline from visible neighbours
    covered: np.ndarray       # (T, E) cells the baseline reconstructed
    times: np.ndarray         # (T,) seconds
    sample_id: str


def build_example(state: MultichannelMotion, hidden: dict[str, np.ndarray]) -> MaskedExample:
    """Flatten rotation channels and compute the leakage-free baseline input."""
    if set(hidden) != set(ROTATION_CHANNELS):
        raise ValueError("hide exactly the rotation channels")
    baseline = interpolation_baseline(state, hidden, allow_inferred=True)
    target, valid, mask, interpolated, covered = [], [], [], [], []
    for name in ROTATION_CHANNELS:
        channel = state.channels[name]
        values = np.where(channel.valid[..., None], channel.values, 0.)
        target.append(values)
        valid.append(channel.valid)
        mask.append(hidden[name])
        recon, cover = baseline[name]
        # Uncovered hidden cells fall back to identity; they are never scored.
        recon = np.where((hidden[name] & ~cover)[..., None], _IDENTITY_6D, recon)
        interpolated.append(np.where(hidden[name][..., None], recon, values))
        covered.append(cover)
    return MaskedExample(np.concatenate(target, axis=1), np.concatenate(valid, axis=1),
                         np.concatenate(mask, axis=1), np.concatenate(interpolated, axis=1),
                         np.concatenate(covered, axis=1), np.asarray(state.timestamps), state.sample_id)


class ResidualMaskedMotion(nn.Module):
    """Bidirectional GRU predicting a zero-initialised 6D correction per element."""

    def __init__(self, elements: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if type(elements) is not int or elements < 1:
            raise ValueError("elements must be a positive integer")
        self.elements = elements
        features = elements * (6 + 6 + 1) + 1  # visible values, baseline, mask, clock
        self.encoder = nn.GRU(features, hidden_dim, batch_first=True, bidirectional=True)
        self.correction = nn.Linear(2 * hidden_dim, elements * 6)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, visible: torch.Tensor, baseline: torch.Tensor, hidden: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        batch, frames, elements, _ = visible.shape
        if elements != self.elements:
            raise ValueError("element count differs from the model")
        clock = (times - times[:, :1]).unsqueeze(-1).to(visible.dtype)
        inputs = torch.cat((visible.reshape(batch, frames, -1), baseline.reshape(batch, frames, -1),
                            hidden.reshape(batch, frames, -1).to(visible.dtype), clock), dim=-1)
        states, _ = self.encoder(inputs)
        correction = self.correction(states).reshape(batch, frames, elements, 6)
        return baseline + correction


def model_inputs(example: MaskedExample, dtype=torch.float32):
    """Tensors for one example; hidden reference values never enter the input."""
    hidden = torch.from_numpy(example.hidden)
    visible = torch.from_numpy(np.where(example.hidden[..., None], 0., example.target)).to(dtype)
    baseline = torch.from_numpy(example.interpolated).to(dtype)
    times = torch.from_numpy(example.times).to(torch.float64)
    return visible[None], baseline[None], hidden[None], times[None]


def scored_cells(example: MaskedExample) -> np.ndarray:
    """Hidden, valid cells that the interpolation baseline also covers."""
    return example.hidden & example.valid & example.covered


def half_angle_loss(prediction: torch.Tensor, target: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Mean sin^2(theta/2) over the given cells, with exact chordal arithmetic."""
    if not bool(cells.any()):
        raise ValueError("no scored cells")
    predicted = rotation_6d_to_matrix(prediction[cells].to(torch.float64))
    observed = rotation_6d_to_matrix(target[cells].to(torch.float64))
    return ((predicted - observed).square().sum(dim=(-1, -2)) / 8.).mean()


def geodesic_degrees(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = rotation_6d_to_matrix(torch.from_numpy(prediction))
    observed = rotation_6d_to_matrix(torch.from_numpy(target))
    relative = observed.transpose(-1, -2) @ predicted
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cosine)).numpy()


def train_epoch(model, examples, optimizer, *, seed: int) -> float:
    """One pass in a seeded order; returns the mean training loss."""
    model.train()
    order = np.random.default_rng(seed).permutation(len(examples))
    total = 0.
    for index in order:
        example = examples[index]
        cells = torch.from_numpy(scored_cells(example))
        if not bool(cells.any()):
            continue
        visible, baseline, hidden, times = model_inputs(example)
        prediction = model(visible, baseline, hidden, times)[0]
        loss = half_angle_loss(prediction, torch.from_numpy(example.target), cells)
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("nonfinite masked-motion loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, len(examples))


@torch.no_grad()
def evaluate(model, examples) -> dict:
    """Pooled geodesic error for the model and the baseline on identical cells."""
    model.eval()
    model_sum = base_sum = 0.
    cells_total = 0
    per_sign = []
    for example in examples:
        cells = scored_cells(example)
        if not cells.any():
            continue
        visible, baseline, hidden, times = model_inputs(example)
        prediction = model(visible, baseline, hidden, times)[0].to(torch.float64).numpy()
        model_errors = geodesic_degrees(prediction[cells], example.target[cells])
        base_errors = geodesic_degrees(example.interpolated[cells], example.target[cells])
        model_sum += float(model_errors.sum())
        base_sum += float(base_errors.sum())
        cells_total += int(cells.sum())
        per_sign.append(float(model_errors.mean() - base_errors.mean()))
    differences = np.array(per_sign)
    # The model runs in float32 and the baseline in float64, so an untrained model
    # differs from interpolation by rounding alone (about 1e-5 degrees). Differences
    # within TIE_DEGREES are ties, never wins or losses.
    better = int(np.sum(differences < -TIE_DEGREES))
    worse = int(np.sum(differences > TIE_DEGREES))
    return dict(cells=cells_total, signs=len(per_sign),
                model_mean_degrees=model_sum / cells_total if cells_total else None,
                interpolation_mean_degrees=base_sum / cells_total if cells_total else None,
                signs_model_better=better, signs_model_worse=worse,
                signs_tied=len(per_sign) - better - worse, tie_degrees=TIE_DEGREES,
                mean_per_sign_difference_degrees=float(differences.mean()) if len(differences) else None)
