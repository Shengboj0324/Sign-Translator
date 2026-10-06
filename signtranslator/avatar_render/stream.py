"""Parameter-stream interface and rendering contracts (docs/AVATAR_RENDER.md §1).

The interface between the linguistic/generation stack and the renderer is a
timestamped stream of body/hand/face parameters plus a per-stream contract
(coordinate convention, handedness, scale, skeleton id, blendshape basis id,
frame rate). Contracts are *validated, never silently coerced*, and rendering is a
pure function of the stream (deterministic replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Real
from typing import Callable, List

import torch

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


class Handedness(IntEnum):
    RIGHT = 0
    LEFT = 1


@dataclass(frozen=True)
class AvatarContract:
    """Exact rendering contract carried by a parameter stream."""

    up_axis: str = "y"
    forward_axis: str = "z"
    handedness: Handedness = Handedness.RIGHT
    scale_m_per_unit: float = 1.0
    skeleton_id: str = "smplx"
    blendshape_basis_id: str = "flame10"
    frame_rate: float = 30.0

    def __post_init__(self) -> None:
        if self.up_axis not in _AXES or self.forward_axis not in _AXES:
            raise ValueError("axes must be x/y/z")
        if self.up_axis == self.forward_axis:
            raise ValueError("up and forward axes must differ")
        if not isinstance(self.handedness, Handedness):
            raise ValueError("handedness must be a Handedness enum")
        for name in ("scale_m_per_unit", "frame_rate"):
            value = getattr(self, name)
            try:
                valid = (isinstance(value, Real) and not isinstance(value, bool)
                         and math.isfinite(value) and value > 0)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("skeleton_id", "blendshape_basis_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be explicitly named")


def contract_basis(contract: AvatarContract) -> torch.Tensor:
    """(3, 3) basis ``[right | up | forward]`` for the contract.

    ``right = up × forward`` for a right-handed contract (``det = +1``);
    ``right = forward × up`` for left-handed (``det = −1``). The determinant is the
    handedness certificate.
    """
    up = torch.tensor(_AXES[contract.up_axis])
    fwd = torch.tensor(_AXES[contract.forward_axis])
    right = torch.cross(up, fwd, dim=-1) if contract.handedness == Handedness.RIGHT \
        else torch.cross(fwd, up, dim=-1)
    return torch.stack((right, up, fwd), dim=1)


def contract_is_self_consistent(contract: AvatarContract) -> bool:
    """The basis determinant sign must match the declared handedness."""
    det = float(torch.linalg.det(contract_basis(contract)))
    return (det > 0) == (contract.handedness == Handedness.RIGHT)


@dataclass
class ParameterStream:
    """A timestamped stream of body-6D / translation / expression parameters."""

    contract: AvatarContract
    timestamps: torch.Tensor          # (T,) seconds, strictly increasing
    rot6d: torch.Tensor               # (T, J, 6)
    gamma: torch.Tensor               # (T, 3) root translation
    expr: torch.Tensor                # (T, E)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, AvatarContract):
            raise ValueError("contract must be an AvatarContract")
        if not all(isinstance(value, torch.Tensor) for value in
                   (self.timestamps, self.rot6d, self.gamma, self.expr)):
            raise ValueError("stream arrays must be tensors")
        if self.timestamps.dim() != 1:
            raise ValueError("timestamps must be (T,)")
        T = self.timestamps.shape[0]
        if self.rot6d.dim() != 3 or self.rot6d.shape[0] != T or self.rot6d.shape[-1] != 6:
            raise ValueError("rot6d must be (T, J, 6)")
        if self.gamma.shape != (T, 3):
            raise ValueError("gamma must be (T, 3)")
        if self.expr.dim() != 2 or self.expr.shape[0] != T:
            raise ValueError("expr must be (T, E)")

    @property
    def num_frames(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.rot6d.shape[1])


def validate_stream(stream: ParameterStream) -> List[str]:
    """Return value-rule violations; raise ValueError for malformed structure.

    Structural checks are repeated because stream tensors can be replaced after
    construction. An empty list certifies only the rules checked here, not
    source provenance, anatomical validity or linguistic correctness.
    """
    v: List[str] = []
    # Streams are mutable: repeat structural checks at the execution boundary.
    stream.__post_init__()
    if not contract_is_self_consistent(stream.contract):
        v.append("handedness_contract_inconsistent")
    ts = stream.timestamps
    if ts.numel() == 0:
        v.append("empty_stream")
    if not bool(torch.isfinite(ts).all()):
        v.append("non_finite_timestamps")
    if ts.numel() >= 2 and not bool(torch.all(ts[1:] > ts[:-1])):
        v.append("timestamps_not_strictly_increasing")
    if not (torch.isfinite(stream.rot6d).all() and torch.isfinite(stream.gamma).all()
            and torch.isfinite(stream.expr).all()):
        v.append("non_finite_parameters")
    return v


def replay(stream: ParameterStream, render_fn: Callable[[int], torch.Tensor]
           ) -> torch.Tensor:
    """Deterministic replay: apply ``render_fn(frame_index)`` for every frame and
    stack. A pure ``render_fn`` + a fixed stream yield byte-identical output."""
    violations = validate_stream(stream)
    if violations:
        raise ValueError("Cannot replay invalid stream: " + ", ".join(violations))
    return torch.stack([render_fn(i) for i in range(stream.num_frames)], dim=0)
