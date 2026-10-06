"""Lossless CPU collation for canonical motion, without granting training rights.

This boundary pads time only. It does not resample clocks, merge joint layouts,
convert rotations into positions, or replace a governed training admission gate.
Padded timestamps are zero and meaningful only where ``frame_valid`` is true.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..pose.multichannel import CHANNELS, MultichannelMotion


@dataclass(frozen=True)
class ChannelLayout:
    labels: tuple[str, ...]
    coordinate_frame: str
    units: str
    convention: str


@dataclass(frozen=True)
class ChannelProvenance:
    source_id: str
    source_sha256: str
    inference_method: str | None


@dataclass(frozen=True)
class ChannelBatch:
    values: torch.Tensor                # B,T,J,D
    valid: torch.Tensor                 # B,T,J
    observed: torch.Tensor              # B,T,J; may include rejected measurements
    confidence: torch.Tensor            # B,T,J
    layout: ChannelLayout
    provenance: tuple[ChannelProvenance, ...]

    def supervision_weights(self, *, allow_inferred: bool = False) -> torch.Tensor:
        """Reliability weights; opt-in is not authorization for inferred targets."""
        if type(allow_inferred) is not bool:
            raise ValueError("allow_inferred must be an explicit boolean")
        mask = self.valid if allow_inferred else self.valid & self.observed
        return torch.where(mask, self.confidence, torch.zeros_like(self.confidence))


@dataclass(frozen=True)
class MotionBatch:
    timestamps: torch.Tensor            # B,T; original float64 clock values
    frame_valid: torch.Tensor           # B,T; padding validity, not channel validity
    lengths: torch.Tensor               # B
    sample_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    channels: dict[str, ChannelBatch]


def collate_multichannel(states: Sequence[MultichannelMotion]) -> MotionBatch:
    """Validate and copy states for a DataLoader, retaining every source channel.

    Layout and numeric dtypes must agree within each channel across samples.
    Explicit adaptation must happen before collation; no implicit dtype narrowing,
    coefficient remapping, clock alignment or ordering assumptions are permitted.
    Different samples retain their individual source provenance and clock IDs.
    The returned tensors own their storage independently of the supplied arrays.
    """
    items = tuple(states)
    if not items:
        raise ValueError("cannot collate an empty motion batch")
    for state in items:
        if not isinstance(state, MultichannelMotion):
            raise ValueError("batch entries must be typed MultichannelMotion states")
        state.validate()  # frozen dataclasses can still contain mutated arrays

    batch_size = len(items)
    lengths = torch.tensor([len(state.timestamps) for state in items], dtype=torch.int64)
    frames = int(lengths.max())
    timestamps = torch.zeros((batch_size, frames), dtype=torch.float64)
    frame_valid = torch.zeros((batch_size, frames), dtype=torch.bool)
    for row, state in enumerate(items):
        count = len(state.timestamps)
        timestamps[row, :count] = torch.from_numpy(state.timestamps.copy())
        frame_valid[row, :count] = True

    channels = {}
    for name in CHANNELS:
        source_channels = [state.channels[name] for state in items]
        first = source_channels[0]
        layout = ChannelLayout(first.labels, first.coordinate_frame, first.units, first.convention)
        for channel in source_channels[1:]:
            candidate = ChannelLayout(
                channel.labels, channel.coordinate_frame, channel.units, channel.convention)
            if candidate != layout:
                raise ValueError(f"{name}: incompatible layouts require an explicit adapter")
            if (channel.values.dtype != first.values.dtype
                    or channel.confidence.dtype != first.confidence.dtype):
                raise ValueError(f"{name}: mixed numeric dtypes require explicit conversion")

        # Allocate each field in its own original dtype; masks never derive from
        # nonzero positions, since zero can be an authentic measurement.
        fields = {}
        for field in ("values", "valid", "observed", "confidence"):
            prototype = torch.from_numpy(getattr(first, field).copy())
            padded = torch.zeros((batch_size, frames, *prototype.shape[1:]), dtype=prototype.dtype)
            for row, channel in enumerate(source_channels):
                array = getattr(channel, field)
                padded[row, :len(array)] = torch.from_numpy(array.copy())
            fields[field] = padded
        channels[name] = ChannelBatch(
            **fields, layout=layout,
            provenance=tuple(ChannelProvenance(
                channel.source_id, channel.source_sha256, channel.inference_method)
                for channel in source_channels),
        )
    return MotionBatch(timestamps, frame_valid, lengths,
                       tuple(state.sample_id for state in items),
                       tuple(state.clock_id for state in items), channels)
