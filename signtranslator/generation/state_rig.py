"""Declared canonical-state to rig posing, without inferring any convention.

The canonical state and a rig are two independent coordinate systems. This
adapter refuses to connect them until the caller declares, explicitly and
completely:

* which channel element drives which bone's local rotation;
* which bone, if any, receives a channel's translation, and in which direction
  the unit conversion runs;
* which bones are deliberately held at their rest transform;
* which canonical channels deliberately do not drive the rig at all;
* the claimed equality between the channel's declared rotation convention and
  the rig's bone-local-after-rest basis.

Nothing in that list is derived from the data. A declared binding is a recorded
claim whose correctness remains a source-specific, reviewable question: this
module verifies that the claim is complete, internally consistent, and applied
exactly, not that it is anatomically or linguistically right. In particular a
missing required channel can never become a silent static bind control, gaze and
contact are never inferred from body rotations, and a generated or otherwise
unobserved value is posed only when the caller opts in explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from ..avatar_render.makehuman import MakeHumanRig, PosedMesh
from ..pose.multichannel import CHANNELS, MultichannelMotion
from ..pose.rotations import rotation_6d_to_matrix

#: The one supported declared equality between a channel's rotation convention
#: and the rig basis. A different source convention requires its own explicit
#: adapter, not a relabelling of this one.
BONE_LOCAL_AFTER_REST = "bone_local_after_rest_v1"
_ROTATION_CHANNELS = tuple(
    name for name, (kind, _, _) in CHANNELS.items() if kind == "rotation_6d_columns")
_TRANSLATION_CHANNELS = tuple(
    name for name, (kind, _, _) in CHANNELS.items() if kind == "translation")


def _named(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be explicitly named")
    return value


@dataclass(frozen=True)
class ChannelBoneMapping:
    """One channel element driving one rig bone."""

    channel: str
    element: str
    bone: str

    def __post_init__(self) -> None:
        if self.channel not in CHANNELS:
            raise ValueError("mapping must name a canonical channel")
        _named(self.element, "element")
        _named(self.bone, "bone")

    def to_dict(self) -> dict:
        return dict(channel=self.channel, element=self.element, bone=self.bone)


@dataclass(frozen=True)
class StateRigBinding:
    """A complete, explicit declaration connecting one state to one rig."""

    rotation_mappings: tuple[ChannelBoneMapping, ...]
    translation_mappings: tuple[ChannelBoneMapping, ...]
    rest_bones: tuple[str, ...]
    unmapped_channels: tuple[str, ...]
    metres_per_source_unit: float
    rotation_basis: str
    asset_commit: str

    def __post_init__(self) -> None:
        for field in ("rotation_mappings", "translation_mappings"):
            value = getattr(self, field)
            if (type(value) is not tuple
                    or any(not isinstance(item, ChannelBoneMapping) for item in value)):
                raise ValueError(f"{field} must be a tuple of typed mappings")
        if self.rotation_basis != BONE_LOCAL_AFTER_REST:
            raise ValueError("the declared rotation basis is not supported by this adapter")
        _named(self.asset_commit, "asset_commit")
        if (type(self.metres_per_source_unit) not in (int, float)
                or isinstance(self.metres_per_source_unit, bool)
                or not math.isfinite(self.metres_per_source_unit)
                or self.metres_per_source_unit <= 0):
            raise ValueError("metres_per_source_unit must be a finite positive scale")
        if (type(self.rest_bones) is not tuple
                or any(type(bone) is not str or not bone.strip() for bone in self.rest_bones)
                or len(set(self.rest_bones)) != len(self.rest_bones)):
            raise ValueError("rest_bones must be distinct explicitly named bones")
        if (type(self.unmapped_channels) is not tuple
                or any(channel not in CHANNELS for channel in self.unmapped_channels)
                or len(set(self.unmapped_channels)) != len(self.unmapped_channels)):
            raise ValueError("unmapped_channels must be distinct canonical channel names")
        for mapping in self.rotation_mappings:
            if mapping.channel not in _ROTATION_CHANNELS:
                raise ValueError(f"{mapping.channel} is not a rotation channel")
        for mapping in self.translation_mappings:
            if mapping.channel not in _TRANSLATION_CHANNELS:
                raise ValueError(f"{mapping.channel} is not a translation channel")
        rotated = [mapping.bone for mapping in self.rotation_mappings]
        if len(set(rotated)) != len(rotated):
            raise ValueError("a bone cannot receive two declared rotations")
        translated = [mapping.bone for mapping in self.translation_mappings]
        if len(set(translated)) != len(translated):
            raise ValueError("a bone cannot receive two declared translations")
        if set(rotated) & set(self.rest_bones):
            raise ValueError("a bone is declared both driven and held at rest")
        keys = [(mapping.channel, mapping.element)
                for mapping in self.rotation_mappings + self.translation_mappings]
        if len(set(keys)) != len(keys):
            raise ValueError("a channel element cannot drive two bones")
        driving = {mapping.channel for mapping in
                   self.rotation_mappings + self.translation_mappings}
        if driving & set(self.unmapped_channels):
            raise ValueError("a channel is declared both driving and unmapped")
        if driving | set(self.unmapped_channels) != set(CHANNELS):
            missing = sorted(set(CHANNELS) - driving - set(self.unmapped_channels))
            raise ValueError(
                "every canonical channel must be declared driving or unmapped; "
                f"undeclared: {missing}")

    def to_dict(self) -> dict:
        return dict(
            rotation_mappings=[m.to_dict() for m in self.rotation_mappings],
            translation_mappings=[m.to_dict() for m in self.translation_mappings],
            rest_bones=list(self.rest_bones),
            unmapped_channels=list(self.unmapped_channels),
            metres_per_source_unit=self.metres_per_source_unit,
            rotation_basis=self.rotation_basis, asset_commit=self.asset_commit)

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def check_against(self, rig: MakeHumanRig, state: MultichannelMotion) -> None:
        """Verify the declaration covers this exact rig and state, or fail."""
        if not isinstance(rig, MakeHumanRig):
            raise ValueError("a typed rig is required")
        rig.validate()
        if not isinstance(state, MultichannelMotion):
            raise ValueError("a typed canonical state is required")
        state.validate()
        if self.asset_commit != rig.asset_commit:
            raise ValueError("binding asset commit does not match the rig")
        if rig.coordinate_units != "unscaled_source_units":
            raise ValueError("this adapter only poses the pinned source-unit rig")
        bones = set(rig.bone_names)
        declared = {mapping.bone for mapping in self.rotation_mappings} | set(self.rest_bones)
        if declared != bones:
            raise ValueError(
                "every rig bone must be declared driven or held at rest; "
                f"undeclared: {sorted(bones - declared)}, unknown: {sorted(declared - bones)}")
        unknown = {mapping.bone for mapping in self.translation_mappings} - bones
        if unknown:
            raise ValueError(f"translation mappings name unknown bones: {sorted(unknown)}")
        for mapping in self.rotation_mappings + self.translation_mappings:
            labels = state.channels[mapping.channel].labels
            if mapping.element not in labels:
                raise ValueError(
                    f"{mapping.channel}: element {mapping.element!r} is absent from the state")
        for channel in self.unmapped_channels:
            if channel not in state.channels:
                raise ValueError(f"unmapped channel {channel} is absent from the state")


@dataclass(frozen=True)
class PosedStateFrame:
    """One posed frame with the provenance needed to reproduce it."""

    mesh: PosedMesh
    frame_index: int
    clock_seconds: float
    sample_id: str
    binding_sha256: str
    asset_commit: str
    posed_inferred_values: bool
    accepted_provenance: bool = False

    def provenance(self) -> dict:
        return dict(frame_index=self.frame_index, clock_seconds=self.clock_seconds,
                    sample_id=self.sample_id, binding_sha256=self.binding_sha256,
                    asset_commit=self.asset_commit,
                    posed_inferred_values=self.posed_inferred_values,
                    accepted_provenance=self.accepted_provenance)


def pose_state_frame(rig: MakeHumanRig, state: MultichannelMotion,
                     binding: StateRigBinding, frame_index: int, *,
                     allow_inferred: bool = False,
                     world_from_source: torch.Tensor | None = None) -> PosedStateFrame:
    """Pose exactly one frame of a canonical state through a declared binding.

    Required channel elements must be valid at that frame. They must also be
    observed unless the caller passes ``allow_inferred=True``, which is the
    explicit opt-in needed to pose a generated or fitted state. An unavailable
    required element fails: it never falls back to the bind pose.

    Translations are converted from canonical metres into the rig's source units
    by dividing by ``metres_per_source_unit``, then applied in that bone's
    rest-local axes, exactly as :meth:`MakeHumanRig.pose` defines. A posed frame
    is geometry, not accepted calibration, retargeting or signing fidelity.
    """
    if type(allow_inferred) is not bool:
        raise ValueError("allow_inferred must be an explicit boolean")
    binding.check_against(rig, state)
    frames = len(state.timestamps)
    if type(frame_index) is not int or isinstance(frame_index, bool) or not 0 <= frame_index < frames:
        raise ValueError("frame_index must select an existing frame")
    dtype, device = rig.rest_vertices.dtype, rig.rest_vertices.device
    bone_index = {name: index for index, name in enumerate(rig.bone_names)}
    rotations = torch.eye(3, dtype=dtype, device=device).expand(
        len(rig.bone_names), 3, 3).clone()
    translations = torch.zeros((len(rig.bone_names), 3), dtype=dtype, device=device)
    posed_inferred = False

    def selected(mapping):
        nonlocal posed_inferred
        channel = state.channels[mapping.channel]
        element = channel.labels.index(mapping.element)
        if not bool(channel.valid[frame_index, element]):
            raise ValueError(
                f"{mapping.channel}:{mapping.element} is unavailable at frame "
                f"{frame_index}; the bind pose is not a substitute")
        observed = bool(channel.observed[frame_index, element])
        if not observed:
            if not allow_inferred:
                raise ValueError(
                    f"{mapping.channel}:{mapping.element} at frame {frame_index} is valid "
                    "but unobserved; posing inferred values requires an explicit opt-in")
            posed_inferred = True
        return torch.from_numpy(channel.values[frame_index, element].copy())

    for mapping in binding.rotation_mappings:
        values = selected(mapping).to(dtype=dtype, device=device)
        rotations[bone_index[mapping.bone]] = rotation_6d_to_matrix(values)
    for mapping in binding.translation_mappings:
        metres = selected(mapping).to(dtype=dtype, device=device)
        source_units = metres / binding.metres_per_source_unit
        if not bool(torch.isfinite(source_units).all()):
            raise ValueError("unit conversion overflowed the rig coordinate range")
        translations[bone_index[mapping.bone]] = source_units

    mesh = rig.pose(rotations, local_translations=translations,
                    world_from_source=world_from_source)
    return PosedStateFrame(
        mesh=mesh, frame_index=frame_index,
        clock_seconds=float(state.timestamps[frame_index]),
        sample_id=state.sample_id, binding_sha256=binding.sha256(),
        asset_commit=rig.asset_commit, posed_inferred_values=posed_inferred)


def pose_state_sequence(rig: MakeHumanRig, state: MultichannelMotion,
                        binding: StateRigBinding, *, allow_inferred: bool = False,
                        world_from_source: torch.Tensor | None = None):
    """Yield posed frames one at a time, bounding mesh memory to one frame.

    Frame order follows the state's own strictly increasing clock. Temporal
    smoothness, contact behaviour and transition plausibility are not checked
    here; they remain separate qualification requirements.
    """
    for frame_index in range(len(state.timestamps)):
        yield pose_state_frame(rig, state, binding, frame_index,
                               allow_inferred=allow_inferred,
                               world_from_source=world_from_source)
