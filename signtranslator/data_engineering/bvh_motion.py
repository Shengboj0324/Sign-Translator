"""Strict Biovision Hierarchy (BVH) adapter for skeleton motion capture.

Prepared for the RIT/CUNY ASL Motion-Capture Corpus, whose stories ship as BVH (and
FBX) with gloss keyframe timing. It is written against the BVH format, not against
that corpus' files, which are not yet available; its joint mapping must therefore
be declared after the real skeleton has been inspected, and nothing here guesses
one.

**Format.** ``HIERARCHY`` declares one ``ROOT`` and nested ``JOINT``/``End Site``
blocks, each with an ``OFFSET`` and (except end sites) a ``CHANNELS`` list. ``MOTION``
gives ``Frames: N``, ``Frame Time: dt`` and ``N`` rows, one value per declared
channel in declaration order. Parsing refuses unbalanced braces, duplicate names,
unknown channels, repeated axes, wrong row lengths, a frame count that disagrees
with the rows, and non-finite values.

**Rotation.** Rotation channels are Euler angles in degrees applied in the order
listed: for ``Zrotation Xrotation Yrotation`` the local rotation is
``R = R_z(z) R_x(x) R_y(y)``, so a vector in the child frame maps to the parent as
``v_parent = OFFSET + T + R v_child``. Each elementary rotation is the standard
right-handed one. A joint without rotation channels has the identity rotation.

**Clock.** ``Frame Time`` is parsed exactly as a decimal and frame ``i`` is at
``i * dt`` seconds; there is no other clock in the file.

Values are solved skeleton rotations, so they are ``observed=False`` with an
explicit inference method, like every other solved source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import math
import re

import numpy as np
import torch

from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d

POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")
ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
_NAME = re.compile(r"[A-Za-z0-9_:.\-]+")
MAX_BVH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class BVHJoint:
    name: str
    parent: int | None
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    end_site: tuple[float, float, float] | None


@dataclass(frozen=True)
class BVHDocument:
    joints: tuple[BVHJoint, ...]
    frame_time: Fraction
    motion: np.ndarray  # (T, total channels), float64
    sha256: str

    def channel_slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for joint in self.joints:
            out[joint.name] = slice(start, start + len(joint.channels))
            start += len(joint.channels)
        return out

    @property
    def times(self) -> np.ndarray:
        return np.array([float(i * self.frame_time) for i in range(len(self.motion))], dtype=np.float64)


def _number(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {token!r}")
    return value


def parse_bvh(payload: bytes) -> BVHDocument:
    """Parse one BVH file strictly; every structural surprise is an error."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_BVH_BYTES:
        raise ValueError("BVH payload must be nonempty bytes within the size limit")
    text = payload.decode("utf-8")
    sections = re.split(r"^[ \t]*MOTION[ \t]*\r?$", text, maxsplit=1, flags=re.MULTILINE)
    if len(sections) != 2:
        raise ValueError("no MOTION section")
    head, motion_text = sections
    tokens = head.split()
    if not tokens or tokens[0] != "HIERARCHY":
        raise ValueError("file must start with HIERARCHY")
    joints: list[dict] = []
    stack: list[int | None] = []
    position = 1

    def take(count=1):
        nonlocal position
        if position + count > len(tokens):
            raise ValueError("truncated hierarchy")
        values = tokens[position:position + count]
        position += count
        return values

    pending_end = False
    while position < len(tokens):
        (word,) = take()
        if word in ("ROOT", "JOINT"):
            if (word == "ROOT") != (not joints) or (word == "ROOT" and stack):
                raise ValueError("exactly one ROOT, first, is required")
            (name,) = take()
            if not _NAME.fullmatch(name) or any(j["name"] == name for j in joints):
                raise ValueError(f"invalid or duplicate joint name {name!r}")
            if take() != ["{"]:
                raise ValueError(f"{name}: expected '{{'")
            if take() != ["OFFSET"]:
                raise ValueError(f"{name}: OFFSET must come first")
            offset = tuple(_number(v) for v in take(3))
            if take() != ["CHANNELS"]:
                raise ValueError(f"{name}: CHANNELS must follow OFFSET")
            (count,) = take()
            if not count.isdigit() or not 0 < int(count) <= 6:
                raise ValueError(f"{name}: channel count must be 1-6")
            channels = tuple(take(int(count)))
            if any(c not in POSITION_CHANNELS + ROTATION_CHANNELS for c in channels) or len(set(channels)) != len(channels):
                raise ValueError(f"{name}: unknown or repeated channel in {channels}")
            joints.append(dict(name=name, parent=stack[-1] if stack else None, offset=offset,
                               channels=channels, end_site=None))
            stack.append(len(joints) - 1)
        elif word == "End":
            if take() != ["Site"] or take() != ["{"] or take() != ["OFFSET"] or not stack:
                raise ValueError("malformed End Site")
            site = tuple(_number(v) for v in take(3))
            if take() != ["}"]:
                raise ValueError("End Site must close after its OFFSET")
            if joints[stack[-1]]["end_site"] is not None:
                raise ValueError(f"{joints[stack[-1]]['name']} has two end sites")
            joints[stack[-1]]["end_site"] = site
        elif word == "}":
            if not stack:
                raise ValueError("unbalanced '}'")
            stack.pop()
        else:
            raise ValueError(f"unexpected token {word!r} in hierarchy")
    if stack or not joints:
        raise ValueError("hierarchy braces are unbalanced or empty")
    lines = [line.strip() for line in motion_text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("MOTION needs Frames and Frame Time")
    frames_match = re.fullmatch(r"Frames:\s*(\d+)", lines[0])
    time_match = re.fullmatch(r"Frame Time:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", lines[1])
    if not frames_match or not time_match:
        raise ValueError("malformed Frames or Frame Time line")
    try:
        frame_time = Fraction(Decimal(time_match.group(1)))
    except InvalidOperation as error:
        raise ValueError("unparsable Frame Time") from error
    if frame_time <= 0:
        raise ValueError("Frame Time must be positive")
    rows = lines[2:]
    if len(rows) != int(frames_match.group(1)) or not rows:
        raise ValueError(f"Frames declares {frames_match.group(1)} rows, file has {len(rows)}")
    width = sum(len(j["channels"]) for j in joints)
    motion = np.empty((len(rows), width), dtype=np.float64)
    for index, row in enumerate(rows):
        values = row.split()
        if len(values) != width:
            raise ValueError(f"frame {index}: expected {width} values, found {len(values)}")
        motion[index] = [_number(v) for v in values]
    return BVHDocument(tuple(BVHJoint(**j) for j in joints), frame_time, motion,
                       hashlib.sha256(payload).hexdigest())


def elementary_rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    """Right-handed rotation matrices about one coordinate axis, (T, 3, 3)."""
    theta = np.radians(np.asarray(degrees, dtype=np.float64))
    c, s = np.cos(theta), np.sin(theta)
    out = np.zeros(theta.shape + (3, 3))
    i = "XYZ".index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    out[..., i, i] = 1.
    out[..., j, j], out[..., j, k] = c, -s
    out[..., k, j], out[..., k, k] = s, c
    return out


def local_transforms(document: BVHDocument) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per joint: (T, 3, 3) rotations in listed channel order and (T, 3) translations."""
    frames = len(document.motion)
    rotations, translations = {}, {}
    for joint, part in zip(document.joints, document.channel_slices().values()):
        values = document.motion[:, part]
        rotation = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        translation = np.broadcast_to(np.asarray(joint.offset), (frames, 3)).copy()
        for column, channel in enumerate(joint.channels):
            if channel in ROTATION_CHANNELS:
                rotation = rotation @ elementary_rotation(channel[0], values[:, column])
            else:
                translation[:, POSITION_CHANNELS.index(channel)] += values[:, column]
        rotations[joint.name], translations[joint.name] = rotation, translation
    return rotations, translations


def world_positions(document: BVHDocument) -> dict[str, np.ndarray]:
    """Forward kinematics: (T, 3) world position of every joint origin."""
    rotations, translations = local_transforms(document)
    world_r, world_p = {}, {}
    for joint in document.joints:
        if joint.parent is None:
            world_r[joint.name], world_p[joint.name] = rotations[joint.name], translations[joint.name]
        else:
            parent = document.joints[joint.parent].name
            world_p[joint.name] = world_p[parent] + np.einsum("tij,tj->ti", world_r[parent], translations[joint.name])
            world_r[joint.name] = world_r[parent] @ rotations[joint.name]
    return world_p


@dataclass(frozen=True)
class BVHMapping:
    """Declared joint-to-channel mapping; must be written after inspecting a real file."""

    mapping_id: str
    root_joint: str
    body: tuple[str, ...]
    left_hand: tuple[str, ...]
    right_hand: tuple[str, ...]
    head: str
    excluded: tuple[str, ...]
    metres_per_unit: float
    coordinate_frame: str

    def joints(self) -> tuple[str, ...]:
        return (*self.body, *self.left_hand, *self.right_hand, self.head)


@dataclass
class BVHReport:
    source_sha256: str
    sample_id: str
    status: str = "pending"
    frames: int = 0
    frame_time_seconds: float = 0.
    unavailable_channels: list = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def adapt_bvh(payload: bytes, *, mapping: BVHMapping, sample_id: str, source_id: str, clock_id: str,
              inference_method: str) -> tuple[MultichannelMotion | None, BVHReport]:
    """Convert one BVH into a canonical state, or refuse with a recorded reason."""
    if not isinstance(inference_method, str) or not inference_method.strip():
        raise ValueError("an inference method is required")
    if type(mapping.metres_per_unit) not in (int, float) or not mapping.metres_per_unit > 0:
        raise ValueError("metres_per_unit must be positive")
    report = BVHReport(hashlib.sha256(bytes(payload)).hexdigest(), sample_id)
    try:
        document = parse_bvh(bytes(payload))
    except (ValueError, UnicodeDecodeError) as error:
        report.status, report.reason = "refused", str(error)
        return None, report
    names = [j.name for j in document.joints]
    declared = set(mapping.joints()) | set(mapping.excluded)
    missing = [j for j in mapping.joints() if j not in names]
    undeclared = sorted(set(names) - declared)
    if missing or undeclared or len(set(mapping.joints())) != len(mapping.joints()):
        report.status = "refused"
        report.reason = f"skeleton differs from mapping: missing {missing}, undeclared {undeclared}"
        return None, report
    root = document.joints[0]
    if root.name != mapping.root_joint or not set(POSITION_CHANNELS) <= set(root.channels):
        report.status, report.reason = "refused", "the declared root must be the BVH root with position channels"
        return None, report
    rotations, translations = local_transforms(document)
    frames = len(document.motion)
    report.frames, report.frame_time_seconds = frames, float(document.frame_time)
    convention = f"bvh-listed-order-euler-degrees-v1;mapping={mapping.mapping_id}"

    def channel(values, labels, units, frame):
        valid = np.ones(values.shape[:2], dtype=bool)
        return MotionChannel(values=values, valid=valid, observed=np.zeros_like(valid),
                             confidence=np.ones(valid.shape), labels=labels, coordinate_frame=frame,
                             units=units, source_id=source_id, source_sha256=report.source_sha256,
                             convention=convention, inference_method=inference_method)

    def six_d(joints):
        stacked = np.stack([rotations[j] for j in joints], axis=1)
        return matrix_to_rotation_6d(torch.from_numpy(stacked)).numpy()

    channels = {
        "root_translation": channel((translations[root.name] * mapping.metres_per_unit)[:, None], (root.name,),
                                    "m", mapping.coordinate_frame),
        "body": channel(six_d(mapping.body), mapping.body, "unitless", mapping.coordinate_frame),
        "left_hand": channel(six_d(mapping.left_hand), mapping.left_hand, "unitless", mapping.coordinate_frame),
        "right_hand": channel(six_d(mapping.right_hand), mapping.right_hand, "unitless", mapping.coordinate_frame),
        "head_rotation": channel(six_d((mapping.head,)), (mapping.head,), "unitless", mapping.coordinate_frame),
    }
    for name in CHANNELS:
        if name not in channels:
            _, width, units = CHANNELS[name]
            channels[name] = MotionChannel(
                values=np.zeros((frames, 1, width)), valid=np.zeros((frames, 1), dtype=bool),
                observed=np.zeros((frames, 1), dtype=bool), confidence=np.zeros((frames, 1)),
                labels=(f"{name}-unavailable",), coordinate_frame="not-recorded-by-source", units=units,
                source_id=source_id, source_sha256=report.source_sha256, convention=convention)
            report.unavailable_channels.append(name)
    state = MultichannelMotion(document.times, clock_id, sample_id, channels)
    report.status = "converted"
    return state, report


def keyframe_seconds(keyframe: int, frame_time: Fraction) -> float:
    """Exact time of a gloss boundary given as a BVH frame number."""
    if type(keyframe) is not int or keyframe < 0:
        raise ValueError("keyframe must be a nonnegative integer")
    return float(keyframe * frame_time)


__all__ = ["BVHJoint", "BVHDocument", "BVHMapping", "BVHReport", "parse_bvh", "elementary_rotation",
           "local_transforms", "world_positions", "adapt_bvh", "keyframe_seconds"]
