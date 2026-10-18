"""Native glTF 2.0 animation adapter into the canonical multichannel state.

Written against the published glTF 2.0 specification and the actual 3D-LEX v1.0
ASL exports (FBX2glTF v0.9.7, a 68-joint Mixamo-style skeleton), not a guessed
schema. Everything the canonical state needs that the source does not record is
declared unavailable; nothing is filled in.

Specification facts relied on, each checked rather than assumed:

* A GLB is a 12-byte header (magic ``glTF``, version 2, total length), a JSON
  chunk, then an optional BIN chunk; chunk lengths are 4-byte aligned.
* Accessors read ``count`` elements of a declared type from a buffer view with
  optional byte stride. Only non-normalized float32 data is accepted here;
  sparse accessors are refused rather than half-supported.
* Animation sampler input is time in seconds. Rotation output is a unit
  quaternion stored ``(x, y, z, w)``; translation and scale are VEC3. LINEAR
  rotation interpolation is spherical; STEP holds values. CUBICSPLINE stores
  tangents in a different layout and is refused.
* A node property without an animation channel keeps its static node value.

One shared clock. The canonical state has one clock for every channel. If the
animation's samplers do not share bit-identical input times the file is refused:
merging different sampling grids would require resampling, which is a modelling
decision this adapter does not make silently.

Solved, not observed. These joint rotations come from an optical-marker and
glove solve exported through FBX2glTF, so every valid value is marked
``observed=False`` with an explicit inference method. Consumers must opt in to
training on inferred targets.

Physical-plausibility rule. A frame whose root position implies a speed above
the declared bound relative to every neighbour is marked invalid for all
channels, with the reason recorded; its values are zeroed as the canonical
contract requires but are reported, never interpolated. The 3D-LEX exports end
with a reset key that moves the hips about 1.36 m in 1/24 s (about 33 m/s); the
default bound of 5 m/s is far above any root speed in seated or standing
signing and far below that artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import struct

import numpy as np
import torch

from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from ..pose.rotations import matrix_to_rotation_6d, quaternion_to_matrix

GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT = 5126
_WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}
_MAX_BYTES = 64 * 1024 * 1024
UNIT_QUATERNION_TOLERANCE = 1e-5


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate glTF JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite glTF JSON number {value}")


@dataclass(frozen=True)
class GLBDocument:
    document: dict
    binary: bytes

    def accessor(self, index: int) -> np.ndarray:
        """Read one float32 accessor with full bounds checking, as float64."""
        accessors = self.document.get("accessors", [])
        if type(index) is not int or not 0 <= index < len(accessors):
            raise ValueError("accessor index out of range")
        spec = accessors[index]
        if "sparse" in spec:
            raise ValueError("sparse accessors are not supported")
        if spec.get("componentType") != _FLOAT or spec.get("normalized", False):
            raise ValueError("only non-normalized float32 accessors are supported")
        width = _WIDTH.get(spec.get("type"))
        count = spec.get("count")
        if width is None or type(count) is not int or count < 1:
            raise ValueError("unsupported accessor type or count")
        views = self.document.get("bufferViews", [])
        view_index = spec.get("bufferView")
        if type(view_index) is not int or not 0 <= view_index < len(views):
            raise ValueError("accessor buffer view out of range")
        view = views[view_index]
        if view.get("buffer", 0) != 0:
            raise ValueError("only the embedded GLB buffer is supported")
        element = 4 * width
        stride = view.get("byteStride", element)
        start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
        view_end = view.get("byteOffset", 0) + view["byteLength"]
        if stride < element or stride % 4 or start % 4:
            raise ValueError("accessor alignment or stride is invalid")
        last = start + stride * (count - 1) + element
        if last > view_end or view_end > len(self.binary):
            raise ValueError("accessor reads beyond its buffer view")
        values = np.lib.stride_tricks.as_strided(
            np.frombuffer(self.binary, dtype="<f4", offset=start,
                          count=(last - start) // 4),
            shape=(count, width), strides=(stride, 4))
        result = values.astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("accessor contains nonfinite values")
        return result


def parse_glb(data: bytes) -> GLBDocument:
    """Strictly parse one GLB container."""
    if not isinstance(data, (bytes, bytearray)) or not 20 <= len(data) <= _MAX_BYTES:
        raise ValueError("GLB must be bounded bytes")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC or version != 2 or length != len(data):
        raise ValueError("not a glTF 2.0 binary of the declared length")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != _CHUNK_JSON or json_length % 4 or 20 + json_length > len(data):
        raise ValueError("first chunk must be an aligned JSON chunk")
    document = json.loads(bytes(data[20:20 + json_length]).decode("utf-8"),
                          object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
        raise ValueError("asset.version must be 2.0")
    offset = 20 + json_length
    binary = b""
    if offset < len(data):
        bin_length, bin_type = struct.unpack_from("<II", data, offset)
        if bin_type != _CHUNK_BIN or offset + 8 + bin_length > len(data):
            raise ValueError("second chunk must be a BIN chunk within the container")
        binary = bytes(data[offset + 8:offset + 8 + bin_length])
    return GLBDocument(document, binary)


@dataclass(frozen=True)
class NodeAnimation:
    """Animated local TRS per node on one shared clock."""

    times: np.ndarray                         # (T,) seconds, float64 widened from float32
    rotations: dict[str, np.ndarray]          # node -> (T, 4) unit (x, y, z, w)
    translations: dict[str, np.ndarray]       # node -> (T, 3)
    scales: dict[str, np.ndarray]             # node -> (T, 3)
    max_quaternion_norm_error: float


def extract_animation(glb: GLBDocument) -> NodeAnimation:
    document = glb.document
    animations = document.get("animations")
    if not animations:
        raise ValueError("the file contains no animation")
    if len(animations) != 1:
        raise ValueError("exactly one animation is required")
    animation = animations[0]
    nodes = document.get("nodes", [])
    names = [node.get("name") for node in nodes]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("every node needs a distinct nonempty name")
    samplers = animation.get("samplers", [])
    inputs = set()
    tracks = {"rotation": {}, "translation": {}, "scale": {}}
    worst = 0.
    for channel in animation.get("channels", []):
        target = channel.get("target", {})
        path, node = target.get("path"), target.get("node")
        if path not in tracks or type(node) is not int or not 0 <= node < len(nodes):
            raise ValueError(f"unsupported animation target {target}")
        sampler = samplers[channel["sampler"]]
        interpolation = sampler.get("interpolation", "LINEAR")
        if interpolation not in ("LINEAR", "STEP"):
            raise ValueError(f"interpolation {interpolation} is not supported")
        inputs.add(sampler["input"])
        values = glb.accessor(sampler["output"])
        if path == "rotation":
            if values.shape[1] != 4:
                raise ValueError("rotation output must be VEC4")
            norms = np.linalg.norm(values, axis=1)
            error = float(np.abs(norms - 1).max())
            if error > UNIT_QUATERNION_TOLERANCE:
                raise ValueError(f"{names[node]}: quaternion norm error {error:.2e} exceeds tolerance")
            worst = max(worst, error)
            values = values / norms[:, None]
        elif values.shape[1] != 3:
            raise ValueError(f"{path} output must be VEC3")
        if names[node] in tracks[path]:
            raise ValueError(f"{names[node]} has two {path} channels")
        tracks[path][names[node]] = values
    if len(inputs) != 1:
        raise ValueError("samplers must share one input clock; resampling is not implied")
    times = glb.accessor(next(iter(inputs)))[:, 0]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("animation clock must strictly increase over at least two keys")
    for path, values in tracks.items():
        for name, array in values.items():
            if len(array) != len(times):
                raise ValueError(f"{name} {path} length differs from the shared clock")
    return NodeAnimation(times, tracks["rotation"], tracks["translation"], tracks["scale"], worst)


def _trs_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        matrix = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
        return matrix
    x, y, z, w = node.get("rotation", (0., 0., 0., 1.))
    rotation = quaternion_to_matrix(torch.tensor([w, x, y, z], dtype=torch.float64)).numpy()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.asarray(node.get("scale", (1., 1., 1.)), dtype=np.float64)
    matrix[:3, 3] = node.get("translation", (0., 0., 0.))
    return matrix


def longest_plausible_subsequence(times: np.ndarray, positions: np.ndarray,
                                  max_speed: float) -> list[int]:
    """Frames of the longest chain whose consecutive members move plausibly.

    Frame ``i`` may precede ``j > i`` iff ``|p_j - p_i| / (t_j - t_i) <= max_speed``
    on the real clock. A longest path in this DAG is found exactly by dynamic
    programming in ``O(n^2)``. Ties prefer the latest admissible predecessor and
    then the latest end frame, so contiguous runs are kept and the result is
    deterministic. For genuine motion below the bound the chain is every frame.
    """
    count = len(times)
    best = np.ones(count, dtype=np.int64)
    previous = np.full(count, -1, dtype=np.int64)
    for j in range(1, count):
        distance = np.linalg.norm(positions[:j] - positions[j], axis=1)
        admissible = distance <= max_speed * (times[j] - times[:j])
        if admissible.any():
            lengths = np.where(admissible, best[:j], 0)
            top = lengths.max()
            i = int(np.flatnonzero(lengths == top)[-1])
            best[j], previous[j] = top + 1, i
    end = int(np.flatnonzero(best == best.max())[-1])
    chain = []
    while end >= 0:
        chain.append(end)
        end = int(previous[end])
    return chain[::-1]


def _constant_ancestor(name: str, animation: NodeAnimation) -> dict | None:
    """Static TRS if every animated channel of ``name`` never changes, else None."""
    found, static = False, {}
    for key, tracks in (("rotation", animation.rotations), ("translation", animation.translations),
                        ("scale", animation.scales)):
        values = tracks.get(name)
        if values is None:
            continue
        found = True
        if key == "rotation":
            dots = np.clip(np.abs(values @ values[0]), 0., 1.)
            if np.degrees(2 * np.arccos(dots.min())) > 1e-4:
                return None
        elif np.abs(values - values[0]).max() > 1e-7 * max(1., float(np.abs(values[0]).max())):
            return None
        static[key] = values[0].tolist()
    return static if found else None


def _static_offset_degrees(node: dict, keyed: dict) -> float | None:
    """None when the keyed constant TRS equals the node's static TRS, else the offset.

    Translation and scale differences are reported as a rotation offset of 0 only
    when they also match; any other mismatch is returned as ``inf``.
    """
    rotation = np.asarray(node.get("rotation", (0., 0., 0., 1.)), dtype=np.float64)
    if "rotation" in keyed:
        dot = min(1., abs(float(np.dot(np.asarray(keyed["rotation"]), rotation))))
        degrees = math.degrees(2 * math.acos(dot))
        if degrees > 1e-4:
            return degrees
    for key, default in (("translation", (0., 0., 0.)), ("scale", (1., 1., 1.))):
        if key in keyed and not np.allclose(keyed[key], node.get(key, default), atol=1e-7):
            return math.inf
    return None


def rest_pose_distance(animation: "NodeAnimation", nodes: list[dict], index: dict[str, int],
                       mapping: "SkeletonMapping") -> np.ndarray | None:
    """Per frame, the median over animated mapped joints (root excluded) of the
    geodesic angle in degrees between the keyed local rotation and the node's rest
    rotation. Joints that are not animated equal their rest value trivially and
    would bias the median to zero, so they are left out. None if none is animated.
    """
    joints = [j for j in mapping.joints() if j != mapping.root_joint and j in animation.rotations]
    if not joints:
        return None
    angles = []
    for joint in joints:
        rest = np.asarray(nodes[index[joint]].get("rotation", (0., 0., 0., 1.)), dtype=np.float64)
        rest = rest / np.linalg.norm(rest)
        dots = np.clip(np.abs(animation.rotations[joint] @ rest), 0., 1.)
        angles.append(np.degrees(2 * np.arccos(dots)))
    return np.median(np.stack(angles, axis=1), axis=1)


def _parent_map(nodes: list[dict]) -> dict[int, int]:
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise ValueError("a glTF node has two parents")
            parents[child] = index
    return parents


@dataclass(frozen=True)
class SkeletonMapping:
    """Declared joint-to-channel mapping for one source skeleton.

    The joint lists are a recorded convention read from the source's actual
    joint names; their linguistic and anatomical adequacy needs review.
    """

    mapping_id: str
    root_joint: str
    body: tuple[str, ...]
    left_hand: tuple[str, ...]
    right_hand: tuple[str, ...]
    head: str
    excluded: tuple[str, ...]
    rotation_convention: str = "gltf-node-local-rotation-including-rest-v1"
    coordinate_frame: str = "gltf-world-y-up-right-handed-metres"

    def joints(self) -> tuple[str, ...]:
        return (*self.body, *self.left_hand, *self.right_hand, self.head)

    def sha256(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, default=list).encode()
        return hashlib.sha256(payload).hexdigest()


_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")

THREE_D_LEX_MIXAMO68 = SkeletonMapping(
    mapping_id="3d-lex-v1.0-asl-mixamo68-v1",
    root_joint="Hips",
    body=("Hips", "Spine", "Spine1", "Spine2", "Neck",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
          "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"),
    left_hand=tuple(f"LeftHand{f}{k}" for f in _FINGERS for k in (1, 2, 3)),
    right_hand=tuple(f"RightHand{f}{k}" for f in _FINGERS for k in (1, 2, 3)),
    head="Head",
    # End effectors carry no deforming rotation; eyes are not animated.
    excluded=("Armature", "HeadTop_End", "LeftEye", "RightEye", "LeftToe_End", "RightToe_End",
              *(f"{side}Hand{f}4" for side in ("Left", "Right") for f in _FINGERS)),
)


@dataclass
class AdapterReport:
    source_sha256: str
    sample_id: str
    status: str = "pending"
    frames: int = 0
    invalid_frames: list[dict] = field(default_factory=list)
    max_quaternion_norm_error: float | None = None
    unrepresented: dict = field(default_factory=dict)
    unavailable_channels: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _channel(values, valid, *, labels, units, frame, source_id, source_sha256, convention,
             inference):
    frames, elements = valid.shape
    observed = np.zeros((frames, elements), dtype=bool)
    confidence = np.where(valid, 1., 0.)
    values = np.where(valid[..., None], values, 0.)
    return MotionChannel(values=values, valid=valid, observed=observed, confidence=confidence,
                         labels=labels, coordinate_frame=frame, units=units,
                         source_id=source_id, source_sha256=source_sha256,
                         convention=convention,
                         inference_method=inference if valid.any() else None)


def _unavailable(name, frames, *, source_id, source_sha256, convention):
    _, width, units = CHANNELS[name]
    return MotionChannel(values=np.zeros((frames, 1, width)),
                         valid=np.zeros((frames, 1), dtype=bool),
                         observed=np.zeros((frames, 1), dtype=bool),
                         confidence=np.zeros((frames, 1)), labels=(f"{name}-unavailable",),
                         coordinate_frame="not-recorded-by-source", units=units,
                         source_id=source_id, source_sha256=source_sha256,
                         convention=convention)


def adapt_glb(data: bytes, *, sample_id: str, source_id: str, clock_id: str,
              mapping: SkeletonMapping = THREE_D_LEX_MIXAMO68,
              inference_method: str,
              max_root_speed_m_per_s: float = 5.,
              max_excluded_fraction: float = .1,
              salvage_reoriented_root: bool = False,
              rest_pose_tolerance_degrees: float | None = 2.,
              ) -> tuple[MultichannelMotion | None, AdapterReport]:
    """Convert one GLB into a canonical state, or refuse with a recorded reason.

    ``salvage_reoriented_root`` is an explicit opt-in for files whose
    axis-correction node above the root is keyed. Such a file's global
    orientation is untrustworthy, and so is the root joint's own rotation, which
    is expressed relative to that node; both are marked unavailable while every
    parent-relative joint below the root is kept. The caller must supply
    independent evidence that the remaining chain is sound before relying on it.
    """
    if type(salvage_reoriented_root) is not bool:
        raise ValueError("salvage_reoriented_root must be an explicit boolean")
    digest = hashlib.sha256(bytes(data)).hexdigest()
    report = AdapterReport(source_sha256=digest, sample_id=sample_id)
    if (type(max_root_speed_m_per_s) not in (int, float) or not math.isfinite(max_root_speed_m_per_s)
            or max_root_speed_m_per_s <= 0):
        raise ValueError("max_root_speed_m_per_s must be finite and positive")
    if (type(max_excluded_fraction) not in (int, float) or not 0 <= max_excluded_fraction < .5):
        raise ValueError("max_excluded_fraction must be in [0, 0.5)")
    if rest_pose_tolerance_degrees is not None and (
            type(rest_pose_tolerance_degrees) not in (int, float)
            or not 0 <= rest_pose_tolerance_degrees < 10):
        raise ValueError("rest_pose_tolerance_degrees must be None or in [0, 10)")
    try:
        glb = parse_glb(data)
        animation = extract_animation(glb)
    except ValueError as error:
        report.status, report.reason = "refused", str(error)
        return None, report
    nodes = glb.document["nodes"]
    names = [node["name"] for node in nodes]
    index = {name: i for i, name in enumerate(names)}
    skins = glb.document.get("skins", [])
    if len(skins) != 1:
        report.status, report.reason = "refused", "exactly one skin is required"
        return None, report
    joints = {names[j] for j in skins[0]["joints"]}
    declared = set(mapping.joints()) | set(mapping.excluded)
    missing = [joint for joint in mapping.joints() if joint not in joints]
    undeclared = sorted(joints - declared)
    animated = (set(animation.rotations) | set(animation.translations) | set(animation.scales))
    undeclared_motion = sorted(animated - declared)
    if missing or undeclared or undeclared_motion:
        report.status = "refused"
        report.reason = (f"skeleton differs from mapping: missing {missing}, undeclared joints "
                         f"{undeclared}, undeclared animated nodes {undeclared_motion}")
        return None, report
    times = animation.times
    frames = len(times)
    report.frames = frames
    report.max_quaternion_norm_error = animation.max_quaternion_norm_error

    def local_rotation(joint):
        if joint in animation.rotations:
            xyzw = animation.rotations[joint]
        else:
            xyzw = np.tile(np.asarray(nodes[index[joint]].get("rotation", (0., 0., 0., 1.))), (frames, 1))
        wxyz = torch.from_numpy(np.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=1))
        return matrix_to_rotation_6d(quaternion_to_matrix(wxyz)).numpy()

    # World root position: static parent chain applied to the animated local translation.
    salvaged = None
    parents = _parent_map(nodes)
    chain = np.eye(4)
    ancestor = parents.get(index[mapping.root_joint])
    while ancestor is not None:
        static = _constant_ancestor(names[ancestor], animation)
        if static is not None:
            # A constant keyed value overrides the node value under glTF rules.
            # Accept it only if it equals that static value; a different constant
            # would move this file into another world frame under the same label.
            offset = _static_offset_degrees(nodes[ancestor], static)
            if offset is not None:
                if not salvage_reoriented_root:
                    report.status = "refused"
                    report.reason = (f"ancestor {names[ancestor]} is keyed to a constant orientation "
                                     f"{offset:.1f} degrees from its static value; its world frame "
                                     f"would differ from the dataset's")
                    return None, report
                salvaged = (f"ancestor {names[ancestor]} keyed to a constant orientation "
                            f"{offset:.1f} degrees from its static value")
        elif any(names[ancestor] in tracks for tracks in
                 (animation.rotations, animation.translations, animation.scales)):
            # A node above the root re-orients the whole skeleton. In the 3D-LEX
            # exports this is the axis-correction node being keyed instead of
            # static, which tips the body through tens of degrees: an export
            # artifact, not performer motion, so it is refused and quantified.
            detail = ""
            track = animation.rotations.get(names[ancestor])
            if track is not None:
                dots = np.clip(np.abs(track @ track[0]), 0., 1.)
                detail = f"; rotation sweeps {np.degrees(2 * np.arccos(dots.min())):.1f} degrees"
            if not salvage_reoriented_root:
                report.status = "refused"
                report.reason = (f"animated ancestor {names[ancestor]} above the root "
                                 f"re-orients the whole skeleton{detail}")
                return None, report
            # Distances are invariant under the static rigid chain, so the
            # plausibility search below still sees true root speeds.
            salvaged = f"animated ancestor {names[ancestor]}{detail}"
        chain = _trs_matrix(nodes[ancestor]) @ chain
        ancestor = parents.get(ancestor)
    local = animation.translations.get(mapping.root_joint)
    if local is None:
        local = np.tile(np.asarray(nodes[index[mapping.root_joint]].get("translation", (0., 0., 0.))),
                        (frames, 1))
    root = (chain[:3, :3] @ local.T).T + chain[:3, 3]

    # Physical plausibility of the root path on the actual clock: keep the
    # longest subsequence whose every consecutive pair is plausible.
    chain = longest_plausible_subsequence(times, root, max_root_speed_m_per_s)
    implausible = set(range(frames)) - set(int(f) for f in chain)
    # Rest-pose resets at either end. Some exports snap every joint back to the
    # bind pose on the first or last key; in files whose root is unavailable the
    # root-speed rule cannot see that. A retained end frame whose median animated
    # joint lies within the declared tolerance of its node rest rotation is not
    # performer motion (measured signing frames sit 11-38 degrees from rest).
    at_rest = set()
    if rest_pose_tolerance_degrees is not None:
        distance = rest_pose_distance(animation, nodes, index, mapping)
        if distance is not None:
            kept = [int(f) for f in chain]
            while kept and distance[kept[-1]] <= rest_pose_tolerance_degrees:
                at_rest.add(kept.pop())
            while kept and distance[kept[0]] <= rest_pose_tolerance_degrees:
                at_rest.add(kept.pop(0))
            chain = np.asarray(kept, dtype=int)
    valid_frame = np.zeros(frames, dtype=bool)
    valid_frame[chain] = True
    excluded = np.flatnonzero(~valid_frame)
    for frame in excluded:
        if int(frame) in at_rest and int(frame) not in implausible:
            reason = (f"end frame at the declared rest pose: median animated joint "
                      f"{distance[frame]:.3f} degrees from its node rest rotation, within "
                      f"{rest_pose_tolerance_degrees} degrees")
        else:
            reason = (f"outside the longest subsequence with root speed at most "
                      f"{max_root_speed_m_per_s} m/s between consecutive retained frames")
        report.invalid_frames.append(dict(frame=int(frame), time_seconds=float(times[frame]),
                                          reason=reason))
    if len(excluded) > max_excluded_fraction * frames or len(chain) < 2:
        report.status = "refused"
        report.reason = (f"ambiguous frame set: the root-speed and rest-pose rules keep "
                         f"{len(chain)} of {frames} frames, excluding more than the declared "
                         f"{max_excluded_fraction:.0%} exclusion limit")
        report.invalid_frames.clear()
        return None, report

    for name, tracks in (("scale", animation.scales),):
        for joint, values in tracks.items():
            report.unrepresented[f"{joint}.{name}"] = dict(
                min=values[valid_frame].min(axis=0).round(6).tolist(),
                max=values[valid_frame].max(axis=0).round(6).tolist(),
                note="canonical state has no scale channel; rotations are scale-invariant")

    convention = mapping.rotation_convention
    common = dict(frame=mapping.coordinate_frame, source_id=source_id, source_sha256=digest,
                  convention=f"{convention};mapping={mapping.mapping_id}", inference=inference_method)

    def rotations(joints):
        stacked = np.stack([local_rotation(joint) for joint in joints], axis=1)
        valid = np.broadcast_to(valid_frame[:, None], stacked.shape[:2]).copy()
        return stacked, valid

    channels = {}
    for name, joints in (("body", mapping.body), ("left_hand", mapping.left_hand),
                         ("right_hand", mapping.right_hand), ("head_rotation", (mapping.head,))):
        values, valid = rotations(joints)
        channels[name] = _channel(values, valid, labels=tuple(joints), units="unitless", **common)
    root_valid = valid_frame[:, None].copy()
    if salvaged is not None:
        root_valid[:] = False
        body = channels["body"]
        element = body.labels.index(mapping.root_joint)
        valid = body.valid.copy()
        valid[:, element] = False
        values = body.values.copy()
        values[:, element] = 0.
        confidence = np.where(valid, body.confidence, 0.)
        channels["body"] = MotionChannel(
            values=values, valid=valid, observed=body.observed, confidence=confidence,
            labels=body.labels, coordinate_frame=body.coordinate_frame, units=body.units,
            source_id=body.source_id, source_sha256=body.source_sha256,
            convention=body.convention, inference_method=body.inference_method)
        report.unrepresented["salvaged_global_orientation"] = dict(
            reason=salvaged, unavailable=[f"body:{mapping.root_joint}", "root_translation"])
    channels["root_translation"] = _channel(root[:, None, :], root_valid, labels=(mapping.root_joint,),
                                            units="m", **common)
    for name in CHANNELS:
        if name not in channels:
            channels[name] = _unavailable(name, frames, source_id=source_id, source_sha256=digest,
                                          convention=f"unavailable;mapping={mapping.mapping_id}")
            report.unavailable_channels.append(name)
    state = MultichannelMotion(times.astype(np.float64), clock_id, sample_id, channels)
    report.status = "converted_partial" if salvaged is not None else "converted"
    return state, report


def joint_world_positions(data: bytes, joints: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """World positions (T, J, 3) of the named joints by vectorised forward kinematics.

    Each node's local matrix per frame composes its keyed TRS (or static value) by
    glTF's ``T R S``; world matrices compose parent before child. Agrees with the
    per-frame composition used by :func:`skeleton_elbow_flexion`.
    """
    glb = parse_glb(data)
    animation = extract_animation(glb)
    nodes = glb.document["nodes"]
    names = [node["name"] for node in nodes]
    parents = _parent_map(nodes)
    frames = len(animation.times)
    cache: dict[int, np.ndarray] = {}

    def local(index):
        node = nodes[index]
        name = names[index]
        animated = any(name in tracks for tracks in
                       (animation.rotations, animation.translations, animation.scales))
        if not animated:
            return np.broadcast_to(_trs_matrix(node), (frames, 4, 4))
        if "matrix" in node:
            raise ValueError(f"{name}: an animated node cannot also carry a matrix")
        xyzw = animation.rotations.get(name)
        if xyzw is None:
            xyzw = np.tile(np.asarray(node.get("rotation", (0., 0., 0., 1.)), dtype=np.float64), (frames, 1))
        wxyz = torch.from_numpy(np.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=1).astype(np.float64))
        rotation = quaternion_to_matrix(wxyz).numpy()
        scale = animation.scales.get(name)
        if scale is None:
            scale = np.tile(np.asarray(node.get("scale", (1., 1., 1.)), dtype=np.float64), (frames, 1))
        translation = animation.translations.get(name)
        if translation is None:
            translation = np.tile(np.asarray(node.get("translation", (0., 0., 0.)), dtype=np.float64), (frames, 1))
        matrix = np.zeros((frames, 4, 4))
        matrix[:, :3, :3] = rotation * scale[:, None, :]
        matrix[:, :3, 3] = translation
        matrix[:, 3, 3] = 1.
        return matrix

    def world(index):
        if index not in cache:
            matrix = local(index)
            if index in parents:
                matrix = world(parents[index]) @ matrix
            cache[index] = matrix
        return cache[index]

    lookup = {name: i for i, name in enumerate(names)}
    missing = [joint for joint in joints if joint not in lookup]
    if missing:
        raise ValueError(f"joints not in the file: {missing}")
    positions = np.stack([world(lookup[joint])[:, :3, 3] for joint in joints], axis=1)
    return animation.times, positions


def skeleton_elbow_flexion(data: bytes, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Elbow flexion of the solved skeleton by full forward kinematics.

    Uses world positions of ``{side}Arm``, ``{side}ForeArm`` and ``{side}Hand``.
    The angle is invariant to any rigid motion of the whole skeleton, so it does
    not depend on the axis-correction defect that :func:`adapt_glb` guards.
    """
    from .marker_validation import joint_angle_degrees
    if side not in ("Left", "Right"):
        raise ValueError("side must be 'Left' or 'Right'")
    glb = parse_glb(data)
    animation = extract_animation(glb)
    nodes = glb.document["nodes"]
    names = [node["name"] for node in nodes]
    parents = _parent_map(nodes)
    frames = len(animation.times)

    def local(index, frame):
        node = dict(nodes[index])
        for key, tracks in (("rotation", animation.rotations),
                            ("translation", animation.translations), ("scale", animation.scales)):
            if names[index] in tracks:
                node[key] = tracks[names[index]][frame].tolist()
                node.pop("matrix", None)
        return _trs_matrix(node)

    def world(index, frame):
        matrix = local(index, frame)
        while index in parents:
            index = parents[index]
            matrix = local(index, frame) @ matrix
        return matrix[:3, 3]

    joints = [names.index(f"{side}{joint}") for joint in ("Arm", "ForeArm", "Hand")]
    points = np.array([[world(joint, frame) for joint in joints] for frame in range(frames)])
    return animation.times, joint_angle_degrees(points[:, 0], points[:, 1], points[:, 2])
