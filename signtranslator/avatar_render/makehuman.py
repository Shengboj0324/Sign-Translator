"""Asset-bound adapter for the audited MakeHuman base mesh and default rig.

This is an independent implementation of rest-frame kinematics and full-influence
linear blend skinning. It does not load MakeHuman application code. Source units
are preserved: no metric calibration, motion retargeting, anatomical joint limits,
morphed-character parity or linguistic qualification is implied.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from ..pose.rotations import matrix_to_rotation_6d
from .rigged import apply_lbs


# Pinned exporter guiexport.py:124-128 maps native decimeters to meters by 0.1.
SOURCE_METERS_PER_UNIT = 0.1
ASSET_COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
_ASSETS = {
    "makehuman/data/3dobjs/base.obj": "8e761e6624b8f54536409135d1636da63b32486a90d4897f84e121d144f6fb4c",
    "makehuman/data/rigs/default.mhskel": "99f179bce0aa850b45d4191a1d0d234c5851f881c057439470ded3bddf729a24",
    "makehuman/data/rigs/default_weights.mhw": "0f3641d651ae3d00ad6b4ccee43142edb109d3bd909d27d9e4139ef1beed8625",
}


def _finite_tensor(value: torch.Tensor, shape: tuple[int, ...], label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite float32 or float64")


def _rigid(value: torch.Tensor, count: int, label: str) -> None:
    _finite_tensor(value, (count, 4, 4), label)
    matrix_to_rotation_6d(value[:, :3, :3])  # Shared finite, orthogonal, det=+1 contract.
    expected = value.new_tensor([0, 0, 0, 1]).expand(count, 4)
    if not torch.equal(value[:, 3], expected):
        raise ValueError(f"{label} must have homogeneous last row [0,0,0,1]")


@dataclass(frozen=True)
class PosedMesh:
    """One posed mesh in the unchanged source coordinate frame and source units."""

    vertices: torch.Tensor
    bone_globals: torch.Tensor
    skin_transforms: torch.Tensor


@dataclass(frozen=True)
class MakeHumanRig:
    """One bound rest mesh; all tensors share dtype/device and are rechecked on use.

    Bone order is topological. A local pose is a rotation after the bone's
    parent-relative rest transform. ``world_from_source`` is applied once to
    the root, allowing explicit whole-body rotation/translation in source units.
    Face polygons and OBJ group membership are preserved, including helpers;
    selecting renderable anatomical surfaces is a separate explicit operation.
    """

    bone_names: tuple[str, ...]
    parents: tuple[int, ...]
    rest_vertices: torch.Tensor
    rest_globals: torch.Tensor
    weights: torch.Tensor
    faces: tuple[tuple[int, ...], ...]
    face_groups: tuple[tuple[str, ...], ...]
    asset_commit: str = ASSET_COMMIT
    coordinate_units: str = "unscaled_source_units"

    def validate(self) -> None:
        count = len(self.bone_names)
        if not count or len(set(self.bone_names)) != count:
            raise ValueError("Bone names must be nonempty and unique")
        if len(self.parents) != count or self.parents[0] != -1:
            raise ValueError("Exactly one first root is required")
        if any(type(p) is not int or not 0 <= p < i for i, p in enumerate(self.parents[1:], 1)):
            raise ValueError("Parents must precede children, with only one root")
        if self.rest_vertices.ndim != 2 or not len(self.rest_vertices):
            raise ValueError("Rest mesh must be nonempty")
        vertices = len(self.rest_vertices)
        _finite_tensor(self.rest_vertices, (vertices, 3), "rest_vertices")
        _rigid(self.rest_globals, count, "rest_globals")
        _finite_tensor(self.weights, (vertices, count), "weights")
        for tensor in (self.rest_globals, self.weights):
            if tensor.dtype != self.rest_vertices.dtype or tensor.device != self.rest_vertices.device:
                raise ValueError("Rig tensors must share dtype and device")
        tolerance = 32 * torch.finfo(self.weights.dtype).eps
        if (self.weights < 0).any() or not torch.allclose(
            self.weights.sum(1), self.weights.new_ones(vertices), atol=tolerance, rtol=0
        ):
            raise ValueError("Weights must be a nonnegative partition of unity")
        if len(self.faces) != len(self.face_groups) or not self.faces:
            raise ValueError("Faces require corresponding OBJ group memberships")
        if any(len(face) < 3 or any(type(v) is not int or not 0 <= v < vertices for v in face)
               for face in self.faces):
            raise ValueError("Face indices must reference the rest mesh")
        if self.asset_commit != ASSET_COMMIT or self.coordinate_units != "unscaled_source_units":
            raise ValueError("This adapter only supports the pinned unscaled source convention")

    def pose(self, local_rotations: torch.Tensor, *, local_translations: torch.Tensor | None = None,
             world_from_source: torch.Tensor | None = None) -> PosedMesh:
        """Pose a single frame, retaining all influences and input gradients.

        A sequence can be processed one frame at a time to bound mesh memory.
        Identity local rotations reproduce rest geometry; a root-world transform
        is composed once, not independently accumulated down the hierarchy.
        Optional translations are in each bone's rest-local axes, in unchanged
        source units, after its parent-relative rest transform. They are not
        world-space joint positions. The local transform is [R,t], so rotating
        that same bone does not additionally rotate its own translation. Parent
        motion carries descendant translations. Zero translation is the default.
        """
        self.validate()
        count = len(self.bone_names)
        _finite_tensor(local_rotations, (count, 3, 3), "local_rotations")
        matrix_to_rotation_6d(local_rotations)
        if local_rotations.dtype != self.rest_vertices.dtype or local_rotations.device != self.rest_vertices.device:
            raise ValueError("Pose and rig must share dtype and device")
        translations = local_rotations.new_zeros(count, 3)
        if local_translations is not None:
            _finite_tensor(local_translations, (count, 3), "local_translations")
            if local_translations.dtype != local_rotations.dtype or local_translations.device != local_rotations.device:
                raise ValueError("Translations and rig must share dtype and device")
            translations = local_translations
        world = torch.eye(4, dtype=local_rotations.dtype, device=local_rotations.device)
        if world_from_source is not None:
            _finite_tensor(world_from_source, (4, 4), "world_from_source")
            _rigid(world_from_source.unsqueeze(0), 1, "world_from_source")
            if world_from_source.dtype != world.dtype or world_from_source.device != world.device:
                raise ValueError("World transform and rig must share dtype and device")
            world = world_from_source
        inverse_rest = torch.linalg.inv(self.rest_globals)
        globals_ = []
        for i, parent in enumerate(self.parents):
            upper = torch.cat((local_rotations[i], translations[i, :, None]), dim=1)
            local = torch.cat((upper, local_rotations.new_tensor([[0, 0, 0, 1]])), dim=0)
            relative_rest = self.rest_globals[i] if parent == -1 else inverse_rest[parent] @ self.rest_globals[i]
            preceding = world if parent == -1 else globals_[parent]
            globals_.append(preceding @ relative_rest @ local)
        posed_globals = torch.stack(globals_)
        skin = posed_globals @ inverse_rest
        vertices = apply_lbs(self.rest_vertices, self.weights, skin)
        if not torch.isfinite(vertices).all() or not torch.isfinite(posed_globals).all() or not torch.isfinite(skin).all():
            raise ValueError("Pose arithmetic overflowed")
        return PosedMesh(vertices, posed_globals, skin)


def load_makehuman_rig(asset_root: str | Path, *, rest_vertices: torch.Tensor | None = None) -> MakeHumanRig:
    """Load the exact audited assets, in float64 on CPU, without changing bytes.

    Hashes are fixed in code rather than trusted from an adjacent editable
    manifest. New mesh/rig versions require a new audit. No influence threshold
    or top-k cap is applied: all positive source influences are retained and
    normalized per vertex. This policy is explicit, not native runtime parity.

    An explicit rest mesh must retain the pinned vertex ordering/topology and
    source units. Its joint locations and every rest frame are rebuilt from the
    pinned helper definitions. This accepts geometry, not its anatomical or
    source provenance; callers must separately establish those facts. Rest
    geometry is a fixed float64 CPU input, not a trainable shape parameter.
    """
    payloads = {}
    for name, digest in _ASSETS.items():
        payload = (Path(asset_root) / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Asset identity mismatch: {name}")
        payloads[name] = payload
    mesh, skeleton, weights_file = payloads.values()
    vertices, faces, groups = [], [], []
    current_group: tuple[str, ...] = ()
    for line in mesh.decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "g":
            current_group = tuple(fields[1:])
        elif fields[0] == "f":
            faces.append(tuple(int(v.split("/")[0])-1 for v in fields[1:]))
            groups.append(current_group)
    coords = torch.tensor(vertices, dtype=torch.float64)
    if rest_vertices is not None:
        _finite_tensor(rest_vertices, tuple(coords.shape), "explicit rest_vertices")
        if rest_vertices.dtype != torch.float64 or rest_vertices.device.type != "cpu":
            raise ValueError("Explicit rest geometry requires float64 CPU coordinates")
        if rest_vertices.requires_grad:
            raise ValueError("Explicit rest geometry must be fixed, not trainable")
        coords = rest_vertices.clone()
    spec = json.loads(skeleton)
    definitions = spec["bones"]
    names: list[str] = []
    pending = set(definitions)
    while pending:
        ready = sorted(name for name in pending if definitions[name]["parent"] is None or definitions[name]["parent"] in names)
        if not ready:
            raise ValueError("Unresolved or cyclic parent hierarchy")
        names.extend(ready)
        pending.difference_update(ready)
    lookup = {name: i for i, name in enumerate(names)}
    parents = tuple(-1 if definitions[name]["parent"] is None else lookup[definitions[name]["parent"]] for name in names)
    joints = {name: coords[indices].mean(0) for name, indices in spec["joints"].items()}

    def unit(vector: torch.Tensor) -> torch.Tensor:
        length = torch.linalg.vector_norm(vector)
        if not torch.isfinite(length) or length <= 1e-12:
            raise ValueError("Degenerate rest geometry; no fallback axis")
        return vector / length

    rest = torch.eye(4, dtype=coords.dtype).repeat(len(names), 1, 1)
    for name, i in lookup.items():
        bone = definitions[name]
        head, tail = joints[bone["head"]], joints[bone["tail"]]
        a, b, c = (joints[j] for j in spec["planes"][bone["rotation_plane"]])
        normal = unit(torch.linalg.cross(unit(c-b), unit(b-a)))
        y = unit(tail-head)
        x = unit(normal - torch.dot(normal, y)*y)
        z = unit(torch.linalg.cross(x, y))
        rest[i, :3, :3] = torch.stack((x, y, z), dim=1)
        rest[i, :3, 3] = head
    weights = torch.zeros(len(coords), len(names), dtype=coords.dtype)
    for name, entries in json.loads(weights_file)["weights"].items():
        for vertex, weight in entries:
            weights[vertex, lookup[name]] += weight
    totals = weights.sum(1, keepdim=True)
    if (totals <= 0).any() or not torch.isfinite(totals).all():
        raise ValueError("Every mesh vertex requires finite positive weight support")
    rig = MakeHumanRig(tuple(names), parents, coords, rest, weights/totals, tuple(faces), tuple(groups))
    rig.validate()
    return rig
