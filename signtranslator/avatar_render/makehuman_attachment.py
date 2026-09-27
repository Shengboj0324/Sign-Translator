"""Independent fitting/skinning of identity-verified MakeHuman attachments.

Signed affine geometry coordinates and nonnegative bone influences are different
quantities. This private fitter is used only after asset-specific hash checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .makehuman import MakeHumanRig, _finite_tensor
from .rigged import apply_lbs

def _rig_identity(rig: MakeHumanRig) -> str:
    value = hashlib.sha256(json.dumps([rig.bone_names, rig.parents, rig.asset_commit, rig.coordinate_units]).encode())
    for tensor in (rig.rest_vertices, rig.rest_globals, rig.weights):
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


@dataclass(frozen=True)
class MakeHumanAttachment:
    rig: MakeHumanRig
    rig_identity: str
    rest_vertices: torch.Tensor
    weights: torch.Tensor
    faces: tuple[tuple[int, ...], ...]
    texcoords: tuple[tuple[float, float], ...]
    face_uvs: tuple[tuple[int, ...], ...]
    texture_path: Path
    negative_geometry_coefficients: int

    def pose(self, local_rotations: torch.Tensor, *, local_translations: torch.Tensor | None = None,
             world_from_source: torch.Tensor | None = None) -> torch.Tensor:
        """Deform the fitted rest eyes, not an offset refitted to posed vertices.

        Refitting fixed-axis offsets after body posing would not rotate those
        offsets correctly. Fitting is therefore bound to a specific rest rig.
        """
        if _rig_identity(self.rig) != self.rig_identity:
            raise ValueError("Rest rig changed after attachment fitting")
        _finite_tensor(self.rest_vertices, (len(self.rest_vertices), 3), "attachment rest vertices")
        _finite_tensor(self.weights, (len(self.rest_vertices), len(self.rig.bone_names)), "attachment weights")
        for tensor in (self.rest_vertices, self.weights):
            if tensor.dtype != self.rig.rest_vertices.dtype or tensor.device != self.rig.rest_vertices.device:
                raise ValueError("Attachment and rig must share dtype/device")
        if (self.weights < 0).any() or not torch.allclose(self.weights.sum(1), self.weights.new_ones(len(self.weights)), atol=1e-12, rtol=0):
            raise ValueError("Attachment weights must be a nonnegative partition of unity")
        posed = self.rig.pose(local_rotations, local_translations=local_translations,
                              world_from_source=world_from_source)
        result = apply_lbs(self.rest_vertices, self.weights, posed.skin_transforms)
        if not torch.isfinite(result).all():
            raise ValueError("Attachment deformation overflowed")
        return result


def _fit_attachment(proxy_bytes: bytes, mesh_bytes: bytes, texture_path: Path, rig: MakeHumanRig) -> MakeHumanAttachment:
    """Fit verified native direct-vertex or three-vertex mappings to rest geometry."""
    rig.validate()
    if rig.rest_vertices.dtype != torch.float64 or rig.rest_vertices.device.type != "cpu":
        raise ValueError("Pinned attachment fitting requires a float64 CPU rest rig")
    proxy = proxy_bytes.decode("utf-8")
    source = rig.rest_vertices
    scales = torch.full((3,), float("nan"), dtype=source.dtype)
    refs, coefficients, offsets = [], [], []
    reading = False
    for line in proxy.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] in ("x_scale", "y_scale", "z_scale"):
            axis = "xyz".index(fields[0][0])
            a, b, denominator = int(fields[1]), int(fields[2]), float(fields[3])
            scales[axis] = abs(source[a, axis]-source[b, axis])/denominator
        elif fields[0] == "verts":
            reading = True
        elif reading:
            if len(fields) == 1:
                vertex = int(fields[0])
                refs.append([vertex, vertex, vertex])
                coefficients.append([1.0, 0.0, 0.0])
                offsets.append([0.0, 0.0, 0.0])
            elif len(fields) == 9:
                refs.append([int(v) for v in fields[:3]])
                coefficients.append([float(v) for v in fields[3:6]])
                offsets.append([float(v) for v in fields[6:9]])
            else:
                raise ValueError("Unsupported proxy row in pinned asset")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("All proxy scale axes must be finite and positive")
    indices = torch.tensor(refs, dtype=torch.long)
    affine = torch.tensor(coefficients, dtype=source.dtype)
    displacement = torch.tensor(offsets, dtype=source.dtype)
    rest = (source[indices]*affine[..., None]).sum(1)+displacement*scales
    contributions = rig.weights[indices]*affine[..., None]
    mapped = torch.where(contributions > 1e-4, contributions, 0).sum(1)
    totals = mapped.sum(1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("Attachment vertex has no mapped bone support")
    mapped = mapped/totals
    mapped = torch.where(mapped > 1e-4, mapped, 0)
    totals = mapped.sum(1, keepdim=True)
    if (totals <= 0).any():
        raise ValueError("Attachment vertex has no support after thresholding")
    mapped = mapped/totals
    faces, uvs, face_uvs = [], [], []
    count = 0
    mesh = mesh_bytes.decode("utf-8")
    for line in mesh.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            count += 1
        elif fields[0] == "vt":
            uvs.append(tuple(float(v) for v in fields[1:3]))
        elif fields[0] == "f":
            parts = [v.split("/") for v in fields[1:]]
            faces.append(tuple(int(v[0])-1 for v in parts))
            face_uvs.append(tuple(int(v[1])-1 for v in parts))
    if count != len(rest) or not faces:
        raise ValueError("Proxy mapping and OBJ vertex inventory disagree")
    if any(not 0 <= i < count for face in faces for i in face) or any(not 0 <= i < len(uvs) for face in face_uvs for i in face):
        raise ValueError("Invalid attachment face/UV index")
    _finite_tensor(rest, (count, 3), "fitted attachment")
    _finite_tensor(mapped, (count, len(rig.bone_names)), "fitted attachment weights")
    return MakeHumanAttachment(rig, _rig_identity(rig), rest, mapped, tuple(faces), tuple(uvs), tuple(face_uvs),
                         texture_path.resolve(), int((affine < 0).sum()))
