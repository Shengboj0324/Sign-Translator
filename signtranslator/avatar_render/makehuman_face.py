"""Decode a pinned authored face-pose catalogue, not observed signing motion.

The BVH rows are independent named poses. Its frame time is file metadata, not
blink duration. The pinned native expression path disables translation, converts
Z-up to Y-up and conjugates rotations into each target bone's rest basis. Source
BVH offsets are inspected but never substituted for the target bind skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

from .makehuman import MakeHumanRig
from .makehuman_attachment import _rig_identity
from ..pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

_FILES = {
    'makehuman/data/poseunits/face-poseunits.json': 'e7224f76e9610049b79e5f9e8ac2985d1e9a367333395c07537919258cb42cc3',
    'makehuman/data/poseunits/face-poseunits.bvh': '42a8e69692e30010371ee6993a6e5d3b468d35065559ec0953c936e31b8d903f',
}


@dataclass(frozen=True)
class FacePoseUnits:
    rig: MakeHumanRig
    rig_identity: str
    names: tuple[str, ...]
    local_rotations: torch.Tensor
    source_rest_max_coordinate_discrepancy: float
    source_frame_time_metadata: float
    source_rest_max_rotation_degrees: float

    def rotations(self, name: str) -> torch.Tensor:
        """Return one full authored pose; unknown names and changed rig fail."""
        if name not in self.names:
            raise ValueError(f'Unknown authored face pose: {name}')
        if _rig_identity(self.rig) != self.rig_identity:
            raise ValueError('Rest rig changed after face-pose binding')
        pose = self.local_rotations[self.names.index(name)]
        if pose.shape != (len(self.rig.bone_names), 3, 3):
            raise ValueError('Face pose has invalid bone layout')
        matrix_to_rotation_6d(pose)
        return pose.clone()


def load_makehuman_face_units(asset_root: str | Path, rig: MakeHumanRig) -> FacePoseUnits:
    """Interpret exactly the audited catalogue with explicit axis/order mapping."""
    rig.validate()
    if rig.rest_vertices.dtype != torch.float64 or rig.rest_vertices.device.type != 'cpu':
        raise ValueError('Face-pose binding requires a float64 CPU rig')
    payloads = {}
    for name, expected in _FILES.items():
        data = (Path(asset_root)/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Face asset identity mismatch: {name}')
        payloads[name] = data
    labels = tuple(json.loads(payloads['makehuman/data/poseunits/face-poseunits.json'])['framemapping'])
    hierarchy, motion = payloads['makehuman/data/poseunits/face-poseunits.bvh'].decode().split('MOTION', 1)
    tokens = iter(re.findall(r'[^\s{}]+|[{}]', hierarchy))
    names, parents, offsets, channels = [], [], [], []

    def expect(wanted: str) -> None:
        actual = next(tokens, None)
        if actual != wanted:
            raise ValueError(f'Expected BVH token {wanted}; got {actual}')

    def joint(parent: int, end_site: bool = False) -> None:
        name = None if end_site else next(tokens)
        expect('{')
        expect('OFFSET')
        offset = [float(next(tokens)) for _ in range(3)]
        if end_site:
            expect('}')
            return
        i = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(offset)
        expect('CHANNELS')
        count = int(next(tokens))
        channels.append([next(tokens) for _ in range(count)])
        while (kind := next(tokens)) != '}':
            if kind == 'JOINT':
                joint(i)
            elif kind == 'End':
                expect('Site')
                joint(i, True)
            else:
                raise ValueError(f'Unexpected BVH token {kind}')

    expect('HIERARCHY')
    expect('ROOT')
    joint(-1)
    if next(tokens, None) is not None or len(set(names)) != len(names) or set(names) != set(rig.bone_names):
        raise ValueError('BVH hierarchy must map every target bone exactly once')
    lines = motion.strip().splitlines()
    if not lines[0].startswith('Frames:') or not lines[1].startswith('Frame Time:'):
        raise ValueError('Invalid BVH motion header')
    count = int(lines[0].split()[-1])
    frame_time = float(lines[1].split()[-1])
    values = torch.tensor([[float(v) for v in line.split()] for line in lines[2:]], dtype=torch.float64)
    if values.shape != (count, sum(map(len, channels))) or not torch.isfinite(values).all():
        raise ValueError('BVH channel inventory or finite domain mismatch')
    if len(labels) != count or len(set(labels)) != count or labels[0] != 'Rest':
        raise ValueError('Face labels and BVH row inventory mismatch')
    # Explicit proper basis conversion (BVH Z-up -> target Y-up).
    conversion = torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float64)
    matrices = {}
    positions = []
    column = 0
    rest_angle_max = 0.0
    for i, name in enumerate(names):
        positions.append(torch.tensor(offsets[i], dtype=torch.float64)+(0 if parents[i] < 0 else positions[parents[i]]))
        expected = ['Xrotation', 'Yrotation', 'Zrotation']
        if parents[i] < 0:
            expected = ['Xposition', 'Yposition', 'Zposition']+expected
        if channels[i] != expected:
            raise ValueError('Unexpected channel order in pinned face catalogue')
        matrix = torch.eye(3, dtype=torch.float64).repeat(count, 1, 1)
        for channel in channels[i]:
            channel_values = values[:, column]
            column += 1
            if channel.endswith('position'):
                # Native expression loading disables translation; this asset
                # also contains exact zeros. Reject rather than silently ignore
                # future nonzero positional content.
                if torch.count_nonzero(channel_values):
                    raise ValueError('Face catalogue unexpectedly contains translation')
            else:
                axis = 'XYZ'.index(channel[0])
                angle = torch.zeros(count, 3, dtype=torch.float64)
                angle[:, axis] = torch.deg2rad(channel_values)
                matrix = matrix @ axis_angle_to_matrix(angle)
                rest_angle_max = max(rest_angle_max, abs(float(channel_values[0])))
        matrices[name] = conversion @ matrix @ conversion.T
    local = []
    errors = []
    for i, name in enumerate(rig.bone_names):
        basis = rig.rest_globals[i, :3, :3]
        local.append(basis.T @ matrices[name] @ basis)
        errors.append(float((conversion @ positions[names.index(name)]-rig.rest_globals[i, :3, 3]).abs().max()))
    rotations = torch.stack(local, dim=1)
    matrix_to_rotation_6d(rotations)
    return FacePoseUnits(rig, _rig_identity(rig), labels, rotations, max(errors), frame_time, rest_angle_max)
