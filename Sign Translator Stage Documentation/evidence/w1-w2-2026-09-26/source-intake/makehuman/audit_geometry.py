"""Reproducible float64 geometry audit of the pinned, unchanged candidate assets.

Independent mathematical checks, not execution of the upstream native loader.
Run from repository root with .venv/bin/python. No network or file writes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from signtranslator.avatar_render.rigged import apply_lbs

ROOT = Path('Sign Translator Stage Documentation/evidence/w1-w2-2026-09-26/source-intake/makehuman')
manifest = json.loads((ROOT / 'intake.json').read_text())
for entry in manifest['files']:
    payload = (ROOT / entry['upstream_path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise ValueError(f"Asset identity changed: {entry['upstream_path']}")
vertices = np.array([
    [float(v) for v in line.split()[1:4]]
    for line in (ROOT / 'makehuman/data/3dobjs/base.obj').read_text().splitlines()
    if line.startswith('v ')
], dtype=np.float64)
spec = json.loads((ROOT / 'makehuman/data/rigs/default.mhskel').read_text())
bones = spec['bones']
names = list(bones)
index = {name: i for i, name in enumerate(names)}
joints = {name: vertices[indices].mean(axis=0) for name, indices in spec['joints'].items()}

def unit(v):
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError('Degenerate geometry; no fallback axis permitted in this audit')
    return v / length

rest = np.repeat(np.eye(4)[None], len(names), axis=0)
lengths = []
plane_sines = []
axis_sines = []
for name, bone in bones.items():
    head, tail = (joints[bone[k]] for k in ('head', 'tail'))
    a, b, c = (joints[j] for j in spec['planes'][bone['rotation_plane']])
    # Explicit native winding: (third-second) cross (second-first).
    cross = np.cross(unit(c-b), unit(b-a))
    plane_sines.append(float(np.linalg.norm(cross)))
    normal = unit(cross)
    y = unit(tail-head)
    # Gram-Schmidt construction, with right-handed completion.
    projected = normal - np.dot(normal, y)*y
    axis_sines.append(float(np.linalg.norm(projected)))
    x = unit(projected)
    z = unit(np.cross(x, y))
    rest[index[name], :3, :3] = np.column_stack((x, y, z))
    rest[index[name], :3, 3] = head
    lengths.append(float(np.linalg.norm(tail-head)))
rotation = rest[:, :3, :3]
orth_error = float(np.max(np.abs(rotation.transpose(0, 2, 1) @ rotation - np.eye(3))))
det_error = float(np.max(np.abs(np.linalg.det(rotation)-1)))
if max(orth_error, det_error) > 1e-12:
    raise ValueError('Rest frames are not proper orthonormal frames')

# Reconstruct global rest frames recursively through parent-relative transforms.
rebuilt = {}
relative = {}
def reconstruct(name, visiting=frozenset()):
    if name in visiting:
        raise ValueError('Hierarchy cycle')
    if name in rebuilt:
        return rebuilt[name]
    parent = bones[name]['parent']
    original = rest[index[name]]
    if parent is None:
        relative[name] = original.copy()
        rebuilt[name] = original.copy()
    else:
        relative[name] = np.linalg.solve(rest[index[parent]], original)
        rebuilt[name] = reconstruct(parent, visiting | {name}) @ relative[name]
    return rebuilt[name]
reconstructed = np.stack([reconstruct(name) for name in names])
hierarchy_error = float(np.max(np.abs(reconstructed-rest)))
relative_skin = reconstructed @ np.linalg.inv(rest)
identity_error = float(np.max(np.abs(relative_skin-np.eye(4))))

raw = np.zeros((len(vertices), len(names)), dtype=np.float64)
weight_spec = json.loads((ROOT / 'makehuman/data/rigs/default_weights.mhw').read_text())
for name, entries in weight_spec['weights'].items():
    for vertex, weight in entries:
        raw[vertex, index[name]] += weight
if not np.isfinite(raw).all() or (raw < 0).any() or (raw.sum(axis=1) <= 0).any():
    raise ValueError('Invalid source weights')
normalized = raw / raw.sum(axis=1, keepdims=True)
retained = np.where(normalized > 1e-4, normalized, 0.0)
removed = (normalized > 0) & (retained == 0)
post_sum = retained.sum(axis=1)
if (post_sum <= 0).any():
    raise ValueError('Threshold removed every influence')
# Explicit full partition-of-unity policy for our mathematical oracle.
partition = retained / post_sum[:, None]
v = torch.from_numpy(vertices)
w = torch.from_numpy(partition)
rest_output = apply_lbs(v, w, torch.from_numpy(relative_skin)).numpy()
# All bones undergo the same known proper rotation + translation. LBS must
# equal direct point transformation, independent of bone count and topology.
angle = 0.37
rigid = np.eye(4)
rigid[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
rigid[:3, 3] = [0.2, -0.4, 0.7]
posed_skin = (rigid[None] @ reconstructed) @ np.linalg.inv(rest)
posed = apply_lbs(v, w, torch.from_numpy(posed_skin)).numpy()
expected = vertices @ rigid[:3, :3].T + rigid[:3, 3]
rest_error = float(np.max(np.abs(rest_output-vertices)))
rigid_error = float(np.max(np.abs(posed-expected)))
if max(hierarchy_error, identity_error, rest_error, rigid_error) > 1e-10:
    raise ValueError('Geometric invariance failed')
counts = (retained > 0).sum(axis=1)
result = {
    'schema_version': 1,
    'asset_commit': manifest['commit'],
    'intake_sha256': hashlib.sha256((ROOT/'intake.json').read_bytes()).hexdigest(),
    'method': 'Independent float64 mathematics on pinned base mesh; upstream loader not executed; source units only',
    'vertices': len(vertices), 'bones': len(names),
    'min_bone_length_source_units': min(lengths),
    'min_plane_cross_norm': min(plane_sines),
    'min_normal_perpendicular_component': min(axis_sines),
    'max_orthogonality_error': orth_error, 'max_determinant_error': det_error,
    'max_rest_hierarchy_error': hierarchy_error,
    'max_rest_removed_identity_error': identity_error,
    'positive_influences_removed_at_native_threshold': int(removed.sum()),
    'post_threshold_sum_min': float(post_sum.min()),
    'post_threshold_sum_max': float(post_sum.max()),
    'max_positive_influences': int(counts.max()),
    'vertices_exceeding_four_influences': int((counts > 4).sum()),
    'max_rest_lbs_coordinate_error_source_units': rest_error,
    'max_common_rigid_lbs_coordinate_error_source_units': rigid_error,
    'weight_policy': 'Retain all influences above native threshold; explicitly renormalize to unity for oracle; raw asset unchanged',
    'eye_weight_group_present': {name: name in weight_spec['weights'] for name in ('eye.L', 'eye.R')},
    'qualification': {'native_float32_parity': False, 'metric_units_verified': False, 'morphed_rest_mesh_verified': False, 'render_verified': False, 'signer_review_accepted': False, 'phase_exit_approved': False},
}
print(json.dumps(result, indent=2, allow_nan=False))
