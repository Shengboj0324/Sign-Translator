"""Surface-crossing diagnostic over authored poses, not an anatomical certificate.

Project Python exports pinned mesh states; a separate Blender process triangulates
and supplies candidate triangle pairs. Float64 segment/triangle tests distinguish
proper interior crossings from boundary/coplanar/degenerate candidates. Counts
are tessellation dependent and are not penetration volumes or clearance margins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def classify_pairs(a, b, eps=1e-10):
    """Return proper-crossing and unresolved-contact masks for paired triangles.

    eps is dimensionless numerical tolerance, not physical contact allowance.
    Strict interior segment hits prove a surface crossing in the given triangles.
    Degenerate or coincident-plane cases are not silently declared safe.
    """
    import numpy as np
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (3, 3):
        raise ValueError('Expected paired arrays (pairs, 3 vertices, 3 coordinates)')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Triangle coordinates must be finite')
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError('Invalid dimensionless tolerance')
    hit = np.zeros(len(a), dtype=bool)
    ambiguous = np.zeros(len(a), dtype=bool)
    for segment_triangle, target in ((a, b), (b, a)):
        e1, e2 = target[:, 1]-target[:, 0], target[:, 2]-target[:, 0]
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        edge_scale = np.linalg.norm(e1, axis=1)*np.linalg.norm(e2, axis=1)
        degenerate = norm <= eps*edge_scale
        ambiguous |= degenerate
        dist = np.einsum('nvc,nc->nv', segment_triangle-target[:, None, 0], normal)
        extent = np.maximum(np.linalg.norm(segment_triangle-target[:, None, 0], axis=2).max(1), np.sqrt(edge_scale))
        coplanar = np.all(np.abs(dist) <= eps*norm[:, None]*extent[:, None], axis=1)
        ambiguous |= coplanar
        for edge in range(3):
            origin = segment_triangle[:, edge]
            direction = segment_triangle[:, (edge+1)%3]-origin
            cross = np.cross(direction, e2)
            determinant = np.einsum('nc,nc->n', e1, cross)
            resolved = np.abs(determinant) > eps*edge_scale*np.linalg.norm(direction, axis=1)
            denominator = np.where(resolved, determinant, 1)
            displacement = origin-target[:, 0]
            u = np.einsum('nc,nc->n', displacement, cross)/denominator
            q = np.cross(displacement, e1)
            v = np.einsum('nc,nc->n', direction, q)/denominator
            t = np.einsum('nc,nc->n', e2, q)/denominator
            inclusive = resolved & (u >= -eps) & (v >= -eps) & (u+v <= 1+eps) & (t >= -eps) & (t <= 1+eps)
            strict = resolved & (u > eps) & (v > eps) & (u+v < 1-eps) & (t > eps) & (t < 1-eps)
            hit |= strict
            ambiguous |= inclusive & ~strict
    return hit, ambiguous & ~hit


def analytic_checks():
    import numpy as np
    base = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]])
    crossing = np.array([[0.5, 0.5, -1], [0.5, 0.5, 1], [1.5, 0.5, 1]])
    disjoint = crossing + [4, 0, 0]
    coplanar = np.array([[0.2, 0.2, 0], [0.8, 0.2, 0], [0.2, 0.8, 0]])
    degenerate = np.zeros((3, 3))
    # Bounding boxes overlap but the triangles are disjoint.
    bbox_only = np.array([[1.5, 1.5, -1], [1.5, 1.5, 1], [2, 2, 0]])
    for other, expected in [(crossing, (True, False)), (disjoint, (False, False)),
                             (coplanar, (False, True)), (degenerate, (False, True)),
                             (bbox_only, (False, False))]:
        actual = classify_pairs(base[None], other[None])
        assert (bool(actual[0][0]), bool(actual[1][0])) == expected
    # Order, vertex winding, uniform scale and proper rigid motion cannot alter a crossing.
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    for scale in [1e-3, 1., 1e3]:
        a = (base*scale)@rotation.T+[3, -2, 1]
        b = (crossing*scale)@rotation.T+[3, -2, 1]
        for x, y in [(a, b), (b, a), (a[::-1], b[::-1])]:
            hits, _ = classify_pairs(x[None], y[None])
            assert hits[0]
    return {'analytical_cases': 5, 'metamorphic_crossing_cases': 9, 'passed': True}


def blender_audit(bundle: Path):
    import bpy
    import numpy as np
    from mathutils.bvhtree import BVHTree
    metadata = json.loads((bundle.parent/'catalogue.json').read_text())
    checks = analytic_checks()
    report = []
    pairs = [('body', 'eyes'), ('body', 'teeth'), ('body', 'tongue'), ('teeth', 'tongue')]
    with np.load(bundle, allow_pickle=False) as data:
        for frame, name in enumerate(metadata['pose_names']):
            geometries = {}
            for part in ['body', 'eyes', 'teeth', 'tongue']:
                coords = data[part][frame]
                mesh = bpy.data.meshes.new('Audit_'+part)
                mesh.from_pydata(coords.tolist(), [], metadata['faces'][part])
                mesh.update()
                mesh.calc_loop_triangles()
                vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
                triangles = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int64)
                tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True, epsilon=0.0)
                geometries[part] = (vertices, triangles, tree)
                bpy.data.meshes.remove(mesh)
            row = {'pose': name, 'pairs': {}}
            for left, right in pairs:
                a, ta, tree_a = geometries[left]
                b, tb, tree_b = geometries[right]
                candidates = np.array(tree_a.overlap(tree_b), dtype=np.int64).reshape(-1, 2)
                x, y = a[ta[candidates[:, 0]]], b[tb[candidates[:, 1]]]
                crossings, unresolved = classify_pairs(x, y)
                counts = {}
                for tolerance in [1e-8, 1e-10, 1e-12]:
                    h, u = classify_pairs(x, y, tolerance)
                    counts[str(tolerance)] = {'proper_crossings': int(h.sum()), 'unresolved_candidates': int(u.sum())}
                row['pairs'][left+'__'+right] = {
                    'candidate_pairs': len(candidates), 'proper_crossings': int(crossings.sum()),
                    'unresolved_candidates': int(unresolved.sum()), 'numerical_tolerance_sensitivity': counts,
                    'first_crossing_triangle_pairs': candidates[crossings][:20].tolist(),
                    'left_degenerate_triangles': int((np.linalg.norm(np.cross(a[ta[:, 1]]-a[ta[:, 0]], a[ta[:, 2]]-a[ta[:, 0]]), axis=1)==0).sum()),
                    'right_degenerate_triangles': int((np.linalg.norm(np.cross(b[tb[:, 1]]-b[tb[:, 0]], b[tb[:, 2]]-b[tb[:, 0]]), axis=1)==0).sum()),
                }
            report.append(row)
            print('Audited', frame+1, '/', len(metadata['pose_names']), name, flush=True)
    (bundle.parent/'contact-audit.json').write_text(json.dumps({
        'blender_version': bpy.app.version_string, 'analytical_checks': checks,
        'coordinate_precision': 'Blender float32 vertices, float64 narrow-phase arithmetic',
        'scope': 'Specified inter-mesh triangle pairs only; no self-intersection, enclosed-volume, swept-motion or clearance certification',
        'candidate_backend': 'Blender BVHTree overlap at zero epsilon',
        'rows': report, 'anatomical_acceptance': False, 'phase_exit_approved': False,
    }, indent=2)+'\n')


def export():
    import numpy as np
    import torch
    from signtranslator.avatar_render.makehuman import load_makehuman_rig
    from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
    from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
    from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units
    from signtranslator.avatar_render.rigged import apply_lbs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.intake_root, args.out_dir.resolve()
    rig = load_makehuman_rig(root/'makehuman')
    face = load_makehuman_face_units(root/'makehuman-face', rig)
    attachments = {'eyes': load_makehuman_eyes(root/'makehuman-eyes', rig), **load_makehuman_mouth(root/'makehuman-mouth', rig)}
    out.mkdir(parents=True, exist_ok=False)
    poses = [('TargetBind', torch.eye(3, dtype=torch.float64).repeat(len(rig.bone_names), 1, 1))]
    poses.extend((name, face.rotations(name)) for name in face.names)
    arrays = {name: [] for name in ['body', *attachments]}
    for name, rotations in poses:
        posed = rig.pose(rotations)
        arrays['body'].append(posed.vertices.numpy())
        for part, attachment in attachments.items():
            result = apply_lbs(attachment.rest_vertices, attachment.weights, posed.skin_transforms)
            if not torch.isfinite(result).all():
                raise ValueError('Attachment arithmetic is not finite')
            arrays[part].append(result.numpy())
    archive = out/'posed-meshes.npz'
    np.savez_compressed(archive, **{key: np.stack(value) for key, value in arrays.items()})
    faces = {'body': [f for f, g in zip(rig.faces, rig.face_groups, strict=True) if 'body' in g]}
    faces.update({key: attachment.faces for key, attachment in attachments.items()})
    (out/'catalogue.json').write_text(json.dumps({'pose_names': [p[0] for p in poses], 'faces': faces,
        'source_units': 'decimeters', 'observed_motion': False,
        'mesh_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, separators=(',', ':'))+'\n')
    command = [str(args.blender), '--background', '--factory-startup', '--python-exit-code', '1',
               '--python', str(Path(__file__).resolve()), '--', '--audit-bundle', str(archive)]
    with (out/'blender.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(out/'contact-audit.json')


if __name__ == '__main__':
    if '--' in sys.argv:
        custom = sys.argv[sys.argv.index('--')+1:]
        if len(custom) != 2 or custom[0] != '--audit-bundle':
            raise ValueError('Expected --audit-bundle PATH')
        blender_audit(Path(custom[1]))
    else:
        export()
