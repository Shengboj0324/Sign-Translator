from dataclasses import replace
from pathlib import Path
import math
import re

import pytest
import torch

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_face import load_makehuman_face_units

ROOT = (Path(__file__).resolve().parents[1]/'Sign Translator Stage Documentation'
        /'evidence/w1-w2-2026-09-26/source-intake')


@pytest.fixture(scope='module')
def catalogue():
    return load_makehuman_face_units(ROOT/'makehuman-face', load_makehuman_rig(ROOT/'makehuman'))


def test_catalogue_is_named_poses_not_observed_time_series(catalogue):
    assert len(catalogue.names) == 60
    assert catalogue.local_rotations.shape == (60, 163, 3, 3)
    assert catalogue.source_frame_time_metadata == 0.041667
    assert catalogue.source_rest_max_rotation_degrees == pytest.approx(0.000164)
    assert catalogue.source_rest_max_coordinate_discrepancy == pytest.approx(0.45375)
    assert not torch.equal(catalogue.rotations('Rest'), torch.eye(3, dtype=torch.float64).expand(163, 3, 3))


@pytest.mark.parametrize('pose_name', ['LeftUpperLidClosed', 'RightUpperLidClosed', 'JawDrop', 'TongueOut'])
def test_raw_euler_trigonometric_oracle_in_target_world_basis(catalogue, pose_name):
    raw = (ROOT/'makehuman-face/makehuman/data/poseunits/face-poseunits.bvh').read_text()
    hierarchy, motion = raw.split('MOTION', 1)
    names = re.findall(r'(?:ROOT|JOINT)\s+([^\s{}]+)', hierarchy)
    row = [float(v) for v in motion.strip().splitlines()[2+catalogue.names.index(pose_name)].split()]
    conversion = torch.tensor([[1., 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float64)
    actual = catalogue.rotations(pose_name)
    for i, name in enumerate(catalogue.rig.bone_names):
        j = names.index(name)
        # Root translation occupies the first three columns; every joint has XYZ angles.
        x, y, z = [math.radians(v) for v in row[3+3*j:6+3*j]]
        cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
        sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
        xyz = torch.tensor([[cy*cz, -cy*sz, sy],
                            [cx*sz+sx*sy*cz, cx*cz-sx*sy*sz, -sx*cy],
                            [sx*sz-cx*sy*cz, sx*cz+cx*sy*sz, cx*cy]], dtype=torch.float64)
        basis = catalogue.rig.rest_globals[i, :3, :3]
        reconstructed_world = basis @ actual[i] @ basis.T
        torch.testing.assert_close(reconstructed_world, conversion @ xyz @ conversion.T, atol=2e-14, rtol=0)


def test_invalid_names_mutation_and_changed_binding_refused(catalogue):
    with pytest.raises(ValueError, match='Unknown authored'):
        catalogue.rotations('invented-blink')
    changed = replace(catalogue.rig, rest_vertices=catalogue.rig.rest_vertices+0.1)
    with pytest.raises(ValueError, match='Rest rig changed'):
        replace(catalogue, rig=changed).rotations('Rest')
    rotations = catalogue.local_rotations.clone()
    rotations[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        replace(catalogue, local_rotations=rotations).rotations('Rest')
    returned = catalogue.rotations('Rest')
    returned.zero_()
    assert torch.count_nonzero(catalogue.rotations('Rest')) > 0


def test_face_asset_identity_refused(tmp_path, catalogue):
    path = tmp_path/'makehuman/data/poseunits/face-poseunits.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='Face asset identity mismatch'):
        load_makehuman_face_units(tmp_path, catalogue.rig)
