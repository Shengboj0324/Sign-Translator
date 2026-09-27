from pathlib import Path

import pytest
import torch

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_mouth import load_makehuman_mouth
from signtranslator.pose.rotations import axis_angle_to_matrix

ROOT = (Path(__file__).resolve().parents[1]/"Sign Translator Stage Documentation"
        /"evidence/w1-w2-2026-09-26/source-intake")


@pytest.fixture(scope="module")
def mouth():
    return load_makehuman_mouth(ROOT/"makehuman-mouth", load_makehuman_rig(ROOT/"makehuman"))


@pytest.mark.parametrize('kind', ['teeth', 'tongue'])
def test_mouth_rest_and_global_rigid_oracle(mouth, kind):
    attachment = mouth[kind]
    rotations = torch.eye(3, dtype=torch.float64).repeat(len(attachment.rig.bone_names), 1, 1)
    torch.testing.assert_close(attachment.pose(rotations), attachment.rest_vertices, atol=1e-12, rtol=0)
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = axis_angle_to_matrix(torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64))
    world[:3, 3] = torch.tensor([1.1, -0.3, 0.8], dtype=torch.float64)
    expected = attachment.rest_vertices @ world[:3, :3].T + world[:3, 3]
    torch.testing.assert_close(attachment.pose(rotations, world_from_source=world), expected, atol=1e-12, rtol=0)


def test_teeth_upper_fixed_lower_rigid_jaw_and_gradient(mouth):
    teeth = mouth['teeth']
    rig = teeth.rig
    jaw = rig.bone_names.index('jaw')
    head = rig.bone_names.index('head')
    lower = teeth.weights[:, jaw] > 0
    upper = teeth.weights[:, head] > 0
    assert int(lower.sum()) == 1884 and int(upper.sum()) == 1984
    assert torch.all(lower ^ upper)
    assert torch.all(teeth.weights[lower, jaw] == 1)
    assert torch.all(teeth.weights[upper, head] == 1)
    angle = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    local = axis_angle_to_matrix(torch.stack((angle, angle*0, angle*0)))
    rotations = torch.eye(3, dtype=torch.float64).repeat(len(rig.bone_names), 1, 1)
    rotations[jaw] = local
    actual = teeth.pose(rotations)
    basis, pivot = rig.rest_globals[jaw, :3, :3], rig.rest_globals[jaw, :3, 3]
    rotated = (teeth.rest_vertices-pivot) @ (basis @ local @ basis.T).T + pivot
    expected = torch.where(lower[:, None], rotated, teeth.rest_vertices)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    a, = torch.autograd.grad(actual.square().sum(), angle, retain_graph=True)
    b, = torch.autograd.grad(expected.square().sum(), angle)
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_tongue_direct_vertex_binding_and_teeth_signed_coefficients(mouth):
    text = (ROOT/'makehuman-mouth/tongue/tongue01/tongue01.mhclo').read_text()
    indices = [int(line) for line in text.split('verts 0', 1)[1].splitlines() if line.strip()]
    tongue = mouth['tongue']
    assert len(indices) == 226 and len(tongue.faces) == 224
    torch.testing.assert_close(tongue.rest_vertices, tongue.rig.rest_vertices[indices], atol=0, rtol=0)
    assert tongue.negative_geometry_coefficients == 0
    assert mouth['teeth'].negative_geometry_coefficients == 2646


def test_mouth_asset_tampering_rejected(tmp_path, mouth):
    mesh = tmp_path/'teeth/teeth_base/teeth_base.obj'
    mesh.parent.mkdir(parents=True)
    mesh.write_text('v 0 0 0\n')
    with pytest.raises(ValueError, match='Mouth asset identity mismatch'):
        load_makehuman_mouth(tmp_path, mouth['teeth'].rig)
