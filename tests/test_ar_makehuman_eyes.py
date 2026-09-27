from dataclasses import replace
from pathlib import Path

import pytest
import torch

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
from signtranslator.pose.rotations import axis_angle_to_matrix

ROOT = (Path(__file__).resolve().parents[1]/"Sign Translator Stage Documentation"
        /"evidence/w1-w2-2026-09-26/source-intake")


@pytest.fixture(scope="module")
def eyes():
    return load_makehuman_eyes(ROOT/"makehuman-eyes", load_makehuman_rig(ROOT/"makehuman"))


def identity(eyes):
    return torch.eye(3,dtype=torch.float64).repeat(len(eyes.rig.bone_names),1,1)


def test_eye_attachment_inventory_and_rest(eyes):
    assert eyes.rest_vertices.shape == (1064,3)
    assert len(eyes.faces) == 1020 and len(eyes.texcoords) == 808
    assert eyes.negative_geometry_coefficients == 102
    # Authoring OBJ y is around 15; fitting must place eyes at the actual head.
    assert 7.1 < float(eyes.rest_vertices[:,1].min()) < 7.2
    assert 7.4 < float(eyes.rest_vertices[:,1].max()) < 7.5
    torch.testing.assert_close(eyes.pose(identity(eyes)),eyes.rest_vertices,atol=1e-12,rtol=0)


def test_fitted_eyes_follow_common_world_transform(eyes):
    rotation=axis_angle_to_matrix(torch.tensor([0.23,-0.51,0.17],dtype=torch.float64))
    world=torch.eye(4,dtype=torch.float64)
    world[:3,:3]=rotation
    world[:3,3]=torch.tensor([1.7,-0.8,2.1],dtype=torch.float64)
    expected=eyes.rest_vertices@rotation.T+world[:3,3]
    torch.testing.assert_close(eyes.pose(identity(eyes),world_from_source=world),expected,atol=1e-12,rtol=0)


def test_single_eye_motion_and_gradient_match_pivot_oracle(eyes):
    rig=eyes.rig
    i=rig.bone_names.index('eye.L')
    assert i not in rig.parents  # This oracle is for a leaf, so no descendants.
    angle=torch.tensor(0.15,dtype=torch.float64,requires_grad=True)
    local=axis_angle_to_matrix(torch.stack((angle*0,angle,angle*0)))
    rotations=identity(eyes)
    rotations[i]=local
    actual=eyes.pose(rotations)
    basis=rig.rest_globals[i,:3,:3]
    pivot=rig.rest_globals[i,:3,3]
    global_rotation=basis@local@basis.T
    rotated=(eyes.rest_vertices-pivot)@global_rotation.T+pivot
    expected=eyes.rest_vertices+eyes.weights[:,i,None]*(rotated-eyes.rest_vertices)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
    unaffected=eyes.weights[:,i]==0
    assert int(unaffected.sum())==532
    torch.testing.assert_close(actual[unaffected],eyes.rest_vertices[unaffected],atol=1e-12,rtol=0)
    a,=torch.autograd.grad(actual.square().sum(),angle,retain_graph=True)
    b,=torch.autograd.grad(expected.square().sum(),angle)
    torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)


def test_changed_rest_binding_refused(eyes):
    modified=replace(eyes.rig,rest_vertices=eyes.rig.rest_vertices+0.1)
    invalid=replace(eyes,rig=modified)
    with pytest.raises(ValueError,match='Rest rig changed'):
        invalid.pose(identity(eyes))


def test_negative_or_unscaled_skin_weights_refused(eyes):
    for weights in (-eyes.weights,eyes.weights*2):
        with pytest.raises(ValueError,match='partition of unity'):
            replace(eyes,weights=weights).pose(identity(eyes))


def test_unbound_eye_asset_rejected(tmp_path,eyes):
    proxy=tmp_path/'makehuman/data/eyes/high-poly/high-poly.mhclo'
    proxy.parent.mkdir(parents=True)
    proxy.write_text('verts 0\n')
    with pytest.raises(ValueError,match='Eye asset identity mismatch'):
        load_makehuman_eyes(tmp_path,eyes.rig)


def test_attachment_follows_rest_local_head_translation(eyes):
    rig=eyes.rig
    bone=rig.bone_names.index('head')
    translations=torch.zeros(len(rig.bone_names),3,dtype=torch.float64)
    translations[bone]=torch.tensor([0.02,-0.03,0.01],dtype=torch.float64)
    descendants={bone}
    for i,parent in enumerate(rig.parents):
        if parent in descendants: descendants.add(i)
    support=eyes.weights[:,sorted(descendants)].sum(1)
    expected=eyes.rest_vertices+support[:,None]*(rig.rest_globals[bone,:3,:3]@translations[bone])
    torch.testing.assert_close(eyes.pose(identity(eyes),local_translations=translations),expected,atol=1e-12,rtol=0)
