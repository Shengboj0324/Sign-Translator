"""Actual-asset invariants and independent articulated-motion oracles."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.pose.rotations import axis_angle_to_matrix


ASSETS = (Path(__file__).resolve().parents[1] / "Sign Translator Stage Documentation"
          / "evidence/w1-w2-2026-09-26/source-intake/makehuman")


@pytest.fixture(scope="module")
def rig():
    return load_makehuman_rig(ASSETS)


def identity(rig):
    return torch.eye(3, dtype=rig.rest_vertices.dtype).repeat(len(rig.bone_names), 1, 1)


def test_actual_asset_rest_and_full_influence_contract(rig):
    assert rig.rest_vertices.shape == (19158, 3)
    assert len(rig.bone_names) == 163 and len(rig.faces) == 18486
    counts = (rig.weights > 0).sum(1)
    assert int(counts.max()) == 12
    assert int((counts > 4).sum()) == 3665
    assert "body" in {g for groups in rig.face_groups for g in groups}
    assert "helper-l-eye" in {g for groups in rig.face_groups for g in groups}
    result = rig.pose(identity(rig))
    torch.testing.assert_close(result.vertices, rig.rest_vertices, atol=1e-12, rtol=0)
    torch.testing.assert_close(result.skin_transforms, torch.eye(4, dtype=torch.float64).expand(163, 4, 4), atol=1e-12, rtol=0)


def test_root_world_transform_is_applied_once_and_has_translation_gradient(rig):
    rotation = axis_angle_to_matrix(torch.tensor([0.2, -0.4, 0.3], dtype=torch.float64))
    translation = torch.tensor([1.2, -0.3, 0.8], dtype=torch.float64, requires_grad=True)
    world = torch.cat((torch.cat((rotation, translation[:, None]), 1), translation.new_tensor([[0, 0, 0, 1]])), 0)
    result = rig.pose(identity(rig), world_from_source=world)
    expected = rig.rest_vertices @ rotation.T + translation
    torch.testing.assert_close(result.vertices, expected, atol=1e-12, rtol=0)
    gradient, = torch.autograd.grad(result.vertices.mean(0).sum(), translation)
    torch.testing.assert_close(gradient, torch.ones_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", ["finger2-1.L", "finger2-3.L"])
def test_articulation_matches_independent_world_pivot_oracle(rig, name):
    bone = rig.bone_names.index(name)
    angle = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    axis_angle = torch.stack((angle, angle*0, angle*0))
    local = axis_angle_to_matrix(axis_angle)
    rotations = identity(rig)
    rotations[bone] = local
    result = rig.pose(rotations)
    affected = {bone}
    for i, parent in enumerate(rig.parents):
        if parent in affected:
            affected.add(i)
    basis = rig.rest_globals[bone, :3, :3]
    pivot = rig.rest_globals[bone, :3, 3]
    world_rotation = basis @ local @ basis.T
    rotated = (rig.rest_vertices-pivot) @ world_rotation.T + pivot
    influence = rig.weights[:, sorted(affected)].sum(1, keepdim=True)
    expected = rig.rest_vertices + influence*(rotated-rig.rest_vertices)
    torch.testing.assert_close(result.vertices, expected, atol=1e-12, rtol=0)
    assert float((expected-rig.rest_vertices).detach().abs().max()) > 1e-4
    actual_gradient, = torch.autograd.grad(result.vertices.square().sum(), angle, retain_graph=True)
    oracle_gradient, = torch.autograd.grad(expected.square().sum(), angle)
    torch.testing.assert_close(actual_gradient, oracle_gradient, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("bad", ["nan", "shear", "reflection", "shape", "dtype"])
def test_pose_refuses_invalid_rotations(rig, bad):
    rotations = identity(rig)
    if bad == "nan":
        rotations[0, 0, 0] = float("nan")
    elif bad == "shear":
        rotations[0, 0, 1] = 0.4  # determinant +1 alone is insufficient.
    elif bad == "reflection":
        rotations[0, 0, 0] = -1
    elif bad == "shape":
        rotations = rotations[:-1]
    else:
        rotations = rotations.float()
    with pytest.raises(ValueError):
        rig.pose(rotations)


@pytest.mark.parametrize("bad", ["weights", "parent", "rest", "world"])
def test_mutated_contracts_are_rechecked_at_pose(rig, bad):
    candidate = rig
    world = torch.eye(4, dtype=torch.float64)
    if bad == "weights":
        weights = rig.weights.clone()
        weights[0] *= 0.5
        candidate = replace(rig, weights=weights)
    elif bad == "parent":
        candidate = replace(rig, parents=(-1, 1, *rig.parents[2:]))
    elif bad == "rest":
        rest = rig.rest_globals.clone()
        rest[0, 3, 0] = 0.1
        candidate = replace(rig, rest_globals=rest)
    else:
        world[3, 3] = 0
    with pytest.raises(ValueError):
        candidate.pose(identity(rig), world_from_source=world)


def test_asset_identity_is_not_trusted_from_external_manifest(tmp_path):
    mesh = tmp_path / "makehuman/data/3dobjs/base.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\n")
    with pytest.raises(ValueError, match="Asset identity mismatch"):
        load_makehuman_rig(tmp_path)


def test_explicit_rest_rebuilds_heads_and_frames_without_aliasing(rig):
    import json
    coords = rig.rest_vertices * torch.tensor([1.2, 0.85, 1.1], dtype=torch.float64)
    coords += torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    changed = load_makehuman_rig(ASSETS, rest_vertices=coords)
    spec = json.loads((ASSETS/'makehuman/data/rigs/default.mhskel').read_text())
    for i, name in enumerate(changed.bone_names):
        bone = spec['bones'][name]
        head = coords[spec['joints'][bone['head']]].mean(0)
        tail = coords[spec['joints'][bone['tail']]].mean(0)
        torch.testing.assert_close(changed.rest_globals[i, :3, 3], head, atol=1e-12, rtol=0)
        expected_y = (tail-head)/torch.linalg.vector_norm(tail-head)
        torch.testing.assert_close(changed.rest_globals[i, :3, 1], expected_y, atol=1e-12, rtol=0)
    assert not torch.allclose(changed.rest_globals[:, :3, :3], rig.rest_globals[:, :3, :3])
    torch.testing.assert_close(changed.weights, rig.weights, atol=0, rtol=0)
    result = changed.pose(identity(changed))
    torch.testing.assert_close(result.vertices, coords, atol=1e-12, rtol=0)
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = axis_angle_to_matrix(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64))
    world[:3, 3] = torch.tensor([1., 2., -1.], dtype=torch.float64)
    torch.testing.assert_close(changed.pose(identity(changed), world_from_source=world).vertices,
                               coords@world[:3, :3].T+world[:3, 3], atol=1e-12, rtol=0)
    coords.zero_()
    torch.testing.assert_close(changed.pose(identity(changed)).vertices, result.vertices, atol=1e-12, rtol=0)


@pytest.mark.parametrize('kind', ['shape', 'float32', 'nan', 'trainable', 'degenerate'])
def test_explicit_rest_rejects_invalid_or_degenerate_geometry(rig, kind):
    coords = rig.rest_vertices.clone()
    if kind == 'shape': coords = coords[:-1]
    elif kind == 'float32': coords = coords.float()
    elif kind == 'nan': coords[0, 0] = float('nan')
    elif kind == 'trainable': coords.requires_grad_()
    elif kind == 'degenerate': coords.zero_()
    with pytest.raises(ValueError):
        load_makehuman_rig(ASSETS, rest_vertices=coords)


def test_explicit_rest_rejects_stale_attachment_binding(rig):
    from signtranslator.avatar_render.makehuman_eyes import load_makehuman_eyes
    eyes = load_makehuman_eyes(ASSETS.parent/'makehuman-eyes', rig)
    changed = load_makehuman_rig(ASSETS, rest_vertices=rig.rest_vertices*1.1)
    with pytest.raises(ValueError, match='Rest rig changed'):
        replace(eyes, rig=changed).pose(identity(changed))
    fitted = load_makehuman_eyes(ASSETS.parent/'makehuman-eyes', changed)
    torch.testing.assert_close(fitted.pose(identity(changed)), fitted.rest_vertices, atol=1e-12, rtol=0)


def test_local_head_translation_matches_subtree_skin_oracle_and_gradient(rig):
    bone = rig.bone_names.index('head')
    delta = torch.tensor([0.02, -0.03, 0.01], dtype=torch.float64, requires_grad=True)
    translations = torch.zeros(len(rig.bone_names), 3, dtype=torch.float64)
    translations[bone] = delta
    rotation = identity(rig)
    # Own rotation must not rotate its own rest-local translation a second time.
    rotation[bone] = axis_angle_to_matrix(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
    baseline = rig.pose(rotation)
    actual = rig.pose(rotation, local_translations=translations)
    descendants = {bone}
    for i, parent in enumerate(rig.parents):
        if parent in descendants: descendants.add(i)
    support = rig.weights[:, sorted(descendants)].sum(1)
    world_delta = rig.rest_globals[bone, :3, :3] @ delta
    torch.testing.assert_close(actual.vertices-baseline.vertices, support[:, None]*world_delta,
                               atol=1e-12, rtol=0)
    gradient, = torch.autograd.grad(actual.vertices.sum(), delta)
    expected = support.sum()*(rig.rest_globals[bone, :3, :3].T@torch.ones(3,dtype=torch.float64))
    torch.testing.assert_close(gradient, expected, atol=1e-9, rtol=1e-12)


def test_parent_rotation_carries_child_translation_once(rig):
    bone = rig.bone_names.index('head')
    parent = rig.parents[bone]
    rotations = identity(rig)
    rotations[parent] = axis_angle_to_matrix(torch.tensor([0.2, 0.1, -0.1], dtype=torch.float64))
    translations = torch.zeros(len(rig.bone_names), 3, dtype=torch.float64)
    translations[bone] = torch.tensor([0.02, 0.01, -0.03], dtype=torch.float64)
    baseline = rig.pose(rotations)
    parent_rest = rig.rest_globals[parent, :3, :3]
    parent_world_rotation = parent_rest @ rotations[parent] @ parent_rest.T
    displacement = parent_world_rotation @ rig.rest_globals[bone, :3, :3] @ translations[bone]
    actual = rig.pose(rotations, local_translations=translations)
    torch.testing.assert_close(actual.bone_globals[bone, :3, 3]-baseline.bone_globals[bone, :3, 3],
                               displacement, atol=1e-12, rtol=0)
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = axis_angle_to_matrix(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))
    world[:3, 3] = torch.tensor([1., 2., 3.], dtype=torch.float64)
    transformed = rig.pose(rotations, local_translations=translations, world_from_source=world)
    torch.testing.assert_close(transformed.vertices, actual.vertices@world[:3, :3].T+world[:3, 3],
                               atol=1e-12, rtol=0)


@pytest.mark.parametrize('kind', ['shape', 'float32', 'nan'])
def test_local_translation_domain(rig, kind):
    t = torch.zeros(len(rig.bone_names), 3, dtype=torch.float64)
    if kind == 'shape': t=t[:-1]
    elif kind == 'float32': t=t.float()
    else: t[0, 0]=float('nan')
    with pytest.raises(ValueError): rig.pose(identity(rig), local_translations=t)
