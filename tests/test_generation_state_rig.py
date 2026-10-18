"""Declared state-to-rig posing against the pinned real rig asset.

Geometry and declaration contracts only: no calibration, retargeting acceptance
or signing fidelity is established here.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.avatar_render.makehuman import load_makehuman_rig
from signtranslator.data.governed_corpus import collate_governed_motion
from signtranslator.generation.state_rig import (
    BONE_LOCAL_AFTER_REST, ChannelBoneMapping, StateRigBinding, pose_state_frame,
    pose_state_sequence,
)
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d
from test_generation_canonical_state import build_model, corpus_fixture

ASSETS = (Path(__file__).resolve().parents[1] / "Sign Translator Stage Documentation"
          / "evidence/w1-w2-2026-09-26/source-intake/makehuman")
DRIVEN_BONE = "upperarm01.L"
ROOT_BONE = "root"


@pytest.fixture(scope="module")
def rig():
    return load_makehuman_rig(ASSETS)


def state_for(rig, *, rotation=None, translation=None, valid=True, observed=True, frames=2):
    """Canonical state whose element labels are the declared rig bone names."""
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        values = np.zeros((frames, 1, width), dtype=np.float64)
        if kind == "rotation_6d_columns":
            values[:] = [1, 0, 0, 0, 1, 0]
        elif kind == "unit_vector":
            values[:] = [1, 0, 0]
        if name == "body" and rotation is not None:
            values[:] = matrix_to_rotation_6d(rotation).numpy()
        if name == "root_translation" and translation is not None:
            values[:] = np.asarray(translation, dtype=np.float64)
        label = DRIVEN_BONE if name == "body" else (
            ROOT_BONE if name == "root_translation" else name)
        channels[name] = MotionChannel(
            values=values,
            valid=np.full((frames, 1), valid, dtype=bool),
            observed=np.full((frames, 1), observed and valid, dtype=bool),
            confidence=np.full((frames, 1), 1. if valid else 0., dtype=np.float64)
            if valid else np.zeros((frames, 1), dtype=np.float64),
            labels=(label,), coordinate_frame="fixture-right-handed", units=units,
            source_id="synthetic-fixture", source_sha256="b" * 64,
            convention="fixture-convention-v1",
            inference_method=None if observed else "fixture-inference")
        if not valid:
            channels[name] = replace(channels[name],
                                     values=np.zeros((frames, 1, width), dtype=np.float64))
    return MultichannelMotion(np.array([0., .1][:frames], dtype=np.float64),
                              "synthetic-clock", "fixture", channels)


def binding_for(rig, *, metres_per_source_unit=.1, **overrides):
    rest = tuple(name for name in rig.bone_names if name != DRIVEN_BONE)
    declared = dict(
        rotation_mappings=(ChannelBoneMapping("body", DRIVEN_BONE, DRIVEN_BONE),),
        translation_mappings=(ChannelBoneMapping("root_translation", ROOT_BONE, ROOT_BONE),),
        rest_bones=rest,
        unmapped_channels=tuple(
            name for name in CHANNELS if name not in ("body", "root_translation")),
        metres_per_source_unit=metres_per_source_unit,
        rotation_basis=BONE_LOCAL_AFTER_REST, asset_commit=rig.asset_commit)
    declared.update(overrides)
    return StateRigBinding(**declared)


def test_identity_declaration_reproduces_rest_geometry(rig):
    posed = pose_state_frame(rig, state_for(rig), binding_for(rig), 0)
    torch.testing.assert_close(posed.mesh.vertices, rig.rest_vertices, atol=1e-12, rtol=0.)
    assert posed.frame_index == 0 and posed.clock_seconds == 0.
    assert posed.posed_inferred_values is False
    assert posed.provenance()["accepted_provenance"] is False
    assert len(posed.binding_sha256) == 64


def test_declared_unit_conversion_is_applied_exactly(rig):
    metres = (.05, -.02, .01)
    state = state_for(rig, translation=metres)
    posed = pose_state_frame(rig, state, binding_for(rig, metres_per_source_unit=.1), 0)
    expected_translations = torch.zeros((len(rig.bone_names), 3), dtype=torch.float64)
    # Compute the oracle in float64: a float32 intermediate would differ from
    # the adapter's float64 conversion by about 1e-8 source units.
    expected_translations[rig.bone_names.index(ROOT_BONE)] = (
        torch.tensor(metres, dtype=torch.float64) / .1)
    reference = rig.pose(
        torch.eye(3, dtype=torch.float64).expand(len(rig.bone_names), 3, 3).clone(),
        local_translations=expected_translations)
    torch.testing.assert_close(posed.mesh.vertices, reference.vertices, atol=1e-12, rtol=0.)
    # A different declared scale must move the mesh differently.
    other = pose_state_frame(rig, state, binding_for(rig, metres_per_source_unit=1.), 0)
    assert not torch.allclose(other.mesh.vertices, posed.mesh.vertices)


def test_declared_rotation_drives_only_its_bone(rig):
    rotation = axis_angle_to_matrix(torch.tensor([.3, -.2, .1], dtype=torch.float64))
    posed = pose_state_frame(rig, state_for(rig, rotation=rotation), binding_for(rig), 0)
    expected = torch.eye(3, dtype=torch.float64).expand(len(rig.bone_names), 3, 3).clone()
    expected[rig.bone_names.index(DRIVEN_BONE)] = rotation
    reference = rig.pose(expected)
    torch.testing.assert_close(posed.mesh.vertices, reference.vertices, atol=1e-12, rtol=0.)
    assert not torch.allclose(posed.mesh.vertices, rig.rest_vertices)


def test_incomplete_declarations_are_rejected(rig):
    with pytest.raises(ValueError, match="driven or held at rest"):
        pose_state_frame(rig, state_for(rig),
                         binding_for(rig, rest_bones=tuple(
                             name for name in rig.bone_names
                             if name not in (DRIVEN_BONE, ROOT_BONE))), 0)
    with pytest.raises(ValueError, match="driving or unmapped"):
        binding_for(rig, unmapped_channels=("blink",))
    with pytest.raises(ValueError, match="rotation basis"):
        binding_for(rig, rotation_basis="assumed")
    with pytest.raises(ValueError, match="not a rotation channel"):
        binding_for(rig, rotation_mappings=(
            ChannelBoneMapping("blink", "blink", DRIVEN_BONE),))


def test_conflicting_declarations_are_rejected(rig):
    with pytest.raises(ValueError, match="two declared rotations"):
        binding_for(rig, rotation_mappings=(
            ChannelBoneMapping("body", DRIVEN_BONE, DRIVEN_BONE),
            ChannelBoneMapping("head_rotation", "head", DRIVEN_BONE)))
    with pytest.raises(ValueError, match="both driven and held at rest"):
        binding_for(rig, rest_bones=tuple(rig.bone_names))
    with pytest.raises(ValueError, match="both driving and unmapped"):
        binding_for(rig, unmapped_channels=tuple(
            name for name in CHANNELS if name != "root_translation"))
    with pytest.raises(ValueError, match="element cannot drive two bones"):
        binding_for(rig, rotation_mappings=(
            ChannelBoneMapping("body", DRIVEN_BONE, DRIVEN_BONE),
            ChannelBoneMapping("body", DRIVEN_BONE, "head")),
            rest_bones=tuple(name for name in rig.bone_names
                             if name not in (DRIVEN_BONE, "head")))


def test_unavailable_required_element_never_falls_back_to_bind_pose(rig):
    with pytest.raises(ValueError, match="unavailable at frame 0"):
        pose_state_frame(rig, state_for(rig, valid=False), binding_for(rig), 0)


def test_unobserved_values_require_an_explicit_opt_in(rig):
    state = state_for(rig, observed=False)
    with pytest.raises(ValueError, match="requires an explicit opt-in"):
        pose_state_frame(rig, state, binding_for(rig), 0)
    posed = pose_state_frame(rig, state, binding_for(rig), 0, allow_inferred=True)
    assert posed.posed_inferred_values is True
    with pytest.raises(ValueError, match="explicit boolean"):
        pose_state_frame(rig, state, binding_for(rig), 0, allow_inferred=1)


def test_frame_selection_and_asset_binding_are_checked(rig):
    state = state_for(rig)
    with pytest.raises(ValueError, match="existing frame"):
        pose_state_frame(rig, state, binding_for(rig), 5)
    with pytest.raises(ValueError, match="asset commit"):
        pose_state_frame(rig, state, binding_for(rig, asset_commit="0" * 40), 0)
    with pytest.raises(ValueError, match="absent from the state"):
        pose_state_frame(rig, state, binding_for(rig, translation_mappings=(
            ChannelBoneMapping("root_translation", "absent-label", ROOT_BONE),)), 0)


def test_binding_identity_changes_with_the_declaration(rig):
    first = binding_for(rig).sha256()
    assert first != binding_for(rig, metres_per_source_unit=.2).sha256()
    assert first == binding_for(rig).sha256()


def test_generated_state_poses_frame_by_frame_through_the_declared_binding(tmp_path, rig):
    """The honest chain: reviewed SIR to generated state to posed rig frames."""
    vocabulary, alphabet, corpus = corpus_fixture(tmp_path)
    batch = next(iter(DataLoader(corpus.split("train"), batch_size=1, shuffle=False,
                                 collate_fn=collate_governed_motion)))
    model = build_model(vocabulary, alphabet)
    torch.manual_seed(41)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.add_(torch.randn_like(parameter) * .05)
    generated = model.generate_canonical_state(batch, inference_method="w4-baseline")[0]
    labels = {name: (DRIVEN_BONE if name == "body" else
                     ROOT_BONE if name == "root_translation" else name)
              for name in CHANNELS}
    relabelled = replace(generated, channels={
        name: replace(channel, labels=(labels[name],))
        for name, channel in generated.channels.items()})
    binding = binding_for(rig)
    # Generated values are valid but unobserved, so posing needs the opt-in.
    with pytest.raises(ValueError, match="requires an explicit opt-in"):
        pose_state_frame(rig, relabelled, binding, 0)
    frames = list(pose_state_sequence(rig, relabelled, binding, allow_inferred=True))
    assert len(frames) == len(generated.timestamps)
    assert [frame.frame_index for frame in frames] == list(range(len(frames)))
    assert [frame.clock_seconds for frame in frames] == list(generated.timestamps)
    for frame in frames:
        assert bool(torch.isfinite(frame.mesh.vertices).all())
        assert frame.posed_inferred_values is True
        assert frame.accepted_provenance is False
