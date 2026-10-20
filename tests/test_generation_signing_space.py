"""Signing-space FK in torch matches the preview skeleton; the location loss is exact and differentiable."""
from dataclasses import replace

import numpy as np
import torch

from signtranslator.data.multichannel import collate_multichannel
from signtranslator.data_engineering.video_landmarks import BODY_LABELS, finger_joints
from signtranslator.generation.canonical_skeleton import skeleton_positions
from signtranslator.generation.signing_space import POINTS, location_loss, signing_space_points
from signtranslator.pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion
from signtranslator.pose.rotations import axis_angle_to_matrix, matrix_to_rotation_6d

LABELS = {"body": BODY_LABELS, "left_hand": tuple(j.name for j in finger_joints("Left")),
          "right_hand": tuple(j.name for j in finger_joints("Right"))}


def signer_state(seed=0, frames=12, amplitude=.6):
    rng = np.random.default_rng(seed)
    channels = {}
    for name, (kind, width, units) in CHANNELS.items():
        if name in LABELS:
            n = len(LABELS[name])
            axes = rng.normal(size=(frames, n, 3)) * amplitude
            values = (matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(axes))).numpy() if amplitude
                      else np.tile([1., 0., 0., 0., 1., 0.], (frames, n, 1)))
            valid, labels = np.ones((frames, n), bool), LABELS[name]
        else:
            values, valid, labels = np.zeros((frames, 1, width)), np.zeros((frames, 1), bool), (f"{name}0",)
        channels[name] = MotionChannel(values, valid, np.zeros_like(valid), valid.astype(float), labels, "f", units,
                                       "src", "a" * 64, "c", inference_method="estimate" if valid.any() else None)
    return MultichannelMotion(np.arange(frames) / 24., "clock", f"s{seed}", channels)


def tensors(state):
    return [torch.from_numpy(state.channels[n].values) for n in LABELS]


def test_torch_fk_equals_the_preview_skeleton_in_the_chest_frame():
    state = signer_state()
    joints, positions, _ = skeleton_positions(state)
    from signtranslator.generation.motion_metrics import _matrices
    body = state.channels["body"]
    chest = _matrices(body.values[:, 0]) @ _matrices(body.values[:, 1])          # Pelvis · Chest
    spine = positions[:, joints.index("Spine2")]
    names = [f"{side}Hand" if p == "Hand" else f"{side}Hand{p[:-3]}4"
             for side in ("Left", "Right") for p in ("Hand", "ThumbTip", "IndexTip", "MiddleTip", "RingTip", "PinkyTip")]
    expected = np.einsum("tji,tpj->tpi", chest, positions[:, [joints.index(n) for n in names]] - spine[:, None])
    points, chains = signing_space_points(*tensors(state), *LABELS.values())
    np.testing.assert_allclose(points.numpy(), expected, atol=1e-10)
    assert len(POINTS) == points.shape[-2] == len(chains) == 12
    assert chains[0] == (("body", "LeftUpperArm"), ("body", "LeftForearm"))


def test_location_loss_is_zero_on_the_reference_and_masks_invalid_chains():
    states = [signer_state(s) for s in range(3)]
    batch = collate_multichannel(states)
    prediction = {n: batch.channels[n].values.clone() for n in LABELS}
    loss, count = location_loss(prediction, batch)
    assert float(loss) == 0. and count == 3 * 12 * 12
    # invalidate one forearm everywhere in one clip: its wrist and five fingertips drop out
    body = states[0].channels["body"]
    valid = body.valid.copy(); valid[:, BODY_LABELS.index("LeftForearm")] = False
    states[0] = replace(states[0], channels={**states[0].channels, "body": replace(
        body, valid=valid, values=np.where(valid[..., None], body.values, 0.), confidence=valid.astype(float))})
    batch = collate_multichannel(states)
    _, count = location_loss(prediction, batch)            # predictions never carry zero payload
    assert count == 3 * 12 * 12 - 12 * 6


def test_location_loss_measures_hand_displacement_and_reaches_the_shoulder():
    batch = collate_multichannel([signer_state(0, amplitude=0.)])             # rest pose
    prediction = {n: batch.channels[n].values.clone().requires_grad_(True) for n in LABELS}
    shoulder = BODY_LABELS.index("LeftUpperArm")
    turned = matrix_to_rotation_6d(axis_angle_to_matrix(torch.tensor([0., 0., np.radians(10)], dtype=torch.float64)))
    body = prediction["body"].detach().clone()
    body[..., shoulder, :] = turned
    body.requires_grad_(True)
    loss, _ = location_loss({**prediction, "body": body}, batch, scale=1.)
    # Left wrist at the rest arm length .53 m moves by the chord 2·.53·sin(5°); Huber(δ=1) of d < 1 is d²/2
    points, _ = signing_space_points(body.detach(), prediction["left_hand"].detach(), prediction["right_hand"].detach(),
                                     *LABELS.values())
    reference, _ = signing_space_points(*(batch.channels[n].values for n in LABELS), *LABELS.values())
    wrist_move = float(torch.linalg.vector_norm(points[0, 0, 0] - reference[0, 0, 0]))
    np.testing.assert_allclose(wrist_move, 2 * .53 * np.sin(np.radians(5)), rtol=1e-9)
    loss.backward()
    assert float(body.grad[..., shoulder, :].abs().sum()) > 0
    assert float(body.grad[..., BODY_LABELS.index("RightUpperArm"), :].abs().sum()) == 0.


def test_generators_add_the_location_term_and_learn_from_it():
    from signtranslator.generation.canonical_state import channel_bindings_from_state
    from signtranslator.generation.motion_tokens import MotionVQ, MotionVQConfig, tokenizer_loss
    from signtranslator.generation.text_to_motion import TextToMotion, TextToMotionConfig, encode_texts, training_loss
    torch.manual_seed(0)
    states = [signer_state(s, frames=16, amplitude=.3) for s in range(2)]
    batch = collate_multichannel(states)
    tokens, mask = encode_texts(["a", "b"])
    trained = ("body", "left_hand", "right_hand")
    for objective in ("regression", "diffusion"):
        config = TextToMotionConfig(bindings=channel_bindings_from_state(states[0]), trained_channels=trained,
                                    d_model=32, heads=4, text_layers=1, motion_layers=1, feedforward=64, dropout=0.,
                                    objective=objective)
        model = TextToMotion(config)
        loss, terms, _ = training_loss(model, tokens, mask, batch)
        assert terms["location"] > 0
        off = replace(config, location_weight=0.)
        model.config = off
        _, terms_off, _ = training_loss(model, tokens, mask, batch)
        assert "location" not in terms_off
    vq = MotionVQ(MotionVQConfig(replace(config, objective="regression"), hidden=16, code_dim=8, codebook=16))
    _, terms = tokenizer_loss(vq, batch)
    assert terms["location"] > 0
    # the location gradient alone moves the decoder towards the reference hand positions
    config = replace(config, objective="regression")
    model = TextToMotion(config)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for _ in range(80):
        optimiser.zero_grad()
        loss, terms, _ = training_loss(model, tokens, mask, batch)
        history.append(terms["location"])
        loss.backward()
        optimiser.step()
    assert history[-1] < .5 * history[0]


def test_location_errors_are_zero_for_identical_states_and_grow_with_arm_error():
    from signtranslator.generation.signing_space import location_errors
    states = [signer_state(s) for s in range(3)]
    same = location_errors(states, states)
    assert same["wrist_cm"] == 0. and same["fingertip_cm"] == 0.
    other = [signer_state(s + 10) for s in range(3)]
    worse = location_errors(other, states)
    assert worse["wrist_cm"] > 5. and worse["fingertip_cm"] > worse["wrist_cm"] * .5
