"""Tokens v2 vectorised training step = the per-part, per-joint implementation it replaced.

The reference functions are verbatim copies of commit 5819cba (``reference_`` prefix;
the quantiser reference is :class:`ResidualQuantizer`, run part by part as
``MotionRVQ.forward`` did). The vectorised step must give identical codes and the same
losses, gradients and codebooks up to float rounding (1e-6).
"""
import copy
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from signtranslator.data.multichannel import MotionBatch, collate_multichannel
from signtranslator.data_engineering.video_landmarks import BODY_JOINTS, finger_joints
from signtranslator.generation.canonical_skeleton import _FINGER_LENGTHS, _KNUCKLES, _LENGTH, _SHOULDER_HALF
from signtranslator.generation.canonical_state import channel_loss_per_example
from signtranslator.generation.motion_rvq import MotionRVQ, rvq_loss
from signtranslator.generation.signing_space import (
    FINGERS, MOTION_SCALES, _vector, applicable, skeleton_motion_loss, skeleton_points,
)
from signtranslator.generation.text_to_motion import LossScales, diffusion_targets, unit_to_raw
from signtranslator.pose.rotations import rotation_6d_to_matrix
from tests.test_generation_motion_rvq import small
from tests.test_generation_signing_space import LABELS, signer_state

TOLERANCE = dict(rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------------
# reference implementation (commit 5819cba)


def reference_skeleton_points(body6: torch.Tensor, left6: torch.Tensor, right6: torch.Tensor,
                    body_labels, left_labels, right_labels):
    """Chest-frame positions (…, 44, 3) of every arm and hand point, their names and chains.

    Per side: elbow, wrist, then for each finger its knuckle (MCP), PIP, DIP and tip —
    the same forward kinematics and nominal lengths as :func:`signing_space_points`,
    which is the wrist-and-tip subset of these points.
    """
    index = {("body", n): k for k, n in enumerate(body_labels)}
    index.update({("left_hand", n): k for k, n in enumerate(left_labels)})
    index.update({("right_hand", n): k for k, n in enumerate(right_labels)})
    tensors = {"body": body6, "left_hand": left6, "right_hand": right6}
    rotation = lambda channel, name: rotation_6d_to_matrix(tensors[channel][..., index[(channel, name)], :])
    like = body6
    eye = torch.eye(3, dtype=like.dtype, device=like.device).expand(*body6.shape[:-2], 3, 3)
    world = {"Chest": eye}
    points, names, chains = [], [], []
    for side, sign in (("Left", 1.), ("Right", -1.)):
        channel = f"{side.lower()}_hand"
        start = _vector([sign * _SHOULDER_HALF, 0., 0.], like).expand(*body6.shape[:-2], 3)
        chain = []
        for joint, part, point in ((f"{side}UpperArm", "UpperArm", f"{side}Elbow"),
                                   (f"{side}Forearm", "Forearm", f"{side}Wrist")):
            spec = next(j for j in BODY_JOINTS if j.name == joint)
            world[joint] = world[spec.parent] @ rotation("body", joint)
            start = start + _LENGTH[part] * (world[joint] @ _vector(spec.rest, like))
            chain.append(("body", joint))
            points.append(start); names.append(point); chains.append(tuple(chain))
        wrist = start
        hand = world[f"{side}Forearm"] @ rotation("body", f"{side}Hand")
        hand_chain = chain + [("body", f"{side}Hand")]
        for finger in FINGERS:
            x, y = _KNUCKLES[finger]
            position = wrist + hand @ _vector([sign * x, y, 0.], like)
            points.append(position); names.append(f"{side}{finger}MCP"); chains.append(tuple(hand_chain))
            frame, finger_chain = hand, list(hand_chain)
            for segment, label in zip((1, 2, 3), ("PIP", "DIP", "Tip")):
                spec = next(j for j in finger_joints(side) if j.name == f"{side}Hand{finger}{segment}")
                frame = frame @ rotation(channel, spec.name)
                position = position + _FINGER_LENGTHS[segment - 1] * (frame @ _vector(spec.rest, like))
                finger_chain.append((channel, spec.name))
                points.append(position); names.append(f"{side}{finger}{label}"); chains.append(tuple(finger_chain))
    return torch.stack(points, dim=-2), tuple(names), tuple(chains)



def reference_skeleton_motion_loss(prediction: dict, motion, *, scales=MOTION_SCALES) -> dict:
    """Huber position, velocity and acceleration losses of all 44 arm/hand points (chest frame).

    Each term is Huber(δ=1) of the error norm divided by its scale, averaged over the
    points whose whole chain is valid in the reference on every frame the difference
    uses (one, two or three consecutive valid frames). Velocities and accelerations are
    finite differences over the batch's own clock, so a generator is held to the
    reference's speed and smoothness, not only its poses. Returns {} if no point is
    supported.
    """
    names = ("body", "left_hand", "right_hand")
    labels = [motion.channels[n].layout.labels for n in names]
    dtype = prediction["body"].dtype
    identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=dtype, device=prediction["body"].device)
    reference = [torch.where(motion.channels[n].valid[..., None], motion.channels[n].values.to(dtype), identity)
                 for n in names]
    predicted, _, chains = reference_skeleton_points(*(prediction[n] for n in names), *labels)
    target, _, _ = reference_skeleton_points(*reference, *labels)
    target = target.detach()
    channels = {n: (motion.channels[n].valid, tuple(motion.channels[n].layout.labels)) for n in names}
    support = reference_support(channels, chains, motion.frame_valid)                  # (B, T, P)
    t = motion.timestamps.to(dtype)
    out = {}

    def huber(error, mask, scale):
        if not bool(mask.any()):
            return None
        d = torch.linalg.vector_norm(error, dim=-1)[mask] / scale
        return F.huber_loss(d, torch.zeros_like(d), delta=1., reduction="mean")

    position = huber(predicted - target, support, scales["position"])
    if position is not None:
        out["position"] = position
    if predicted.shape[1] >= 2:
        dt = (t[:, 1:] - t[:, :-1]).clamp_min(1e-3)[..., None, None]
        vp, vt = (predicted[:, 1:] - predicted[:, :-1]) / dt, (target[:, 1:] - target[:, :-1]) / dt
        both = support[:, 1:] & support[:, :-1]
        velocity = huber(vp - vt, both, scales["velocity"])
        if velocity is not None:
            out["velocity"] = velocity
        if predicted.shape[1] >= 3:
            dt2 = ((t[:, 2:] - t[:, :-2]) / 2).clamp_min(1e-3)[..., None, None]
            ap, at = (vp[:, 1:] - vp[:, :-1]) / dt2, (vt[:, 1:] - vt[:, :-1]) / dt2
            three = both[:, 1:] & both[:, :-1]
            acceleration = huber(ap - at, three, scales["acceleration"])
            if acceleration is not None:
                out["acceleration"] = acceleration
    return out


def reference_support(channels: dict, chains, frame_valid: torch.Tensor) -> torch.Tensor:
    """(B, T, 12) reference validity of every point's whole chain on valid frames."""
    masks = []
    for chain in chains:
        ok = frame_valid.clone()
        for channel, label in chain:
            ok = ok & channels[channel][0][..., channels[channel][1].index(label)]
        masks.append(ok)
    return torch.stack(masks, dim=-1)


def reference_forward(model, unit, frame_valid, stages=None):
    """``MotionRVQ.forward`` as it was: each part's quantiser and EMA update in turn."""
    latents = model.encode(unit)
    quantized, indices, inputs = {}, {}, {}
    for part, _ in model.parts:
        quantized[part], indices[part], inputs[part] = model.quantizers[part](latents[part], stages)
    codes = torch.stack([indices[part] for part, _ in model.parts], dim=-1).flatten(-2)
    valid = model.token_valid(frame_valid)
    if model.training:
        per_part = model.split_codes(codes)
        for part, _ in model.parts:
            model.quantizers[part].update(inputs[part], per_part[part], valid)
    straight = {part: latents[part] + (quantized[part] - latents[part]).detach() for part, _ in model.parts}
    return model.decode_latents(straight, unit.shape[1]), latents, quantized, codes, valid


def reference_rvq_loss(model: MotionRVQ, motion: MotionBatch, *, stages: int | None = None):
    """Per-channel reconstruction loss + skeleton position/velocity/acceleration + commitment.

    In training, quantiser dropout draws the number of stages; ``stages`` forces it.
    """
    config = model.config
    if stages is None and model.training and torch.rand(()) < config.quantizer_dropout:
        stages = int(torch.randint(1, config.stages + 1, ()))
    unit = diffusion_targets(config.motion, motion)
    reconstruction, latents, quantized, codes, valid = reference_forward(model, unit, motion.frame_valid, stages)
    raw = unit_to_raw(config.motion, reconstruction.double())
    scales = LossScales(**config.motion.loss_config)
    total = torch.zeros((), dtype=torch.float64, device=unit.device)
    terms = {}
    for name in config.motion.trained_channels:
        values, support, _ = channel_loss_per_example(name, raw[name], motion.channels[name], motion.frame_valid,
                                                      config=scales, timestamps=motion.timestamps, allow_inferred=True)
        if bool(support.any()):
            term = values[support].mean()
            terms[name] = float(term.detach())
            total = total + term
    if applicable(config.motion.bindings, config.motion.trained_channels):
        # float64 was needed for the SO(3) projection (unit_to_raw); the projected 6D columns are
        # orthonormal, so the forward kinematics runs in float32 (GPUs run float64 at a fraction of the rate).
        skeleton = reference_skeleton_motion_loss({n: raw[n].float() for n in ("body", "left_hand", "right_hand")}, motion)
        for key, value in skeleton.items():
            terms[f"skeleton_{key}"] = float(value.detach())
            total = total + config.skeleton_weights[key] * value.to(total.dtype)
    commitment = torch.stack([((latents[p] - quantized[p].detach()) ** 2).mean(-1)[valid].mean()
                              for p, _ in model.parts]).mean()
    terms["commitment"] = float(commitment.detach())
    terms["stages"] = float(stages or config.stages)
    per_part = model.split_codes(codes)
    for p, _ in model.parts:
        terms[f"codes_used_{p}"] = float(per_part[p][..., 0][valid].unique().numel())
    return total + config.commitment * commitment.to(total.dtype), terms


# --------------------------------------------------------------------------------------
# equivalence


def batch_with_gaps(seed=0):
    """Clips of different lengths (padding) with some invalid finger and arm rotations."""
    states = [signer_state(seed + s, frames=frames, amplitude=.5) for s, frames in enumerate((24, 17, 9))]
    rng = np.random.default_rng(seed)
    for state in states[:2]:
        for name in ("left_hand", "body"):
            channel = state.channels[name]
            gap = rng.random(channel.valid.shape) < .1
            channel.valid[gap], channel.confidence[gap], channel.values[gap] = False, 0., 0.
    return collate_multichannel(states)


def test_skeleton_points_and_motion_loss_equal_the_per_joint_reference():
    torch.manual_seed(0)
    batch = batch_with_gaps()
    labels = [batch.channels[n].layout.labels for n in LABELS]
    prediction = {n: (batch.channels[n].values.float() + .3 * torch.randn_like(batch.channels[n].values.float()))
                  .requires_grad_(True) for n in LABELS}
    points, names, chains = skeleton_points(*prediction.values(), *labels)
    expected, expected_names, expected_chains = reference_skeleton_points(*prediction.values(), *labels)
    assert (names, chains) == (expected_names, expected_chains)
    torch.testing.assert_close(points, expected, **TOLERANCE)
    channels = {n: (batch.channels[n].valid, tuple(labels[k])) for k, n in enumerate(LABELS)}
    from signtranslator.generation.signing_space import _support
    assert torch.equal(_support(channels, chains, batch.frame_valid), reference_support(channels, chains, batch.frame_valid))
    assert not _support(channels, chains, batch.frame_valid).all()                 # the gaps are exercised

    loss, reference = skeleton_motion_loss(prediction, batch), reference_skeleton_motion_loss(prediction, batch)
    assert set(loss) == set(reference) == {"position", "velocity", "acceleration"}
    for key in loss:
        torch.testing.assert_close(loss[key], reference[key], **TOLERANCE)
    grads = torch.autograd.grad(sum(loss.values()), list(prediction.values()))
    expected_grads = torch.autograd.grad(sum(reference.values()), list(prediction.values()))
    for g, e in zip(grads, expected_grads):
        torch.testing.assert_close(g, e, **TOLERANCE)


def latents_for(model, seed):
    generator = torch.Generator().manual_seed(seed)
    return {part: torch.randn(3, 10, model.config.code_dim, generator=generator) * .3 for part, _ in model.parts}


def test_vectorised_quantizer_gives_the_per_part_codes_and_codebooks():
    torch.manual_seed(0)
    state = signer_state(0, frames=12)
    model = MotionRVQ(replace(small(state), codebook=16)).train()
    reference = copy.deepcopy(model)
    valid = torch.ones(3, 10, dtype=torch.bool); valid[1, 6:] = False; valid[2, 3:] = False
    with torch.no_grad():                                         # codes about to die, so re-seeding is exercised
        for m in (model, reference):
            for quantizer in m.quantizers.values():
                quantizer.code_mass[:, :5] = 1e-4
    reseeded = False
    for step, stages in enumerate((None, 2, None, 1, None)):
        latents = latents_for(model, step)
        quantized, codes, inputs = model.quantize(latents, stages)
        expected_q, expected_i, expected_inputs = {}, {}, {}
        for part, _ in reference.parts:
            expected_q[part], expected_i[part], expected_inputs[part] = reference.quantizers[part](latents[part], stages)
        expected_codes = torch.stack([expected_i[part] for part, _ in reference.parts], -1).flatten(-2)
        assert torch.equal(codes, expected_codes)
        for part, _ in model.parts:
            assert torch.equal(quantized[part], expected_q[part])
        before = model.quantizers["body"].codes.clone()
        torch.manual_seed(100 + step)
        model.update_codebooks(inputs, codes, valid)
        torch.manual_seed(100 + step)
        per_part = reference.split_codes(expected_codes)
        for part, _ in reference.parts:
            reference.quantizers[part].update(expected_inputs[part], per_part[part], valid)
        for name, buffer in model.named_buffers():                # bit-identical, re-seeded codes included
            assert torch.equal(buffer, dict(reference.named_buffers())[name]), name
        reseeded |= bool((model.quantizers["body"].code_mass == 1.).any()) and not torch.equal(before, model.quantizers["body"].codes)
    assert reseeded
    # a batch without valid tokens changes nothing, in either
    snapshot = {k: v.clone() for k, v in model.named_buffers()}
    quantized, codes, inputs = model.quantize(latents_for(model, 9))
    model.update_codebooks(inputs, codes, torch.zeros(3, 10, dtype=torch.bool))
    assert all(torch.equal(v, snapshot[k]) for k, v in model.named_buffers())


def test_vectorised_training_step_gives_the_reference_codes_losses_and_gradients():
    batch = batch_with_gaps(seed=3)
    torch.manual_seed(0)
    model = MotionRVQ(small(signer_state(0, frames=12))).train()
    reference = copy.deepcopy(model)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in (model, reference)]
    for step in range(4):                                         # quantiser dropout draws differ between steps
        torch.manual_seed(step)
        loss, terms = rvq_loss(model, batch)
        torch.manual_seed(step)
        expected, expected_terms = reference_rvq_loss(reference, batch)
        torch.testing.assert_close(loss, expected, **TOLERANCE)
        assert set(terms) == set(expected_terms) and "head_rotation" not in terms      # unsupported channel skipped
        for key, value in terms.items():
            torch.testing.assert_close(torch.as_tensor(float(value), dtype=torch.float64),
                                       torch.as_tensor(expected_terms[key], dtype=torch.float64), **TOLERANCE)
        for m, optimizer, value in zip((model, reference), optimizers, (loss, expected)):
            optimizer.zero_grad(); value.backward()
        for (name, p), q in zip(model.named_parameters(), reference.parameters()):   # float32 rounding of sums
            assert float((p.grad - q.grad).norm()) <= 1e-5 * float(q.grad.norm()) + 1e-9, name
        for optimizer in optimizers:
            optimizer.step()
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, dict(reference.named_buffers())[name], **TOLERANCE)
    with torch.no_grad():
        unit = diffusion_targets(model.config.motion, batch)
        assert torch.equal(model.eval()(unit, batch.frame_valid)[3], reference_forward(reference.eval(), unit, batch.frame_valid)[3])


def test_weight_ema_foreach_update_equals_the_per_tensor_update():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("train_tokens", Path(__file__).resolve().parents[1] / "scripts/train_motion_tokens.py")
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.LayerNorm(7), torch.nn.Linear(7, 3))
    ema = script.WeightEMA(model, decay=.99)
    shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
    for update in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        d = min(.99, (1 + update) / (10 + update))
        for n, p in model.named_parameters():
            shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)
        ema.update(model)
    assert all(torch.equal(ema.shadow[n], shadow[n]) for n in shadow)
