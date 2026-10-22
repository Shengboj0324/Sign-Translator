"""Where the hands are: differentiable signing-space locations and their loss.

Per-joint rotation losses treat every joint alike, but sign location (hand at the
chin, the chest, a point in neutral space) is the composition of the shoulder,
elbow, wrist and finger rotations: a 10° shoulder error moves the hand ~9 cm while
costing the same as a 10° error of a fingertip joint. This module runs the forward
kinematics of :mod:`canonical_skeleton` in torch, in the **chest frame** (signing
space is body-relative, so the pelvis and the root translation are left out), with
the same nominal bone lengths — signer-normalised locations, not measurements — and
scores the wrists and fingertips of generated motion against the reference.

The loss per point is Huber(‖p̂ − p‖ / ``scale``), averaged over the points whose
whole kinematic chain is valid in the reference on valid frames. Gradients reach
every rotation of the chain, weighted by its lever arm.
"""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from ..data_engineering.video_landmarks import BODY_JOINTS, finger_joints
from ..pose.rotations import _matmul3, rotation_6d_to_matrix
from .canonical_skeleton import _FINGER_LENGTHS, _KNUCKLES, _LENGTH, _SHOULDER_HALF

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
POINTS = tuple(f"{side}{p}" for side in ("Left", "Right") for p in ("Hand", *(f"{f}Tip" for f in FINGERS)))
LOCATION_SCALE_METRES = .1


def _vector(values, like):
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def signing_space_points(body6: torch.Tensor, left6: torch.Tensor, right6: torch.Tensor,
                         body_labels, left_labels, right_labels):
    """Chest-frame positions (…, 12, 3) of the wrists and fingertips, and each point's chain.

    ``body6`` (…, J_body, 6) and the hand tensors (…, 15, 6) are 6D local rotations
    labelled as in the canonical channels. The chain of a point lists the
    (channel, joint) rotations it depends on, for masking.
    """
    index = {("body", n): k for k, n in enumerate(body_labels)}
    index.update({("left_hand", n): k for k, n in enumerate(left_labels)})
    index.update({("right_hand", n): k for k, n in enumerate(right_labels)})
    tensors = {"body": body6, "left_hand": left6, "right_hand": right6}
    rotation = lambda channel, name: rotation_6d_to_matrix(tensors[channel][..., index[(channel, name)], :])
    like = body6
    eye = torch.eye(3, dtype=like.dtype, device=like.device).expand(*body6.shape[:-2], 3, 3)
    world = {"Chest": eye}
    points, chains = [], []
    for side, sign in (("Left", 1.), ("Right", -1.)):
        channel = f"{side.lower()}_hand"
        start = _vector([sign * _SHOULDER_HALF, 0., 0.], like).expand(*body6.shape[:-2], 3)
        chain = []
        for joint, part in ((f"{side}UpperArm", "UpperArm"), (f"{side}Forearm", "Forearm")):
            spec = next(j for j in BODY_JOINTS if j.name == joint)
            world[joint] = world[spec.parent] @ rotation("body", joint)
            start = start + _LENGTH[part] * (world[joint] @ _vector(spec.rest, like))
            chain.append(("body", joint))
        wrist = start
        hand = world[f"{side}Forearm"] @ rotation("body", f"{side}Hand")
        chain.append(("body", f"{side}Hand"))
        points.append(wrist)
        chains.append(tuple(chain[:2]))
        for finger in FINGERS:
            x, y = _KNUCKLES[finger]
            position = wrist + hand @ _vector([x, y, 0.], like)       # hand x runs pinky -> index on both hands
            frame, finger_chain = hand, list(chain)
            for segment in (1, 2, 3):
                spec = next(j for j in finger_joints(side) if j.name == f"{side}Hand{finger}{segment}")
                frame = frame @ rotation(channel, spec.name)
                position = position + _FINGER_LENGTHS[segment - 1] * (frame @ _vector(spec.rest, like))
                finger_chain.append((channel, spec.name))
            points.append(position)
            chains.append(tuple(finger_chain))
    return torch.stack(points, dim=-2), tuple(chains)


@functools.lru_cache(maxsize=None)
def _skeleton_layout(body_labels: tuple, left_labels: tuple, right_labels: tuple) -> dict:
    """The static part of :func:`skeleton_points`: gathered joints, FK constants, point names and chains.

    Per side the arm is Chest → UpperArm → Forearm → Hand, then five three-segment fingers.
    """
    index = {("body", n): k for k, n in enumerate(body_labels)}
    index.update({("left_hand", n): k for k, n in enumerate(left_labels)})
    index.update({("right_hand", n): k for k, n in enumerate(right_labels)})
    spec = {j.name: j for j in BODY_JOINTS}
    out = dict(arm=[], left_hand=[], right_hand=[], shoulder=[], arm_rest=[], knuckle=[], finger_rest=[])
    names, chains = [], []
    for side, sign in (("Left", 1.), ("Right", -1.)):
        channel = f"{side.lower()}_hand"
        if spec[f"{side}UpperArm"].parent != "Chest" or spec[f"{side}Forearm"].parent != f"{side}UpperArm":
            raise ValueError("the arm chain is Chest -> UpperArm -> Forearm")
        out["arm"] += [index[("body", f"{side}{j}")] for j in ("UpperArm", "Forearm", "Hand")]
        out["shoulder"].append([sign * _SHOULDER_HALF, 0., 0.])
        out["arm_rest"].append([spec[f"{side}UpperArm"].rest, spec[f"{side}Forearm"].rest])
        chain = [("body", f"{side}UpperArm")]
        names.append(f"{side}Elbow"); chains.append(tuple(chain))
        chain.append(("body", f"{side}Forearm"))
        names.append(f"{side}Wrist"); chains.append(tuple(chain))
        hand_chain = chain + [("body", f"{side}Hand")]
        joints = {j.name: j for j in finger_joints(side)}
        knuckles, rests = [], []
        for finger in FINGERS:
            x, y = _KNUCKLES[finger]
            knuckles.append([x, y, 0.])                               # as canonical_skeleton (not mirrored)
            names.append(f"{side}{finger}MCP"); chains.append(tuple(hand_chain))
            finger_chain, finger_rest = list(hand_chain), []
            for segment, label in zip((1, 2, 3), ("PIP", "DIP", "Tip")):
                name = f"{side}Hand{finger}{segment}"
                out[channel].append(index[(channel, name)])
                finger_rest.append(joints[name].rest)
                finger_chain.append((channel, name))
                names.append(f"{side}{finger}{label}"); chains.append(tuple(finger_chain))
            rests.append(finger_rest)
        out["knuckle"].append(knuckles)
        out["finger_rest"].append(rests)
    return dict(out, names=tuple(names), chains=tuple(chains))


@functools.lru_cache(maxsize=None)
def _skeleton_tensors(labels: tuple, dtype: torch.dtype, device: torch.device) -> dict:
    """:func:`_skeleton_layout` as tensors on ``device``, made once (a host→device copy waits for the GPU)."""
    layout = _skeleton_layout(*labels)
    out = {key: torch.tensor(layout[key], dtype=dtype, device=device)
           for key in ("shoulder", "arm_rest", "knuckle", "finger_rest")}
    out.update({key: torch.tensor(layout[key], dtype=torch.long, device=device)
                for key in ("arm", "left_hand", "right_hand")})
    return out


def _apply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """(…, 3, 3) · (…, 3) -> (…, 3), broadcasting the leading dimensions (elementwise, not a batched GEMM)."""
    return (matrix * vector[..., None, :]).sum(-1)


def skeleton_points(body6: torch.Tensor, left6: torch.Tensor, right6: torch.Tensor,
                    body_labels, left_labels, right_labels):
    """Chest-frame positions (…, 44, 3) of every arm and hand point, their names and chains.

    Per side: elbow, wrist, then for each finger its knuckle (MCP), PIP, DIP and tip —
    the same forward kinematics and nominal lengths as :func:`signing_space_points`,
    which is the wrist-and-tip subset of these points. Both sides and all five fingers
    are composed at once (one 6D→matrix conversion, three finger-segment steps), so the
    cost in kernel launches does not grow with the number of points.
    """
    labels = (tuple(body_labels), tuple(left_labels), tuple(right_labels))
    layout, like = _skeleton_layout(*labels), body6
    tensors = _skeleton_tensors(labels, like.dtype, like.device)
    constant = tensors.__getitem__
    lead = body6.shape[:-2]
    rotations = rotation_6d_to_matrix(torch.cat((body6[..., tensors["arm"], :], left6[..., tensors["left_hand"], :],
                                                 right6[..., tensors["right_hand"], :]), dim=-2))
    arm = rotations[..., :6, :, :].reshape(*lead, 2, 3, 3, 3)                      # (…, side, joint, 3, 3)
    finger = rotations[..., 6:, :, :].reshape(*lead, 2, len(FINGERS), 3, 3, 3)    # (…, side, finger, segment, 3, 3)
    arm_rest = constant("arm_rest")
    eye = torch.eye(3, dtype=like.dtype, device=like.device).expand(*lead, 2, 3, 3)
    upper = _matmul3(eye, arm[..., 0, :, :])
    elbow = constant("shoulder") + _LENGTH["UpperArm"] * _apply(upper, arm_rest[:, 0])
    forearm = _matmul3(upper, arm[..., 1, :, :])
    wrist = elbow + _LENGTH["Forearm"] * _apply(forearm, arm_rest[:, 1])
    hand = _matmul3(forearm, arm[..., 2, :, :])[..., None, :, :]                         # (…, side, 1, 3, 3)
    position = wrist[..., None, :] + _apply(hand, constant("knuckle"))
    finger_rest = constant("finger_rest")
    frame, along = hand, [position]
    for segment in range(3):
        frame = _matmul3(frame, finger[..., segment, :, :])
        position = position + _FINGER_LENGTHS[segment] * _apply(frame, finger_rest[:, :, segment])
        along.append(position)
    fingers = torch.stack(along, dim=-2).flatten(-3, -2)                           # (…, side, 20, 3)
    points = torch.cat((torch.stack((elbow, wrist), dim=-2), fingers), dim=-2)
    return points.flatten(-3, -2), layout["names"], layout["chains"]


MOTION_SCALES = dict(position=LOCATION_SCALE_METRES, velocity=.5, acceleration=10.)   # m, m/s, m/s²


def skeleton_motion_loss(prediction: dict, motion, *, scales=MOTION_SCALES) -> dict:
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
    # the prediction and the reference go through the kinematics as one batch
    points, _, chains = skeleton_points(*(torch.stack((prediction[n], r)) for n, r in zip(names, reference)), *labels)
    predicted, target = points[0], points[1].detach()
    channels = {n: (motion.channels[n].valid, tuple(motion.channels[n].layout.labels)) for n in names}
    support = _support(channels, chains, motion.frame_valid)                  # (B, T, P)
    t = motion.timestamps.to(dtype)
    masks, errors = {"position": support}, {"position": predicted - target}
    if predicted.shape[1] >= 2:
        dt = (t[:, 1:] - t[:, :-1]).clamp_min(1e-3)[..., None, None]
        vp, vt = (predicted[:, 1:] - predicted[:, :-1]) / dt, (target[:, 1:] - target[:, :-1]) / dt
        masks["velocity"], errors["velocity"] = support[:, 1:] & support[:, :-1], vp - vt
        if predicted.shape[1] >= 3:
            dt2 = ((t[:, 2:] - t[:, :-2]) / 2).clamp_min(1e-3)[..., None, None]
            ap, at = (vp[:, 1:] - vp[:, :-1]) / dt2, (vt[:, 1:] - vt[:, :-1]) / dt2
            masks["acceleration"] = masks["velocity"][:, 1:] & masks["velocity"][:, :-1]
            errors["acceleration"] = ap - at
    # which terms have support: one host sync for all of them; the means are masked sums, not boolean selections
    present = torch.stack([m.any() for m in masks.values()]).tolist()
    out = {}
    for (key, mask), keep in zip(masks.items(), present):
        if keep:
            d = torch.linalg.vector_norm(errors[key], dim=-1) / scales[key]
            cells = F.huber_loss(d, torch.zeros_like(d), delta=1., reduction="none")
            out[key] = torch.where(mask, cells, torch.zeros_like(cells)).sum() / mask.sum()
    return out


def _support(channels: dict, chains, frame_valid: torch.Tensor) -> torch.Tensor:
    """(…, P) reference validity of every point's whole chain on valid frames."""
    layout = tuple((name, tuple(labels)) for name, (_, labels) in channels.items())
    index = _chain_index(layout, tuple(chains), frame_valid.device)
    # one gather of every chain's joints (padded with an always-valid column), then all() along the chain
    valid = torch.cat([valid for valid, _ in channels.values()] + [torch.ones_like(frame_valid)[..., None]], dim=-1)
    return valid[..., index].all(-1) & frame_valid[..., None]


@functools.lru_cache(maxsize=None)
def _chain_index(layout: tuple, chains: tuple, device: torch.device) -> torch.Tensor:
    """(P, longest chain) columns of the concatenated channel masks; the padding column is one past the last."""
    column, offset = {}, 0
    for name, labels in layout:
        column.update({(name, label): offset + k for k, label in enumerate(labels)})
        offset += len(labels)
    width = max(len(chain) for chain in chains)
    return torch.tensor([[column[link] for link in chain] + [offset] * (width - len(chain)) for chain in chains],
                        device=device)


def location_loss(prediction: dict, motion, *, scale: float = LOCATION_SCALE_METRES):
    """(mean Huber location loss, supported point count) of predicted 6D rotations against a :class:`MotionBatch`.

    ``prediction`` maps ``body``, ``left_hand`` and ``right_hand`` to (B, T, J, 6).
    Returns ``None`` for the loss when no point is supported.
    """
    names = ("body", "left_hand", "right_hand")
    labels = [motion.channels[n].layout.labels for n in names]
    identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=prediction["body"].dtype, device=prediction["body"].device)
    # unavailable reference rotations are zero payload: identity keeps the FK defined, the mask drops them
    reference = [torch.where(motion.channels[n].valid[..., None], motion.channels[n].values.to(identity.dtype), identity)
                 for n in names]
    predicted, chains = signing_space_points(*(prediction[n] for n in names), *labels)
    target, _ = signing_space_points(*reference, *labels)
    channels = {n: (motion.channels[n].valid, tuple(motion.channels[n].layout.labels)) for n in names}
    support = _support(channels, chains, motion.frame_valid)
    if not bool(support.any()):
        return None, 0
    distance = torch.linalg.vector_norm(predicted - target.detach(), dim=-1)[support] / scale
    loss = F.huber_loss(distance, torch.zeros_like(distance), delta=1., reduction="mean")
    return loss, int(support.sum())


def location_errors(generated, reference) -> dict:
    """Median chest-frame wrist and fingertip distance (cm, nominal skeleton) of paired states on a shared clock."""
    import numpy as np
    wrists, tips = [], []
    names = ("body", "left_hand", "right_hand")
    for g, r in zip(generated, reference):
        frames = min(len(g.timestamps), len(r.timestamps))
        def points(state):
            identity = np.array([1., 0., 0., 0., 1., 0.])
            values = [torch.from_numpy(np.where(state.channels[n].valid[:frames, :, None],
                                                state.channels[n].values[:frames], identity)) for n in names]
            return signing_space_points(*values, *(state.channels[n].labels for n in names))
        (pg, chains), (pr, _) = points(g), points(r)
        channels = {n: (torch.from_numpy(r.channels[n].valid[:frames] & g.channels[n].valid[:frames]),
                        tuple(r.channels[n].labels)) for n in names}
        support = _support(channels, chains, torch.ones(frames, dtype=torch.bool)).numpy()
        distance = torch.linalg.vector_norm(pg - pr, dim=-1).numpy() * 100
        for k, sink in enumerate([wrists] + [tips] * 5 + [wrists] + [tips] * 5):
            sink.append(distance[support[:, k], k])
    median = lambda parts: float(np.median(np.concatenate(parts))) if parts and sum(map(len, parts)) else float("nan")
    return dict(wrist_cm=median(wrists), fingertip_cm=median(tips))


def applicable(bindings, trained_channels) -> bool:
    """Whether the body and both hands are trained with the canonical joint labels the FK needs."""
    labels = {b.name: set(b.labels) for b in bindings}
    needed = {"body": {"LeftUpperArm", "LeftForearm", "LeftHand", "RightUpperArm", "RightForearm", "RightHand"},
              "left_hand": {j.name for j in finger_joints("Left")}, "right_hand": {j.name for j in finger_joints("Right")}}
    return all(n in trained_channels and needed[n] <= labels.get(n, set()) for n in needed)


__all__ = ["skeleton_points", "skeleton_motion_loss", "MOTION_SCALES", "signing_space_points", "location_loss", "location_errors", "applicable", "POINTS", "LOCATION_SCALE_METRES"]
