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

import torch
import torch.nn.functional as F

from ..data_engineering.video_landmarks import BODY_JOINTS, finger_joints
from ..pose.rotations import rotation_6d_to_matrix
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
            position = wrist + hand @ _vector([sign * x, y, 0.], like)
            frame, finger_chain = hand, list(chain)
            for segment in (1, 2, 3):
                spec = next(j for j in finger_joints(side) if j.name == f"{side}Hand{finger}{segment}")
                frame = frame @ rotation(channel, spec.name)
                position = position + _FINGER_LENGTHS[segment - 1] * (frame @ _vector(spec.rest, like))
                finger_chain.append((channel, spec.name))
            points.append(position)
            chains.append(tuple(finger_chain))
    return torch.stack(points, dim=-2), tuple(chains)


def _support(channels: dict, chains, frame_valid: torch.Tensor) -> torch.Tensor:
    """(B, T, 12) reference validity of every point's whole chain on valid frames."""
    masks = []
    for chain in chains:
        ok = frame_valid.clone()
        for channel, label in chain:
            ok = ok & channels[channel][0][..., channels[channel][1].index(label)]
        masks.append(ok)
    return torch.stack(masks, dim=-1)


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


__all__ = ["signing_space_points", "location_loss", "location_errors", "applicable", "POINTS", "LOCATION_SCALE_METRES"]
