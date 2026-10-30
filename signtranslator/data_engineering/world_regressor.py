"""Learned image-to-world landmark regressors for the Sem-Lex adapter (``semlex_pose``).

MediaPipe's pose and hand landmarkers output, from one image crop, both normalised image landmarks (x, y, relative z)
and world landmarks (metres). Sem-Lex stored only the former; our canonical states are solved from the latter. These
small MLPs learn world from image on pairs produced by our own Tasks trackers (``scripts/collect_world_pairs.py``), so
the adapter's states are in our world-landmark representation rather than a geometric pseudo-3D.

**Hand model** (one per side; the side is the pose wrist the hand is attached to). Input: the 21 image points as
``q = (x, y / a, z)`` (image widths), wrist-centred, rotated in the image plane so that the wrist -> middle-MCP vector
points up, divided by the hand's mean bone length ``s`` (3D, depth weighted by ``HAND_SCALE_Z``). Output: the 21 world
points minus the world wrist, in the same in-plane-rotated frame, in decimetres; the rotation is undone afterwards (a
rotation about the camera axis), which makes the model equivariant to in-plane hand rotation by construction.

**Pose model.** Input: points 0..24 as ``q`` relative to the shoulder midpoint, divided by the image shoulder width,
plus the context (shoulder midpoint x and y, shoulder width, aspect). Outputs: points 0..24 of the world landmarks
minus the world shoulder midpoint (decimetres), and 33 visibility logits (MediaPipe's visibility as a soft target).

Losses: position MSE plus ``1 - cos`` on the bone directions the IK uses (fingers and palm; arms, shoulder line, torso).

**z-free variant** (``FEATURES_XY``, ``"world-regressor-features-xy-v1"``) for sources that stored only image ``x, y``
(the UWB YouTube-ASL Clip Keypoint Dataset: MediaPipe Holistic 2D in pixels, no relative z, no visibility). The
features are the same constructions on ``(x, y / a)`` alone: the hand is wrist-centred, turned so that wrist -> middle
MCP points up and divided by its mean *image-plane* bone length (no depth weighting), 20 points x 2 = 40 inputs; the
pose is 25 points x 2 relative to the shoulder midpoint over the image shoulder width plus the same 4 context values,
54 inputs. Any third coordinate in the input is ignored, so a z-free bundle reads xyz and xy arrays identically. The
targets, outputs and losses are unchanged. A bundle records its feature version and :class:`WorldRegressors` applies
the matching feature functions; the v1 constructions, the v1/v2 bundles and their SHA-256 are untouched.

**Framing-invariant z-free variant** (``FEATURES_XY2``, ``"world-regressor-features-xy-v2"``): the xy-v1 pose context
(shoulder midpoint x and y, image shoulder width, aspect) is absolute crop framing. The pairs are Sem-Lex-framed
crops, so on UWB full frames these inputs fall far outside the training range (review 2026-10-30: midpoint-y z-score
median -5.0, p10 -9.1; width -1.65; aspect +1.3). xy-v2 keeps only the 50 shoulder-relative point inputs, so the pose
model (world points and visibility head alike) is invariant to image translation, scale and frame shape by
construction; the hand features are already invariant. :func:`input_zscores` measures how far a source's inputs lie
from a bundle's training distribution (the gate asks p99 |z| <= 4 for every input).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

FEATURES_VERSION = "world-regressor-features-v1"      # (x, y / a, z): the v1 and v2 bundles
FEATURES_XY = "world-regressor-features-xy-v1"       # (x, y / a) only: sources without relative z (UWB 2D keypoints)
FEATURES_XY2 = "world-regressor-features-xy-v2"      # xy without the absolute framing context (translation/scale free)
Z_FREE_FEATURES = frozenset((FEATURES_XY, FEATURES_XY2))
# The bundle that passed the declared gate (2026-10-28; scripts/semlex_recognition_shift.py, learned_gate):
# trained on 8 How2Sign signers' Sem-Lex-framed crops (train 1, 3, 5, 8, 11; val 2, 4, 9), 30 epochs.
WORLD_REGRESSORS_V1 = dict(
    key="acquisition/2026-10-17-v1/sem-lex-mirror-v1/adapter-dev/world-regressors-v1/world-regressors-v1.pt",
    sha256="2d17b7434a5360f850d5910c284ffee3ab040d9132933127e3a7ef6a1b3ce793")
# Superseded: v1 used How2Sign signer 9 (the generator's final-test signer) for validation and split signers 1 and 2
# (shared source videos) across train and val, and it learned from Tasks image landmarks, not legacy Holistic.
# v2 (2026-10-28): inputs are legacy MediaPipe Holistic (mediapipe 0.10.14, model_complexity 1, refined face) on the
# same Sem-Lex-framed How2Sign crops, targets our Tasks world landmarks and visibility; train signers 3, 4, 5, 8, 11,
# validation 1 and 2 (one source-video component); signer 9 excluded entirely; seed 0, 30 epochs. It passed gate v2
# (docs/NEXT_PHASE_PLAN.md, 09:30 UTC) on 300 ASL Citizen val clips through legacy Holistic arrays.
WORLD_REGRESSORS_V2 = dict(
    key="acquisition/2026-10-17-v1/sem-lex-mirror-v1/adapter-dev/world-regressors-v2/world-regressors-v2.pt",
    sha256="0953cd2cb2b48506808dff2d7199061eac3218ad555ea35327ff34edac3acab1")
WORLD_REGRESSORS = WORLD_REGRESSORS_V2    # the pinned bundle the converter defaults to
HAND_SCALE_Z = 1.5
POSE_POINTS = 25                      # 0..24: face, shoulders, arms, hand points, hips
HAND_BONES = tuple((a, b) for f in range(5) for a, b in zip((0, *range(1 + 4 * f, 4 + 4 * f)), range(1 + 4 * f, 5 + 4 * f))) \
    + ((5, 17), (5, 9), (9, 13), (13, 17))
POSE_BONES = ((11, 13), (13, 15), (12, 14), (14, 16), (12, 11), (24, 23), (15, 19), (16, 20), (0, 11), (0, 12))
DECIMETRES = 10.


def _rotation_2d(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.stack((np.stack((c, -s), -1), np.stack((s, c), -1)), -2)      # (..., 2, 2)


def hand_features(image: np.ndarray, aspect: float):
    """(N, 21, 3) normalised image points -> features (N, 60), in-plane rotation (N, 2, 2), scale (N,), valid (N,)."""
    q = np.array(image, dtype=np.float64)
    q[..., 1] /= aspect
    q = q - q[:, :1]
    up = q[:, 9, :2]
    theta = np.arctan2(up[:, 0], -up[:, 1])          # angle that turns (wrist -> middle MCP) onto image up (0, -1)
    rotation = _rotation_2d(-theta)
    q[..., :2] = np.einsum("nij,nkj->nki", rotation, q[..., :2])
    weighted = q * [1., 1., HAND_SCALE_Z]
    scale = np.mean([np.linalg.norm(weighted[:, b] - weighted[:, a], axis=-1) for a, b in HAND_BONES[:20]], axis=0)
    valid = np.isfinite(q).all((1, 2)) & (scale > 1e-6) & (np.linalg.norm(up, axis=-1) > 1e-6)
    features = np.where(valid[:, None, None], q / np.where(valid, scale, 1.)[:, None, None], 0.)[:, 1:].reshape(len(q), -1)
    return features.astype(np.float32), rotation, scale, valid


def hand_target(world: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    w = np.array(world, dtype=np.float64) - world[:, :1]
    w[..., :2] = np.einsum("nij,nkj->nki", rotation, w[..., :2])
    return (w * DECIMETRES).astype(np.float32)


def hand_output_to_world(output: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """(N, 21, 3) model output -> world points (metres, wrist at the origin, camera-aligned axes)."""
    w = np.array(output, dtype=np.float64) / DECIMETRES
    w[..., :2] = np.einsum("nji,nkj->nki", rotation, w[..., :2])            # inverse rotation (transpose)
    return w


def pose_features(image: np.ndarray, aspect: float):
    """(N, 33, 3) normalised image points -> features (N, 25*3 + 4), valid (N,)."""
    q = np.array(image[:, :POSE_POINTS], dtype=np.float64)
    q[..., 1] /= aspect
    mid = (q[:, 11] + q[:, 12]) / 2
    width = np.linalg.norm((q[:, 11] - q[:, 12])[:, :2], axis=-1)
    valid = np.isfinite(q).all((1, 2)) & (width > 1e-6)
    safe = np.where(valid, width, 1.)
    relative = (q - mid[:, None]) / safe[:, None, None]
    context = np.stack((mid[:, 0], mid[:, 1] * aspect, width, np.full(len(q), aspect)), axis=1)
    features = np.concatenate((relative.reshape(len(q), -1), context), axis=1)
    return np.where(valid[:, None], features, 0.).astype(np.float32), valid


def hand_features_xy(image: np.ndarray, aspect: float):
    """(N, 21, >= 2) normalised image points (any z ignored) -> features (N, 40), in-plane rotation (N, 2, 2),
    image-plane scale (N,), valid (N,). The z-free twin of :func:`hand_features`."""
    q = np.array(np.asarray(image)[..., :2], dtype=np.float64)
    q[..., 1] /= aspect
    q = q - q[:, :1]
    up = q[:, 9].copy()
    theta = np.arctan2(up[:, 0], -up[:, 1])
    rotation = _rotation_2d(-theta)
    q = np.einsum("nij,nkj->nki", rotation, q)
    scale = np.mean([np.linalg.norm(q[:, b] - q[:, a], axis=-1) for a, b in HAND_BONES[:20]], axis=0)
    valid = np.isfinite(q).all((1, 2)) & (scale > 1e-6) & (np.linalg.norm(up, axis=-1) > 1e-6)
    features = np.where(valid[:, None, None], q / np.where(valid, scale, 1.)[:, None, None], 0.)[:, 1:].reshape(len(q), -1)
    return features.astype(np.float32), rotation, scale, valid


def pose_features_xy(image: np.ndarray, aspect: float):
    """(N, 33, >= 2) normalised image points (any z ignored) -> features (N, 25*2 + 4), valid (N,). The z-free twin of
    :func:`pose_features` (same shoulder normalisation and context)."""
    q = np.array(np.asarray(image)[:, :POSE_POINTS, :2], dtype=np.float64)
    q[..., 1] /= aspect
    mid = (q[:, 11] + q[:, 12]) / 2
    width = np.linalg.norm(q[:, 11] - q[:, 12], axis=-1)
    valid = np.isfinite(q).all((1, 2)) & (width > 1e-6)
    safe = np.where(valid, width, 1.)
    relative = (q - mid[:, None]) / safe[:, None, None]
    context = np.stack((mid[:, 0], mid[:, 1] * aspect, width, np.full(len(q), aspect)), axis=1)
    features = np.concatenate((relative.reshape(len(q), -1), context), axis=1)
    return np.where(valid[:, None], features, 0.).astype(np.float32), valid


def pose_features_xy2(image: np.ndarray, aspect: float):
    """(N, 33, >= 2) normalised image points (any z ignored) -> features (N, 25*2), valid (N,): the xy-v1 relative
    block without the framing context, invariant to image translation, scale and frame shape."""
    features, valid = pose_features_xy(image, aspect)
    return np.ascontiguousarray(features[:, :POSE_POINTS * 2]), valid


# feature version -> (hand feature function, pose feature function, hand input width, pose input width)
FEATURE_FUNCTIONS = {FEATURES_VERSION: (hand_features, pose_features, 60, POSE_POINTS * 3 + 4),
                     FEATURES_XY: (hand_features_xy, pose_features_xy, 40, POSE_POINTS * 2 + 4),
                     FEATURES_XY2: (hand_features_xy, pose_features_xy2, 40, POSE_POINTS * 2)}


def input_names(version: str) -> dict:
    """Names of each model's inputs: {"hand": [...], "pose": [...]}."""
    dims = ("x", "y", "z") if version == FEATURES_VERSION else ("x", "y")
    hand = [f"h{k}.{d}" for k in range(1, 21) for d in dims]
    pose = [f"p{k}.{d}" for k in range(POSE_POINTS) for d in dims]
    if version != FEATURES_XY2:
        pose += ["context.mid_x", "context.mid_y", "context.shoulder_width", "context.aspect"]
    return dict(hand=hand, pose=pose)


def pose_target(world: np.ndarray) -> np.ndarray:
    w = np.array(world[:, :POSE_POINTS], dtype=np.float64)
    return ((w - (w[:, 11:12] + w[:, 12:13]) / 2) * DECIMETRES).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int = 512, depth: int = 3):
        super().__init__()
        self.shape = [inputs, outputs, width, depth]
        layers, size = [], inputs
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.GELU()]
            size = width
        layers.append(nn.Linear(size, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def direction_loss(pred: torch.Tensor, target: torch.Tensor, bones) -> torch.Tensor:
    a = torch.stack([pred[:, j] - pred[:, i] for i, j in bones], 1)
    b = torch.stack([target[:, j] - target[:, i] for i, j in bones], 1)
    return (1 - nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-6)).mean()


@dataclass
class WorldRegressors:
    """Trained hand (per side) and pose models with their normalisation statistics."""

    hand: dict            # side -> (MLP, mean (60,) or (40,), std)
    pose: tuple           # (MLP, mean, std)
    sha256: str | None = None
    features: str = FEATURES_VERSION     # which feature functions the models were trained on (FEATURE_FUNCTIONS)

    @torch.no_grad()
    def hands(self, image: np.ndarray, aspect: float, side: str):
        """(N, 21, 3) image points (xy features: (N, 21, >= 2)) -> (N, 21, 3) world (metres, wrist-centred), valid (N,)."""
        features, rotation, _, valid = FEATURE_FUNCTIONS[self.features][0](image, aspect)
        model, mean, std = self.hand[side]
        out = model(torch.from_numpy((features - mean) / std)).numpy().reshape(-1, 21, 3)
        return np.where(valid[:, None, None], hand_output_to_world(out, rotation), 0.), valid

    @torch.no_grad()
    def pose_world(self, image: np.ndarray, aspect: float):
        """(N, 33, 3) image points (xy features: (N, 33, >= 2)) -> world (N, 33, 3) (points 25..32 zero), visibility
        (N, 33) in [0, 1], valid (N,)."""
        features, valid = FEATURE_FUNCTIONS[self.features][1](image, aspect)
        model, mean, std = self.pose
        out = model(torch.from_numpy((features - mean) / std)).numpy()
        world = np.zeros((len(image), 33, 3))
        world[:, :POSE_POINTS] = out[:, :POSE_POINTS * 3].reshape(-1, POSE_POINTS, 3) / DECIMETRES
        visibility = 1. / (1. + np.exp(-out[:, POSE_POINTS * 3:]))
        return np.where(valid[:, None, None], world, 0.), np.where(valid[:, None], visibility, 0.), valid

    def save(self) -> bytes:
        if self.features not in FEATURE_FUNCTIONS:
            raise ValueError(f"world regressors: unknown feature version {self.features!r}")
        blob = dict(version=self.features,
                    hand={s: dict(state=m.state_dict(), mean=torch.from_numpy(mu), std=torch.from_numpy(sd),
                                  shape=list(m.shape))
                          for s, (m, mu, sd) in self.hand.items()},
                    pose=dict(state=self.pose[0].state_dict(), mean=torch.from_numpy(self.pose[1]),
                              std=torch.from_numpy(self.pose[2]),
                              shape=list(self.pose[0].shape)))
        buffer = io.BytesIO()
        torch.save(blob, buffer)
        return buffer.getvalue()

    @classmethod
    def load(cls, payload: bytes, *, expected_sha256: str | None = None) -> "WorldRegressors":
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("world regressors: content SHA-256 differs from the pinned one")
        blob = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        version = blob.get("version")
        if version not in FEATURE_FUNCTIONS:
            raise ValueError(f"world regressors: unknown feature version {version!r}")
        _, _, hand_width, pose_width = FEATURE_FUNCTIONS[version]
        if any(e["shape"][0] != hand_width for e in blob["hand"].values()) or blob["pose"]["shape"][0] != pose_width:
            raise ValueError(f"world regressors: model input widths do not match feature version {version!r}")

        def build(entry):
            model = MLP(*entry["shape"])
            model.load_state_dict(entry["state"])
            return model.eval(), entry["mean"].numpy(), entry["std"].numpy()
        return cls(hand={s: build(e) for s, e in blob["hand"].items()}, pose=build(blob["pose"]), sha256=digest,
                   features=version)


def input_zscores(regressors: "WorldRegressors", pose: np.ndarray | None, hands: dict, aspect: float) -> dict:
    """Standardised model inputs ``(features - mean) / std`` (the bundle's own normalisation) of the valid rows:
    {"pose": (N, D), "Left": (M, 40), "Right": ...}. ``pose``: (N, 33, >= 2) normalised image points; ``hands``:
    side -> (M, 21, >= 2)."""
    hand_fn, pose_fn, _, _ = FEATURE_FUNCTIONS[regressors.features]
    out = {}
    if pose is not None and len(pose):
        features, valid = pose_fn(pose, aspect)
        _, mean, std = regressors.pose
        out["pose"] = ((features - mean) / std)[valid]
    for side, image in hands.items():
        if image is not None and len(image):
            features, _, _, valid = hand_fn(image, aspect)
            _, mean, std = regressors.hand[side]
            out[side] = ((features - mean) / std)[valid]
    return out


def load_regressors(path: Path, expected_sha256: str | None = None) -> WorldRegressors:
    return WorldRegressors.load(Path(path).read_bytes(), expected_sha256=expected_sha256)


__all__ = ["FEATURES_VERSION", "FEATURES_XY", "FEATURES_XY2", "Z_FREE_FEATURES", "FEATURE_FUNCTIONS",
           "hand_features_xy", "pose_features_xy", "pose_features_xy2", "input_names", "input_zscores", "WORLD_REGRESSORS", "WORLD_REGRESSORS_V1", "WORLD_REGRESSORS_V2", "HAND_BONES", "POSE_BONES", "POSE_POINTS", "MLP", "WorldRegressors", "hand_features",
           "hand_target", "hand_output_to_world", "pose_features", "pose_target", "direction_loss", "load_regressors"]
