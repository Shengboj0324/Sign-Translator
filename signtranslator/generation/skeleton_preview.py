"""Skeleton preview videos so reviewers can see each performed sign.

3D-LEX ships motion, not video, yet a gloss reviewer must watch the sign. This
draws the solved skeleton in four panels — front and side of the upper body, and a
close-up of each hand relative to its wrist — and encodes it on the motion's own
clock. It is an inspection aid, not an appearance render.

**Body frame.** Captures differ in world orientation, so positions are expressed in
a frame fixed to the signer: up is world ``+y`` (glTF), lateral is the mean
right-to-left shoulder direction projected onto the ground plane, and forward
completes a right-handed frame. Rotating the whole capture about the vertical axis
therefore yields the identical video (tested). Scale is fixed per sign from the
bounding box of all frames, so motion is never rescaled frame to frame.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from .stream_render import DEFAULT_TIME_BASE, presentation_timestamps

_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_SPINE = ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End")


def mixamo_bones() -> tuple[tuple[str, str], ...]:
    bones = list(zip(_SPINE, _SPINE[1:]))
    for side in ("Left", "Right"):
        arm = ("Spine2", f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand")
        bones += list(zip(arm, arm[1:]))
        for finger in _FINGERS:
            chain = (f"{side}Hand", *(f"{side}Hand{finger}{k}" for k in (1, 2, 3, 4)))
            bones += list(zip(chain, chain[1:]))
    return tuple(bones)


def preview_joints(bones) -> tuple[str, ...]:
    return tuple(dict.fromkeys(joint for bone in bones for joint in bone))


def body_frame(positions: np.ndarray, joints: tuple[str, ...]) -> np.ndarray:
    """(T, J, 3) world positions -> (T, J, 3) in (lateral, up, forward) body axes."""
    index = {name: i for i, name in enumerate(joints)}
    lateral = (positions[:, index["LeftShoulder"]] - positions[:, index["RightShoulder"]]).mean(axis=0)
    lateral[1] = 0.
    norm = np.linalg.norm(lateral)
    if norm < 1e-9:
        raise ValueError("shoulders do not define a lateral direction")
    lateral /= norm
    up = np.array([0., 1., 0.])
    forward = np.cross(lateral, up)
    origin = positions[:, index["Spine2"]].mean(axis=0)
    basis = np.stack((lateral, up, forward), axis=1)  # columns are body axes
    return (positions - origin) @ basis


def _draw_segment(image, a, b, color, radius):
    length = int(np.ceil(np.hypot(*(b - a)))) + 1
    points = np.linspace(a, b, max(length, 2))
    height, width = image.shape[:2]
    for x, y in np.rint(points).astype(int):
        y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
        x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
        if y0 < y1 and x0 < x1:
            image[y0:y1, x0:x1] = color


def _project(points2d, low, high, size, margin=.08):
    span = float(max((high - low).max(), 1e-6))
    scale = (1 - 2 * margin) * size / span
    centre = (low + high) / 2
    u = (points2d[..., 0] - centre[0]) * scale + size / 2
    v = size / 2 - (points2d[..., 1] - centre[1]) * scale
    return np.stack((u, v), axis=-1)


def _colour(bone):
    name = bone[1]
    if name.startswith("Left"):
        return np.array([40, 110, 230], np.uint8)
    if name.startswith("Right"):
        return np.array([220, 60, 50], np.uint8)
    return np.array([90, 90, 90], np.uint8)


def _bounds(points, quantile):
    flat = points.reshape(-1, 2)
    if quantile:
        return np.quantile(flat, quantile, axis=0), np.quantile(flat, 1 - quantile, axis=0)
    return flat.min(0), flat.max(0)


def render_preview_frames(positions: np.ndarray, joints: tuple[str, ...], bones, *, panel: int = 320,
                          radius: int = 2, bounds_quantile: float = 0.) -> list[np.ndarray]:
    """Four-panel RGB frames (2 panel x 2 panel), white background.

    Pass only frames that belong to the motion (for example the adapter's valid
    frames); ``bounds_quantile`` > 0 trims extreme points from the fixed per-sign
    scale when no validity mask exists.
    """
    if panel % 2:
        raise ValueError("panel size must be even for the encoder")
    if not 0 <= bounds_quantile < .5:
        raise ValueError("bounds_quantile must be in [0, 0.5)")
    body = body_frame(positions, joints)
    index = {name: i for i, name in enumerate(joints)}
    pairs = [(index[a], index[b]) for a, b in bones]
    # Panel coordinates per view: front (lateral, up), side (forward, up), each hand
    # relative to its wrist in front coordinates.
    views = [body[..., [0, 1]], body[..., [2, 1]]]
    hands = []
    for side in ("Left", "Right"):
        members = [index[j] for j in joints if j.startswith(f"{side}Hand")]
        wrist = body[:, index[f"{side}Hand"]][:, None, :]
        local = (body - wrist)[..., [0, 1]]
        hands.append((local, members))
    ranges = [_bounds(v, bounds_quantile) for v in views]
    hand_ranges = [_bounds(h[:, m], bounds_quantile) for h, m in hands]
    projected = [_project(v, lo, hi, panel) for v, (lo, hi) in zip(views, ranges)]
    projected += [_project(h, lo, hi, panel) for (h, _), (lo, hi) in zip(hands, hand_ranges)]
    hand_bones = [[(a, b) for a, b in pairs if a in set(m) and b in set(m)] for _, m in hands]
    frames = []
    for t in range(len(positions)):
        canvas = np.full((2 * panel, 2 * panel, 3), 255, np.uint8)
        for k, points in enumerate(projected):
            row, col = divmod(k, 2)
            tile = canvas[row * panel:(row + 1) * panel, col * panel:(col + 1) * panel]
            chosen = pairs if k < 2 else hand_bones[k - 2]
            for (a, b), bone in ((p, (joints[p[0]], joints[p[1]])) for p in chosen):
                _draw_segment(tile, points[t, a], points[t, b], _colour(bone), radius)
            tile[:, -1] = 200
            tile[-1, :] = 200
        frames.append(canvas)
    return frames


def encode_preview(frames, clock, path: Path, *, crf: int = 23,
                   time_base: Fraction = DEFAULT_TIME_BASE) -> dict:
    """H.264 preview with presentation timestamps taken from the motion's clock."""
    import av
    ticks = presentation_timestamps(clock, time_base=time_base)
    if len(ticks) != len(frames):
        raise ValueError("frame count must equal clock length")
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=None)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame, tick in zip(frames, ticks):
            video = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video.pts, video.time_base = tick, time_base
            for packet in stream.encode(video):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return dict(frames=len(frames), width=width, height=height, ticks_last=ticks[-1])


__all__ = ["mixamo_bones", "preview_joints", "body_frame", "render_preview_frames", "encode_preview"]
