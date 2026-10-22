"""A schematic face panel, so reviewers of skeleton previews can see the non-manual grammar.

The four-panel skeleton preview (``skeleton_preview``) draws body and hands only, yet ASL grammar lives largely on the
face and head: brow raise (yes/no questions, topics), brow furrow (wh-questions), headshake (negation), eye gaze and
mouth morphemes. This panel draws, frame by frame from the state itself:

- the head turned by ``head_rotation`` (camera frame, y up, the face looking along +z): yaw, pitch and roll move the
  features against the outline, so a nod or a shake is visible;
- brows from ``browInnerUp`` / ``browOuterUp*`` (raise) and ``browDown*`` (lower and draw together);
- eyelids from ``blink`` (or ``eyeBlink*``), widened by ``eyeWide*`` and narrowed by ``eyeSquint*``;
- pupils from ``left_gaze`` / ``right_gaze`` when valid (rotated into the head frame), else from ``eyeLook*``;
- the mouth from ``jawOpen`` (opening), ``mouthSmile*`` / ``mouthStretch*`` (width, corners up), ``mouthFrown*``
  (corners down), ``mouthPucker`` / ``mouthFunnel`` (narrow and round), ``cheekPuff`` (cheek arcs).

It is a diagram of the coefficients, not an appearance render; frames whose face channel is invalid show the outline
only, in grey. The signer's left is drawn on the viewer's right (a camera view of the signer, as the skeleton front
panel), assuming the blendshapes name the signer's own side.
"""
from __future__ import annotations

import numpy as np

from .skeleton_preview import _draw_segment

_INK = np.array([30, 30, 30], np.uint8)
_GREY = np.array([185, 185, 185], np.uint8)
_BROW = np.array([90, 55, 30], np.uint8)
_LIP = np.array([170, 50, 70], np.uint8)


def _coefficients(state, frames: int):
    """name -> (T,) blendshape coefficient (0 where invalid or absent) and the (T,) face validity."""
    face = state.channels["face"]
    valid = face.valid[:, 0].astype(bool) if face.valid.ndim == 2 else np.ones(frames, bool)
    values = {name: np.where(valid, face.values[:, k, 0], 0.) for k, name in enumerate(face.labels)}
    blink = state.channels.get("blink")
    if blink is not None and blink.valid.any():
        for k, name in enumerate(blink.labels):
            values[name] = np.where(blink.valid[:, k], blink.values[:, k, 0], values.get(name, np.zeros(frames)))

    cache: dict[str, np.ndarray] = {}

    def get(name):
        if name not in cache:
            cache[name] = np.clip(np.asarray(values.get(name, np.zeros(frames)), float), 0., 1.)
        return cache[name]
    return get, valid


def _head_matrices(state, frames: int) -> np.ndarray:
    from signtranslator.pose.rotations import rotation_6d_to_matrix
    import torch
    head = state.channels["head_rotation"]
    out = np.tile(np.eye(3), (frames, 1, 1))
    ok = head.valid[:, 0].astype(bool)
    if ok.any():
        out[ok] = rotation_6d_to_matrix(torch.as_tensor(head.values[ok, 0], dtype=torch.float64)).numpy()
    return out


def _gaze_offsets(state, get, rotations, frames: int):
    """(T, 2, 2) pupil offset per eye (signer left, right) in face units, x to the viewer's right, y up."""
    out = np.zeros((frames, 2, 2))
    for e, side in enumerate(("Left", "Right")):
        # blendshape fallback: look-out moves the signer's left pupil to the viewer's right, look-in the other way
        outward = 1. if side == "Left" else -1.
        horizontal = outward * (get(f"eyeLookOut{side}") - get(f"eyeLookIn{side}"))
        vertical = get(f"eyeLookUp{side}") - get(f"eyeLookDown{side}")
        out[:, e] = np.stack((horizontal, vertical), -1) * .07
        channel = state.channels.get(f"{side.lower()}_gaze")
        if channel is not None and channel.valid.any():
            ok = channel.valid[:, 0].astype(bool)
            g = channel.values[:, 0]
            local = np.einsum("tji,tj->ti", rotations, g)        # R^T g: gaze in the head frame
            out[ok, e] = local[ok, :2] * .09
    return out


def _ellipse(cx, cy, rx, ry, n=24):
    a = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack((cx + rx * np.cos(a), cy + ry * np.sin(a), np.zeros_like(a)), -1)


def _polyline(tile, points, colour, radius):
    for a, b in zip(points[:-1], points[1:]):
        _draw_segment(tile, a, b, colour, radius)


def render_face_frames(state, *, width: int = 320, height: int = 320) -> list[np.ndarray]:
    """One RGB face panel (height x width) per frame of ``state``."""
    if width % 2 or height % 2:
        raise ValueError("panel sizes must be even for the encoder")
    frames = len(state.timestamps)
    get, valid = _coefficients(state, frames)
    rotations = _head_matrices(state, frames)
    gaze = _gaze_offsets(state, get, rotations, frames)
    scale = .36 * min(width, height * .8)

    def project(points, t, depth=0.):
        p = np.asarray(points, float).copy()
        p[..., 2] += depth
        q = p @ rotations[t].T
        return np.stack((width / 2 + q[..., 0] * scale, height / 2 - q[..., 1] * scale), -1)

    outline = _ellipse(0., 0., 1., 1.3, 40)
    out = []
    for t in range(frames):
        def c(name, t=t):
            return float(get(name)[t])
        tile = np.full((height, width, 3), 255, np.uint8)
        tile[:, 0] = 200
        _polyline(tile, project(outline, t), _INK if valid[t] else _GREY, 1)
        if not valid[t]:
            out.append(tile)
            continue
        z = .55                                                     # features sit in front of the outline plane
        # nose
        _polyline(tile, project([[0., .12, 0.], [-.07, -.2, 0.], [.06, -.22, 0.]], t, z + .1), _GREY, 1)
        for e, (side, sx) in enumerate((("Left", 1.), ("Right", -1.))):
            down = c(f"browDown{side}")
            inner = .45 + .22 * c("browInnerUp") - .16 * down
            outer = .47 + .22 * c(f"browOuterUp{side}") - .07 * down
            brow = [[sx * (.14 + .05 * (1 - down)), inner, 0.], [sx * .38, (inner + outer) / 2 + .04, 0.],
                    [sx * .62, outer, 0.]]
            _polyline(tile, project(brow, t, z), _BROW, 2)
            closure = c(f"eyeBlink{side}")
            opening = .11 * (1 - closure) * (1 + .6 * c(f"eyeWide{side}") - .4 * c(f"eyeSquint{side}"))
            cx, cy = sx * .38, .17
            a = np.linspace(0, np.pi, 9)
            upper = np.stack((cx + .17 * np.cos(a), cy + max(opening, .006) * np.sin(a), 0 * a), -1)
            lower = np.stack((cx + .17 * np.cos(a), cy - max(opening, .006) * .7 * np.sin(a), 0 * a), -1)
            _polyline(tile, project(upper, t, z), _INK, 1)
            _polyline(tile, project(lower, t, z), _INK, 1)
            if closure < .6:
                pupil = project([[cx + gaze[t, e, 0], cy + gaze[t, e, 1] * .6, 0.]], t, z)[0]
                _draw_segment(tile, pupil, pupil, _INK, 3)
        smile = (c("mouthSmileLeft") + c("mouthSmileRight")) / 2
        stretch = (c("mouthStretchLeft") + c("mouthStretchRight")) / 2
        frown = (c("mouthFrownLeft") + c("mouthFrownRight")) / 2
        pucker, funnel = c("mouthPucker"), c("mouthFunnel")
        half = .3 * (1 + .35 * smile + .3 * stretch - .45 * pucker - .3 * funnel)
        gap = .03 + .45 * c("jawOpen") + .08 * funnel
        corner = .1 * smile - .1 * frown
        my = -.58
        a = np.linspace(-1, 1, 9)
        upper = np.stack((half * a, my + .04 * (1 - a ** 2) + corner * a ** 2, 0 * a), -1)
        lower = np.stack((half * a, my - gap * (1 - a ** 2) + corner * a ** 2, 0 * a), -1)
        _polyline(tile, project(upper, t, z - .05), _LIP, 2)
        _polyline(tile, project(lower, t, z - .05), _LIP, 2)
        puff = c("cheekPuff")
        if puff > .2:
            for sx in (1., -1.):
                b = np.linspace(-.5, .5, 7)
                arc = np.stack((sx * (.78 + .12 * puff * np.cos(b * np.pi)), -.35 + .3 * b, 0 * b), -1)
                _polyline(tile, project(arc, t, z - .2), _GREY, 1)
        out.append(tile)
    return out


def with_face_panel(skeleton_frames: list[np.ndarray], state, *, width: int | None = None) -> list[np.ndarray]:
    """Skeleton preview frames with the face panel appended on the right, at the skeleton's height."""
    height = skeleton_frames[0].shape[0]
    face = render_face_frames(state, width=width or height // 4 * 2, height=height)
    n = min(len(skeleton_frames), len(face))
    return [np.concatenate((skeleton_frames[t], face[t]), axis=1) for t in range(n)]


__all__ = ["render_face_frames", "with_face_panel"]
