"""Render-side temporal filter: remove shake from a canonical state before retargeting.

Applied by ``scripts/render_avatar.py`` to every state, with one fixed parameter table (:data:`FILTER_V1`), so no arm can
get another. The filter never sees a text, a label or a source: only the state's values on its own clock. It is for drawing
only; the result is never a training or scoring input.

**Why world frames for the arms.** A canonical arm rotation is stored relative to its parent, so noise in the chest
frame appears, inverted, in the upper arm's local rotation, and the two cancel in the world. Filtering the chest hard and the
arm lightly in local frames would break that cancellation and push the chest's noise into the wrist. So pelvis, chest, upper
arm, forearm and hand are filtered as world rotations, and locals are recomposed afterwards. Fingers are filtered as
stored, relative to the hand.

**Gaps.** Invalid frames are filled first, because a filter over a held value and then a step smears the step: rotations by
geodesic interpolation between the valid neighbours whatever the gap's length, other values linearly, the clip's ends held.
A joint that is never valid stays invalid (the retarget gives it the rig's rest relation). Every fill is counted.

**Methods.** ``SavitzkyGolay`` fits a polynomial in a sliding window of the rotation's log map about the centre frame. It is
zero-phase, and it keeps a ramp or a parabola exactly. ``OneEuro`` runs Casiez's adaptive low-pass forward and backward on the
same manifold and takes the midpoint: its cutoff rises with speed, so it smooths holds hard and passes fast changes, and
running it both ways cancels its lag.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json

import numpy as np
from scipy.signal import savgol_coeffs
from scipy.spatial.transform import Rotation

from ..pose.multichannel import MultichannelMotion

BODY_WORLD_ORDER = ("Pelvis", "Chest", "LeftUpperArm", "LeftForearm", "LeftHand", "RightUpperArm", "RightForearm", "RightHand")
BODY_PARENT = {"Pelvis": None, "Chest": "Pelvis", "LeftUpperArm": "Chest", "LeftForearm": "LeftUpperArm", "LeftHand": "LeftForearm",
               "RightUpperArm": "Chest", "RightForearm": "RightUpperArm", "RightHand": "RightForearm"}


@dataclass(frozen=True)
class SavitzkyGolay:
    window: int             # odd, frames
    order: int

    def describe(self) -> dict:
        return dict(method="savitzky-golay", window=self.window, order=self.order)


@dataclass(frozen=True)
class OneEuro:
    min_cutoff_hz: float
    beta: float             # Hz of cutoff per unit of speed (rad/s for rotations, the value's unit per second otherwise)
    derivative_cutoff_hz: float = 1.

    def describe(self) -> dict:
        return dict(method="one-euro-two-way", min_cutoff_hz=self.min_cutoff_hz, beta=self.beta,
                    derivative_cutoff_hz=self.derivative_cutoff_hz)


@dataclass(frozen=True)
class FilterSettings:
    """One method per group of channels; ``None`` leaves a group as it is."""
    torso: object = None            # Pelvis, Chest (world)
    arm: object = None              # upper arm, forearm (world)
    hand: object = None             # hand frame (world)
    finger: object = None           # all 15 finger joints per hand (local to the hand)
    leg: object = None              # thighs, shins (local; held at rest by default, filtered for completeness)
    head: object = None             # head_rotation (world)
    translation: object = None      # root_translation, head_translation (not drawn, filtered for completeness)
    face: object = None             # 52 blendshape coefficients
    blink: object = None
    gaze: object = None

    def describe(self) -> dict:
        return {k: (None if v is None else v.describe()) for k, v in self.__dict__.items()}

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()


_LIGHT = SavitzkyGolay(5, 2)        # -3 dB at 5.7 Hz; gain 0.97 at 3 Hz, 0.91 at 4 Hz, 0.66 at 6 Hz, 0.23 at 8 Hz, 0.19 at 10 Hz, 0.37 at 12 Hz (a window's sidelobe)
_TORSO = SavitzkyGolay(9, 2)        # -3 dB at 2.9 Hz; How2Sign's torso is jitter-dominated (velocity share at 6 Hz and up: 0.68)

#: Chosen by measurement (docs/AVATAR_RENDER.md section 13). The only table the renderer uses; no argument changes it.
FILTER_V1 = FilterSettings(torso=_TORSO, arm=_LIGHT, hand=_LIGHT, finger=_LIGHT, leg=_LIGHT, head=_LIGHT, translation=_TORSO,
                           face=_LIGHT, blink=_LIGHT, gaze=_LIGHT)


def _from_6d(values: np.ndarray) -> np.ndarray:
    a, b = values[..., :3], values[..., 3:]
    r1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (r1 * b).sum(-1, keepdims=True) * r1
    r2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.stack((r1, r2, np.cross(r1, r2)), axis=-1)


def _to_6d(matrices: np.ndarray) -> np.ndarray:
    return np.concatenate((matrices[..., 0], matrices[..., 1]), axis=-1)


def _log(matrices: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(*matrices.shape[:-2], 3)


def _exp(rotvecs: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix().reshape(*rotvecs.shape[:-1], 3, 3)


def _savgol_weights(frames: int, window: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, W) weights and offsets: the fit's value at each frame, on an asymmetric window at the clip's ends."""
    window = min(window, frames if frames % 2 else frames - 1)
    half = window // 2
    centre = savgol_coeffs(window, min(order, window - 1), pos=half, use="dot")
    offsets = np.arange(-half, half + 1)
    weights = np.tile(centre, (frames, 1))
    for t in range(half):
        coefficient = savgol_coeffs(window, min(order, window - 1), pos=t, use="dot")
        weights[t] = coefficient
        weights[frames - 1 - t] = coefficient[::-1]
    return weights, offsets


def _clip_index(t: int, offset: np.ndarray, frames: int, half: int) -> np.ndarray:
    return t + offset + (half - t if t < half else 0) - (t + half - frames + 1 if t + half >= frames else 0)


def _savgol_rotations(rotations: np.ndarray, spec: SavitzkyGolay) -> np.ndarray:
    frames = len(rotations)
    if frames < 3 or spec.window < 3:
        return rotations.copy()
    weights, offsets = _savgol_weights(frames, spec.window, spec.order)
    half = len(offsets) // 2
    out = np.empty_like(rotations)
    for t in range(frames):
        neighbours = _clip_index(t, offsets, frames, half)
        relative = np.swapaxes(rotations[t], -1, -2)[None] @ rotations[neighbours]
        correction = np.einsum("w,wjk->jk", weights[t], _log(relative))
        out[t] = rotations[t] @ _exp(correction)
    return out


def _savgol_values(values: np.ndarray, spec: SavitzkyGolay) -> np.ndarray:
    frames = len(values)
    if frames < 3 or spec.window < 3:
        return values.copy()
    weights, offsets = _savgol_weights(frames, spec.window, spec.order)
    half = len(offsets) // 2
    out = np.empty_like(values)
    for t in range(frames):
        out[t] = np.einsum("w,w...->...", weights[t], values[_clip_index(t, offsets, frames, half)])
    return out


def _alpha(cutoff: np.ndarray, fps: float) -> np.ndarray:
    tau = 1. / (2. * np.pi * cutoff)
    return 1. / (1. + tau * fps)


def _one_euro_pass_rotations(rotations: np.ndarray, spec: OneEuro, fps: float) -> np.ndarray:
    frames, joints = rotations.shape[:2]
    out = np.empty_like(rotations)
    out[0] = rotations[0]
    speed = np.zeros(joints)
    alpha_d = float(_alpha(np.array(spec.derivative_cutoff_hz), fps))
    for t in range(1, frames):
        step = _log(np.swapaxes(rotations[t - 1], -1, -2) @ rotations[t])
        speed = speed + alpha_d * (np.linalg.norm(step, axis=-1) * fps - speed)
        cutoff = spec.min_cutoff_hz + spec.beta * speed
        difference = _log(np.swapaxes(out[t - 1], -1, -2) @ rotations[t])
        out[t] = out[t - 1] @ _exp(difference * _alpha(cutoff, fps)[..., None])
    return out


def _one_euro_rotations(rotations: np.ndarray, spec: OneEuro, fps: float) -> np.ndarray:
    forward = _one_euro_pass_rotations(rotations, spec, fps)
    backward = _one_euro_pass_rotations(rotations[::-1], spec, fps)[::-1]
    half = _log(np.swapaxes(forward, -1, -2) @ backward) * .5
    return forward @ _exp(half)


def _one_euro_pass_values(values: np.ndarray, spec: OneEuro, fps: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    speed = np.zeros(values.shape[1:])
    alpha_d = float(_alpha(np.array(spec.derivative_cutoff_hz), fps))
    for t in range(1, len(values)):
        speed = speed + alpha_d * (np.abs(values[t] - values[t - 1]) * fps - speed)
        out[t] = out[t - 1] + _alpha(spec.min_cutoff_hz + spec.beta * speed, fps) * (values[t] - out[t - 1])
    return out


def filter_rotations(rotations: np.ndarray, spec, fps: float) -> np.ndarray:
    """(T, J, 3, 3) rotations through ``spec``; ``None`` copies."""
    if spec is None or len(rotations) < 3:
        return rotations.copy()
    if isinstance(spec, SavitzkyGolay):
        return _savgol_rotations(rotations, spec)
    if isinstance(spec, OneEuro):
        return _one_euro_rotations(rotations, spec, fps)
    raise TypeError(f"unknown filter {spec!r}")


def filter_values(values: np.ndarray, spec, fps: float) -> np.ndarray:
    """(T, ...) real values through ``spec``; ``None`` copies."""
    if spec is None or len(values) < 3:
        return values.copy()
    if isinstance(spec, SavitzkyGolay):
        return _savgol_values(values, spec)
    if isinstance(spec, OneEuro):
        forward = _one_euro_pass_values(values, spec, fps)
        backward = _one_euro_pass_values(values[::-1], spec, fps)[::-1]
        return (forward + backward) / 2.
    raise TypeError(f"unknown filter {spec!r}")


def _fill_rotations(clock: np.ndarray, matrices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Geodesic interpolation between valid neighbours on the clock; the ends held."""
    frames = len(clock)
    indices = np.arange(frames)
    previous = np.maximum.accumulate(np.where(valid, indices, -1))
    following = np.minimum.accumulate(np.where(valid, indices, frames)[::-1])[::-1]
    p = np.where(previous >= 0, previous, following)
    n = np.where(following < frames, following, previous)
    span = clock[n] - clock[p]
    fraction = np.where(span > 0, (clock - clock[p]) / np.where(span > 0, span, 1.), 0.)
    relative = np.swapaxes(matrices[p], -1, -2) @ matrices[n]
    filled = matrices[p] @ _exp(_log(relative) * fraction[:, None])
    return np.where(valid[:, None, None], matrices, filled)


def _fill_values(clock: np.ndarray, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1).copy()
    for d in range(flat.shape[1]):
        flat[:, d] = np.interp(clock, clock[valid], flat[valid, d])
    return flat.reshape(values.shape)


def _gap_stats(clock: np.ndarray, valid: np.ndarray) -> dict:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (~valid).astype(int), [0]))))
    lengths = [float(clock[min(b, len(clock) - 1)] - clock[max(a - 1, 0)]) for a, b in zip(edges[::2], edges[1::2])]
    return dict(valid_frames=int(valid.sum()), filled_frames=int((~valid).sum()), gaps=len(lengths),
                longest_gap_seconds=max(lengths) if lengths else 0.)


ROTATION_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")


def fill_gaps(state: MultichannelMotion) -> tuple[MultichannelMotion, dict]:
    """Every element's invalid frames filled (module docstring); elements never valid are left invalid. Returns the record."""
    clock = np.asarray(state.timestamps, dtype=np.float64)
    channels, record = {}, {}
    for name, channel in state.channels.items():
        if name == "contact" or not channel.valid.any():
            channels[name] = channel
            continue
        values = channel.values.astype(np.float64).copy()
        valid = channel.valid.copy()
        for j, label in enumerate(channel.labels):
            ok = valid[:, j]
            if ok.all() or not ok.any():
                continue
            record[f"{name}/{label}"] = _gap_stats(clock, ok)
            if name in ROTATION_CHANNELS:
                matrices = np.tile(np.eye(3), (len(clock), 1, 1))
                matrices[ok] = _from_6d(values[ok, j])
                values[:, j] = _to_6d(_fill_rotations(clock, matrices, ok))
            else:
                values[:, j] = _fill_values(clock, values[:, j], ok)
                if name.endswith("_gaze"):
                    values[:, j] /= np.linalg.norm(values[:, j], axis=-1, keepdims=True).clip(1e-9)
            valid[:, j] = True
        channels[name] = replace(channel, values=values.astype(channel.values.dtype), valid=valid)
    return replace(state, channels=channels), record


def _filter_body(channel, settings: FilterSettings, fps: float):
    labels = channel.labels
    values = channel.values.astype(np.float64).copy()
    frames = len(values)
    usable = {name: labels.index(name) for name in BODY_WORLD_ORDER if name in labels and channel.valid[:, labels.index(name)].all()}
    local = {name: _from_6d(values[:, k]) for name, k in usable.items()}
    world, filtered = {}, {}
    for name in BODY_WORLD_ORDER:                                     # world rotations of the chain, parents first
        if name not in usable:
            continue
        parent = BODY_PARENT[name]
        if parent is None:
            world[name] = local[name]
        elif parent in world:
            world[name] = world[parent] @ local[name]
    groups = (("torso", ("Pelvis", "Chest")), ("arm", ("LeftUpperArm", "LeftForearm", "RightUpperArm", "RightForearm")),
              ("hand", ("LeftHand", "RightHand")))
    for group, names in groups:
        present = [n for n in names if n in world]
        if not present:
            continue
        stack = filter_rotations(np.stack([world[n] for n in present], axis=1), getattr(settings, group), fps)
        filtered.update({n: stack[:, i] for i, n in enumerate(present)})
    for name in BODY_WORLD_ORDER:                                     # locals recomposed from filtered worlds
        if name not in world:
            continue
        parent = BODY_PARENT[name]
        if parent is None:
            recomposed = filtered[name]
        else:
            recomposed = np.swapaxes(filtered[parent], -1, -2) @ filtered[name]
        values[:, usable[name]] = _to_6d(recomposed)
    legs = [k for k, name in enumerate(labels) if name not in BODY_WORLD_ORDER and channel.valid[:, k].all()]
    if legs and settings.leg is not None:
        stack = filter_rotations(_from_6d(values[:, legs]), settings.leg, fps)
        values[:, legs] = _to_6d(stack)
    if frames != len(channel.values):
        raise AssertionError
    return replace(channel, values=values.astype(channel.values.dtype))


def _filter_rotation_channel(channel, spec, fps: float):
    keep = [k for k in range(len(channel.labels)) if channel.valid[:, k].all()]
    if not keep or spec is None:
        return channel
    values = channel.values.astype(np.float64).copy()
    values[:, keep] = _to_6d(filter_rotations(_from_6d(values[:, keep]), spec, fps))
    return replace(channel, values=values.astype(channel.values.dtype))


def _filter_value_channel(channel, spec, fps: float, *, clip01: bool = False, unit: bool = False):
    keep = [k for k in range(len(channel.labels)) if channel.valid[:, k].all()]
    if not keep or spec is None:
        return channel
    values = channel.values.astype(np.float64).copy()
    smooth = filter_values(values[:, keep], spec, fps)
    if clip01:
        smooth = np.clip(smooth, 0., 1.)
    if unit:
        smooth = smooth / np.linalg.norm(smooth, axis=-1, keepdims=True).clip(1e-9)
    values[:, keep] = smooth
    return replace(channel, values=values.astype(channel.values.dtype))


def filter_state(state: MultichannelMotion, settings: FilterSettings) -> MultichannelMotion:
    """A gap-free state's channels through ``settings``, on the state's own frame rate. No value's clock or length changes."""
    if len(state.timestamps) < 2:
        return state
    fps = 1. / float(np.median(np.diff(state.timestamps)))
    channels = dict(state.channels)
    channels["body"] = _filter_body(channels["body"], settings, fps)
    for name in ("left_hand", "right_hand"):
        channels[name] = _filter_rotation_channel(channels[name], settings.finger, fps)
    channels["head_rotation"] = _filter_rotation_channel(channels["head_rotation"], settings.head, fps)
    for name in ("root_translation", "head_translation"):
        if name in channels:
            channels[name] = _filter_value_channel(channels[name], settings.translation, fps)
    if "face" in channels:
        channels["face"] = _filter_value_channel(channels["face"], settings.face, fps, clip01=True)
    if "blink" in channels:
        channels["blink"] = _filter_value_channel(channels["blink"], settings.blink, fps, clip01=True)
    for name in ("left_gaze", "right_gaze"):
        if name in channels:
            channels[name] = _filter_value_channel(channels[name], settings.gaze, fps, unit=True)
    return replace(state, channels=channels)


__all__ = ["SavitzkyGolay", "OneEuro", "FilterSettings", "FILTER_V1", "fill_gaps", "filter_state", "filter_rotations", "filter_values"]
