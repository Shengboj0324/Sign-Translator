"""Timestamp-exact frame streams from posed canonical states.

Two concerns are kept separate. The *bundle* fixes what is rendered: posed
vertices per frame, rig faces, the state's own clock and full provenance. The
*encoder* fixes when each frame is displayed: every frame's presentation
timestamp (PTS) is derived from the canonical clock, so a nonuniform source clock
survives into the video instead of being resampled to a nominal frame rate.

Timestamp arithmetic is exact. With time base ``b`` (seconds per tick) and clock
values ``t_i``, the encoder uses ``PTS_i = round((t_i - t_0) / b)``, computed on
the exact binary rationals of the float64 inputs with ``fractions.Fraction``, so
the representation error satisfies ``|PTS_i b - (t_i - t_0)| <= b / 2``. Two
clock values closer than one tick could round to the same PTS; the encoder then
refuses rather than merging or reordering frames. ``t_0`` is recorded so the
absolute clock is recoverable.

A rendered stream is geometry made visible, not accepted signing: provenance
records whether inferred values were posed and keeps ``accepted_provenance``
false.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..avatar_render.makehuman import MakeHumanRig
from ..pose.multichannel import MultichannelMotion
from .state_rig import StateRigBinding, pose_state_sequence

DEFAULT_TIME_BASE = Fraction(1, 90000)


def presentation_timestamps(clock: Sequence[float], *,
                            time_base: Fraction = DEFAULT_TIME_BASE) -> list[int]:
    """Exact rounded ticks relative to the first clock value; strictly increasing."""
    if not isinstance(time_base, Fraction) or time_base <= 0:
        raise ValueError("time_base must be a positive exact Fraction")
    values = [float(value) for value in clock]
    if not values or not all(np.isfinite(values)):
        raise ValueError("clock must contain finite values")
    origin = Fraction(values[0])
    ticks = [round((Fraction(value) - origin) / time_base) for value in values]
    for previous, current in zip(ticks, ticks[1:]):
        if current <= previous:
            raise ValueError(
                "two frames round to the same or earlier tick; refine the time base "
                "instead of merging frames")
    return ticks


def timestamp_error_bound(clock: Sequence[float], ticks: Sequence[int],
                          time_base: Fraction = DEFAULT_TIME_BASE) -> Fraction:
    """Exact maximum |PTS * b - (t - t0)| over all frames."""
    origin = Fraction(float(clock[0]))
    return max(abs(tick * time_base - (Fraction(float(value)) - origin))
               for tick, value in zip(ticks, clock))


@dataclass(frozen=True)
class FrameBundle:
    path: Path
    sha256: str
    frames: int
    provenance: dict


def write_frame_bundle(rig: MakeHumanRig, state: MultichannelMotion,
                       binding: StateRigBinding, path: Path, *,
                       allow_inferred: bool) -> FrameBundle:
    """Pose every frame and write a no-overwrite NPZ bundle with provenance.

    Vertices are stored as float32 for the renderer, which bounds bundle size;
    posing itself runs in the rig's float64. The rounding of stored positions
    is at most half a float32 unit in the last place of each coordinate.
    """
    frames, provenances = [], []
    for posed in pose_state_sequence(rig, state, binding, allow_inferred=allow_inferred):
        vertices = posed.mesh.vertices.detach().cpu().numpy()
        if not np.isfinite(vertices).all():
            raise ValueError("posed vertices are not finite")
        frames.append(vertices.astype(np.float32))
        provenances.append(posed.provenance())
    faces = np.full((len(rig.faces), max(len(face) for face in rig.faces)), -1, dtype=np.int64)
    for index, face in enumerate(rig.faces):
        faces[index, :len(face)] = face
    renderable = np.array(["body" in groups for groups in rig.face_groups], dtype=bool)
    provenance = dict(
        schema_version=1, sample_id=state.sample_id, clock_id=state.clock_id,
        binding_sha256=binding.sha256(), asset_commit=rig.asset_commit,
        rig_units=rig.coordinate_units, frames=len(frames),
        posed_inferred_values=any(item["posed_inferred_values"] for item in provenances),
        accepted_provenance=False,
        renderable_faces="faces whose OBJ groups include 'body'; helpers excluded")
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, vertices=np.stack(frames), faces=faces,
                            renderable=renderable,
                            clock=np.asarray(state.timestamps, dtype=np.float64),
                            provenance=np.array(json.dumps(provenance, sort_keys=True)))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return FrameBundle(Path(path), digest, len(frames), provenance)


def encode_timestamped_video(frames: Iterable[np.ndarray], clock: Sequence[float],
                             path: Path, *, time_base: Fraction = DEFAULT_TIME_BASE,
                             codec: str = "libx264", quality: str = "lossless") -> dict:
    """Encode RGB frames with PTS taken from the clock; refuse any reordering.

    ``quality='lossless'`` (default) is H.264 4:4:4 at CRF 0. ``'review'`` is H.264
    4:2:0 at CRF 16, which common players (QuickTime, browsers) decode; timestamps
    are identical either way. Returns the ticks, time base, absolute clock origin
    and exact error bound.
    """
    if quality not in ("lossless", "review"):
        raise ValueError("quality must be 'lossless' or 'review'")
    import av

    ticks = presentation_timestamps(clock, time_base=time_base)
    frames = list(frames)
    if len(frames) != len(ticks):
        raise ValueError("frame count must equal clock length")
    height, width = frames[0].shape[:2]
    if any(frame.shape != (height, width, 3) or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("frames must share one uint8 HxWx3 shape")
    if height % 2 or width % 2:
        raise ValueError("encoder requires even frame dimensions")
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=None)
        stream.width, stream.height = width, height
        lossless = codec == "libx264" and quality == "lossless"
        stream.pix_fmt = "yuv444p" if lossless else "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        if lossless:
            stream.options = {"crf": "0", "preset": "veryslow"}
        elif codec == "libx264":
            stream.options = {"crf": "16", "preset": "slow"}
        for tick, image in zip(ticks, frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = tick
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    bound = timestamp_error_bound(clock, ticks, time_base)
    return dict(ticks=ticks, time_base=str(time_base), clock_origin_seconds=float(clock[0]),
                codec=codec, pixel_format="yuv444p" if (codec == "libx264" and quality == "lossless") else "yuv420p",
                max_timestamp_error_seconds=float(bound),
                max_timestamp_error_exact=str(bound), frames=len(ticks))


def verify_timestamped_video(path: Path, clock: Sequence[float], *,
                             time_base: Fraction = DEFAULT_TIME_BASE) -> dict:
    """Decode the stream and check frame count, order and every timestamp."""
    import av

    expected = presentation_timestamps(clock, time_base=time_base)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = [(frame.pts, Fraction(frame.time_base)) for frame in container.decode(stream)]
    if len(decoded) != len(expected):
        raise ValueError(f"decoded {len(decoded)} frames, expected {len(expected)}")
    observed = []
    for pts, base in decoded:
        seconds = pts * base
        observed.append(seconds)
    origin = observed[0]
    worst = Fraction(0)
    for tick, seconds in zip(expected, observed):
        worst = max(worst, abs((seconds - origin) - tick * time_base))
    if any(b <= a for a, b in zip(observed, observed[1:])):
        raise ValueError("decoded timestamps are not strictly increasing")
    clock_error = max(abs((seconds - origin) - (Fraction(float(value)) - Fraction(float(clock[0]))))
                      for seconds, value in zip(observed, clock))
    if clock_error > time_base / 2:
        raise ValueError("a decoded timestamp exceeds the half-tick error bound")
    return dict(frames=len(decoded), max_pts_drift_seconds=float(worst),
                max_clock_error_seconds=float(clock_error),
                half_tick_bound_seconds=float(time_base / 2), strictly_increasing=True)
