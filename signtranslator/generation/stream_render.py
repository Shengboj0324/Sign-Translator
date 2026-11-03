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
#: ``quality='constant'``: H.264 4:2:0 held below 2 Mb/s by x264's VBV (``encoder_fraction`` of the rate and the 1 s buffer, one value for
#: every clip, so no clip's rate or quality follows its content), with a fixed frame-type pattern; x264's own settings and version SEI is
#: dropped and the muxer writes no version; then SEI units of incompressible user data, spread over the samples, top the stream up to
#: exactly ``rate * frames / fps + buffer`` bits, and the file is laid out one sample to a chunk. Stream size, stream and file bit rate,
#: frame types, and every MP4 box's size, and its bytes but for the sample sizes and sample offsets, are then a function of the frame count
#: alone, and a zip can't shrink the top-up back to the content's size, as it would x264's own CBR filler (0xFF bytes). On a busy synthetic
#: 768 px avatar clip it scored above CRF 16 on PSNR, hands included (docs/AVATAR_RENDER.md section 13).
CONSTANT_RATE = dict(bits_per_second=2_000_000, buffer_bits=2_000_000, keyint=48, preset="slow", encoder_fraction=.9)


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
    4:2:0 at CRF 16, which common players (QuickTime, browsers) decode. ``'constant'``
    is :data:`CONSTANT_RATE`. Timestamps are identical either way. Returns the ticks,
    time base, absolute clock origin and exact error bound.
    """
    if quality not in ("lossless", "review", "constant"):
        raise ValueError("quality must be 'lossless', 'review' or 'constant'")
    import av
    if quality == "constant":
        return _encode_constant_rate(frames, clock, Path(path), time_base=time_base, codec=codec)

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


def constant_rate_budget(frames: int, fps: float) -> int:
    """The stream bytes of a ``quality='constant'`` encode of ``frames`` frames at ``fps``: what the rate sends over the clip plus one buffer."""
    import math
    return math.ceil((CONSTANT_RATE["bits_per_second"] * frames / fps + CONSTANT_RATE["buffer_bits"]) / 8)


def _nominal_rate(clock: Sequence[float]) -> Fraction:
    steps = np.diff(np.asarray(clock, dtype=np.float64))
    return Fraction(1. / float(np.median(steps))).limit_denominator(1001) if len(steps) else Fraction(24)


def _encode_constant_rate(frames, clock, path: Path, *, time_base: Fraction, codec: str) -> dict:
    """``quality='constant'``: encode under :data:`CONSTANT_RATE` to a sibling file, then copy every packet to ``path`` with SEI units of
    unregistered user data, spread over the samples, that bring the stream to :func:`constant_rate_budget`, and lay the file out one sample
    to a chunk. Packets, timestamps and pictures are unchanged; decoders skip the SEI units.

    x264 is held to ``encoder_fraction`` of the budget's rate and buffer, and its buffer starts 90% full, so a stream can run past the rate
    by up to that buffer; the budget's own rate and buffer above it leave 0.2 Mb/s of the clip's length plus 0.38 Mb of room. A stream
    still over its budget, or too close to it to hold one filler unit, is refused, never encoded again at a lower rate: a clip at another
    rate would carry another quality, and an arm whose clips run over more often would be marked by it. One rate holds for every clip
    because it is in the settings hash, which a packet holds every render to. The room left is recorded. Nothing is left at ``path`` or
    beside it after a failure."""
    import av
    if codec != "libx264":
        raise ValueError("quality='constant' is libx264's")
    rate = CONSTANT_RATE
    ticks = presentation_timestamps(clock, time_base=time_base)
    frames = list(frames)
    if len(frames) != len(ticks):
        raise ValueError("frame count must equal clock length")
    height, width = frames[0].shape[:2]
    if any(frame.shape != (height, width, 3) or frame.dtype != np.uint8 for frame in frames):
        raise ValueError("frames must share one uint8 HxWx3 shape")
    if height % 2 or width % 2:
        raise ValueError("encoder requires even frame dimensions")
    fps = _nominal_rate(clock)
    budget = constant_rate_budget(len(frames), float(fps))
    raw = path.with_name(path.name + ".rate")
    attempts = []
    try:
        bits, buffer = round(rate["bits_per_second"] * rate["encoder_fraction"]), round(rate["buffer_bits"] * rate["encoder_fraction"])
        with av.open(str(raw), mode="w", format="mp4") as container:
            stream = container.add_stream(codec, rate=None)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.codec_context.framerate = fps
            stream.options = {"preset": rate["preset"], "b": str(bits), "maxrate": str(bits), "bufsize": str(buffer),
                              "x264-params": f"force-cfr=1:b-adapt=0:scenecut=0:keyint={rate['keyint']}:min-keyint={rate['keyint']}"}
            for tick, image in zip(ticks, frames):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = tick
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(raw)) as source:
            template = source.streams.video[0]
            packets = [(p.pts, p.dts, p.duration, p.is_keyframe, _without_sei(bytes(p))) for p in source.demux(template) if p.size]
        stream_bytes = sum(len(p[4]) for p in packets)
        room = budget - stream_bytes
        attempts.append(dict(encoder_bits_per_second=bits, encoder_buffer_bits=buffer, stream_bytes_before_filler=stream_bytes, room=room))
        if room < 0 or 0 < room < _SEI_SMALLEST:
            raise ValueError(f"constant-rate encoding failed: the stream is {stream_bytes} bytes against a budget of {budget}; the packet's one "
                             "rate isn't lowered for one clip")
        seed = hashlib.sha256(b"".join(p[4] for p in packets)).digest()       # unique to the clip: no archiver can share it between files
        shares = _sei_shares(budget - stream_bytes, len(packets))
        with av.open(str(raw)) as source:
            template = source.streams.video[0]
            with av.open(str(path), mode="w", format="mp4", container_options={"fflags": "+bitexact"}) as target:
                out = target.add_stream_from_template(template)
                for n, (pts, dts, duration, key, data) in enumerate(packets):
                    packet = av.Packet(_with_sei(data, shares[n], seed, n))
                    packet.pts, packet.dts, packet.duration, packet.time_base = pts, dts, duration, template.time_base
                    packet.is_keyframe = key
                    packet.stream = out
                    target.mux(packet)
        one_sample_per_chunk(path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    bound = timestamp_error_bound(clock, ticks, time_base)
    used = attempts[-1]
    return dict(ticks=ticks, time_base=str(time_base), clock_origin_seconds=float(clock[0]), codec=codec, pixel_format="yuv420p",
                max_timestamp_error_seconds=float(bound), max_timestamp_error_exact=str(bound), frames=len(ticks),
                constant_rate=dict(rate, fps=str(fps), encoder_bits_per_second=used["encoder_bits_per_second"], attempts=attempts,
                                   room_share=used["room"] / budget,
                                   stream_bytes_before_filler=used["stream_bytes_before_filler"], stream_bytes=budget, filler_bytes=used["room"]))


def _mp4_boxes(data: bytes, start: int, end: int) -> list[tuple[str, int, int, int]]:
    """(kind, offset, size, header length) of every box from ``start`` to ``end``, which must end exactly there."""
    import struct
    out, i = [], start
    while i < end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < header or i + size > end:
            raise ValueError(f"a broken MP4 box chain at {i}")
        out.append((kind.decode("latin-1"), i, size, header))
        i += size
    return out


def one_sample_per_chunk(path: Path) -> None:
    """Rewrite an MP4's one video track to one sample per chunk (``stsc`` one entry, ``stco`` one offset a sample). FFmpeg starts a new
    chunk wherever a chunk would pass 1 MiB, so its layout follows the content; this one follows the frame count. ``moov`` must come after
    ``mdat``, so no sample moves; only ``moov`` is rewritten."""
    import struct
    data = Path(path).read_bytes()
    top = _mp4_boxes(data, 0, len(data))
    kinds = [b[0] for b in top]
    if kinds.count("moov") != 1 or kinds.count("mdat") != 1 or kinds.index("moov") < kinds.index("mdat"):
        raise ValueError(f"{path}: expected one mdat, then one moov")

    def rebuild(offset, size, header, trail):
        """The box at ``offset`` rebuilt, descending along ``trail`` (kinds) to stbl, where stsc and stco are replaced."""
        kind = data[offset + header - 4:offset + header].decode("latin-1")
        children = _mp4_boxes(data, offset + header, offset + size)
        if not trail:
            boxes = {k: (o, s, h) for k, o, s, h in children}
            o, s, h = boxes["stsz"]
            uniform, count = struct.unpack(">II", data[o + h + 4:o + h + 12])
            sizes = [uniform] * count if uniform else list(struct.unpack(f">{count}I", data[o + h + 12:o + h + 12 + 4 * count]))
            o, s, h = boxes["stsc"]
            entries = struct.unpack(">I", data[o + h + 4:o + h + 8])[0]
            runs = [struct.unpack(">III", data[o + h + 8 + 12 * k:o + h + 20 + 12 * k]) for k in range(entries)]
            if "co64" in boxes or {r[2] for r in runs} != {1}:
                raise ValueError(f"{path}: 64-bit offsets or several sample descriptions")
            o, s, h = boxes["stco"]
            chunks = struct.unpack(">I", data[o + h + 4:o + h + 8])[0]
            starts = struct.unpack(f">{chunks}I", data[o + h + 8:o + h + 8 + 4 * chunks])
            offsets, sample = [], 0
            for c, start in enumerate(starts, start=1):
                per = next(r[1] for r in reversed(runs) if r[0] <= c)
                for _ in range(per):
                    offsets.append(start)
                    start += sizes[sample]
                    sample += 1
            if sample != count:
                raise ValueError(f"{path}: the chunk table holds {sample} samples, the size table {count}")
            replaced = {"stsc": b"stsc" + bytes(4) + struct.pack(">IIII", 1, 1, 1, 1),
                        "stco": b"stco" + bytes(4) + struct.pack(f">I{count}I", count, *offsets)}
            body = b"".join(struct.pack(">I", len(replaced[k]) + 4) + replaced[k] if k in replaced else data[o:o + s] for k, o, s, _ in children)
        else:
            body = b"".join(rebuild(o, s, h, trail[1:]) if k == trail[0] else data[o:o + s] for k, o, s, h in children)
            if sum(k == trail[0] for k, _, _, _ in children) != 1:
                raise ValueError(f"{path}: expected one {trail[0]} in {kind}")
        return struct.pack(">I", len(body) + 8) + kind.encode("latin-1") + body

    _, offset, size, header = top[kinds.index("moov")]
    moov = rebuild(offset, size, header, ("trak", "mdia", "minf", "stbl"))
    Path(path).write_bytes(data[:offset] + moov + data[offset + size:])


_SEI_SMALLEST = 4 + 1 + 2 + 16 + 1               # length prefix, NAL header, one message's type and size, its UUID, trailing bits


def _sei_unit(total: int, seed: bytes, counter: int) -> bytes:
    """One length-prefixed SEI NAL unit (type 6), exactly ``total`` bytes, of unregistered user-data messages (payload type 5) of at most
    254 bytes each, every UUID and data byte SHA-256 output with no zero byte. So no start code can appear, no size field runs to 0xFF, and
    nothing in it repeats: it doesn't compress, and decoders skip it."""
    body = total - 4 - 1 - 1                                  # the messages
    if body < 18:
        raise ValueError(f"an SEI unit of {total} bytes is too small")
    count = -(-body // 256)
    sizes = [body // count + (k < body % count) - 2 for k in range(count)]
    random = b"".join(hashlib.sha256(seed + counter.to_bytes(8, "big") + i.to_bytes(8, "big")).digest() for i in range((sum(sizes) + 31) // 32))
    random = (np.frombuffer(random[:sum(sizes)], np.uint8) % 255 + 1).astype(np.uint8).tobytes()
    parts, at = [b"\x06"], 0
    for size in sizes:
        parts.append(b"\x05" + bytes([size]) + random[at:at + size])
        at += size
    unit = b"".join(parts) + b"\x80"
    return len(unit).to_bytes(4, "big") + unit


def _sei_shares(need: int, samples: int) -> list[int]:
    """``need`` bytes as one SEI unit's size per sample, as even as they go over as many samples as can hold one (0: none)."""
    if need < 0:
        raise ValueError(f"the stream is {-need} bytes over its budget: constant-rate encoding failed")
    used = min(samples, need // _SEI_SMALLEST)
    if need and not used:
        raise ValueError(f"the stream is {need} bytes under its budget, too few for one filler unit: constant-rate encoding failed")
    return [need // used + (n < need % used) if n < used else 0 for n in range(samples)]


def _without_sei(sample: bytes) -> bytes:
    """``sample`` (length-prefixed NAL units) without its SEI units: x264 writes its version and every setting, the rate included, into one."""
    out, i = [], 0
    while i < len(sample):
        end = i + 4 + int.from_bytes(sample[i:i + 4], "big")
        if sample[i + 4] & 31 != 6:
            out.append(sample[i:end])
        i = end
    return b"".join(out)


def _with_sei(sample: bytes, size: int, seed: bytes, counter: int) -> bytes:
    """``sample`` (length-prefixed NAL units) with an SEI unit of ``size`` bytes put before its first slice, where an SEI unit belongs."""
    if not size:
        return sample
    i = 0
    while i < len(sample) and (sample[i + 4] & 31) not in (1, 5):          # past the access unit delimiter, parameter sets and SEI
        i += 4 + int.from_bytes(sample[i:i + 4], "big")
    return sample[:i] + _sei_unit(size, seed, counter) + sample[i:]


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
