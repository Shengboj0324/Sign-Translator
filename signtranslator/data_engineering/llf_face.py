"""Live Link Face (ARKit) CSV adapter: face coefficients, gaze and eyelid state.

The 3D-LEX ``ASL_LLF`` archive holds one CSV per sign from Epic's Live Link Face
app: a device timecode, a blendshape count, 52 ARKit coefficients and nine
rotations (head and each eye, yaw/pitch/roll). This adapter was written after
measuring every one of the 997 files; each rule below answers a measured defect.

**Clock.** ``HH:MM:SS:FF.fff`` is a device timecode in frames of the declared rate
(60 here) with a fractional subframe. Time is computed exactly in rationals,
``t = 3600 h + 60 m + s + (FF + fff)/rate``, then rounded once to float64. The clock
must strictly increase; dropped frames stay dropped (no resampling).

**Truncated tails.** 824 files end mid-row without a final newline, so their last
line may be cut inside a number and still parse. A file without a terminating
newline therefore loses its last line unconditionally, and the loss is reported.
Any other malformed row refuses the file.

**Head-mounted camera.** In every measured file the head rotations are exactly
zero, so the device was worn, not facing the signer: head pose was not recorded.
``head_rotation`` stays unavailable rather than becoming a fictitious identity.
A file with nonzero head rotation is refused, because its frame convention would
then matter and is undeclared.

**Clock relationship to the skeleton.** None is established. The timecode is the
phone's, the skeleton's take clock is Shogun's; measured take start times differ
by an hour plus seconds that vary between takes, and no observable is shared
(head motion is not recorded). These states therefore carry their own clock and
must never be merged frame-wise with a skeleton state.

Values are ARKit's own solve from the TrueDepth camera, so every value is valid
but ``observed=False`` with an explicit inference method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import re

import numpy as np

from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion

ARKIT_BLENDSHAPES = (
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
    "JawRight", "JawLeft", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
    "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft",
    "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
    "NoseSneerRight", "TongueOut",
)
ROTATIONS = ("HeadYaw", "HeadPitch", "HeadRoll", "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
             "RightEyeYaw", "RightEyePitch", "RightEyeRoll")
LLF_COLUMNS = ("Timecode", "BlendshapeCount", *ARKIT_BLENDSHAPES, *ROTATIONS)
BLENDSHAPE_COUNT = len(ARKIT_BLENDSHAPES) + len(ROTATIONS)  # 61, as the file declares
_TIMECODE = re.compile(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
GAZE_FRAME = "llf-eye-yaw-pitch-v1:(cos p sin y, sin p, cos p cos y)"
CONVENTION = "arkit-live-link-face-csv-v1"


@dataclass(frozen=True)
class LLFRecording:
    times: np.ndarray        # (T,) float64 seconds on the device timecode
    blendshapes: np.ndarray  # (T, 52) in [0, 1]
    rotations: np.ndarray    # (T, 9) radians, column order ROTATIONS
    dropped_tail: bool       # an unterminated last line was discarded
    rate: int


def _timecode_seconds(text: str, rate: int) -> Fraction:
    match = _TIMECODE.fullmatch(text)
    if not match:
        raise ValueError(f"malformed timecode {text!r}")
    hours, minutes, seconds, frame, sub = (int(g) for g in match.groups())
    if minutes > 59 or seconds > 59 or frame >= rate:
        raise ValueError(f"timecode field out of range {text!r}")
    return Fraction(3600 * hours + 60 * minutes + seconds) + (frame + Fraction(sub, 1000)) / rate


def parse_llf_csv(payload: bytes, *, rate: int = 60) -> LLFRecording:
    """Strictly parse one Live Link Face CSV."""
    if type(rate) is not int or rate <= 0:
        raise ValueError("rate must be a positive integer frame rate")
    text = payload.decode("utf-8")
    lines = text.split("\n")
    terminated = text.endswith("\n")
    if terminated:
        lines.pop()  # the empty string after the final newline
    if any(line.endswith("\r") for line in lines):
        lines = [line[:-1] if line.endswith("\r") else line for line in lines]
    if not lines or tuple(lines[0].split(",")) != LLF_COLUMNS:
        raise ValueError("header is not the declared Live Link Face column set")
    rows = lines[1:]
    dropped_tail = False
    if not terminated and rows:
        rows = rows[:-1]
        dropped_tail = True
    if not rows:
        raise ValueError("no complete data rows")
    times, values = [], []
    for number, line in enumerate(rows, start=2):
        fields = line.split(",")
        if len(fields) != len(LLF_COLUMNS):
            raise ValueError(f"line {number}: expected {len(LLF_COLUMNS)} fields")
        if fields[1] != str(BLENDSHAPE_COUNT):
            raise ValueError(f"line {number}: blendshape count {fields[1]!r}")
        if not all(_NUMBER.fullmatch(item) for item in fields[2:]):
            raise ValueError(f"line {number}: non-numeric value")
        times.append(_timecode_seconds(fields[0], rate))
        values.append([float(item) for item in fields[2:]])
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timecode does not strictly increase")
    data = np.asarray(values, dtype=np.float64)
    blendshapes, rotations = data[:, :len(ARKIT_BLENDSHAPES)], data[:, len(ARKIT_BLENDSHAPES):]
    if not np.isfinite(data).all():
        raise ValueError("non-finite value")
    if np.any((blendshapes < 0) | (blendshapes > 1)):
        raise ValueError("ARKit coefficient outside [0, 1]")
    if np.any(np.abs(rotations) > math.pi / 2):
        raise ValueError("rotation beyond a quarter turn; radians are required")
    clock = np.array([float(t) for t in times], dtype=np.float64)
    if np.any(np.diff(clock) <= 0):
        raise ValueError("timecode collapses under float64")
    return LLFRecording(clock, blendshapes, rotations, dropped_tail, rate)


def gaze_vectors(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Unit gaze directions in the declared yaw/pitch frame; exact unit norm."""
    yaw, pitch = np.asarray(yaw, np.float64), np.asarray(pitch, np.float64)
    vectors = np.stack((np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)), -1)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@dataclass
class LLFReport:
    source_sha256: str
    sample_id: str
    status: str = "pending"
    frames: int = 0
    span_seconds: float = 0.
    dropped_tail: bool = False
    dropped_frames: int = 0
    unavailable_channels: list = field(default_factory=list)
    unrepresented: dict = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def adapt_llf(payload: bytes, *, sample_id: str, source_id: str, clock_id: str,
              inference_method: str, rate: int = 60) -> tuple[MultichannelMotion | None, LLFReport]:
    """Convert one LLF CSV into a face-only canonical state, or refuse with a reason."""
    digest = hashlib.sha256(bytes(payload)).hexdigest()
    report = LLFReport(digest, sample_id)
    if not isinstance(inference_method, str) or not inference_method.strip():
        raise ValueError("an inference method is required")
    try:
        recording = parse_llf_csv(bytes(payload), rate=rate)
    except (ValueError, UnicodeDecodeError) as error:
        report.status, report.reason = "refused", str(error)
        return None, report
    if np.any(recording.rotations[:, :3] != 0):
        report.status = "refused"
        report.reason = "head rotation is recorded but its frame convention is undeclared"
        return None, report
    frames = len(recording.times)
    report.frames, report.dropped_tail = frames, recording.dropped_tail
    report.span_seconds = float(recording.times[-1] - recording.times[0])
    expected = round(report.span_seconds * rate) + 1
    report.dropped_frames = int(expected - frames)

    def channel(values, labels, frame):
        valid = np.ones(values.shape[:2], dtype=bool)
        return MotionChannel(values=values, valid=valid, observed=np.zeros_like(valid),
                             confidence=np.ones(valid.shape), labels=labels, coordinate_frame=frame,
                             units="unitless", source_id=source_id, source_sha256=digest,
                             convention=CONVENTION, inference_method=inference_method)

    rotations = recording.rotations
    blink = recording.blendshapes[:, [ARKIT_BLENDSHAPES.index("EyeBlinkLeft"),
                                       ARKIT_BLENDSHAPES.index("EyeBlinkRight")]]
    channels = {
        "face": channel(recording.blendshapes[..., None], ARKIT_BLENDSHAPES, "arkit-blendshape-basis"),
        "left_gaze": channel(gaze_vectors(rotations[:, 3], rotations[:, 4])[:, None], ("LeftEye",),
                             GAZE_FRAME),
        "right_gaze": channel(gaze_vectors(rotations[:, 6], rotations[:, 7])[:, None], ("RightEye",),
                              GAZE_FRAME),
        "blink": channel(blink[..., None], ("EyeBlinkLeft", "EyeBlinkRight"),
                         "arkit-eyeBlink-coefficient-as-eyelid-closure"),
    }
    for name in CHANNELS:
        if name not in channels:
            _, width, units = CHANNELS[name]
            channels[name] = MotionChannel(
                values=np.zeros((frames, 1, width)), valid=np.zeros((frames, 1), dtype=bool),
                observed=np.zeros((frames, 1), dtype=bool), confidence=np.zeros((frames, 1)),
                labels=(f"{name}-unavailable",), coordinate_frame="not-recorded-by-source",
                units=units, source_id=source_id, source_sha256=digest, convention=CONVENTION)
            report.unavailable_channels.append(name)
    report.unrepresented = {
        "eye_roll": "a gaze direction has no roll; LeftEyeRoll/RightEyeRoll are not represented",
        "head_rotation": "exactly zero throughout: head-mounted device, head pose not recorded",
    }
    state = MultichannelMotion(recording.times, clock_id, sample_id, channels)
    report.status = "converted"
    return state, report


__all__ = ["ARKIT_BLENDSHAPES", "ROTATIONS", "LLF_COLUMNS", "LLFRecording", "LLFReport",
           "parse_llf_csv", "gaze_vectors", "adapt_llf"]
