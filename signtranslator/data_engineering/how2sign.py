"""Strict ingestion of the downloaded How2Sign frontal-view training subset.

This module does not infer gloss labels, signer identities, depth, missing media,
or timestamps.  It joins clips by the published sentence name, reads exact video
presentation timestamps, and converts OpenPose BODY_25/hand/face observations to
an explicitly two-dimensional landmark track.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .exporter import LandmarkTrack, decode_video_clock


HOW2SIGN_FIELDS = (
    "VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
    "START_REALIGNED", "END_REALIGNED", "SENTENCE",
)
HOW2SIGN_CLIP_PATTERN = re.compile(
    r"^(?P<sentence_id>.+)-(?P<filename_code>\d+)-rgb_front$"
)
OPENPOSE_PARTS = (
    ("pose_keypoints_2d", 25),
    ("hand_left_keypoints_2d", 21),
    ("hand_right_keypoints_2d", 21),
    ("face_keypoints_2d", 70),
)
OPENPOSE_3D_KEYS = (
    "pose_keypoints_3d", "hand_left_keypoints_3d",
    "hand_right_keypoints_3d", "face_keypoints_3d",
)
OPENPOSE_JOINT_NAMES = tuple(
    f"{part.removesuffix('_keypoints_2d')}:{index}"
    for part, count in OPENPOSE_PARTS for index in range(count)
)
OPENPOSE_LANDMARK_PARTS = {
    "body": tuple(range(0, 25)),
    "left_hand": tuple(range(25, 46)),
    "right_hand": tuple(range(46, 67)),
    "face": tuple(range(67, 137)),
}


def _offset_edges(edges: Iterable[tuple[int, int]], offset: int) -> tuple[tuple[int, int], ...]:
    return tuple((left + offset, right + offset) for left, right in edges)


# OpenPose's published BODY_25, hand, and face render connections. The source
# components are separate estimators, so the final entries are explicit fusion
# edges that make their graph connected for ST-GCN message passing. Facial
# fusion follows anatomical anchors: nose bridge, brows/jaw, eyes, nose base,
# lips, and pupils. Source topology:
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/include/openpose
_BODY_25_EDGES = (
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19),
    (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
)
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
_FACE_EDGES = (
    *tuple((index, index + 1) for index in range(0, 16)),
    *tuple((index, index + 1) for index in range(17, 21)),
    *tuple((index, index + 1) for index in range(22, 26)),
    *tuple((index, index + 1) for index in range(27, 30)),
    *tuple((index, index + 1) for index in range(31, 35)),
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    *tuple((index, index + 1) for index in range(48, 59)), (59, 48),
    *tuple((index, index + 1) for index in range(60, 67)), (67, 60),
)
_FACE_OFFSET = 67
OPENPOSE_HOLISTIC_EDGES = (
    *_BODY_25_EDGES,
    *_offset_edges(_HAND_EDGES, 25),
    *_offset_edges(_HAND_EDGES, 46),
    (7, 25),                         # left body wrist -> left hand wrist
    (4, 46),                         # right body wrist -> right hand wrist
    *_offset_edges(_FACE_EDGES, _FACE_OFFSET),
    (0, _FACE_OFFSET + 27),          # body nose -> facial nose bridge
    (_FACE_OFFSET + 27, _FACE_OFFSET + 21),
    (_FACE_OFFSET + 27, _FACE_OFFSET + 22),
    (_FACE_OFFSET + 17, _FACE_OFFSET + 0),
    (_FACE_OFFSET + 26, _FACE_OFFSET + 16),
    (_FACE_OFFSET + 30, _FACE_OFFSET + 33),
    (_FACE_OFFSET + 27, _FACE_OFFSET + 39),
    (_FACE_OFFSET + 27, _FACE_OFFSET + 42),
    (_FACE_OFFSET + 33, _FACE_OFFSET + 51),
    (_FACE_OFFSET + 51, _FACE_OFFSET + 62),
    (_FACE_OFFSET + 68, _FACE_OFFSET + 39),
    (_FACE_OFFSET + 69, _FACE_OFFSET + 42),
)
OPENPOSE_HOLISTIC_CENTER = 1


def openpose_holistic_graph():
    """Return the connected 137-node graph matching this module's joint order."""
    from ..skeleton.graph import SkeletonGraph

    return SkeletonGraph(
        num_nodes=len(OPENPOSE_JOINT_NAMES), edges=OPENPOSE_HOLISTIC_EDGES,
        center=OPENPOSE_HOLISTIC_CENTER,
    )


@dataclass(frozen=True)
class How2SignRow:
    video_id: str
    video_name: str
    sentence_id: str
    sentence_name: str
    start_realigned: float
    end_realigned: float
    sentence: str
    filename_code: str


@dataclass(frozen=True)
class How2SignInventory:
    metadata_rows: int
    raw_mp4_clips: int
    rendered_mp4_clips: int
    openpose_directories: int
    complete_clip_joins: int
    metadata_without_raw: Tuple[str, ...]
    raw_without_metadata: Tuple[str, ...]
    raw_without_openpose_json: Tuple[str, ...]
    raw_without_rendered_video: Tuple[str, ...]
    openpose_json_without_raw: Tuple[str, ...]
    rendered_video_without_raw: Tuple[str, ...]
    unexpected_raw_entries: Tuple[str, ...]
    gloss_artifacts: Tuple[str, ...]
    license_artifacts: Tuple[str, ...]

    @property
    def structurally_ready(self) -> bool:
        """Whether metadata, raw clips, and both OpenPose products match exactly."""
        return not (
            self.metadata_without_raw or self.raw_without_metadata
            or self.raw_without_openpose_json or self.raw_without_rendered_video
            or self.openpose_json_without_raw or self.rendered_video_without_raw
            or self.unexpected_raw_entries
        )

    @property
    def translation_ready(self) -> bool:
        """Whether structure plus independently supplied gloss/license evidence exist."""
        return bool(
            self.structurally_ready and self.gloss_artifacts and self.license_artifacts
        )


@dataclass(frozen=True)
class OpenPoseDiagnostics:
    frame_count: int
    empty_person_frames: int
    invalid_confidences: int
    below_threshold_observations: int
    out_of_frame_coordinates: int
    valid_observations: int


@dataclass(frozen=True)
class How2SignClip:
    row: How2SignRow
    track: LandmarkTrack
    diagnostics: OpenPoseDiagnostics
    joint_names: Tuple[str, ...] = OPENPOSE_JOINT_NAMES
    coordinate_system: str = "image-normalized-xy-top-left"


def read_how2sign_metadata(path: str | os.PathLike[str]) -> Tuple[How2SignRow, ...]:
    """Read the tab-delimited realignment table and reject schema ambiguity."""
    rows = []
    sentence_names = set()
    with open(path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HOW2SIGN_FIELDS:
            raise ValueError(
                f"How2Sign metadata header must be exactly {HOW2SIGN_FIELDS}; "
                f"received {tuple(reader.fieldnames or ())}"
            )
        for line_number, raw in enumerate(reader, start=2):
            if None in raw or any(raw[field] is None for field in HOW2SIGN_FIELDS):
                raise ValueError(f"metadata line {line_number}: malformed column count")
            required = HOW2SIGN_FIELDS[:-1]
            if any(not raw[field].strip() for field in required):
                raise ValueError(f"metadata line {line_number}: required field is empty")
            sentence_name = raw["SENTENCE_NAME"].strip()
            match = HOW2SIGN_CLIP_PATTERN.fullmatch(sentence_name)
            if match is None or match.group("sentence_id") != raw["SENTENCE_ID"].strip():
                raise ValueError(
                    f"metadata line {line_number}: SENTENCE_NAME does not encode "
                    "SENTENCE_ID and a numeric filename code"
                )
            if sentence_name in sentence_names:
                raise ValueError(f"metadata line {line_number}: duplicate SENTENCE_NAME")
            sentence_names.add(sentence_name)
            try:
                start = float(raw["START_REALIGNED"])
                end = float(raw["END_REALIGNED"])
            except ValueError as error:
                raise ValueError(
                    f"metadata line {line_number}: realigned interval is not numeric"
                ) from error
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(
                    f"metadata line {line_number}: require 0 <= START_REALIGNED < END_REALIGNED"
                )
            rows.append(How2SignRow(
                video_id=raw["VIDEO_ID"].strip(),
                video_name=raw["VIDEO_NAME"].strip(),
                sentence_id=raw["SENTENCE_ID"].strip(),
                sentence_name=sentence_name,
                start_realigned=start,
                end_realigned=end,
                sentence=raw["SENTENCE"].strip(),
                filename_code=match.group("filename_code"),
            ))
    if not rows:
        raise ValueError("How2Sign metadata contains no records")
    return tuple(rows)


def _artifact_names(root: Path, patterns: Iterable[str]) -> Tuple[str, ...]:
    names = set()
    for pattern in patterns:
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern))
    return tuple(sorted(names))


def inspect_how2sign_root(root: str | os.PathLike[str]) -> How2SignInventory:
    """Inventory the expected download without recursively hashing large media."""
    root_path = Path(root)
    metadata_path = root_path / "how2sign_realigned_train.csv"
    raw_dir = root_path / "raw_videos"
    rendered_dir = root_path / "openpose_output" / "video"
    json_root = root_path / "openpose_output" / "json"
    for required in (metadata_path, raw_dir, rendered_dir, json_root):
        if not required.exists():
            raise FileNotFoundError(f"required How2Sign path is absent: {required}")
    rows = read_how2sign_metadata(metadata_path)
    metadata_names = {row.sentence_name for row in rows}

    raw_names = set()
    unexpected = []
    for path in raw_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".mp4" \
                and HOW2SIGN_CLIP_PATTERN.fullmatch(path.stem):
            raw_names.add(path.stem)
        else:
            unexpected.append(path.name)
    rendered_names = {
        path.stem for path in rendered_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }
    json_names = {path.name for path in json_root.iterdir() if path.is_dir()}
    complete = metadata_names & raw_names & rendered_names & json_names

    # Deliberately shallow: recursive globs through millions of frame JSON files
    # are both wasteful and incapable of proving annotation semantics.
    gloss = _artifact_names(root_path, ("*gloss*", "*GLOSS*"))
    license_files = _artifact_names(
        root_path,
        ("LICENSE*", "license*", "*terms*"),
    )
    return How2SignInventory(
        metadata_rows=len(rows),
        raw_mp4_clips=len(raw_names),
        rendered_mp4_clips=len(rendered_names),
        openpose_directories=len(json_names),
        complete_clip_joins=len(complete),
        metadata_without_raw=tuple(sorted(metadata_names - raw_names)),
        raw_without_metadata=tuple(sorted(raw_names - metadata_names)),
        raw_without_openpose_json=tuple(sorted(raw_names - json_names)),
        raw_without_rendered_video=tuple(sorted(raw_names - rendered_names)),
        openpose_json_without_raw=tuple(sorted(json_names - raw_names)),
        rendered_video_without_raw=tuple(sorted(rendered_names - raw_names)),
        unexpected_raw_entries=tuple(sorted(unexpected)),
        gloss_artifacts=gloss,
        license_artifacts=license_files,
    )


def _json_frame_paths(json_dir: Path, clip_name: str) -> Tuple[Path, ...]:
    pattern = re.compile(
        rf"^{re.escape(clip_name)}_(?P<index>\d{{12}})_keypoints\.json$"
    )
    indexed = []
    unexpected = []
    for path in json_dir.iterdir():
        if not path.is_file():
            unexpected.append(path.name)
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            unexpected.append(path.name)
        else:
            indexed.append((int(match.group("index")), path))
    if unexpected:
        raise ValueError(f"{clip_name}: unexpected OpenPose entries: {sorted(unexpected)!r}")
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indices))):
        raise ValueError(f"{clip_name}: OpenPose frame indices are not contiguous from zero")
    return tuple(path for _, path in indexed)


def decode_how2sign_openpose(
    video_path: str | os.PathLike[str],
    json_dir: str | os.PathLike[str],
    *,
    confidence_threshold: float = 0.3,
) -> tuple[LandmarkTrack, OpenPoseDiagnostics]:
    """Decode one 2D OpenPose track on the video's exact presentation clock.

    Coordinates in the half-open image domain are normalized by ``width`` and
    ``height``. Any
    non-finite/out-of-contract confidence, sub-threshold confidence, or
    out-of-frame coordinate is invalidated with zero value and confidence.
    Values are never clipped, imputed, triangulated, or promoted to fake 3D.
    """
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    video = Path(video_path)
    keypoint_dir = Path(json_dir)
    if not keypoint_dir.is_dir():
        raise FileNotFoundError(f"OpenPose directory is absent: {keypoint_dir}")
    clock = decode_video_clock(video)
    if clock.width < 2 or clock.height < 2:
        raise ValueError(f"{video.stem}: video dimensions must both be at least two pixels")
    frame_paths = _json_frame_paths(keypoint_dir, video.stem)
    if len(frame_paths) != clock.timestamps.shape[0]:
        raise ValueError(
            f"{video.stem}: OpenPose/video frame-count mismatch "
            f"({len(frame_paths)} != {clock.timestamps.shape[0]})"
        )

    frame_count = len(frame_paths)
    joint_count = len(OPENPOSE_JOINT_NAMES)
    values = np.zeros((2, frame_count, joint_count), dtype=np.float32)
    confidence = np.zeros((frame_count, joint_count), dtype=np.float32)
    validity = np.zeros((frame_count, joint_count), dtype=np.bool_)
    empty_people = invalid_confidences = below_threshold = outside = 0

    for frame_index, path in enumerate(frame_paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{video.stem}: unreadable OpenPose JSON {path.name}") from error
        if payload.get("version") != 1.3:
            raise ValueError(f"{video.stem}: unsupported OpenPose JSON version")
        people = payload.get("people")
        if not isinstance(people, list):
            raise ValueError(f"{video.stem}: OpenPose people must be a list")
        if len(people) > 1:
            raise ValueError(f"{video.stem}: multiple people make signer selection ambiguous")
        if not people:
            empty_people += 1
            continue
        person = people[0]
        if not isinstance(person, dict):
            raise ValueError(f"{video.stem}: OpenPose person must be an object")
        if any(person.get(key) not in (None, []) for key in OPENPOSE_3D_KEYS):
            raise ValueError(f"{video.stem}: unexpected 3D keypoints in the 2D ingestion path")

        joint_offset = 0
        for key, count in OPENPOSE_PARTS:
            raw = person.get(key)
            if not isinstance(raw, list) or len(raw) != count * 3:
                raise ValueError(
                    f"{video.stem}: {key} must contain exactly {count * 3} values"
                )
            try:
                points = np.asarray(raw, dtype=np.float64).reshape(count, 3)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{video.stem}: {key} contains non-numeric values") from error
            x, y, score = points.T
            finite_score = np.isfinite(score)
            in_contract = finite_score & (score >= 0) & (score <= 1)
            invalid_confidences += int((~in_contract).sum())
            finite_xy = np.isfinite(x) & np.isfinite(y)
            in_frame = finite_xy & (x >= 0) & (x < clock.width) & (y >= 0) & (y < clock.height)
            outside += int((~in_frame).sum())
            below = in_contract & (score < confidence_threshold)
            below_threshold += int(below.sum())
            valid = in_contract & ~below & in_frame
            target = slice(joint_offset, joint_offset + count)
            values[0, frame_index, target][valid] = (x[valid] / clock.width).astype(np.float32)
            values[1, frame_index, target][valid] = (y[valid] / clock.height).astype(np.float32)
            confidence[frame_index, target][valid] = score[valid].astype(np.float32)
            validity[frame_index, target] = valid
            joint_offset += count

    diagnostics = OpenPoseDiagnostics(
        frame_count=frame_count,
        empty_person_frames=empty_people,
        invalid_confidences=invalid_confidences,
        below_threshold_observations=below_threshold,
        out_of_frame_coordinates=outside,
        valid_observations=int(validity.sum()),
    )
    return LandmarkTrack(values, confidence, validity, clock.timestamps), diagnostics


def load_how2sign_clip(
    root: str | os.PathLike[str],
    row: How2SignRow,
    *,
    confidence_threshold: float = 0.3,
) -> How2SignClip:
    """Load a metadata-resolved raw clip; realignment times are not re-applied."""
    root_path = Path(root)
    video = root_path / "raw_videos" / f"{row.sentence_name}.mp4"
    json_dir = root_path / "openpose_output" / "json" / row.sentence_name
    if not video.is_file():
        raise FileNotFoundError(f"raw sentence clip is absent: {video}")
    track, diagnostics = decode_how2sign_openpose(
        video, json_dir, confidence_threshold=confidence_threshold,
    )
    return How2SignClip(row=row, track=track, diagnostics=diagnostics)
