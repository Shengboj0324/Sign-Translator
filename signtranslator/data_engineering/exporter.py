"""Fail-closed bridge from governed records to active corpus shards.

The exporter accepts already extracted, timestamped holistic motion. Raw-video pose
estimation is intentionally outside this module: it requires a separately licensed and
versioned extractor. This boundary never invents landmarks, timestamps, confidence, or
labels when an upstream extractor did not provide them.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..data.corpus import CorpusSpec, ctc_min_input_length, subsampled_length
from .schema import ConsentState, Sample, validate_sample
from .splitting import certify_no_group_leakage, grouped_split


CORPUS_FORMAT_VERSION = 2


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray       # (num_samples, channels), float32 in [-1, 1]
    timestamps: np.ndarray    # (num_samples,), seconds from stream start
    sample_rate: int


def decode_pcm_wav(path: str | os.PathLike[str]) -> DecodedAudio:
    """Decode uncompressed 16- or 32-bit PCM WAV without losing timestamps.

    Unsupported compression and sample widths are rejected. There is deliberately no
    guessed resampling, channel mixing, or timestamp synthesis beyond the exact PCM
    sample clock.
    """
    with wave.open(os.fspath(path), "rb") as stream:
        if stream.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is unsupported: {stream.getcomptype()}")
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
        raw = stream.readframes(frame_count)
    if channels <= 0 or sample_rate <= 0:
        raise ValueError("WAV channels and sample rate must be positive")
    if sample_width == 2:
        integer = np.frombuffer(raw, dtype="<i2")
        scale = float(1 << 15)
    elif sample_width == 4:
        integer = np.frombuffer(raw, dtype="<i4")
        scale = float(1 << 31)
    else:
        raise ValueError("only 16-bit and 32-bit integer PCM WAV are supported")
    if integer.size != frame_count * channels:
        raise ValueError("WAV payload length does not match its header")
    samples = (integer.astype(np.float64) / scale).astype(np.float32)
    samples = samples.reshape(frame_count, channels)
    timestamps = np.arange(frame_count, dtype=np.float64) / sample_rate
    return DecodedAudio(samples=samples, timestamps=timestamps,
                        sample_rate=sample_rate)


@dataclass(frozen=True)
class LandmarkTrack:
    values: np.ndarray        # (C, T, V)
    confidence: np.ndarray    # (T, V)
    validity_mask: np.ndarray # (T, V), boolean
    timestamps: np.ndarray    # (T,), seconds


def assemble_holistic_track(parts: Dict[str, LandmarkTrack]) -> LandmarkTrack:
    """Concatenate timestamp-aligned body, hand, and face extractor outputs.

    No temporal interpolation or coordinate conversion is performed implicitly. Upstream
    extractors must produce the same frame clock and coordinate system before assembly.
    """
    if not parts:
        raise ValueError("at least one landmark part is required")
    ordered_names = tuple(parts)
    for name in ordered_names:
        _validate_track(parts[name], sample_id=name)
    reference = parts[ordered_names[0]]
    for name in ordered_names[1:]:
        part = parts[name]
        if part.values.shape[:2] != reference.values.shape[:2]:
            raise ValueError(f"{name}: channels and frames do not match the reference")
        if not np.array_equal(part.timestamps, reference.timestamps):
            raise ValueError(f"{name}: timestamps do not exactly match the reference")
    return LandmarkTrack(
        values=np.concatenate([parts[name].values for name in ordered_names], axis=2),
        confidence=np.concatenate([parts[name].confidence for name in ordered_names], axis=1),
        validity_mask=np.concatenate(
            [parts[name].validity_mask for name in ordered_names], axis=1),
        timestamps=reference.timestamps.copy(),
    )


def decode_landmark_npz(path: str | os.PathLike[str]) -> LandmarkTrack:
    """Read the strict interchange format produced by a versioned extractor."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"values", "confidence", "validity_mask", "timestamps"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"landmark archive missing arrays: {sorted(missing)}")
        track = LandmarkTrack(
            values=np.array(archive["values"], copy=True),
            confidence=np.array(archive["confidence"], copy=True),
            validity_mask=np.array(archive["validity_mask"], copy=True),
            timestamps=np.array(archive["timestamps"], copy=True),
        )
    _validate_track(track, sample_id=os.fspath(path))
    return track


@dataclass(frozen=True)
class ExtractedSample:
    governance: Sample
    track: LandmarkTrack
    gloss_tokens: Tuple[str, ...]
    source_tokens: Tuple[str, ...]
    media_sha256: str
    extractor_id: str
    coordinate_system: str
    speech_features: Optional[np.ndarray] = None  # (T_speech, F)
    speech_timestamps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ExportResult:
    corpus_dir: str
    manifest_path: str
    review_path: str
    spec: CorpusSpec
    split_assignment: Dict[int, str]


def _validate_track(track: LandmarkTrack, sample_id: str) -> None:
    values = np.asarray(track.values)
    confidence = np.asarray(track.confidence)
    validity = np.asarray(track.validity_mask)
    timestamps = np.asarray(track.timestamps)
    if values.ndim != 3:
        raise ValueError(f"{sample_id}: motion values must be (C, T, V)")
    channels, frames, joints = values.shape
    if channels < 3 or frames < 1 or joints < 1:
        raise ValueError(f"{sample_id}: motion dimensions must be C>=3, T>=1, V>=1")
    if confidence.shape != (frames, joints) or validity.shape != (frames, joints):
        raise ValueError(f"{sample_id}: confidence/validity must be (T, V)")
    if validity.dtype != np.bool_:
        raise TypeError(f"{sample_id}: validity_mask must have boolean dtype")
    if timestamps.shape != (frames,):
        raise ValueError(f"{sample_id}: timestamps must be (T,)")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{sample_id}: timestamps must be finite and strictly increasing")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError(f"{sample_id}: confidence must be finite and in [0, 1]")
    expanded_validity = np.broadcast_to(validity[None], values.shape)
    if not np.isfinite(values[expanded_validity]).all():
        raise ValueError(f"{sample_id}: valid motion observations must be finite")
    if np.any(confidence[~validity] != 0):
        raise ValueError(f"{sample_id}: invalid observations must have zero confidence")


def _validate_record(record: ExtractedSample, speech_subsample: int) -> None:
    sample_id = record.governance.sample_id
    violations = validate_sample(record.governance)
    if violations:
        raise ValueError(f"{sample_id}: governed sample violations: {violations}")
    if record.governance.consent is not ConsentState.GRANTED:
        raise PermissionError(f"{sample_id}: consent is not granted")
    if not record.gloss_tokens or not record.source_tokens:
        raise ValueError(f"{sample_id}: source and gloss token sequences are required")
    if any(not token for token in record.gloss_tokens + record.source_tokens):
        raise ValueError(f"{sample_id}: empty token labels are forbidden")
    if len(record.media_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in record.media_sha256.lower()):
        raise ValueError(f"{sample_id}: media_sha256 must be a 64-character hex digest")
    if not record.extractor_id or not record.coordinate_system:
        raise ValueError(f"{sample_id}: extractor_id and coordinate_system are required")
    _validate_track(record.track, sample_id)
    motion_minimum = ctc_min_input_length(record.gloss_tokens)
    if record.track.values.shape[1] < motion_minimum:
        raise ValueError(
            f"{sample_id}: motion CTC target needs {motion_minimum} frames, got "
            f"{record.track.values.shape[1]}")
    if (record.speech_features is None) != (record.speech_timestamps is None):
        raise ValueError(f"{sample_id}: speech features and timestamps must appear together")
    if record.speech_features is not None:
        features = np.asarray(record.speech_features)
        timestamps = np.asarray(record.speech_timestamps)
        if features.ndim != 2 or features.shape[0] < 1:
            raise ValueError(f"{sample_id}: speech_features must be (T, F)")
        if timestamps.shape != (features.shape[0],):
            raise ValueError(f"{sample_id}: speech timestamps must match feature frames")
        if not np.isfinite(features).all() or not np.isfinite(timestamps).all():
            raise ValueError(f"{sample_id}: speech arrays must be finite")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{sample_id}: speech timestamps must strictly increase")
        usable = subsampled_length(features.shape[0], speech_subsample)
        minimum = ctc_min_input_length(record.source_tokens)
        if usable < minimum:
            raise ValueError(
                f"{sample_id}: speech CTC target needs {minimum} post-subsample "
                f"frames, got {usable}")


def _weighted_normalization(records: Sequence[ExtractedSample],
                            train_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    channels, _, joints = records[0].track.values.shape
    weight_sum = np.zeros((1, 1, joints), dtype=np.float64)
    weighted_sum = np.zeros((channels, 1, joints), dtype=np.float64)
    weighted_square = np.zeros((channels, 1, joints), dtype=np.float64)
    for index in train_indices:
        record = records[index]
        values = np.asarray(record.track.values, dtype=np.float64)
        weights = (record.track.confidence * record.track.validity_mask).astype(np.float64)
        weights = weights[None]
        safe_values = np.where(np.broadcast_to(record.track.validity_mask[None], values.shape),
                               values, 0.0)
        weight_sum += weights.sum(axis=1, keepdims=True)
        weighted_sum += (safe_values * weights).sum(axis=1, keepdims=True)
        weighted_square += ((safe_values ** 2) * weights).sum(axis=1, keepdims=True)
    if np.any(weight_sum <= 0):
        missing_joints = np.flatnonzero(weight_sum.reshape(-1) <= 0).tolist()
        raise ValueError(f"training split has no valid support for joints {missing_joints}")
    mean = weighted_sum / weight_sum
    variance = weighted_square / weight_sum - mean ** 2
    variance = np.maximum(variance, 0.0)
    if np.any(variance <= 1e-12):
        locations = np.argwhere(variance <= 1e-12).tolist()
        raise ValueError(f"training normalization has zero variance at {locations}")
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _write_review(path: Path, records: Sequence[ExtractedSample],
                  assignment: Dict[int, str]) -> None:
    rows = []
    for index, record in enumerate(records):
        valid = record.track.validity_mask
        rows.append(
            "<tr>"
            f"<td>{html.escape(record.governance.sample_id)}</td>"
            f"<td>{html.escape(assignment[index])}</td>"
            f"<td>{record.track.values.shape[1]}</td>"
            f"<td>{record.track.values.shape[2]}</td>"
            f"<td>{float(valid.mean()):.3f}</td>"
            f"<td>{float(record.track.confidence[valid].mean()) if valid.any() else 0.0:.3f}</td>"
            f"<td>{html.escape(record.media_sha256)}</td>"
            f"<td>{html.escape(record.governance.video_uri or 'N/A')}</td>"
            "</tr>")
    document = (
        "<!doctype html><meta charset='utf-8'><title>Corpus review</title>"
        "<h1>Corpus review queue</h1>"
        "<p>This report supports human inspection; it is not evidence that review occurred.</p>"
        "<table border='1'><thead><tr><th>sample</th><th>split</th><th>frames</th>"
        "<th>joints</th><th>valid fraction</th><th>mean confidence</th><th>media SHA-256</th>"
        "<th>source video</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")
    path.write_text(document, encoding="utf-8")


def export_corpus(records: Sequence[ExtractedSample], out_dir: str | os.PathLike[str],
                  *, joint_names: Sequence[str], split_ratios=(0.7, 0.15, 0.15),
                  seed: int = 0, speech_subsample: int = 2) -> ExportResult:
    """Export validated records into versioned, active-loader-compatible shards."""
    if not records:
        raise ValueError("at least one extracted sample is required")
    destination = Path(out_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"refusing to export into non-empty path {destination}")
    for record in records:
        _validate_record(record, speech_subsample)
    languages = {record.governance.target_language for record in records}
    coordinates = {record.coordinate_system for record in records}
    extractors = {record.extractor_id for record in records}
    shapes = {(record.track.values.shape[0], record.track.values.shape[2])
              for record in records}
    if len(languages) != 1 or len(coordinates) != 1 or len(extractors) != 1:
        raise ValueError("language, coordinate system, and extractor must be uniform")
    if len(shapes) != 1:
        raise ValueError("all records must share channel and joint dimensions")
    channels, joints = next(iter(shapes))
    if len(joint_names) != joints or len(set(joint_names)) != joints:
        raise ValueError("joint_names must be unique and match the motion joint count")
    speech_presence = {record.speech_features is not None for record in records}
    if len(speech_presence) != 1:
        raise ValueError("speech modality must be present for every record or none")

    samples = [record.governance for record in records]
    assignment = grouped_split(samples, ratios=split_ratios, seed=seed)
    certificate = certify_no_group_leakage(samples, assignment)
    if not certificate.certified:
        raise RuntimeError(f"split leakage detected: {certificate.offending_groups}")
    counts = {name: sum(split == name for split in assignment.values())
              for name in ("train", "val", "test")}
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"every split must be non-empty; got {counts}")

    gloss_labels = sorted({token for record in records for token in record.gloss_tokens})
    source_labels = sorted({token for record in records for token in record.source_tokens})
    gloss_to_id = {token: index for index, token in enumerate(gloss_labels)}
    source_to_id = {token: index for index, token in enumerate(source_labels)}
    max_gloss = max(len(record.gloss_tokens) for record in records)
    max_source = max(len(record.source_tokens) for record in records)
    max_frames = max(record.track.values.shape[1] for record in records)
    max_speech = max((record.speech_features.shape[0] for record in records
                      if record.speech_features is not None), default=0)
    speech_dim = (records[0].speech_features.shape[1]
                  if records[0].speech_features is not None else 1)
    if records[0].speech_features is not None and any(
            record.speech_features.shape[1] != speech_dim for record in records):
        raise ValueError("speech feature width must be uniform")

    spec = CorpusSpec(
        num_concepts=len(gloss_labels), seq_len=max_gloss,
        num_joints=joints, in_channels=channels, num_frames=max_frames,
        src_vocab=len(source_labels) + 3, gloss_vocab=len(gloss_labels) + 3,
        num_glosses=len(gloss_labels), speech_frames=max_speech or 1,
        speech_dim=speech_dim, num_source_tokens=len(source_labels))
    train_indices = [index for index, split in assignment.items() if split == "train"]
    pose_mean, pose_std = _weighted_normalization(records, train_indices)

    destination.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        indices = [index for index, split in assignment.items() if split == split_name]
        count = len(indices)
        pose = np.zeros((count, channels, max_frames, joints), dtype=np.float32)
        validity = np.zeros((count, max_frames, joints), dtype=np.bool_)
        confidence = np.zeros((count, max_frames, joints), dtype=np.float32)
        timestamps = np.full((count, max_frames), -1.0, dtype=np.float64)
        motion_lengths = np.zeros(count, dtype=np.int64)
        concepts = np.zeros((count, max_gloss), dtype=np.int64)
        src_concepts = np.zeros((count, max_source), dtype=np.int64)
        lengths = np.zeros(count, dtype=np.int64)
        source_lengths = np.zeros(count, dtype=np.int64)
        sample_ids = np.empty(count, dtype=f"<U{max(len(records[i].governance.sample_id) for i in indices)}")
        if max_speech:
            speech = np.zeros((count, max_speech, speech_dim), dtype=np.float32)
            speech_timestamps = np.full((count, max_speech), -1.0, dtype=np.float64)
            speech_lengths = np.zeros(count, dtype=np.int64)
        for row, index in enumerate(indices):
            record = records[index]
            frames = record.track.values.shape[1]
            length = len(record.gloss_tokens)
            source_length = len(record.source_tokens)
            pose[row, :, :frames] = np.where(
                np.broadcast_to(record.track.validity_mask[None], record.track.values.shape),
                record.track.values, 0.0)
            validity[row, :frames] = record.track.validity_mask
            confidence[row, :frames] = record.track.confidence
            timestamps[row, :frames] = record.track.timestamps
            motion_lengths[row] = frames
            concepts[row, :length] = [gloss_to_id[token] for token in record.gloss_tokens]
            src_concepts[row, :source_length] = [source_to_id[token]
                                                  for token in record.source_tokens]
            lengths[row] = length
            source_lengths[row] = source_length
            sample_ids[row] = record.governance.sample_id
            if max_speech:
                speech_length = record.speech_features.shape[0]
                speech[row, :speech_length] = record.speech_features
                speech_timestamps[row, :speech_length] = record.speech_timestamps
                speech_lengths[row] = speech_length
        arrays = dict(
            pose=pose, concepts=concepts, src_concepts=src_concepts, lengths=lengths,
            source_lengths=source_lengths,
            motion_lengths=motion_lengths, validity_mask=validity,
            confidence=confidence, frame_timestamps=timestamps, sample_ids=sample_ids)
        if max_speech:
            arrays.update(speech=speech, speech_lengths=speech_lengths,
                          speech_timestamps=speech_timestamps)
        np.savez_compressed(destination / f"{split_name}.npz", **arrays)

    review_path = destination / "review.html"
    _write_review(review_path, records, assignment)
    shard_hashes = {f"{name}.npz": sha256_file(destination / f"{name}.npz")
                    for name in ("train", "val", "test")}
    manifest = {
        "format_version": CORPUS_FORMAT_VERSION,
        "spec": asdict(spec),
        "splits": counts,
        "seed": seed,
        "split_ratios": list(split_ratios),
        "split_group": "signer_id_hash+source_id",
        "leakage_certified": certificate.certified,
        "language": next(iter(languages)),
        "coordinate_system": next(iter(coordinates)),
        "extractor_id": next(iter(extractors)),
        "joint_names": list(joint_names),
        "gloss_vocabulary": gloss_labels,
        "source_vocabulary": source_labels,
        "pose_mean": pose_mean.tolist(),
        "pose_std": pose_std.tolist(),
        "normalization_fit_split": "train",
        "speech_subsample": speech_subsample,
        "shard_sha256": shard_hashes,
        "records": [{
            "sample_id": record.governance.sample_id,
            "source_id": record.governance.source_id,
            "signer_id_hash": record.governance.signer_id_hash,
            "video_uri": record.governance.video_uri,
            "audio_uri": record.governance.audio_uri,
            "media_sha256": record.media_sha256,
            "provenance": record.governance.provenance,
            "split": assignment[index],
        } for index, record in enumerate(records)],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    return ExportResult(os.fspath(destination), os.fspath(manifest_path),
                        os.fspath(review_path), spec, assignment)
