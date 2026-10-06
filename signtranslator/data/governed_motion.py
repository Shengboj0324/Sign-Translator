"""Source-bound SIR/motion loading with an explicit, recorded clock transform.

No gloss projection or temporal interpolation occurs here. This validates file
identities and recorded governance claims, not the truth of a human attestation
or physical calibration. A loaded pair is not empirical phase acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..data_engineering.phase2_policy import AuthorizationEvidence, Phase2Scope
from ..data_engineering.schema import Sample
from ..data_engineering.source_portfolio import SourceCandidate
from ..data_engineering.state_ingestion import validate_phase2_state
from ..planning.supervision import GovernedSIRAnnotation, certify_supervision_batch
from ..pose.multichannel import MultichannelMotion
from ..reproducibility import sha256_file

MAX_TRANSCRIPT_BYTES = 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate alignment field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class GovernedMotionPair:
    motion: MultichannelMotion
    annotation: GovernedSIRAnnotation
    annotation_times: np.ndarray
    event_ids: tuple[int, ...]
    event_frame_membership: np.ndarray   # E,T; graph order, half-open intervals
    alignment_sha256: str
    transcript_payload: bytes          # exact verified source bytes; format not inferred
    annotation_extent: tuple[float, float]  # declared source interval, not event extrema


def load_governed_motion_pair(
    *, motion_path: Path, motion_sha256: str,
    annotation: GovernedSIRAnnotation, sample: Sample,
    video_path: Path, transcript_path: Path,
    annotation_authorization_path: Path,
    alignment_path: Path, alignment_sha256: str,
    sources: tuple[SourceCandidate, ...],
    authorizations: dict[str, AuthorizationEvidence],
    source_files: dict[str, Path],
) -> GovernedMotionPair:
    """Load a research pair only with complete scoped source evidence.

    The alignment document declares ``annotation_time = scale * motion_time +
    offset_seconds``. Its interval is the full annotated sample extent in seconds,
    not an inferred frame duration. Hash binding does not validate calibration;
    that remains a qualified source-specific requirement. Event membership means
    point-sampling at frame timestamps, not duration weights or frame exposures.
    Events between samples are retained with an all-false membership row.
    """
    if not isinstance(annotation, GovernedSIRAnnotation) or not isinstance(sample, Sample):
        raise ValueError("a governed annotation and canonical sample are required")
    # Reconstruct to recheck all review/content bindings rather than trusting type.
    annotation = GovernedSIRAnnotation.from_manifest(annotation.to_manifest())
    if sample.intended_use != "research":
        raise ValueError("this loader supports explicit research scope only")
    video_hash = sha256_file(video_path)
    if transcript_path.is_symlink() or not transcript_path.is_file():
        raise ValueError("transcript must be a regular local file")
    with transcript_path.open('rb') as stream:
        transcript_payload = stream.read(MAX_TRANSCRIPT_BYTES + 1)
    if len(transcript_payload) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("transcript exceeds the one-MiB per-sample size limit")
    # Hash and retain the same read. Reopening after a hash check could consume
    # different bytes if a source file changed between the two reads.
    transcript_hash = hashlib.sha256(transcript_payload).hexdigest()
    certificate = certify_supervision_batch(
        [annotation], {sample.sample_id: sample},
        {sample.sample_id: video_hash}, {sample.sample_id: transcript_hash})
    if not certificate.approved_for_research_training:
        raise PermissionError(f"governed supervision rejected: {certificate.violations}")
    if sha256_file(annotation_authorization_path) != annotation.source.authorization_evidence_sha256:
        raise PermissionError("annotation authorization evidence content hash mismatch")
    if motion_path.is_symlink() or not motion_path.is_file():
        raise ValueError("motion must be a regular local file")
    motion = MultichannelMotion.load(motion_path, expected_sha256=motion_sha256)
    if motion.sample_id != sample.sample_id:
        raise ValueError("motion sample identity differs from governed annotation")
    validate_phase2_state(motion, sources=sources, scope=Phase2Scope.RESEARCH,
                          authorizations=authorizations, source_files=source_files)

    if alignment_path.is_symlink() or not alignment_path.is_file():
        raise ValueError("alignment must be a regular local file")
    with alignment_path.open("rb") as stream:
        payload = stream.read(65537)
    if len(payload) > 65536:
        raise ValueError("alignment document exceeds size limit")
    if hashlib.sha256(payload).hexdigest() != alignment_sha256:
        raise ValueError("alignment content hash mismatch")
    alignment = json.loads(payload, object_pairs_hook=_unique_object)
    identities = {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "source_recording_id": sample.source_id,
        "motion_sha256": motion_sha256,
        "annotation_sha256": annotation.content_sha256(),
        "video_sha256": video_hash,
        "clock_id": motion.clock_id,
    }
    numeric = {"scale", "offset_seconds", "interval_start_seconds", "interval_end_seconds"}
    if not isinstance(alignment, dict) or set(alignment) != set(identities) | numeric:
        raise ValueError("alignment fields do not match the versioned contract")
    if type(alignment["schema_version"]) is not int:
        raise ValueError("alignment schema version must be an integer")
    if any(alignment[key] != value for key, value in identities.items()):
        raise ValueError("alignment identity does not bind this motion and annotation")
    for key in numeric:
        value = alignment[key]
        if type(value) not in (int, float):
            raise ValueError(f"{key} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{key} must be a finite number")
    scale, offset = float(alignment["scale"]), float(alignment["offset_seconds"])
    start, end = float(alignment["interval_start_seconds"]), float(alignment["interval_end_seconds"])
    if scale <= 0 or start >= end:
        raise ValueError("clock scale and annotation interval must be positive")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        times = motion.timestamps * scale + offset
        intervals = np.diff(times)
    if (not np.isfinite(times).all() or not np.isfinite(intervals).all()
            or np.any(intervals <= 0)):
        raise ValueError("mapped clock must remain finite and strictly increasing")
    if np.any(times < start) or np.any(times >= end):
        raise ValueError("mapped motion timestamps lie outside the annotation interval")
    graph = annotation.graph()
    if any(event.t_start < start or event.t_end > end for event in graph.events):
        raise ValueError("SIR event lies outside the declared annotation interval")
    membership = np.stack([(times >= event.t_start) & (times < event.t_end)
                           for event in graph.events])
    return GovernedMotionPair(motion, annotation, times,
                              tuple(event.id for event in graph.events), membership,
                              alignment_sha256, transcript_payload, (start, end))
