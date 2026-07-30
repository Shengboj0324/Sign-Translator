"""Executable Stage B exit gate for a governed real mini-corpus.

Structural correctness is necessary but not sufficient: Stage B also requires
verifiable source bytes, a declared dataset charter, reported annotation
agreement, qualified visual review, and successful consumption by the canonical
loader.  This module keeps those claims fail-closed and machine-checkable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlparse

import numpy as np
from torch.utils.data import DataLoader

from ..data.corpus import SignDataset, collate_corpus, validate_corpus
from .exporter import decode_video, sha256_file
from .schema import ConsentState, DataAuthorization, validate_authorization


@dataclass(frozen=True)
class StageBCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class StageBReadinessReport:
    checks: List[StageBCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(StageBCheck(name, passed, detail))

    def summary(self) -> str:
        lines = ["Stage B real-data readiness", "=" * 68]
        lines.extend(
            f"  [{'PASS' if check.passed else 'FAIL'}] {check.name:<28} {check.detail}"
            for check in self.checks)
        lines.append("-" * 68)
        lines.append(f"  APPROVED TO PROCEED: {'YES' if self.passed else 'NO'}")
        return "\n".join(lines)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _local_media_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("file URI must be local")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"non-local media URI cannot be byte-verified: {parsed.scheme}")
    return Path(uri)


def _validate_charter(corpus_dir: Path, manifest: dict) -> tuple[bool, str]:
    path = corpus_dir / "dataset_charter.json"
    if not path.is_file():
        return False, "dataset_charter.json is absent"
    try:
        charter = _load_json(path)
        required = {
            "schema_version", "target_language", "dialect", "translation_direction",
            "task", "output_representation", "allowed_uses", "primary_population",
            "unacceptable_error_definition",
        }
        missing = required - set(charter)
        if missing:
            raise ValueError(f"missing fields {sorted(missing)}")
        if charter["schema_version"] != 1:
            raise ValueError("unsupported schema_version")
        if charter["target_language"] != manifest["language"]:
            raise ValueError("target_language disagrees with corpus manifest")
        for key in required - {"schema_version", "allowed_uses"}:
            if not isinstance(charter[key], str) or not charter[key].strip():
                raise ValueError(f"{key} must be a non-empty string")
        uses = charter["allowed_uses"]
        if not isinstance(uses, list) or not uses or any(
                not isinstance(use, str) or not use.strip() for use in uses):
            raise ValueError("allowed_uses must be a non-empty string list")
        undeclared = sorted({
            record["intended_use"] for record in manifest["records"]
            if record.get("intended_use") not in uses
        })
        if undeclared:
            raise ValueError(f"record intended uses are absent from charter: {undeclared}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, "target, task, output, use, population, and safety error are declared"


def _validate_sources(corpus_dir: Path, manifest: dict) -> tuple[bool, str]:
    verified: dict[str, str] = {}
    verified_authorizations: dict[str, str] = {}
    try:
        track_timestamps: dict[str, np.ndarray] = {}
        for split in manifest["splits"]:
            with np.load(corpus_dir / f"{split}.npz", allow_pickle=False) as shard:
                for row, sample_id in enumerate(shard["sample_ids"].tolist()):
                    length = int(shard["motion_lengths"][row])
                    track_timestamps[str(sample_id)] = np.array(
                        shard["frame_timestamps"][row, :length], copy=True)
        for record in manifest["records"]:
            uri = record.get("video_uri")
            if not isinstance(uri, str) or not uri:
                raise ValueError(f"{record['sample_id']}: source video URI is absent")
            path = _local_media_path(uri)
            if not path.is_file():
                raise ValueError(f"{record['sample_id']}: source media is not a local file")
            digest = sha256_file(path)
            if digest != record["media_sha256"].lower():
                raise ValueError(f"{record['sample_id']}: source media SHA-256 mismatch")
            decoded = decode_video(path)
            if decoded.frames.shape[0] < 2:
                raise ValueError(f"{record['sample_id']}: source video has fewer than two frames")
            extracted_timestamps = track_timestamps[record["sample_id"]]
            matched = np.isclose(extracted_timestamps[:, None], decoded.timestamps[None, :],
                                 rtol=0.0, atol=1e-9).any(axis=1)
            if not matched.all():
                raise ValueError(
                    f"{record['sample_id']}: extracted frame clock is not traceable to video PTS")
            source_id = record["source_id"]
            previous = verified.setdefault(source_id, digest)
            if previous != digest:
                raise ValueError(f"{source_id}: inconsistent immutable source bytes")
            if not record.get("license") or not record.get("intended_use"):
                raise ValueError(f"{record['sample_id']}: license/intended use is absent")
            try:
                consent = ConsentState[record.get("consent", "")]
                authorization = DataAuthorization.from_manifest(record.get("authorization"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{record['sample_id']}: invalid authorization") from error
            violations = validate_authorization(
                authorization, consent, record["intended_use"],
                requested_actions=("download", "create_derivatives", "model_training"))
            if authorization.license_identifier != record["license"] or violations:
                raise ValueError(
                    f"{record['sample_id']}: authorization is inconsistent: {violations}")
            evidence_path = _local_media_path(authorization.evidence_uri)
            if not evidence_path.is_absolute():
                evidence_path = corpus_dir / evidence_path
            if not evidence_path.is_file():
                raise ValueError(
                    f"{record['sample_id']}: authorization evidence is not a local file")
            evidence_digest = sha256_file(evidence_path)
            if evidence_digest != authorization.evidence_sha256.lower():
                raise ValueError(
                    f"{record['sample_id']}: authorization evidence SHA-256 mismatch")
            previous_evidence = verified_authorizations.setdefault(
                authorization.evidence_uri, evidence_digest)
            if previous_evidence != evidence_digest:
                raise ValueError("authorization evidence URI maps to inconsistent bytes")
            provenance = record.get("provenance", "")
            if len(provenance) != 64 or any(
                    char not in "0123456789abcdef" for char in provenance.lower()):
                raise ValueError(f"{record['sample_id']}: invalid provenance root")
    except (OSError, ValueError) as error:
        return False, str(error)
    return True, (
        f"verified {len(verified)} immutable source file(s) and "
        f"{len(verified_authorizations)} authorization evidence file(s)")


def _validate_agreement(corpus_dir: Path) -> tuple[bool, str]:
    path = corpus_dir / "annotation_agreement.json"
    if not path.is_file():
        return False, "annotation_agreement.json is absent"
    try:
        report = _load_json(path)
        if report.get("schema_version") != 1:
            raise ValueError("unsupported annotation agreement schema_version")
        tiers = report.get("tiers")
        if not isinstance(tiers, dict) or "gloss" not in tiers or not tiers:
            raise ValueError("agreement must be reported per tier and include gloss")
        for tier, result in tiers.items():
            kappa = result.get("kappa") if isinstance(result, dict) else None
            count = result.get("item_count") if isinstance(result, dict) else None
            if (not isinstance(kappa, (int, float)) or not np.isfinite(kappa)
                    or not -1 <= kappa <= 1 or not isinstance(count, int) or count < 2):
                raise ValueError(f"{tier}: invalid kappa or item_count")
        if not isinstance(report.get("uncertainty_and_adjudication"), str) or not report[
                "uncertainty_and_adjudication"].strip():
            raise ValueError("uncertainty_and_adjudication must be documented")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, f"agreement and uncertainty reported for {len(tiers)} tier(s)"


def _validate_review(corpus_dir: Path, manifest: dict) -> tuple[bool, str]:
    path = corpus_dir / "review_attestation.json"
    if not path.is_file():
        return False, "review_attestation.json is absent; review.html is only a queue"
    try:
        attestation = _load_json(path)
        if attestation.get("schema_version") != 1 or attestation.get("decision") != "approved":
            raise ValueError("review attestation is not schema v1 with decision=approved")
        if attestation.get("manifest_sha256") != sha256_file(corpus_dir / "manifest.json"):
            raise ValueError("review attestation does not bind this manifest")
        roles = attestation.get("reviewer_roles", [])
        if "qualified_target_language_signer" not in roles:
            raise ValueError("qualified target-language signer review is not attested")
        stages = set(attestation.get("reviewed_stages", []))
        required_stages = {"source_video", "extracted_landmarks", "exported_shard"}
        if not required_stages.issubset(stages):
            raise ValueError(f"reviewed_stages must include {sorted(required_stages)}")
        reviewed = set(attestation.get("reviewed_sample_ids", []))
        available = {record["sample_id"] for record in manifest["records"]}
        minimum = min(10, len(available))
        if not reviewed.issubset(available) or len(reviewed) < minimum:
            raise ValueError(f"review must cover at least {minimum} valid sample ids")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, f"qualified review covers {len(reviewed)} sample(s) at all stages"


def _validate_loader(corpus_dir: Path, manifest: dict) -> tuple[bool, str]:
    traced_ids: set[str] = set()
    try:
        for split, count in manifest["splits"].items():
            dataset = SignDataset(str(corpus_dir), split)
            if len(dataset) != count:
                raise ValueError(f"{split}: loader count mismatch")
            batch = next(iter(DataLoader(dataset, batch_size=len(dataset),
                                         collate_fn=collate_corpus)))
            if int(batch["frame_mask"].sum()) != int(batch["motion_lengths"].sum()):
                raise ValueError(f"{split}: frame mask and lengths disagree")
            if "sample_ids" not in batch:
                raise ValueError(f"{split}: batch lost sample provenance")
            traced_ids.update(batch["sample_ids"])
        manifest_ids = {record["sample_id"] for record in manifest["records"]}
        if traced_ids != manifest_ids:
            raise ValueError("loaded batches do not trace every manifest record")
    except (OSError, ValueError, RuntimeError) as error:
        return False, str(error)
    return True, f"canonical loader consumed and traced {len(traced_ids)} sample(s)"


def assess_stage_b_corpus(corpus_dir: str | Path) -> StageBReadinessReport:
    """Evaluate every Stage B exit claim; never infer readiness from unit tests."""
    root = Path(corpus_dir)
    report = StageBReadinessReport()
    try:
        validate_corpus(str(root))
        manifest = _load_json(root / "manifest.json")
        report.add("schema_and_shard_integrity", True, "v2 schema and shard hashes verified")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        report.add("schema_and_shard_integrity", False, str(error))
        return report

    report.add("holistic_extraction", set(manifest["landmark_parts"]) == {
        "body", "left_hand", "right_hand", "face"},
        "body, dense left/right hands, and face are explicitly partitioned")
    for name, validator in (
            ("dataset_charter", lambda: _validate_charter(root, manifest)),
            ("immutable_source_trace", lambda: _validate_sources(root, manifest)),
            ("annotation_agreement", lambda: _validate_agreement(root)),
            ("qualified_visual_review", lambda: _validate_review(root, manifest)),
            ("active_loader_trace", lambda: _validate_loader(root, manifest))):
        passed, detail = validator()
        report.add(name, passed, detail)
    return report
