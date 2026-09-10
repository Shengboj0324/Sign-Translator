"""Evidence-bound certification of How2Sign Green Screen signer identifiers.

The How2Sign supplemental material states that signer IDs are supplied per video
and reports exact Green Screen utterance counts by signer.  The released names use
the form ``<source>-<signer_id>-rgb_front``.  This module accepts that interpretation
only when the audited, non-missing local rows reconcile exactly with the published
per-signer counts.  It does not infer personal identity or sensitive attributes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

from ..reproducibility import (
    canonical_json_bytes,
    implementation_identity,
    sha256_file,
)
from .how2sign import (
    HOW2SIGN_GREEN_SIGNER_IDS,
    HOW2SIGN_SIGNER_EVIDENCE_SHA256,
    HOW2SIGN_SIGNER_EVIDENCE_URL,
    HOW2SIGN_TRAIN_UTTERANCES_BY_SIGNER,
)


SIGNER_CERTIFICATE_SCHEMA_VERSION = 2
_AUDITED_SOURCE_STATUSES = frozenset({
    "valid", "quality_warning", "structural_failure", "missing_source",
})
_ALL_AUDIT_STATUSES = frozenset({*_AUDITED_SOURCE_STATUSES, "unjoinable_artifact"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _checked_regular_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be a non-symlink regular file: {path}")
    return path.resolve()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("audit manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("audit manifest must be a JSON object")
    required = {
        "schema_version", "audit_complete", "metadata_rows", "audit_database",
        "identity", "status_counts",
    }
    if not required.issubset(manifest):
        raise ValueError(f"audit manifest is missing fields: {sorted(required - set(manifest))}")
    if manifest["schema_version"] != 1 or manifest["audit_complete"] is not True:
        raise ValueError("signer certification requires a complete audit schema v1")
    metadata_rows = manifest["metadata_rows"]
    if (isinstance(metadata_rows, bool) or not isinstance(metadata_rows, int)
            or metadata_rows <= 0):
        raise ValueError("audit manifest metadata_rows must be a positive integer")
    status_counts = manifest["status_counts"]
    if (not isinstance(status_counts, dict)
            or any(status not in _ALL_AUDIT_STATUSES for status in status_counts)
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                   for count in status_counts.values())):
        raise ValueError("audit manifest status_counts is malformed")
    source_count = sum(status_counts.get(status, 0)
                       for status in _AUDITED_SOURCE_STATUSES)
    if source_count != metadata_rows:
        raise ValueError("audit manifest statuses do not account for metadata_rows")
    return manifest


def _read_audited_rows(
    database: Path,
) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    uri = f"{database.as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT sample_id,video_id,filename_code,status FROM clips "
            "WHERE video_id IS NOT NULL ORDER BY sample_id"
        ).fetchall()
        database_status_counts = dict(connection.execute(
            "SELECT status,COUNT(*) FROM clips GROUP BY status ORDER BY status"
        ).fetchall())
    normalized: list[tuple[str, str, str, str]] = []
    previous_sample: str | None = None
    for row in rows:
        if (len(row) != 4 or any(not isinstance(value, str) or not value for value in row)):
            raise ValueError("audit contains a malformed metadata row")
        sample_id, video_id, signer_id, status = row
        if previous_sample is not None and sample_id <= previous_sample:
            raise ValueError("audit sample identifiers are not unique and ordered")
        previous_sample = sample_id
        if signer_id not in HOW2SIGN_GREEN_SIGNER_IDS:
            raise ValueError(f"audit contains unsupported Green Screen signer ID: {signer_id}")
        if status not in _AUDITED_SOURCE_STATUSES:
            raise ValueError(f"audit contains unsupported metadata status: {status}")
        normalized.append((sample_id, video_id, signer_id, status))
    if (any(not isinstance(status, str) or status not in _ALL_AUDIT_STATUSES
            for status in database_status_counts)
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                   for count in database_status_counts.values())):
        raise ValueError("audit database contains malformed status counts")
    return normalized, database_status_counts


def _mapping_csv(rows: list[tuple[str, str, str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("sample_id", "video_id", "signer_id", "audit_status"))
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def certify_how2sign_train_signers(
    audit_dir: str | os.PathLike[str],
    evidence_pdf: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create a compact signer map only after exact published-count reconciliation."""
    audit_root = Path(audit_dir)
    manifest_path = _checked_regular_file(
        audit_root / "audit_manifest.json", "audit manifest")
    database_path = _checked_regular_file(
        audit_root / "audit.sqlite3", "audit database")
    evidence_path = _checked_regular_file(Path(evidence_pdf), "signer evidence PDF")
    destination = Path(output_dir)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("signer-certificate output must be a real directory")
        if any(destination.iterdir()):
            raise FileExistsError("refusing non-empty signer-certificate output")

    manifest = _read_manifest(manifest_path)
    expected_database = manifest["audit_database"]
    if (not isinstance(expected_database, dict)
            or set(expected_database) != {"sha256", "size"}
            or not _is_sha256(expected_database["sha256"])
            or isinstance(expected_database["size"], bool)
            or not isinstance(expected_database["size"], int)
            or expected_database["size"] <= 0):
        raise ValueError("audit database manifest is malformed")
    manifest_before = sha256_file(manifest_path)
    database_before = sha256_file(database_path)
    if (database_before != expected_database["sha256"]
            or database_path.stat().st_size != expected_database["size"]):
        raise ValueError("audit database does not match its immutable manifest")

    rows, database_status_counts = _read_audited_rows(database_path)
    database_after = sha256_file(database_path)
    if database_after != database_before:
        raise RuntimeError("audit database changed during signer certification")
    if sha256_file(manifest_path) != manifest_before:
        raise RuntimeError("audit manifest changed during signer certification")
    if database_status_counts != manifest["status_counts"]:
        raise ValueError("audit database status counts do not match the manifest")
    if len(rows) != manifest["metadata_rows"]:
        raise ValueError("audited signer rows do not account for every metadata record")
    identity = manifest["identity"]
    if (not isinstance(identity, dict)
            or not _is_sha256(identity.get("metadata_sha256"))):
        raise ValueError("audit manifest metadata identity is malformed")
    evidence_sha256 = sha256_file(evidence_path)
    if evidence_sha256 != HOW2SIGN_SIGNER_EVIDENCE_SHA256:
        raise ValueError("signer evidence PDF does not match the pinned official artifact")

    available_counts = {signer_id: 0 for signer_id in HOW2SIGN_GREEN_SIGNER_IDS}
    missing_counts = {signer_id: 0 for signer_id in HOW2SIGN_GREEN_SIGNER_IDS}
    recording_ids: dict[str, set[str]] = {
        signer_id: set() for signer_id in HOW2SIGN_GREEN_SIGNER_IDS
    }
    for _, video_id, signer_id, status in rows:
        if status == "missing_source":
            missing_counts[signer_id] += 1
        else:
            available_counts[signer_id] += 1
            recording_ids[signer_id].add(video_id)
    if available_counts != HOW2SIGN_TRAIN_UTTERANCES_BY_SIGNER:
        raise ValueError(
            "audited non-missing rows do not match supplemental Table 2 signer counts")
    expected_missing = manifest["metadata_rows"] - sum(
        HOW2SIGN_TRAIN_UTTERANCES_BY_SIGNER.values())
    if expected_missing < 0 or sum(missing_counts.values()) != expected_missing:
        raise ValueError("missing-source rows do not reconcile metadata and Table 2 totals")

    mapping_bytes = _mapping_csv(rows)
    module_path = Path(__file__).resolve()
    code_identity = implementation_identity(
        (module_path, module_path.with_name("how2sign.py")),
        repo_root=module_path.parents[2],
    )
    destination.mkdir(parents=True, exist_ok=True)
    mapping_path = destination / "how2sign_train_signers.csv"
    _atomic_write(mapping_path, mapping_bytes)
    certificate = {
        "schema_version": SIGNER_CERTIFICATE_SCHEMA_VERSION,
        "certified": True,
        "claim": "Green Screen filename code is the official pseudonymous signer ID",
        "personal_identity_inferred": False,
        "sensitive_attributes_inferred": False,
        "final_split_created": False,
        "source": {
            "audit_manifest_sha256": manifest_before,
            "audit_database_sha256": database_before,
            "metadata_sha256": identity["metadata_sha256"],
        },
        "implementation": code_identity,
        "official_evidence": {
            "url": HOW2SIGN_SIGNER_EVIDENCE_URL,
            "local_pdf_sha256": evidence_sha256,
        },
        "published_available_utterances_by_signer": available_counts,
        "audited_missing_source_rows_by_signer": missing_counts,
        "available_recordings_by_signer": {
            signer_id: len(recording_ids[signer_id])
            for signer_id in HOW2SIGN_GREEN_SIGNER_IDS
        },
        "mapping": {
            "path": mapping_path.name,
            "sha256": hashlib.sha256(mapping_bytes).hexdigest(),
            "rows": len(rows),
        },
        "limitations": [
            "This certifies a stable pseudonymous grouping key, not a person's identity.",
            f"The {expected_missing} missing clips remain grouped but are not usable media.",
            "A final signer-and-source-disjoint split has not been created.",
        ],
    }
    certificate_path = destination / "certificate.json"
    _atomic_write(certificate_path, canonical_json_bytes(certificate) + b"\n")
    return certificate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--evidence-pdf", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    result = certify_how2sign_train_signers(
        arguments.audit_dir, arguments.evidence_pdf, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
