"""Content-addressed, fail-closed pseudo-gloss model and candidate artifacts."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .calibration import (
    AbstentionConfig,
    LogisticAcceptanceCalibrator,
)
from .contracts import (
    GlossLexicon,
    WeakGlossCandidateRecord,
    canonical_json_bytes,
    sha256_bytes,
)
from .model import (
    NeuralTextProposalModel,
    SourceTokenizer,
    TextProposalConfig,
    VideoCTCEvidenceModel,
    VideoEvidenceConfig,
    state_dict_sha256,
)
from .pipeline import FusionConfig, HybridPseudoGlossPipeline
from .security import InputSecurityPolicy, runtime_environment, strict_json_loads


BUNDLE_SCHEMA_VERSION = 2
CANDIDATE_BATCH_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ModelGovernance:
    text_model_id: str
    text_model_license: str
    text_model_source: str
    text_model_artifact_sha256: str
    text_model_license_evidence_sha256: str
    video_model_id: str
    video_model_license: str
    video_model_source: str
    video_model_artifact_sha256: str
    video_model_license_evidence_sha256: str
    training_data_manifest_sha256: str
    dependency_lock_sha256: str
    sbom_sha256: str
    intended_use: str

    def __post_init__(self) -> None:
        for name in (
            "text_model_id", "text_model_license", "text_model_source",
            "video_model_id", "video_model_license", "video_model_source",
            "intended_use",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        for name in (
            "text_model_artifact_sha256", "text_model_license_evidence_sha256",
            "video_model_artifact_sha256", "video_model_license_evidence_sha256",
            "training_data_manifest_sha256", "dependency_lock_sha256", "sbom_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 \
                    or any(character not in "0123456789abcdef"
                                       for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_safe_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("bundle destination cannot be a symlink")
    current = destination.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("bundle destination has a symlinked parent")
        current = current.parent
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("refusing to overwrite a non-empty bundle destination")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_record_chain(path: Path, records: Sequence[WeakGlossCandidateRecord]) -> None:
    with path.open("wb") as stream:
        previous_hash = "0" * 64
        for record in records:
            payload = record.to_dict()
            event_hash = hashlib.sha256(
                previous_hash.encode("ascii") + canonical_json_bytes(payload)).hexdigest()
            event = {"previous_event_sha256": previous_hash,
                     "event_sha256": event_hash, "record": payload}
            stream.write(canonical_json_bytes(event) + b"\n")
            previous_hash = event_hash
        stream.flush()
        os.fsync(stream.fileno())


def _validate_decision(decision: dict[str, Any]) -> None:
    required = {
        "accepted", "reason", "calibrated_probability",
        "selected_annotation_id", "dropped_text_probability_mass",
    }
    if set(decision) != required:
        raise ValueError("candidate decision schema mismatch")
    if not isinstance(decision["accepted"], bool):
        raise ValueError("candidate decision accepted field must be boolean")
    if not isinstance(decision["reason"], str) or not decision["reason"]:
        raise ValueError("candidate decision reason is required")
    probability = decision["calibrated_probability"]
    if probability is not None and (
            isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
        raise ValueError("calibrated probability must be null or finite in [0, 1]")
    dropped_mass = decision["dropped_text_probability_mass"]
    if (isinstance(dropped_mass, bool) or not isinstance(dropped_mass, (int, float))
            or not math.isfinite(float(dropped_mass)) or not 0 <= float(dropped_mass) <= 1):
        raise ValueError("dropped text probability mass must be finite in [0, 1]")
    selected = decision["selected_annotation_id"]
    if selected is not None and (not isinstance(selected, str) or not selected):
        raise ValueError("selected annotation ID must be null or a non-empty string")
    if decision["accepted"] and selected is None:
        raise ValueError("accepted candidate decision requires a selected annotation")
    if not decision["accepted"] and selected is not None:
        raise ValueError("abstained candidate decision cannot select an annotation")


def pipeline_configuration(pipeline: HybridPseudoGlossPipeline) -> dict[str, Any]:
    tokenizer = pipeline.text_model.tokenizer
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "tokenizer": {
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokens": list(tokenizer.tokens),
            "source_sha256": tokenizer.source_sha256,
        },
        "text_model": asdict(pipeline.text_model.config),
        "video_model": asdict(pipeline.video_model.config),
        "fusion": asdict(pipeline.fusion),
        "abstention": asdict(pipeline.abstention),
        "security": asdict(pipeline.security),
    }


def write_bundle(destination: str | os.PathLike[str], *,
                 pipeline: HybridPseudoGlossPipeline,
                 records: Sequence[WeakGlossCandidateRecord],
                 governance: ModelGovernance, seed: int, code_revision: str) -> Path:
    """Transactionally write weights, config, provenance, records, and hashes."""
    if not isinstance(governance, ModelGovernance):
        raise TypeError("model governance must use the validated ModelGovernance type")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("bundle seed must be an integer")
    if not isinstance(code_revision, str) or not code_revision:
        raise ValueError("bundle code revision must be a non-empty string")
    target = Path(destination)
    _assert_safe_destination(target)
    parent = target.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        pipeline._assert_frozen()
        text_model = pipeline.text_model
        video_model = pipeline.video_model
        lexicon = text_model.lexicon
        if lexicon != video_model.lexicon:
            raise ValueError("bundle models do not share an exact lexicon")
        for record in records:
            record.validate_against(lexicon)
        _write_json(temporary / "lexicon.json", {
            "lexicon_id": lexicon.lexicon_id,
            "convention_id": lexicon.convention_id,
            "tokens": list(lexicon.tokens),
            "source_sha256": lexicon.source_sha256,
        })
        _write_json(temporary / "config.json", pipeline_configuration(pipeline))
        _write_record_chain(temporary / "weak_gloss_candidates.jsonl", records)
        torch.save(text_model.state_dict(), temporary / "text_model.pt")
        torch.save(video_model.state_dict(), temporary / "video_model.pt")
        if pipeline.calibrator is not None:
            _write_json(temporary / "calibrator.json", pipeline.calibrator.state())
        artifact_names = sorted(path.name for path in temporary.iterdir())
        hashes = {name: sha256_file(temporary / name) for name in artifact_names}
        environment = runtime_environment()
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "code_revision": code_revision,
            "seed": seed,
            "machine_only_candidates": True,
            "translation_claim": False,
            "linguistic_validation_claim": False,
            "production_gloss_export_authorized": False,
            "model_governance": asdict(governance),
            "environment": environment,
            "environment_sha256": sha256_bytes(canonical_json_bytes(
                environment)),
            "lexicon_content_sha256": lexicon.content_sha256(),
            "text_state_sha256": state_dict_sha256(text_model),
            "video_state_sha256": state_dict_sha256(video_model),
            "candidate_count": len(records),
            "artifacts": hashes,
        }
        _write_json(temporary / "manifest.json", manifest)
        if target.exists():
            target.rmdir()  # prevalidated empty directory only
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "manifest.json"


def verify_bundle(bundle: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(bundle)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bundle must be a real directory")
    manifest_path = root / "manifest.json"
    manifest = strict_json_loads(manifest_path.read_bytes())
    required = {
        "schema_version", "code_revision", "seed", "machine_only_candidates",
        "translation_claim", "linguistic_validation_claim",
        "production_gloss_export_authorized", "model_governance",
        "environment", "environment_sha256",
        "lexicon_content_sha256",
        "text_state_sha256", "video_state_sha256", "candidate_count", "artifacts",
    }
    if set(manifest) != required \
            or isinstance(manifest["schema_version"], bool) \
            or not isinstance(manifest["schema_version"], int) \
            or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("bundle manifest schema mismatch")
    if (manifest["machine_only_candidates"] is not True
            or manifest["translation_claim"] is not False
            or manifest["linguistic_validation_claim"] is not False
            or manifest["production_gloss_export_authorized"] is not False):
        raise ValueError("bundle claim boundary was modified")
    if not isinstance(manifest["code_revision"], str) or not manifest["code_revision"] \
            or isinstance(manifest["seed"], bool) \
            or not isinstance(manifest["seed"], int) \
            or isinstance(manifest["candidate_count"], bool) \
            or not isinstance(manifest["candidate_count"], int) \
            or manifest["candidate_count"] < 0:
        raise ValueError("bundle revision, seed, or candidate count has an invalid type")
    if not isinstance(manifest["artifacts"], dict) or not manifest["artifacts"]:
        raise ValueError("bundle artifact map is empty or invalid")
    ModelGovernance(**manifest["model_governance"])
    if not isinstance(manifest["environment"], dict) \
            or manifest["environment_sha256"] != sha256_bytes(
                canonical_json_bytes(manifest["environment"])):
        raise ValueError("bundle environment metadata or hash is invalid")
    expected_names = set(manifest["artifacts"]) | {"manifest.json"}
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("bundle contains a symlink or non-file entry")
    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        raise ValueError("bundle contains missing or undeclared files")
    for name, expected in manifest["artifacts"].items():
        path = root / name
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"bundle artifact SHA-256 mismatch: {name}")
    lexicon_value = strict_json_loads((root / "lexicon.json").read_bytes())
    required_lexicon = {"lexicon_id", "convention_id", "tokens", "source_sha256"}
    if not isinstance(lexicon_value, dict) or set(lexicon_value) != required_lexicon:
        raise ValueError("bundle lexicon schema mismatch")
    lexicon = GlossLexicon(
        lexicon_id=lexicon_value["lexicon_id"],
        convention_id=lexicon_value["convention_id"],
        tokens=tuple(lexicon_value["tokens"]),
        source_sha256=lexicon_value["source_sha256"],
    )
    if lexicon.content_sha256() != manifest["lexicon_content_sha256"]:
        raise ValueError("bundle lexicon content hash mismatch")
    previous_hash = "0" * 64
    record_count = 0
    with (root / "weak_gloss_candidates.jsonl").open("rb") as stream:
        for line in stream:
            event = strict_json_loads(line)
            if set(event) != {"previous_event_sha256", "event_sha256", "record"}:
                raise ValueError("candidate audit event schema mismatch")
            if event["previous_event_sha256"] != previous_hash:
                raise ValueError("candidate audit hash chain is discontinuous")
            expected_event_hash = hashlib.sha256(
                previous_hash.encode("ascii") + canonical_json_bytes(event["record"])
            ).hexdigest()
            if event["event_sha256"] != expected_event_hash:
                raise ValueError("candidate audit event hash mismatch")
            record = WeakGlossCandidateRecord.from_dict(event["record"])
            record.validate_against(lexicon)
            if record.provenance.environment_sha256 != manifest["environment_sha256"]:
                raise ValueError("candidate environment binding mismatch")
            previous_hash = expected_event_hash
            record_count += 1
    if record_count != manifest["candidate_count"]:
        raise ValueError("candidate count does not match the bundle manifest")
    return manifest


def load_pipeline_bundle(bundle: str | os.PathLike[str]) -> HybridPseudoGlossPipeline:
    """Verify every byte, reconstruct models, and verify stable tensor hashes."""
    root = Path(bundle)
    manifest = verify_bundle(root)
    current_environment = runtime_environment()
    for name in ("python", "python_implementation", "torch", "numpy",
                 "platform_system", "platform_machine", "byteorder"):
        if manifest["environment"].get(name) != current_environment[name]:
            raise RuntimeError(f"runtime environment mismatch: {name}")
    lexicon_value = strict_json_loads((root / "lexicon.json").read_bytes())
    lexicon = GlossLexicon(
        lexicon_id=lexicon_value["lexicon_id"],
        convention_id=lexicon_value["convention_id"],
        tokens=tuple(lexicon_value["tokens"]),
        source_sha256=lexicon_value["source_sha256"],
    )
    config = strict_json_loads((root / "config.json").read_bytes())
    required_config = {
        "schema_version", "tokenizer", "text_model", "video_model",
        "fusion", "abstention", "security",
    }
    if set(config) != required_config \
            or isinstance(config["schema_version"], bool) \
            or not isinstance(config["schema_version"], int) \
            or config["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("pipeline configuration schema mismatch")
    tokenizer_value = config["tokenizer"]
    if not isinstance(tokenizer_value, dict) or set(tokenizer_value) != {
            "tokenizer_id", "tokens", "source_sha256"}:
        raise ValueError("source tokenizer configuration schema mismatch")
    tokenizer = SourceTokenizer(
        tokenizer_id=tokenizer_value["tokenizer_id"],
        tokens=tuple(tokenizer_value["tokens"]),
        source_sha256=tokenizer_value["source_sha256"],
    )
    text_model = NeuralTextProposalModel(
        tokenizer, lexicon, TextProposalConfig(**config["text_model"]))
    video_model = VideoCTCEvidenceModel(
        lexicon, VideoEvidenceConfig(**config["video_model"]))
    text_model.load_state_dict(torch.load(
        root / "text_model.pt", map_location="cpu", weights_only=True))
    video_model.load_state_dict(torch.load(
        root / "video_model.pt", map_location="cpu", weights_only=True))
    if state_dict_sha256(text_model) != manifest["text_state_sha256"] \
            or state_dict_sha256(video_model) != manifest["video_state_sha256"]:
        raise ValueError("reloaded model tensor hash mismatch")
    calibrator = None
    if (root / "calibrator.json").exists():
        calibrator = LogisticAcceptanceCalibrator.from_state(
            strict_json_loads((root / "calibrator.json").read_bytes()))
    pipeline = HybridPseudoGlossPipeline(
        text_model, video_model, FusionConfig(**config["fusion"]),
        AbstentionConfig(**config["abstention"]),
        InputSecurityPolicy(**config["security"]), calibrator,
    )
    pipeline.freeze_for_inference()
    return pipeline


def write_candidate_batch(destination: str | os.PathLike[str], *,
                          records: Sequence[WeakGlossCandidateRecord],
                          decision: dict[str, Any], model_bundle_manifest_sha256: str,
                          dataset_authorization_sha256: str,
                          source_sample_id: str, transcript_sha256: str,
                          source_video_sha256: str, landmark_track_sha256: str,
                          code_revision: str) -> Path:
    """Write an inference result without duplicating model weights or source media."""
    for name, value in (
        ("model_bundle_manifest_sha256", model_bundle_manifest_sha256),
        ("dataset_authorization_sha256", dataset_authorization_sha256),
        ("transcript_sha256", transcript_sha256),
        ("source_video_sha256", source_video_sha256),
        ("landmark_track_sha256", landmark_track_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if not isinstance(source_sample_id, str) or not source_sample_id:
        raise ValueError("source_sample_id must be a non-empty string")
    if any(record.provenance.source_sample_id != source_sample_id
           or record.provenance.transcript_sha256 != transcript_sha256
           or record.provenance.source_video_sha256 != source_video_sha256
           for record in records):
        raise ValueError("candidate records do not bind the declared batch inputs")
    _validate_decision(decision)
    target = Path(destination)
    _assert_safe_destination(target)
    parent = target.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        _write_record_chain(temporary / "weak_gloss_candidates.jsonl", records)
        records_hash = sha256_file(temporary / "weak_gloss_candidates.jsonl")
        manifest = {
            "schema_version": CANDIDATE_BATCH_SCHEMA_VERSION,
            "code_revision": code_revision,
            "model_bundle_manifest_sha256": model_bundle_manifest_sha256,
            "dataset_authorization_sha256": dataset_authorization_sha256,
            "source_sample_id": source_sample_id,
            "transcript_sha256": transcript_sha256,
            "source_video_sha256": source_video_sha256,
            "landmark_track_sha256": landmark_track_sha256,
            "record_count": len(records),
            "records_sha256": records_hash,
            "decision": decision,
            "machine_only_candidates": True,
            "translation_claim": False,
            "linguistic_validation_claim": False,
            "production_gloss_export_authorized": False,
        }
        _write_json(temporary / "manifest.json", manifest)
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "manifest.json"


def _verify_candidate_batch_preverified(
        batch: str | os.PathLike[str], *,
        model_bundle: str | os.PathLike[str],
        dataset_authorization: str | os.PathLike[str],
        verified_model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(batch)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate batch must be a real directory")
    manifest = strict_json_loads((root / "manifest.json").read_bytes())
    required = {
        "schema_version", "code_revision", "model_bundle_manifest_sha256",
        "dataset_authorization_sha256",
        "source_sample_id", "transcript_sha256",
        "source_video_sha256", "landmark_track_sha256", "record_count",
        "records_sha256", "decision", "machine_only_candidates",
        "translation_claim", "linguistic_validation_claim",
        "production_gloss_export_authorized",
    }
    if set(manifest) != required \
            or isinstance(manifest["schema_version"], bool) \
            or not isinstance(manifest["schema_version"], int) \
            or manifest["schema_version"] != CANDIDATE_BATCH_SCHEMA_VERSION:
        raise ValueError("candidate batch manifest schema mismatch")
    if (manifest["machine_only_candidates"] is not True
            or manifest["translation_claim"] is not False
            or manifest["linguistic_validation_claim"] is not False
            or manifest["production_gloss_export_authorized"] is not False):
        raise ValueError("candidate batch claim boundary was modified")
    if not isinstance(manifest["decision"], dict):
        raise ValueError("candidate decision must be an object")
    _validate_decision(manifest["decision"])
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries) \
            or {path.name for path in entries} != {
            "manifest.json", "weak_gloss_candidates.jsonl"}:
        raise ValueError("candidate batch contains undeclared artifacts")
    if sha256_file(root / "weak_gloss_candidates.jsonl") != manifest["records_sha256"]:
        raise ValueError("candidate records SHA-256 mismatch")
    model_root = Path(model_bundle)
    model_manifest = model_root / "manifest.json"
    if not isinstance(verified_model_manifest, Mapping):
        raise TypeError("preverified model manifest must be a mapping")
    if sha256_file(model_manifest) != manifest["model_bundle_manifest_sha256"]:
        raise ValueError("candidate batch model-bundle binding mismatch")
    from .readiness import validate_dataset_authorization_artifact
    authorization_path = Path(dataset_authorization)
    validate_dataset_authorization_artifact(
        authorization_path, requested_actions=("create_derivatives",))
    if sha256_file(authorization_path) != manifest["dataset_authorization_sha256"]:
        raise ValueError("candidate batch dataset-authorization binding mismatch")
    lexicon_value = strict_json_loads((model_root / "lexicon.json").read_bytes())
    lexicon = GlossLexicon(
        lexicon_id=lexicon_value["lexicon_id"],
        convention_id=lexicon_value["convention_id"],
        tokens=tuple(lexicon_value["tokens"]),
        source_sha256=lexicon_value["source_sha256"],
    )
    previous_hash = "0" * 64
    count = 0
    annotation_ids: list[str] = []
    candidate_ranks: list[int] = []
    expected_model_weight_sha256 = sha256_bytes(canonical_json_bytes({
        "text": verified_model_manifest["text_state_sha256"],
        "video": verified_model_manifest["video_state_sha256"],
    }))
    model_config = strict_json_loads((model_root / "config.json").read_bytes())
    expected_tokenizer_sha256 = model_config["tokenizer"]["source_sha256"]
    expected_decoding_sha256 = sha256_bytes(canonical_json_bytes({
        "abstention": model_config["abstention"],
        "fusion": model_config["fusion"],
        "security": model_config["security"],
    }))
    governance = verified_model_manifest["model_governance"]
    expected_model_id = f"{governance['text_model_id']}+{governance['video_model_id']}"
    with (root / "weak_gloss_candidates.jsonl").open("rb") as stream:
        for line in stream:
            event = strict_json_loads(line)
            if set(event) != {"previous_event_sha256", "event_sha256", "record"} \
                    or event["previous_event_sha256"] != previous_hash:
                raise ValueError("candidate batch hash chain is invalid")
            expected = hashlib.sha256(
                previous_hash.encode("ascii") + canonical_json_bytes(event["record"])
            ).hexdigest()
            if event["event_sha256"] != expected:
                raise ValueError("candidate batch event hash mismatch")
            record = WeakGlossCandidateRecord.from_dict(event["record"])
            record.validate_against(lexicon)
            if record.provenance.source_sample_id != manifest["source_sample_id"] \
                    or record.provenance.transcript_sha256 != manifest["transcript_sha256"]:
                raise ValueError("candidate record input binding mismatch")
            if record.provenance.source_video_sha256 != manifest["source_video_sha256"]:
                raise ValueError("candidate record source video binding mismatch")
            if record.provenance.code_revision != manifest["code_revision"] \
                    or record.provenance.code_revision != verified_model_manifest["code_revision"]:
                raise ValueError("candidate record code-revision binding mismatch")
            if record.provenance.model_weight_sha256 != expected_model_weight_sha256:
                raise ValueError("candidate record model-weight binding mismatch")
            if record.provenance.tokenizer_sha256 != expected_tokenizer_sha256:
                raise ValueError("candidate record tokenizer binding mismatch")
            if record.provenance.decoding_config_sha256 != expected_decoding_sha256:
                raise ValueError("candidate record decoding-configuration binding mismatch")
            if record.provenance.generator_model_id != expected_model_id:
                raise ValueError("candidate record model-identity binding mismatch")
            if record.provenance.environment_sha256 \
                    != verified_model_manifest["environment_sha256"]:
                raise ValueError("candidate record environment binding mismatch")
            annotation_ids.append(record.annotation_id)
            candidate_ranks.append(record.candidate_rank)
            previous_hash = expected
            count += 1
    if count != manifest["record_count"]:
        raise ValueError("candidate batch record count mismatch")
    if len(annotation_ids) != len(set(annotation_ids)) \
            or sorted(candidate_ranks) != list(range(1, count + 1)):
        raise ValueError("candidate IDs or ranks are not unique and contiguous")
    selected = manifest["decision"]["selected_annotation_id"]
    if manifest["decision"]["accepted"] and selected not in annotation_ids:
        raise ValueError("accepted decision does not select a recorded candidate")
    return manifest


def verify_candidate_batch(batch: str | os.PathLike[str], *,
                           model_bundle: str | os.PathLike[str],
                           dataset_authorization: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify a batch and independently re-hash its complete model bundle."""
    return _verify_candidate_batch_preverified(
        batch, model_bundle=model_bundle,
        dataset_authorization=dataset_authorization,
        verified_model_manifest=verify_bundle(model_bundle))
