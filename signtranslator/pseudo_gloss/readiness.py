"""Activation gate for the external evidence required by the research dossier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Mapping, Sequence

from ..data_engineering.schema import ConsentState, DataAuthorization, validate_authorization
from .artifacts import load_pipeline_bundle, sha256_file, verify_bundle
from .contracts import GlossLexicon, canonical_json_bytes, sha256_bytes
from .evaluation import REQUIRED_FALSIFICATION_TESTS
from .security import strict_json_loads


def _schema_version_one(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


@dataclass(frozen=True)
class ArtifactBinding:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path \
                or not isinstance(self.sha256, str) or len(self.sha256) != 64 \
                or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("artifact binding requires a path and lowercase SHA-256")


@dataclass(frozen=True)
class ActivationCharter:
    schema_version: int
    lexicon: ArtifactBinding
    annotation_convention: ArtifactBinding
    dataset_authorization: ArtifactBinding
    calibration_fit_reference_set: ArtifactBinding
    calibration_fit_attestation: ArtifactBinding
    human_reference_set: ArtifactBinding
    qualified_reference_attestation: ArtifactBinding
    preregistration: ArtifactBinding
    label_provenance_policy: ArtifactBinding
    review_workflow: ArtifactBinding
    falsification_report: ArtifactBinding
    calibration_evaluation: ArtifactBinding
    training_data_manifest: ArtifactBinding
    dependency_lock: ArtifactBinding
    sbom: ArtifactBinding
    text_model_artifact: ArtifactBinding
    text_model_license_evidence: ArtifactBinding
    video_model_artifact: ArtifactBinding
    video_model_license_evidence: ArtifactBinding
    model_bundle: ArtifactBinding
    signer_mapping: ArtifactBinding | None
    signer_generalization_claim: bool

    def __post_init__(self) -> None:
        if not _schema_version_one(self.schema_version):
            raise ValueError("unsupported activation charter schema")
        if self.signer_generalization_claim and self.signer_mapping is None:
            raise ValueError("signer-generalization claim requires authoritative signer mapping")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActivationCharter":
        required = {
            "schema_version", "lexicon", "annotation_convention",
            "dataset_authorization",
            "calibration_fit_reference_set", "calibration_fit_attestation",
            "human_reference_set",
            "qualified_reference_attestation",
            "preregistration", "label_provenance_policy", "review_workflow",
            "falsification_report", "calibration_evaluation",
            "training_data_manifest", "dependency_lock", "sbom",
            "text_model_artifact",
            "text_model_license_evidence", "video_model_artifact",
            "video_model_license_evidence", "model_bundle", "signer_mapping",
            "signer_generalization_claim",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(f"activation charter fields must be exactly {sorted(required)}")

        def binding(name: str, *, optional: bool = False):
            item = value[name]
            if item is None and optional:
                return None
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise ValueError(f"{name} must be an exact artifact binding")
            return ArtifactBinding(**item)

        if not isinstance(value["signer_generalization_claim"], bool):
            raise ValueError("signer_generalization_claim must be boolean")
        return cls(
            schema_version=value["schema_version"], lexicon=binding("lexicon"),
            annotation_convention=binding("annotation_convention"),
            dataset_authorization=binding("dataset_authorization"),
            calibration_fit_reference_set=binding("calibration_fit_reference_set"),
            calibration_fit_attestation=binding("calibration_fit_attestation"),
            human_reference_set=binding("human_reference_set"),
            qualified_reference_attestation=binding("qualified_reference_attestation"),
            preregistration=binding("preregistration"),
            label_provenance_policy=binding("label_provenance_policy"),
            review_workflow=binding("review_workflow"),
            falsification_report=binding("falsification_report"),
            calibration_evaluation=binding("calibration_evaluation"),
            training_data_manifest=binding("training_data_manifest"),
            dependency_lock=binding("dependency_lock"),
            sbom=binding("sbom"),
            text_model_artifact=binding("text_model_artifact"),
            text_model_license_evidence=binding("text_model_license_evidence"),
            video_model_artifact=binding("video_model_artifact"),
            video_model_license_evidence=binding("video_model_license_evidence"),
            model_bundle=binding("model_bundle"),
            signer_mapping=binding("signer_mapping", optional=True),
            signer_generalization_claim=value["signer_generalization_claim"],
        )


def load_activation_charter(path: str | Path) -> ActivationCharter:
    charter_path = Path(path)
    if charter_path.is_symlink() or not charter_path.is_file():
        raise ValueError("activation charter must be a non-symlink regular file")
    return ActivationCharter.from_dict(strict_json_loads(charter_path.read_bytes()))


def validate_dataset_authorization_artifact(
        path: str | Path, *,
        requested_actions: Sequence[str] = ("create_derivatives", "model_training"),
) -> DataAuthorization:
    authorization_path = Path(path)
    if authorization_path.is_symlink() or not authorization_path.is_file():
        raise ValueError("dataset authorization must be a non-symlink regular file")
    value = strict_json_loads(authorization_path.read_bytes())
    required = {"schema_version", "intended_use", "consent", "authorization"}
    if not isinstance(value, Mapping) or set(value) != required \
            or not _schema_version_one(value["schema_version"]) \
            or not isinstance(value["intended_use"], str) \
            or not value["intended_use"]:
        raise ValueError("dataset authorization artifact schema mismatch")
    try:
        consent = ConsentState[value["consent"]]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset authorization consent state is invalid") from error
    authorization = DataAuthorization.from_manifest(value["authorization"])
    violations = validate_authorization(
        authorization, consent, value["intended_use"],
        requested_actions=requested_actions)
    if violations:
        raise PermissionError(f"dataset authorization violations: {violations}")
    if "commercial_use" in authorization.permitted_actions \
            or "redistribution" in authorization.permitted_actions \
            or "identity_use" in authorization.permitted_actions:
        raise PermissionError("pseudo-gloss scope forbids commercial, redistribution, or identity use")
    evidence = Path(authorization.evidence_uri)
    if evidence.is_symlink() or not evidence.is_file() \
            or sha256_file(evidence) != authorization.evidence_sha256:
        raise ValueError("dataset license evidence is absent, symlinked, or hash-mismatched")
    return authorization


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PseudoGlossReadinessReport:
    checks: tuple[ReadinessCheck, ...]
    activation_approved: bool
    linguistic_validation_approved: bool = False
    production_gloss_export_approved: bool = False


def _bound_path(binding: ArtifactBinding, name: str) -> Path:
    path = Path(binding.path)
    if path.is_symlink() or not path.exists():
        raise ValueError(f"{name} is missing or symlinked")
    if path.is_file() and sha256_file(path) != binding.sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    if path.is_dir() and sha256_file(path / "manifest.json") != binding.sha256:
        raise ValueError(f"{name} manifest SHA-256 mismatch")
    return path


def _require_nonempty_file(path: Path, name: str) -> bool:
    if not path.is_file() or path.stat().st_size < 1:
        raise ValueError(f"{name} is empty or not a regular file")
    return True


def _require_disjoint(left: frozenset[str], right: frozenset[str], message: str) -> bool:
    if not left.isdisjoint(right):
        raise ValueError(message)
    return True


def assess_activation(charter: ActivationCharter) -> PseudoGlossReadinessReport:
    """Verify all activation artifacts; collect failures rather than bypassing them."""
    checks: list[ReadinessCheck] = []

    def check(name: str, operation) -> Any | None:
        try:
            result = operation()
        except (OSError, TypeError, ValueError, RuntimeError, PermissionError) as error:
            checks.append(ReadinessCheck(name, False, str(error)))
            return None
        checks.append(ReadinessCheck(name, True, "verified"))
        return result

    lexicon_path = check("versioned_closed_lexicon",
                         lambda: _bound_path(charter.lexicon, "lexicon"))
    convention_path = check("annotation_convention",
                            lambda: _bound_path(charter.annotation_convention,
                                                "annotation convention"))
    dataset_authorization_path = check("dataset_authorization", lambda: _bound_path(
        charter.dataset_authorization, "dataset authorization"))
    calibration_fit_path = check("calibration_fit_reference_set",
                                 lambda: _bound_path(
                                     charter.calibration_fit_reference_set,
                                     "calibration fit reference set"))
    calibration_fit_attestation_path = check(
        "calibration_fit_attestation", lambda: _bound_path(
            charter.calibration_fit_attestation, "calibration fit attestation"))
    reference_path = check("independent_human_reference_set",
                           lambda: _bound_path(charter.human_reference_set,
                                               "human reference set"))
    attestation_path = check("qualified_asl_attestation",
                             lambda: _bound_path(charter.qualified_reference_attestation,
                                                 "qualified reference attestation"))
    preregistration_path = check("preregistration",
                                 lambda: _bound_path(charter.preregistration,
                                                     "preregistration"))
    label_policy_path = check("label_provenance_policy", lambda: _bound_path(
        charter.label_provenance_policy, "label provenance policy"))
    review_workflow_path = check("qualified_review_workflow", lambda: _bound_path(
        charter.review_workflow, "review workflow"))
    falsification_path = check("falsification_report", lambda: _bound_path(
        charter.falsification_report, "falsification report"))
    calibration_evaluation_path = check("calibration_evaluation", lambda: _bound_path(
        charter.calibration_evaluation, "calibration evaluation"))
    training_manifest_path = check("training_data_manifest", lambda: _bound_path(
        charter.training_data_manifest, "training data manifest"))
    dependency_lock_path = check("dependency_lock", lambda: _bound_path(
        charter.dependency_lock, "dependency lock"))
    sbom_path = check("sbom", lambda: _bound_path(charter.sbom, "SBOM"))
    check("text_model_artifact", lambda: _bound_path(
        charter.text_model_artifact, "text model artifact"))
    check("text_model_license_evidence", lambda: _bound_path(
        charter.text_model_license_evidence, "text model license evidence"))
    check("video_model_artifact", lambda: _bound_path(
        charter.video_model_artifact, "video model artifact"))
    check("video_model_license_evidence", lambda: _bound_path(
        charter.video_model_license_evidence, "video model license evidence"))
    model_path = check("licensed_model_bundle",
                       lambda: _bound_path(charter.model_bundle, "model bundle"))
    if charter.signer_generalization_claim:
        signer_mapping = charter.signer_mapping
        if signer_mapping is None:  # Defensive even though __post_init__ rejects it.
            raise RuntimeError("signer mapping vanished after charter validation")
        check("authoritative_signer_mapping",
              lambda: _bound_path(signer_mapping, "signer mapping"))
    else:
        checks.append(ReadinessCheck(
            "authoritative_signer_mapping", True,
            "not required because no signer-generalization claim is declared"))

    lexicon = None
    if lexicon_path is not None and lexicon_path.is_file():
        def validate_lexicon():
            value = strict_json_loads(lexicon_path.read_bytes())
            if set(value) != {"lexicon_id", "convention_id", "tokens", "source_sha256"}:
                raise ValueError("lexicon schema mismatch")
            return GlossLexicon(value["lexicon_id"], value["convention_id"],
                                tuple(value["tokens"]), value["source_sha256"])
        lexicon = check("lexicon_schema", validate_lexicon)

    human_reference_info = None
    calibration_fit_info = None
    if lexicon is not None:
        def validate_reference_set(path: Path):
            value = strict_json_loads(path.read_bytes())
            required = {"schema_version", "language", "convention_id", "records"}
            if set(value) != required or not _schema_version_one(value["schema_version"]) \
                    or value["language"] != "ASL" \
                    or value["convention_id"] != lexicon.convention_id \
                    or not isinstance(value["records"], list) or not value["records"]:
                raise ValueError("human reference set schema or ASL convention is invalid")
            annotation_ids = []
            for record in value["records"]:
                record_fields = {
                    "annotation_id", "source_id", "tokens", "label_type",
                    "reviewer_pseudonym",
                }
                if not isinstance(record, Mapping) or set(record) != record_fields:
                    raise ValueError("human reference record schema mismatch")
                if record["label_type"] not in {"official_human", "project_human"}:
                    raise PermissionError(
                        "independent references must be official or project-human labels")
                if not record["annotation_id"] or not record["source_id"] \
                        or not record["reviewer_pseudonym"]:
                    raise ValueError("human reference record lacks provenance")
                lexicon.encode(record["tokens"])
                annotation_ids.append(record["annotation_id"])
            if len(annotation_ids) != len(set(annotation_ids)):
                raise ValueError("human reference annotation IDs are not unique")
            return len(value["records"]), frozenset(
                record["source_id"] for record in value["records"])
        if reference_path is not None:
            human_reference_info = check(
                "human_reference_schema",
                lambda: validate_reference_set(reference_path))
        if calibration_fit_path is not None:
            calibration_fit_info = check(
                "calibration_fit_reference_schema",
                lambda: validate_reference_set(calibration_fit_path))
    if human_reference_info is not None and calibration_fit_info is not None:
        check("calibration_fit_evaluation_source_disjointness", lambda: _require_disjoint(
            human_reference_info[1], calibration_fit_info[1],
            "calibration fit and held-out evaluation sources overlap"))

    convention = None
    if convention_path is not None:
        def validate_convention():
            value = strict_json_loads(convention_path.read_bytes())
            required = {"schema_version", "convention_id", "language", "token_rules",
                        "nonmanual_policy", "fingerspelling_policy", "spatial_policy"}
            if set(value) != required or not _schema_version_one(value["schema_version"]) \
                    or value["language"] != "ASL" \
                    or any(not value[name] for name in required - {"schema_version"}):
                raise ValueError("annotation convention schema or ASL scope is invalid")
            return value
        convention = check("annotation_convention_schema", validate_convention)

    attestation = None
    calibration_fit_attestation = None
    if attestation_path is not None or calibration_fit_attestation_path is not None:
        def validate_attestation(path: Path, reference_sha256: str):
            value = strict_json_loads(path.read_bytes())
            required = {
                "schema_version", "qualified_asl_reference",
                "independent_from_candidate_generation", "source_disjoint",
                "reference_set_sha256", "convention_id", "review_protocol",
                "reviewer_pseudonyms",
            }
            if set(value) != required or not _schema_version_one(value["schema_version"]):
                raise ValueError("qualified-reference attestation schema mismatch")
            if (value["qualified_asl_reference"] is not True
                    or value["independent_from_candidate_generation"] is not True
                    or value["source_disjoint"] is not True):
                raise PermissionError("reference attestation does not satisfy independence gates")
            if value["reference_set_sha256"] != reference_sha256:
                raise ValueError("attestation does not bind the human reference set")
            if not value["review_protocol"] or not value["reviewer_pseudonyms"]:
                raise ValueError("reference attestation lacks reviewer evidence")
            return value
        if attestation_path is not None:
            attestation = check("qualified_reference_schema", lambda: validate_attestation(
                attestation_path, charter.human_reference_set.sha256))
        if calibration_fit_attestation_path is not None:
            calibration_fit_attestation = check(
                "calibration_fit_attestation_schema", lambda: validate_attestation(
                    calibration_fit_attestation_path,
                    charter.calibration_fit_reference_set.sha256))

    preregistration = None
    if preregistration_path is not None:
        def validate_preregistration():
            value = strict_json_loads(preregistration_path.read_bytes())
            required = {"schema_version", "primary_endpoint", "stop_rules",
                        "falsification_thresholds", "thresholds_frozen",
                        "falsification_specifications",
                        "calibration_bin_edges",
                        "test_set_locked", "analysis_plan",
                        "decoding_config_sha256"}
            if set(value) != required or not _schema_version_one(value["schema_version"]) \
                    or value["thresholds_frozen"] is not True \
                    or value["test_set_locked"] is not True \
                    or any(not value[name] for name in (
                        "primary_endpoint", "stop_rules", "falsification_thresholds",
                        "analysis_plan")):
                raise ValueError("preregistration is incomplete or not frozen")
            thresholds = value["falsification_thresholds"]
            if not isinstance(thresholds, Mapping) \
                    or set(thresholds) != set(REQUIRED_FALSIFICATION_TESTS) \
                    or any(isinstance(number, bool) or not isinstance(number, (int, float))
                           or not math.isfinite(float(number)) or float(number) < 0
                           for number in thresholds.values()):
                raise ValueError("preregistered falsification thresholds are incomplete")
            specifications = value["falsification_specifications"]
            specification_fields = {
                "minimum_mean_decline", "null_hypothesis", "effect_measure",
                "stop_rule", "confidence_level", "bootstrap_replicates", "seed",
            }
            if not isinstance(specifications, Mapping) \
                    or set(specifications) != set(REQUIRED_FALSIFICATION_TESTS):
                raise ValueError("preregistered falsification specifications are incomplete")
            for name, specification in specifications.items():
                if not isinstance(specification, Mapping) \
                        or set(specification) != specification_fields \
                        or float(specification["minimum_mean_decline"]) \
                        != float(thresholds[name]) \
                        or any(not specification[field] for field in (
                            "null_hypothesis", "effect_measure", "stop_rule")) \
                        or not isinstance(specification["bootstrap_replicates"], int) \
                        or isinstance(specification["bootstrap_replicates"], bool) \
                        or specification["bootstrap_replicates"] < 100 \
                        or not isinstance(specification["seed"], int) \
                        or isinstance(specification["seed"], bool) \
                        or not isinstance(specification["confidence_level"], (int, float)) \
                        or isinstance(specification["confidence_level"], bool) \
                        or not 0 < float(specification["confidence_level"]) < 1:
                    raise ValueError("a preregistered falsification specification is invalid")
            digest = value["decoding_config_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 \
                    or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("preregistered decoding configuration hash is invalid")
            calibration_edges = value["calibration_bin_edges"]
            if not isinstance(calibration_edges, list) or len(calibration_edges) < 2 \
                    or any(isinstance(edge, bool) or not isinstance(edge, (int, float))
                           or not math.isfinite(float(edge)) for edge in calibration_edges) \
                    or float(calibration_edges[0]) != 0 \
                    or float(calibration_edges[-1]) != 1 \
                    or any(float(right) <= float(left) for left, right in zip(
                        calibration_edges, calibration_edges[1:])):
                raise ValueError("preregistered calibration bins are invalid")
            return value
        preregistration = check("preregistration_schema", validate_preregistration)

    if label_policy_path is not None:
        def validate_label_policy():
            value = strict_json_loads(label_policy_path.read_bytes())
            required = {
                "schema_version", "policy_id", "policy_owner",
                "allowed_training_label_types", "unreviewed_may_enter_gloss_tokens",
                "promotion_function", "source_video_review_required",
                "machine_parent_preservation_required",
            }
            if not isinstance(value, Mapping) or set(value) != required \
                    or not _schema_version_one(value["schema_version"]) \
                    or value["allowed_training_label_types"] != [
                        "official_human", "project_human", "human_corrected_pseudo"] \
                    or value["unreviewed_may_enter_gloss_tokens"] is not False \
                    or value["promotion_function"] != "promote_reviewed_weak_candidate" \
                    or value["source_video_review_required"] is not True \
                    or value["machine_parent_preservation_required"] is not True \
                    or not value["policy_id"] or not value["policy_owner"]:
                raise ValueError("label provenance policy is incomplete or unsafe")
            return value
        check("label_provenance_policy_schema", validate_label_policy)

    if review_workflow_path is not None:
        def validate_review_workflow():
            value = strict_json_loads(review_workflow_path.read_bytes())
            required = {
                "schema_version", "protocol_id", "qualified_asl_required",
                "source_video_visible", "independent_review_required",
                "reviewer_attestation_required", "rejection_and_abstention_recorded",
            }
            if not isinstance(value, Mapping) or set(value) != required \
                    or not _schema_version_one(value["schema_version"]) \
                    or not value["protocol_id"] \
                    or any(value[name] is not True for name in required - {
                        "schema_version", "protocol_id"}):
                raise ValueError("qualified review workflow is incomplete")
            return value
        check("qualified_review_workflow_schema", validate_review_workflow)

    if dependency_lock_path is not None:
        check("dependency_lock_nonempty", lambda: _require_nonempty_file(
            dependency_lock_path, "dependency lock"))
    if dataset_authorization_path is not None:
        check("dataset_authorization_schema", lambda:
              validate_dataset_authorization_artifact(dataset_authorization_path))

    training_source_ids = None
    if training_manifest_path is not None:
        def validate_training_manifest():
            value = strict_json_loads(training_manifest_path.read_bytes())
            required = {
                "schema_version", "source_group_type",
                "local_text_training_source_ids", "local_video_training_source_ids",
                "external_pretraining_dataset_ids",
                "external_pretraining_source_overlap_assessed",
                "signer_disjoint_claim",
            }
            if not isinstance(value, Mapping) or set(value) != required \
                    or not _schema_version_one(value["schema_version"]) \
                    or value["source_group_type"] != "VIDEO_ID" \
                    or value["external_pretraining_source_overlap_assessed"] is not True \
                    or not isinstance(value["signer_disjoint_claim"], bool):
                raise ValueError("training data manifest schema or source policy is invalid")
            lists = (
                value["local_text_training_source_ids"],
                value["local_video_training_source_ids"],
                value["external_pretraining_dataset_ids"],
            )
            if any(not isinstance(items, list) or len(items) != len(set(items))
                   or any(not isinstance(item, str) or not item for item in items)
                   for items in lists):
                raise ValueError("training data manifest lists must contain unique IDs")
            local_sources = frozenset(lists[0]) | frozenset(lists[1])
            if not local_sources:
                raise ValueError("training data manifest contains no local training sources")
            if value["signer_disjoint_claim"] != charter.signer_generalization_claim:
                raise ValueError("training manifest signer claim contradicts the charter")
            return local_sources
        training_source_ids = check(
            "training_data_manifest_schema", validate_training_manifest)

    if training_source_ids is not None and human_reference_info is not None:
        check("training_human_reference_source_disjointness", lambda: _require_disjoint(
            training_source_ids, human_reference_info[1],
            "model training and human-reference VIDEO_ID groups overlap"))
    if training_source_ids is not None and calibration_fit_info is not None:
        check("training_calibration_fit_source_disjointness", lambda: _require_disjoint(
            training_source_ids, calibration_fit_info[1],
            "model training and calibration-fit VIDEO_ID groups overlap"))

    if sbom_path is not None:
        def validate_sbom():
            value = strict_json_loads(sbom_path.read_bytes())
            if not isinstance(value, Mapping) or not (
                    value.get("bomFormat") == "CycloneDX" or "spdxVersion" in value):
                raise ValueError("SBOM must be CycloneDX or SPDX JSON")
            return value
        check("sbom_schema", validate_sbom)

    pipeline = None
    model_manifest = None
    if model_path is not None and model_path.is_dir():
        model_manifest = check("model_bundle_integrity", lambda: verify_bundle(model_path))
        pipeline = check("model_checkpoint_reload", lambda: load_pipeline_bundle(model_path))
    if pipeline is not None:
        def validate_calibration():
            if pipeline.calibrator is None or pipeline.calibrator.certificate is None:
                raise PermissionError("model bundle lacks qualified-reference calibration")
            certificate = pipeline.calibrator.certificate
            if certificate.reference_set_sha256 \
                    != charter.calibration_fit_reference_set.sha256:
                raise ValueError("calibrator is not bound to the calibration fit set")
            if not certificate.qualified_asl_reference or not certificate.source_disjoint:
                raise PermissionError("calibrator certificate fails reference gates")
            if calibration_fit_attestation is None \
                    or certificate.protocol_id \
                    != calibration_fit_attestation["review_protocol"]:
                raise ValueError("calibrator protocol does not bind the fit attestation")
            if calibration_fit_info is not None \
                    and certificate.reference_count != calibration_fit_info[0]:
                raise ValueError("calibrator/fit-reference record counts do not match")
            return certificate
        check("qualified_acceptance_calibration", validate_calibration)

    def require_equal(left: object, right: object, message: str) -> bool:
        if left != right:
            raise ValueError(message)
        return True

    if pipeline is not None and lexicon is not None:
        check("model_lexicon_binding", lambda: require_equal(
            pipeline.text_model.lexicon.content_sha256(), lexicon.content_sha256(),
            "model/external lexicon mismatch"))
    if model_path is not None and preregistration is not None:
        config = strict_json_loads((model_path / "config.json").read_bytes())
        decoding_config_sha256 = sha256_bytes(canonical_json_bytes({
            "abstention": config["abstention"],
            "fusion": config["fusion"],
            "security": config["security"],
        }))
        check("preregistered_decoding_policy_binding", lambda: require_equal(
            decoding_config_sha256, preregistration["decoding_config_sha256"],
            "model decoding policy was not the preregistered policy"))

    if falsification_path is not None and preregistration is not None:
        def validate_falsification_report():
            value = strict_json_loads(falsification_path.read_bytes())
            required = {
                "schema_version", "model_manifest_sha256", "preregistration_sha256",
                "human_reference_set_sha256", "results", "source_holdout_completed",
                "human_reference_completed", "untouched_test_set_evaluations",
                "stop_rules_triggered",
            }
            if not isinstance(value, Mapping) or set(value) != required \
                    or not _schema_version_one(value["schema_version"]):
                raise ValueError("falsification report schema mismatch")
            if value["model_manifest_sha256"] != charter.model_bundle.sha256 \
                    or value["preregistration_sha256"] != charter.preregistration.sha256 \
                    or value["human_reference_set_sha256"] != charter.human_reference_set.sha256:
                raise ValueError("falsification report artifact bindings mismatch")
            if value["source_holdout_completed"] is not True \
                    or value["human_reference_completed"] is not True \
                    or value["untouched_test_set_evaluations"] != 1 \
                    or value["stop_rules_triggered"] != []:
                raise PermissionError("falsification or untouched-test gates did not pass")
            if not isinstance(value["results"], list):
                raise ValueError("falsification results must be a list")
            by_name = {}
            result_fields = {
                "name", "passed", "mean_decline", "lower_confidence_bound",
                "upper_confidence_bound", "required_decline", "sample_count",
                "source_group_count",
                "specification_sha256",
            }
            for result in value["results"]:
                if not isinstance(result, Mapping) or set(result) != result_fields \
                        or result["name"] in by_name:
                    raise ValueError("falsification result schema or uniqueness failure")
                numeric = (result["mean_decline"], result["lower_confidence_bound"],
                           result["upper_confidence_bound"], result["required_decline"])
                if result["passed"] is not True \
                        or any(isinstance(number, bool) or not isinstance(number, (int, float))
                               or not math.isfinite(float(number)) for number in numeric) \
                        or isinstance(result["sample_count"], bool) \
                        or isinstance(result["source_group_count"], bool) \
                        or not isinstance(result["sample_count"], int) \
                        or not isinstance(result["source_group_count"], int) \
                        or result["sample_count"] < 1 or result["source_group_count"] < 2 \
                        or float(result["lower_confidence_bound"]) \
                        > float(result["upper_confidence_bound"]) \
                        or float(result["lower_confidence_bound"]) \
                        <= float(result["required_decline"]):
                    raise PermissionError("a falsification result failed or is invalid")
                by_name[result["name"]] = result
            if set(by_name) != set(REQUIRED_FALSIFICATION_TESTS):
                raise ValueError("falsification report does not contain all required tests")
            thresholds = preregistration["falsification_thresholds"]
            for name, result in by_name.items():
                if float(result["required_decline"]) != float(thresholds[name]):
                    raise ValueError("falsification result does not use preregistered threshold")
                expected_specification_sha256 = sha256_bytes(canonical_json_bytes({
                    "name": name, **preregistration["falsification_specifications"][name],
                }))
                if result["specification_sha256"] != expected_specification_sha256:
                    raise ValueError("falsification result specification hash mismatch")
            return value
        check("complete_falsification_suite", validate_falsification_report)

    if calibration_evaluation_path is not None and preregistration is not None \
            and pipeline is not None and pipeline.calibrator is not None:
        def validate_calibration_evaluation():
            value = strict_json_loads(calibration_evaluation_path.read_bytes())
            required = {
                "schema_version", "reference_set_sha256", "calibrator_state_sha256",
                "held_out", "bin_edges", "count", "source_group_count",
                "brier_score", "log_loss", "expected_calibration_error",
                "maximum_calibration_error", "confidence_level",
                "bootstrap_replicates", "brier_interval",
                "expected_calibration_error_interval",
            }
            if not isinstance(value, Mapping) or set(value) != required \
                    or not _schema_version_one(value["schema_version"]) \
                    or value["held_out"] is not True \
                    or value["reference_set_sha256"] != charter.human_reference_set.sha256 \
                    or value["calibrator_state_sha256"] \
                    != pipeline.calibrator.state_sha256() \
                    or value["bin_edges"] != preregistration["calibration_bin_edges"] \
                    or human_reference_info is None \
                    or value["count"] != human_reference_info[0] \
                    or not isinstance(value["source_group_count"], int) \
                    or isinstance(value["source_group_count"], bool) \
                    or value["source_group_count"] < 2 \
                    or not isinstance(value["bootstrap_replicates"], int) \
                    or isinstance(value["bootstrap_replicates"], bool) \
                    or value["bootstrap_replicates"] < 100:
                raise ValueError("held-out calibration evaluation schema or bindings fail")
            scalar_fields = (
                "brier_score", "log_loss", "expected_calibration_error",
                "maximum_calibration_error", "confidence_level",
            )
            if any(isinstance(value[name], bool)
                   or not isinstance(value[name], (int, float))
                   or not math.isfinite(float(value[name])) for name in scalar_fields):
                raise ValueError("calibration evaluation contains non-finite metrics")
            if not 0 < float(value["confidence_level"]) < 1 \
                    or any(not 0 <= float(value[name]) <= 1 for name in (
                        "brier_score", "expected_calibration_error",
                        "maximum_calibration_error")):
                raise ValueError("calibration evaluation metrics are out of bounds")
            for name in ("brier_interval", "expected_calibration_error_interval"):
                interval = value[name]
                if not isinstance(interval, list) or len(interval) != 2 \
                        or any(isinstance(bound, bool)
                               or not isinstance(bound, (int, float))
                               or not math.isfinite(float(bound)) for bound in interval) \
                        or float(interval[0]) > float(interval[1]):
                    raise ValueError("calibration uncertainty interval is invalid")
            return value
        check("held_out_calibration_metrics", validate_calibration_evaluation)

    if lexicon is not None and convention is not None:
        check("lexicon_convention_binding", lambda: require_equal(
            lexicon.convention_id, convention["convention_id"],
            "lexicon/convention ID mismatch"))
    if lexicon is not None and attestation is not None:
        check("reference_convention_binding", lambda: require_equal(
            lexicon.convention_id, attestation["convention_id"],
            "reference/lexicon convention mismatch"))
    if model_manifest is not None:
        check("model_license_and_training_provenance", lambda: ModelBundleEvidence(
            model_manifest["model_governance"]))
        governance = model_manifest["model_governance"]
        check("model_artifact_and_license_bindings", lambda: (
            require_equal(governance["text_model_artifact_sha256"],
                          charter.text_model_artifact.sha256,
                          "text model artifact hash mismatch"),
            require_equal(governance["text_model_license_evidence_sha256"],
                          charter.text_model_license_evidence.sha256,
                          "text model license evidence hash mismatch"),
            require_equal(governance["video_model_artifact_sha256"],
                          charter.video_model_artifact.sha256,
                          "video model artifact hash mismatch"),
            require_equal(governance["video_model_license_evidence_sha256"],
                          charter.video_model_license_evidence.sha256,
                          "video model license evidence hash mismatch"),
            require_equal(governance["training_data_manifest_sha256"],
                          charter.training_data_manifest.sha256,
                          "training data manifest hash mismatch"),
            require_equal(governance["dependency_lock_sha256"],
                          charter.dependency_lock.sha256,
                          "dependency lock hash mismatch"),
            require_equal(governance["sbom_sha256"], charter.sbom.sha256,
                          "SBOM hash mismatch"),
        ))

    approved = bool(checks) and all(item.passed for item in checks)
    return PseudoGlossReadinessReport(tuple(checks), activation_approved=approved)


class ModelBundleEvidence:
    """Strict validator used by the readiness gate for model governance fields."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        required = {
            "text_model_id", "text_model_license", "text_model_source",
            "text_model_artifact_sha256", "text_model_license_evidence_sha256",
            "video_model_id", "video_model_license", "video_model_source",
            "video_model_artifact_sha256", "video_model_license_evidence_sha256",
            "training_data_manifest_sha256", "dependency_lock_sha256", "sbom_sha256",
            "intended_use",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("model governance schema mismatch")
        if any(not value[name] for name in required):
            raise ValueError("model governance contains an empty field")
