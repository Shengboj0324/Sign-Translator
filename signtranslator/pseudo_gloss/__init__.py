"""Fail-closed weak pseudo-gloss candidate research subsystem.

Unreviewed candidates produced here are not authentic gloss annotations and are
never exported through :mod:`signtranslator.data_engineering.exporter`.
"""

from .contracts import (
    CandidateHypothesis,
    CandidateProvenance,
    GlossLexicon,
    HumanGlossAnnotation,
    LabelType,
    ReviewStatus,
    WeakGlossCandidateRecord,
)
from .mathematics import (
    CTCAlignmentDiagnostics,
    FusedCandidate, NoisyLabelObjectiveCertificate,
    ctc_alignment_diagnostics,
    ctc_log_probability,
    ctc_minimum_frames,
    fuse_candidate_lattice,
    multi_candidate_marginal_loss,
    confidence_weighted_loss,
    selective_risk_curve,
)
from .security import (
    InputSecurityPolicy, runtime_environment, strict_json_loads, validate_transcript,
)
from .calibration import (
    AbstentionConfig, AcceptanceDecision, AcceptanceFeatures,
    CalibrationCertificate, CalibrationEvaluation, CalibrationSlice,
    CalibrationUncertainty, LogisticAcceptanceCalibrator, ReliabilityBin,
    decide_acceptance, evaluate_calibration, source_cluster_calibration_uncertainty,
)
from .model import (
    CandidateLatticeProposal, NeuralTextProposalModel, SourceTokenizer,
    TextProposalConfig, VideoCTCEvidenceModel, VideoEvidenceConfig,
    prepare_openpose_features, state_dict_sha256,
)
from .pipeline import (
    CandidateGenerationResult, FusionConfig, HybridPseudoGlossPipeline,
    apply_video_intervention,
)
from .training import (
    CrossFitAssignment, CrossFitTrainedFold, OptimizationConfig,
    TextTrainingExample, TrainingHistory,
    TextInitializationEvidence,
    VideoCandidateLatticeTrainingExample, VideoTrainingExample,
    WeakSupervisionEvidence, assign_source_folds, candidate_lattice_video_loss,
    certify_cross_fit,
    fit_cross_fold_text_models, fit_cross_fold_video_models,
    fit_text_model, fit_video_model, fit_video_model_on_candidate_lattices,
    training_indices_for_fold,
    validate_training_annotation,
)
from .evaluation import (
    ConstructionSlice, FalsificationSuiteReport, HumanReferenceCase,
    HumanReferenceCaseResult, HumanReferenceEvaluation, InterventionResult,
    InterventionSpecification, TokenErrorCounts,
    REQUIRED_FALSIFICATION_TESTS, build_falsification_report,
    candidate_deletion_abstains, certify_vocabulary_holdout,
    deterministic_source_derangement, evaluate_human_references,
    paired_source_bootstrap, token_error_counts,
)
from .artifacts import (
    BUNDLE_SCHEMA_VERSION, CANDIDATE_BATCH_SCHEMA_VERSION, ModelGovernance,
    load_pipeline_bundle, pipeline_configuration, verify_bundle,
    verify_candidate_batch, write_bundle, write_candidate_batch,
)
from .readiness import (
    ActivationCharter, ArtifactBinding, PseudoGlossReadinessReport,
    ReadinessCheck, assess_activation, load_activation_charter,
    validate_dataset_authorization_artifact,
)
from .corpus import (
    CORPUS_INFERENCE_SCHEMA_VERSION, CorpusInferenceRecord,
    load_corpus_manifest, run_corpus_inference,
)

__all__ = [
    "CandidateHypothesis", "CandidateProvenance", "GlossLexicon",
    "HumanGlossAnnotation", "LabelType",
    "ReviewStatus", "WeakGlossCandidateRecord", "CTCAlignmentDiagnostics",
    "FusedCandidate", "NoisyLabelObjectiveCertificate",
    "ctc_alignment_diagnostics", "ctc_log_probability",
    "ctc_minimum_frames", "fuse_candidate_lattice",
    "multi_candidate_marginal_loss", "confidence_weighted_loss",
    "selective_risk_curve", "InputSecurityPolicy", "strict_json_loads",
    "runtime_environment", "validate_transcript", "AbstentionConfig",
    "AcceptanceFeatures", "CalibrationCertificate", "LogisticAcceptanceCalibrator",
    "CalibrationEvaluation", "CalibrationSlice", "CalibrationUncertainty",
    "ReliabilityBin", "decide_acceptance", "evaluate_calibration",
    "source_cluster_calibration_uncertainty", "CandidateLatticeProposal",
    "NeuralTextProposalModel",
    "SourceTokenizer", "TextProposalConfig", "VideoCTCEvidenceModel",
    "VideoEvidenceConfig", "prepare_openpose_features", "state_dict_sha256",
    "CandidateGenerationResult", "FusionConfig", "HybridPseudoGlossPipeline",
    "apply_video_intervention", "CrossFitAssignment", "CrossFitTrainedFold",
    "OptimizationConfig",
    "TextTrainingExample", "TrainingHistory", "TextInitializationEvidence",
    "VideoCandidateLatticeTrainingExample", "VideoTrainingExample",
    "WeakSupervisionEvidence",
    "assign_source_folds", "certify_cross_fit", "fit_cross_fold_text_models",
    "fit_cross_fold_video_models", "candidate_lattice_video_loss",
    "fit_text_model", "fit_video_model", "fit_video_model_on_candidate_lattices",
    "training_indices_for_fold", "validate_training_annotation",
    "ConstructionSlice", "FalsificationSuiteReport", "HumanReferenceCase",
    "HumanReferenceCaseResult", "HumanReferenceEvaluation", "InterventionResult",
    "InterventionSpecification", "TokenErrorCounts",
    "REQUIRED_FALSIFICATION_TESTS", "build_falsification_report",
    "candidate_deletion_abstains", "certify_vocabulary_holdout",
    "deterministic_source_derangement", "evaluate_human_references",
    "paired_source_bootstrap", "token_error_counts",
    "BUNDLE_SCHEMA_VERSION", "ModelGovernance", "load_pipeline_bundle",
    "CANDIDATE_BATCH_SCHEMA_VERSION", "pipeline_configuration", "verify_bundle",
    "verify_candidate_batch", "write_bundle", "write_candidate_batch",
    "ActivationCharter", "ArtifactBinding", "PseudoGlossReadinessReport",
    "ReadinessCheck", "assess_activation", "load_activation_charter",
    "validate_dataset_authorization_artifact",
    "CORPUS_INFERENCE_SCHEMA_VERSION", "CorpusInferenceRecord",
    "load_corpus_manifest", "run_corpus_inference",
]
