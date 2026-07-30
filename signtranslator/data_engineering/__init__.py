"""Dataset and data-engineering layer (Document 10).

Implements `10_dataset_and_data_engineering.md` (see docs/DATA_ENGINEERING.md): a
canonical sample schema + dataset map, an evidence-backed authorization gate with a
Merkle-style provenance chain, quality mathematics (multi-view DLT triangulation,
confidence propagation, weighted robust reprojection), deduplication, per-tier
inter-annotator agreement + stratified QC, a leakage-certified grouped split, data
governance (consent/retention/policy + sensitive-trait non-inference), and
datasheets. Reuses Doc-04 (camera/reprojection), Doc-03 (kappa), and the existing
data-quality/readiness modules.
"""

from .schema import (
    ConsentState, AuthorizationBasis, PersonalityRightsStatus, DataAuthorization,
    AUTHORIZATION_ACTIONS, validate_authorization, Sample, validate_sample,
    DatasetResource, DATASET_MAP, dataset_map_is_complete, VALID_SPLITS,
)
from .provenance import (
    sha256_hex, content_hash, GateDecision, gate_download, ProvenanceStep,
    ProvenanceChain,
)
from .quality import (
    projection_matrix, triangulate_dlt, triangulation_confidence,
    weighted_reprojection_residual,
)
from .dedup import (
    average_hash, difference_hash, hamming_distance, jaccard_similarity,
    normalized_edit_distance, cluster_duplicates, near_threshold_pairs,
)
from .qc import (
    per_tier_kappa, pooled_kappa, weakest_tier, stratify, stratified_qc_sample,
)
from .splitting import (
    group_samples, grouped_split, certify_no_group_leakage, LeakageCertificate,
    Window, windows_inherit_split, certify_window_split_consistency,
)
from .governance import (
    transition_consent, ConsentError, apply_withdrawal, apply_retention,
    UsagePolicy, gate_action, infer_sensitive_trait, SensitiveInferenceError,
    SENSITIVE_TRAITS,
)
from .datasheet import (
    Datasheet, DATASHEET_SECTIONS, PreprocessingManifest,
)
from .exporter import (
    CORPUS_FORMAT_VERSION, DecodedAudio, DecodedVideo, DecodedVideoClock,
    decode_pcm_wav, decode_video, decode_video_clock,
    LandmarkTrack,
    assemble_holistic_track, decode_landmark_npz, ExtractedSample, ExportResult,
    export_corpus, sha256_file,
)
from .readiness import (
    StageBCheck, StageBReadinessReport, assess_stage_b_corpus,
)
from .how2sign import (
    HOW2SIGN_FIELDS, HOW2SIGN_LICENSE_ID, HOW2SIGN_LICENSE_URL,
    HOW2SIGN_PUBLISHER_EVIDENCE_URL, HOW2SIGN_CITATION_KEY,
    OPENPOSE_JOINT_NAMES, OPENPOSE_LANDMARK_PARTS,
    OPENPOSE_HOLISTIC_EDGES, OPENPOSE_HOLISTIC_CENTER, openpose_holistic_graph,
    How2SignRow, How2SignInventory,
    OpenPoseDiagnostics, How2SignClip, read_how2sign_metadata,
    inspect_how2sign_root, decode_how2sign_openpose, load_how2sign_clip,
    how2sign_authorization,
)

__all__ = [
    "ConsentState", "AuthorizationBasis", "PersonalityRightsStatus",
    "DataAuthorization", "AUTHORIZATION_ACTIONS", "validate_authorization",
    "Sample", "validate_sample", "DatasetResource", "DATASET_MAP",
    "dataset_map_is_complete", "VALID_SPLITS",
    "sha256_hex", "content_hash", "GateDecision", "gate_download",
    "ProvenanceStep", "ProvenanceChain",
    "projection_matrix", "triangulate_dlt", "triangulation_confidence",
    "weighted_reprojection_residual",
    "average_hash", "difference_hash", "hamming_distance", "jaccard_similarity",
    "normalized_edit_distance", "cluster_duplicates", "near_threshold_pairs",
    "per_tier_kappa", "pooled_kappa", "weakest_tier", "stratify",
    "stratified_qc_sample",
    "group_samples", "grouped_split", "certify_no_group_leakage",
    "LeakageCertificate", "Window", "windows_inherit_split",
    "certify_window_split_consistency",
    "transition_consent", "ConsentError", "apply_withdrawal", "apply_retention",
    "UsagePolicy", "gate_action", "infer_sensitive_trait",
    "SensitiveInferenceError", "SENSITIVE_TRAITS",
    "Datasheet", "DATASHEET_SECTIONS", "PreprocessingManifest",
    "CORPUS_FORMAT_VERSION", "DecodedAudio", "DecodedVideo", "DecodedVideoClock",
    "decode_pcm_wav", "decode_video", "decode_video_clock", "LandmarkTrack",
    "assemble_holistic_track", "decode_landmark_npz", "ExtractedSample",
    "ExportResult", "export_corpus",
    "sha256_file",
    "StageBCheck", "StageBReadinessReport", "assess_stage_b_corpus",
    "HOW2SIGN_FIELDS", "HOW2SIGN_LICENSE_ID", "HOW2SIGN_LICENSE_URL",
    "HOW2SIGN_PUBLISHER_EVIDENCE_URL", "HOW2SIGN_CITATION_KEY",
    "OPENPOSE_JOINT_NAMES", "OPENPOSE_LANDMARK_PARTS",
    "OPENPOSE_HOLISTIC_EDGES", "OPENPOSE_HOLISTIC_CENTER",
    "openpose_holistic_graph",
    "How2SignRow",
    "How2SignInventory", "OpenPoseDiagnostics", "How2SignClip",
    "read_how2sign_metadata", "inspect_how2sign_root",
    "decode_how2sign_openpose", "load_how2sign_clip",
    "how2sign_authorization",
]
