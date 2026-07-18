"""Speech foundation layer (see docs/SPEECH_FOUNDATION.md).

Stage 1 (implemented): waveform -> log-Mel features, prosody, streaming with an
explicit latency model, the resampler + gated projection into planner width, and
LoRA adapters with the freeze-first protocol.

Stages 2-5 (planned): CTC beam search / N-best lattice with revision, word
timestamps, confidence calibration and the fail-closed policy, the full training
objective, and the evaluation harness.
"""

from .features import (
    LogMelSpectrogram, mel_filterbank, hz_to_mel, mel_to_hz, stft_power,
    triangular_response, num_frames,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS,
)
from .prosody import (
    ProsodyExtractor, estimate_f0_yin, F0Result, rms_energy, detect_pauses,
    frame_signal, difference_function, cumulative_mean_normalized,
)
from .streaming import (
    StreamingFeatureExtractor, LatencyModel, LatencyMeasurement,
    measure_emission_latency, percentile,
)
from .projection import (
    TemporalResampler, GatedProjection, SpeechPathways, SpeechProjector,
)
from .lora import (
    LoRALinear, inject_lora, merge_all_lora, iter_lora_modules,
    mark_only_lora_trainable, unfreeze_upper_blocks, freeze_all,
    trainable_parameter_summary,
)
from .decoding import (
    ctc_prefix_beam_search, ctc_exact_posteriors, ctc_greedy_path, collapse,
    Hypothesis, NBestList, Lattice,
)
from .alignment import (
    ctc_forced_alignment, token_timings, align_and_time, Alignment,
    TokenTiming, FrameTimeMapper, extended_targets, minimum_frames_required,
)
from .revision import (
    StreamingDecoder, StreamingHypothesis, RevisionStats,
    longest_common_prefix, commitment_error_count,
)
from .calibration import (
    brier_score, binary_brier_score, brier_decomposition, BrierDecomposition,
    reliability_diagram, CalibrationBin, expected_calibration_error,
    maximum_calibration_error, negative_log_likelihood,
    TemperatureScaler, BrierLoss,
)
from .policy import (
    Action, PolicyDecision, PolicyOutcome, FailClosedPolicy,
    SelectivePoint, selective_metrics, risk_coverage_curve,
    area_under_risk_coverage,
)
from .objective import (
    BoundaryHead, boundary_loss, balanced_pos_weight,
    boundary_targets_from_alignment, ObjectiveWeights, ObjectiveOutput,
    SpeechTrainingObjective, speech_sign_retrieval,
)
from .schedule import FreezeFirstSchedule, FreezeFirstConfig, Phase
from .evaluation import (
    EditOps, edit_ops, word_error_rate as corpus_word_error_rate,
    character_error_rate, TimestampError, timestamp_error,
    Condition, STANDARD_CONDITIONS, ConditionProfile, characterise_condition,
    ArmResult, EvaluationReport,
)

__all__ = [
    # features
    "LogMelSpectrogram", "mel_filterbank", "hz_to_mel", "mel_to_hz",
    "stft_power", "triangular_response", "num_frames",
    "SAMPLE_RATE", "N_FFT", "HOP_LENGTH", "N_MELS",
    # prosody
    "ProsodyExtractor", "estimate_f0_yin", "F0Result", "rms_energy",
    "detect_pauses", "frame_signal", "difference_function",
    "cumulative_mean_normalized",
    # streaming
    "StreamingFeatureExtractor", "LatencyModel", "LatencyMeasurement",
    "measure_emission_latency", "percentile",
    # projection
    "TemporalResampler", "GatedProjection", "SpeechPathways", "SpeechProjector",
    # adaptation
    "LoRALinear", "inject_lora", "merge_all_lora", "iter_lora_modules",
    "mark_only_lora_trainable", "unfreeze_upper_blocks", "freeze_all",
    "trainable_parameter_summary",
    # decoding / lattice
    "ctc_prefix_beam_search", "ctc_exact_posteriors", "ctc_greedy_path",
    "collapse", "Hypothesis", "NBestList", "Lattice",
    # alignment / timestamps
    "ctc_forced_alignment", "token_timings", "align_and_time", "Alignment",
    "TokenTiming", "FrameTimeMapper", "extended_targets",
    "minimum_frames_required",
    # streaming revision
    "StreamingDecoder", "StreamingHypothesis", "RevisionStats",
    "longest_common_prefix", "commitment_error_count",
    # calibration
    "brier_score", "binary_brier_score", "brier_decomposition",
    "BrierDecomposition", "reliability_diagram", "CalibrationBin",
    "expected_calibration_error", "maximum_calibration_error",
    "negative_log_likelihood", "TemperatureScaler", "BrierLoss",
    # fail-closed policy
    "Action", "PolicyDecision", "PolicyOutcome", "FailClosedPolicy",
    "SelectivePoint", "selective_metrics", "risk_coverage_curve",
    "area_under_risk_coverage",
    # training objective
    "BoundaryHead", "boundary_loss", "balanced_pos_weight",
    "boundary_targets_from_alignment", "ObjectiveWeights", "ObjectiveOutput",
    "SpeechTrainingObjective", "speech_sign_retrieval",
    # adaptation schedule
    "FreezeFirstSchedule", "FreezeFirstConfig", "Phase",
    # evaluation harness
    "EditOps", "edit_ops", "corpus_word_error_rate", "character_error_rate",
    "TimestampError", "timestamp_error", "Condition", "STANDARD_CONDITIONS",
    "ConditionProfile", "characterise_condition", "ArmResult",
    "EvaluationReport",
]
