"""Self-supervised and weakly supervised pretraining layer (Document 11).

Implements `11_self_supervised_pretraining.md` (see docs/PRETRAINING.md): masking
strategies with an interpolation-defeating certificate, MAE-style masked motion
modeling, cross-modal contrast with linguistically-grounded hard negatives and a
shortcut falsification, temporal/part consistency with a handedness-preserving
augmentation guard, an evidence battery (probes, scaling, cross-signer retrieval,
leakage, loss-vs-usefulness dissociation), and the five-stage curriculum. Reuses
Doc-04 (rotations/LinearProbe), Doc-03 (minimal-pair oracle), Doc-06 (VQ tokens),
`models/alignment.py` (InfoNCE), and Doc-10 (leakage-free split).
"""

from .masking import (
    random_point_mask, span_mask, part_mask, semantic_boundary_mask, mask_ratio,
    mask_interpolation_error_floor, worst_masked_floor, typical_masked_floor,
    is_certified_hard, linear_interpolate_reconstruction,
)
from .masked_modeling import (
    masked_token_nll, masked_rotation_geodesic, codebook_diversity_loss,
    copy_through_logits, MaskedMotionModel,
)
from .contrast import (
    info_nce_loss, l2_normalize, recall_at_k, retrieval_recall,
    info_nce_against_negatives,
)
from .hard_negatives import (
    HARD_NEGATIVE_DIMENSIONS, hard_negative, is_minimal_linguistic_contrast,
    is_licensed_contrast, contrast_changed_fields,
    signer_shortcut_embedding, length_shortcut_embedding, content_embedding,
)
from .consistency import (
    recover_order_from_timestamps, pairwise_precedence_accuracy, align_views,
    view_retrieval_recall1, AugmentationError, LinguisticDirection,
    augment_appearance, horizontal_flip,
)
from .evaluation import (
    chance_accuracy, linear_probe_accuracy, low_resource_scaling_curve,
    cross_signer_retrieval_recall, signer_leakage_accuracy, is_leaky,
    loss_usefulness_dissociation,
)
from .curriculum import (
    Stage, CurriculumStage, CURRICULUM, is_monotone_unlock, stage_objective,
    FrozenBaseline,
)

__all__ = [
    "random_point_mask", "span_mask", "part_mask", "semantic_boundary_mask",
    "mask_ratio", "mask_interpolation_error_floor", "worst_masked_floor",
    "typical_masked_floor", "is_certified_hard", "linear_interpolate_reconstruction",
    "masked_token_nll", "masked_rotation_geodesic", "codebook_diversity_loss",
    "copy_through_logits", "MaskedMotionModel",
    "info_nce_loss", "l2_normalize", "recall_at_k", "retrieval_recall",
    "info_nce_against_negatives",
    "HARD_NEGATIVE_DIMENSIONS", "hard_negative", "is_minimal_linguistic_contrast",
    "is_licensed_contrast", "contrast_changed_fields", "signer_shortcut_embedding",
    "length_shortcut_embedding", "content_embedding",
    "recover_order_from_timestamps", "pairwise_precedence_accuracy", "align_views",
    "view_retrieval_recall1", "AugmentationError", "LinguisticDirection",
    "augment_appearance", "horizontal_flip",
    "chance_accuracy", "linear_probe_accuracy", "low_resource_scaling_curve",
    "cross_signer_retrieval_recall", "signer_leakage_accuracy", "is_leaky",
    "loss_usefulness_dissociation",
    "Stage", "CurriculumStage", "CURRICULUM", "is_monotone_unlock",
    "stage_objective", "FrozenBaseline",
]
