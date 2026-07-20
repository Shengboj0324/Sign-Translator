"""Facial and non-manual modeling layer (Document 09).

Implements `09_facial_expression_modeling.md` (see docs/FACIAL_NMM.md): concurrent
non-manual grammatical channels as scoped intervals, a conditioned multi-label
interval decoder, a loss suite (BCE / boundary / scope / smoothness /
disentanglement / focal / uncertainty), and articulation to FLAME / SMPL-X
expression coefficients. Builds on Doc-03 (non-manual + SIR + SignBLEU), Doc-04
(jaw / eye / expression), and Doc-08 (blendshapes).
"""

from .channels import (
    Channel, Marker, MARKER_CHANNELS, GRAMMATICAL_MARKERS, NonmanualEvent,
    scope_relation, is_properly_nested, nesting_parents,
)
from .decoder import MultiChannelDecoder
from .losses import (
    nmm_bce, boundary_targets, boundary_loss, scope_loss, temporal_smoothness,
    total_nmm_loss, NMMWeights,
)
from .uncertainty import (
    focal_loss, focal_modulation, class_balanced_weights, heteroscedastic_nll,
    agreement_to_target_logvar,
)
from .disentangle import grad_reverse, AffectAdversary, affect_leakage
from .articulate import (
    MarkerArticulator, jaw_rotation, eye_rotation, articulate_blendshapes,
    marker_one_hot,
)
from .evaluation import (
    minimal_pair_distinguishes, minimal_pair_accuracy, scope_boundary_error,
    gaze_locus_agreement, head_manual_offset, synchronisation_rate,
    channel_ablation_drop,
)
from .integration import (
    build_sir_with_nonmanual, articulate_frames, events_to_frame_intensities,
)

__all__ = [
    "Channel", "Marker", "MARKER_CHANNELS", "GRAMMATICAL_MARKERS", "NonmanualEvent",
    "scope_relation", "is_properly_nested", "nesting_parents",
    "MultiChannelDecoder",
    "nmm_bce", "boundary_targets", "boundary_loss", "scope_loss",
    "temporal_smoothness", "total_nmm_loss", "NMMWeights",
    "focal_loss", "focal_modulation", "class_balanced_weights",
    "heteroscedastic_nll", "agreement_to_target_logvar",
    "grad_reverse", "AffectAdversary", "affect_leakage",
    "MarkerArticulator", "jaw_rotation", "eye_rotation", "articulate_blendshapes",
    "marker_one_hot",
    "minimal_pair_distinguishes", "minimal_pair_accuracy", "scope_boundary_error",
    "gaze_locus_agreement", "head_manual_offset", "synchronisation_rate",
    "channel_ablation_drop",
    "build_sir_with_nonmanual", "articulate_frames", "events_to_frame_intensities",
]
