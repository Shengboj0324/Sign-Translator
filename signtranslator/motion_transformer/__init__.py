"""Temporal motion transformer layer (Document 06).

Implements `06_temporal_motion_transformers.md` (see docs/MOTION_TRANSFORMER.md):
a VQ / residual-VQ motion tokenizer, a hierarchical temporal backbone, the motion
loss (geodesic + velocity + acceleration + contact + semantic + commitment),
anti-oversmoothing diagnostics, and streaming with rotation-space chunk blending.
"""

from .quantizer import VectorQuantizer
from .residual_vq import kmeans, ResidualVQ, PartitionedVQ
from .autoencoder import (
    velocity, acceleration, velocity_l1, acceleration_l1, geodesic_motion_loss,
    motion_loss, MotionLossWeights, TemporalEncoder, TemporalDecoder, MotionVQVAE,
)
from .spectral import (
    power_spectrum, parseval_energy, band_energy, spectral_energy_by_part,
    spectral_energy_matching_loss, duration_calibration_error,
)
from .backbone import (
    causal_mask, ClausePlanner, DurationModel, MotionDecoder, RecurrentMemory,
)
from .streaming import (
    bounded_right_context_mask, streaming_latency_frames, slerp,
    crossfade_rotations, overlap_add_rotations,
)
from .decoding import (
    MotionTokenGPT, MaskedMotionModel, compare_reconstruction, compare_shared_vs_part,
)
from .chain import MotionGenerationChain

__all__ = [
    "VectorQuantizer", "kmeans", "ResidualVQ", "PartitionedVQ",
    "velocity", "acceleration", "velocity_l1", "acceleration_l1",
    "geodesic_motion_loss", "motion_loss", "MotionLossWeights",
    "TemporalEncoder", "TemporalDecoder", "MotionVQVAE",
    "power_spectrum", "parseval_energy", "band_energy", "spectral_energy_by_part",
    "spectral_energy_matching_loss", "duration_calibration_error",
    "causal_mask", "ClausePlanner", "DurationModel", "MotionDecoder",
    "RecurrentMemory",
    "bounded_right_context_mask", "streaming_latency_frames", "slerp",
    "crossfade_rotations", "overlap_add_rotations",
    "MotionTokenGPT", "MaskedMotionModel", "compare_reconstruction",
    "compare_shared_vs_part", "MotionGenerationChain",
]
