"""Diffusion motion generation layer (Document 07).

Implements `07_diffusion_motion_generation.md` (see docs/DIFFUSION_GEN.md): a
conditional temporal-DiT diffusion model with ε/x₀/v parameterizations,
classifier-free guidance, part-aware schedules, inpainting, kinematic constraints,
and consistency/rectified-flow distillation. Builds on the audited DDPM core in
``models/diffusion.py``.
"""

from .schedule import NoiseSchedule
from .dit import DiTBlock, TemporalDiT, modulate
from .guidance import (
    drop_condition_mask, apply_condition_dropout, classifier_free_guidance,
    guidance_weight_schedule,
)
from .part_aware import part_loss_weights, weighted_mse, PartAwareSchedule
from .inpaint import (
    forward_diffuse_known, merge_known_unknown, inpaint_step, streaming_overlap_mask,
)
from .constraints import (
    joint_limit_penalty, project_joint_limits, collision_penalty, contact_penalty,
    temporal_boundary_penalty, project_feasible,
)
from .consistency import (
    consistency_coeffs, ConsistencyModel, self_consistency_loss,
    rectified_flow_interpolant, rectified_flow_velocity,
    rectified_flow_x0_from_velocity, rectified_flow_sample,
)
from .evaluation import (
    multimodality, semantic_preservation_verified_multimodality, jerk,
    p95_generation_time, semantic_accuracy_diagnostic, compare_generators,
)
from .generator import DiffusionMotionGenerator

__all__ = [
    "NoiseSchedule", "DiTBlock", "TemporalDiT", "modulate",
    "drop_condition_mask", "apply_condition_dropout", "classifier_free_guidance",
    "guidance_weight_schedule",
    "part_loss_weights", "weighted_mse", "PartAwareSchedule",
    "forward_diffuse_known", "merge_known_unknown", "inpaint_step",
    "streaming_overlap_mask",
    "joint_limit_penalty", "project_joint_limits", "collision_penalty",
    "contact_penalty", "temporal_boundary_penalty", "project_feasible",
    "consistency_coeffs", "ConsistencyModel", "self_consistency_loss",
    "rectified_flow_interpolant", "rectified_flow_velocity",
    "rectified_flow_x0_from_velocity", "rectified_flow_sample",
    "multimodality", "semantic_preservation_verified_multimodality", "jerk",
    "p95_generation_time", "semantic_accuracy_diagnostic", "compare_generators",
    "DiffusionMotionGenerator",
]
