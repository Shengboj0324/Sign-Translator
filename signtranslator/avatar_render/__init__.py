"""Avatar rendering pipeline layer (Document 08).

Implements `08_avatar_rendering_pipeline.md` (see docs/AVATAR_RENDER.md): a
parameter-stream interface with exact contracts, three rendering tracks (rigged
mesh with LBS / dual-quaternion skinning, 3D Gaussian Splatting, NeRF volume
rendering), SO(3) frame pacing, a linguistically-aware LOD, and a strict
separation of rendering from linguistic evaluation.
"""

from .stream import (
    Handedness, AvatarContract, contract_basis, contract_is_self_consistent,
    ParameterStream, validate_stream, replay,
)
from .rigged import (
    apply_lbs, RetargetMap, build_retarget, retarget_residual, prioritized_joints,
    correct_joint_limits, apply_blendshapes,
)
from .dqs import (
    quat_mul, quat_conj, dq_from_transform, transform_from_dq, apply_dq_to_point,
    dq_normalize, dlb,
)
from .gaussian import (
    covariance_3d, projection_jacobian, covariance_2d, gaussian_2d_value,
    alpha_composite, render_pixel,
)
from .nerf import (
    deltas_from_samples, alphas_from_density, transmittance, volume_render,
    expected_depth,
)
from .pacing import (
    target_timeline, resample_rotations, resample_linear, pace,
)
from .lod import (
    ImportanceTier, PROTECTED_TIERS, lod_keep_mask, fingers_face_always_kept,
    budget_curve, AppearanceConsent, can_render_identity, requires_synthetic_marker,
)
from .evaluation import (
    motion_to_photon_p95, dropped_frame_rate, temporal_flicker, penetration_rate,
    silhouette_iou_error, psnr, ssim_global, AppearanceReport, SigningReport,
    signing_quality_from_appearance, combined_report,
)

__all__ = [
    "Handedness", "AvatarContract", "contract_basis", "contract_is_self_consistent",
    "ParameterStream", "validate_stream", "replay",
    "apply_lbs", "RetargetMap", "build_retarget", "retarget_residual",
    "prioritized_joints", "correct_joint_limits", "apply_blendshapes",
    "quat_mul", "quat_conj", "dq_from_transform", "transform_from_dq",
    "apply_dq_to_point", "dq_normalize", "dlb",
    "covariance_3d", "projection_jacobian", "covariance_2d", "gaussian_2d_value",
    "alpha_composite", "render_pixel",
    "deltas_from_samples", "alphas_from_density", "transmittance", "volume_render",
    "expected_depth",
    "target_timeline", "resample_rotations", "resample_linear", "pace",
    "ImportanceTier", "PROTECTED_TIERS", "lod_keep_mask", "fingers_face_always_kept",
    "budget_curve", "AppearanceConsent", "can_render_identity",
    "requires_synthetic_marker",
    "motion_to_photon_p95", "dropped_frame_rate", "temporal_flicker",
    "penetration_rate", "silhouette_iou_error", "psnr", "ssim_global",
    "AppearanceReport", "SigningReport", "signing_quality_from_appearance",
    "combined_report",
]
