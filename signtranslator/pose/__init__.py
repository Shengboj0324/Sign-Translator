"""3D human representation layer (SMPL-X style).

Implements `04_3d_human_representation_smplx.md` (see docs/HUMAN_REPRESENTATION.md).
Rotations are carried in the continuous 6D representation (Zhou et al.,
arXiv:1812.07035) and converted to SO(3); the body is a differentiable mesh
`M(q, β)` built from blend shapes, a joint regressor, pose correctives, and
linear blend skinning (SMPL-X, arXiv:1904.05866).
"""

from .rotations import (
    rotation_6d_to_matrix, matrix_to_rotation_6d,
    axis_angle_to_matrix, matrix_to_axis_angle,
    quaternion_to_matrix, matrix_to_quaternion,
    rotation_6d_to_axis_angle, axis_angle_to_rotation_6d,
    geodesic_distance, is_rotation_matrix,
)
from .state import SMPLXLayout, MotionSequence, ROT_DIM
from .body_model import (
    BodyModelTensors, BodyOutput, SMPLXBodyModel, forward_kinematics,
    make_toy_model, rest_pose_sequence,
)
from .camera import (
    PerspectiveCamera, WeakPerspectiveCamera,
    geman_mcclure, geman_mcclure_influence, reprojection_loss,
)
from .fitting import (
    GaussianPosePrior, GMMPosePrior, temporal_smoothness,
    self_collision_penalty, fitting_terms, FittingWeights, FittingTerms,
)
from .metrics import (
    mpjpe, kabsch, pa_mpjpe, mean_geodesic_rotation_error, v2v,
    fingertip_weighted_mpjpe,
)
from .leakage import world_joint_rotations, LinearProbe, normalised_recovery_error
from .integration import (
    build_joint_map, smplx_joints_to_skeleton, motion_to_skeleton, to_stgcn_layout,
)

__all__ = [
    "rotation_6d_to_matrix", "matrix_to_rotation_6d",
    "axis_angle_to_matrix", "matrix_to_axis_angle",
    "quaternion_to_matrix", "matrix_to_quaternion",
    "rotation_6d_to_axis_angle", "axis_angle_to_rotation_6d",
    "geodesic_distance", "is_rotation_matrix",
    "SMPLXLayout", "MotionSequence", "ROT_DIM",
    "BodyModelTensors", "BodyOutput", "SMPLXBodyModel", "forward_kinematics",
    "make_toy_model", "rest_pose_sequence",
    "PerspectiveCamera", "WeakPerspectiveCamera",
    "geman_mcclure", "geman_mcclure_influence", "reprojection_loss",
    "GaussianPosePrior", "GMMPosePrior", "temporal_smoothness",
    "self_collision_penalty", "fitting_terms", "FittingWeights", "FittingTerms",
    "mpjpe", "kabsch", "pa_mpjpe", "mean_geodesic_rotation_error", "v2v",
    "fingertip_weighted_mpjpe",
    "world_joint_rotations", "LinearProbe", "normalised_recovery_error",
    "build_joint_map", "smplx_joints_to_skeleton", "motion_to_skeleton",
    "to_stgcn_layout",
]
