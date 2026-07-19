"""Bridge: SMPL-X body-model joints -> the existing 27-joint sign skeleton.

The recognition/generation pipeline (docs 00-03) operates on the 27-joint
upper-body + two-hand skeleton in ``skeleton/graph.py``, and ``data/adapters.py``
already names *"an SMPL-X joint regressor"* as a valid source. This module is that
regressor: a linear, row-stochastic map ``(V_skel, J_smplx)`` taking SMPL-X joints
to skeleton joints, plus a helper to lay the result out as the ``(N, C, T, V)``
tensor the ST-GCN motion encoder consumes.

Being a fixed linear map, the bridge inherits the body model's rigid
equivariance: a global rotation/translation of the motion rigidly transforms the
skeleton joints too -- verified, not assumed.
"""

from __future__ import annotations

import torch

from ..skeleton.graph import NUM_DEFAULT_JOINTS
from .body_model import SMPLXBodyModel
from .state import MotionSequence


def build_joint_map(n_smplx_joints: int, num_skeleton_joints: int = NUM_DEFAULT_JOINTS,
                    seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A row-stochastic (V_skel, J_smplx) regressor.

    Each skeleton joint is a convex combination of SMPL-X joints (rows sum to 1),
    so skeleton joints lie in the convex hull of the SMPL-X joints -- the toy
    stand-in for the fixed anatomical correspondence of the real model.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_skeleton_joints, n_smplx_joints, generator=g, dtype=dtype)
    return torch.softmax(logits, dim=-1)


def smplx_joints_to_skeleton(joints: torch.Tensor, joint_map: torch.Tensor
                             ) -> torch.Tensor:
    """(T, J, 3) SMPL-X joints -> (T, V_skel, 3) skeleton joints via the regressor."""
    if joints.shape[-2] != joint_map.shape[1]:
        raise ValueError(
            f"joint count {joints.shape[-2]} != map cols {joint_map.shape[1]}")
    return torch.einsum("kj,tjc->tkc", joint_map, joints)


def motion_to_skeleton(model: SMPLXBodyModel, seq: MotionSequence,
                       joint_map: torch.Tensor) -> torch.Tensor:
    """MotionSequence -> (T, V_skel, 3) skeleton joints through the body model."""
    out = model(seq)
    return smplx_joints_to_skeleton(out.joints, joint_map)


def to_stgcn_layout(skeleton_joints: torch.Tensor) -> torch.Tensor:
    """(T, V, 3) -> (1, 3, T, V) as expected by the ST-GCN motion encoder."""
    T, V, C = skeleton_joints.shape
    return skeleton_joints.permute(2, 0, 1).reshape(1, C, T, V)
