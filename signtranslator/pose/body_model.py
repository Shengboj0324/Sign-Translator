"""Differentiable SMPL-X forward model: blend shapes + joint regression + pose
correctives + kinematic-tree FK + linear blend skinning.

Implements ``M(q, beta)`` of docs/HUMAN_REPRESENTATION.md §3. The generative
equation (Loper et al. SMPL; Pavlakos et al. SMPL-X):

    T_P(beta, theta, psi) = Tbar + B_S(beta) + B_E(psi) + B_P(theta)
    J(beta)               = J_regressor @ (Tbar + B_S(beta))
    G_j                   = G_{parent(j)} @ [[R_j, J_j - J_{parent}], [0, 1]]
    G'_j                  = G_j @ [[I, -J_j], [0, 1]]           (rest -> identity)
    v_i                   = sum_j w_{ij} (G'_j [T_{P,i}; 1])_{1:3}

Global orientation is the root rotation inside ``G``; root translation ``gamma``
is added last.

HONEST SCOPE: the real SMPL-X model tensors are licensed and not downloaded here
(see docs §0). This module implements the pipeline exactly and is validated on a
**controllable toy model** (``make_toy_model``) whose tensors we set, so rest-pose
identity, blendshape linearity, rigid equivariance, partition-of-unity skinning,
and differentiability are proved exactly. The realistic mesh drops in later as
data with no code change.

Pose correctives deliberately **exclude the root joint** (features from joints
1..J-1), so a change of *global orientation* creates no pose-dependent deformation
-- which is exactly what makes the global-rigid-equivariance property hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .rotations import rotation_6d_to_matrix
from .state import MotionSequence, SMPLXLayout


@dataclass
class BodyModelTensors:
    """The (licensed, here toy) model parameters. Shapes documented inline."""

    template: torch.Tensor        # (Nv, 3)     Tbar
    shape_dirs: torch.Tensor      # (Nv, 3, n_shape)   B_S basis
    expr_dirs: torch.Tensor       # (Nv, 3, n_expr)    B_E basis
    pose_dirs: torch.Tensor       # (Nv, 3, 9*(J-1))   B_P basis (excludes root)
    joint_regressor: torch.Tensor # (J, Nv)     J_regressor (rows sum to 1)
    weights: torch.Tensor         # (Nv, J)     LBS weights (rows sum to 1, >= 0)
    parents: torch.Tensor         # (J,) long   parents[0] = -1, parents[j] < j
    layout: SMPLXLayout

    def __post_init__(self) -> None:
        Nv = self.template.shape[0]
        J = self.layout.n_joints
        assert self.template.shape == (Nv, 3)
        assert self.shape_dirs.shape == (Nv, 3, self.layout.n_shape)
        assert self.expr_dirs.shape == (Nv, 3, self.layout.n_expr)
        assert self.pose_dirs.shape == (Nv, 3, 9 * (J - 1))
        assert self.joint_regressor.shape == (J, Nv)
        assert self.weights.shape == (Nv, J)
        assert self.parents.shape == (J,)
        assert int(self.parents[0]) == -1
        assert torch.all(self.parents[1:] < torch.arange(1, J))

    @property
    def num_vertices(self) -> int:
        return self.template.shape[0]


@dataclass
class BodyOutput:
    vertices: torch.Tensor    # (T, Nv, 3)
    joints: torch.Tensor      # (T, J, 3)


def _rigid(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Assemble (..., 4, 4) homogeneous transform from (..., 3, 3) and (..., 3)."""
    shape = R.shape[:-2]
    M = torch.zeros(shape + (4, 4), dtype=R.dtype, device=R.device)
    M[..., :3, :3] = R
    M[..., :3, 3] = t
    M[..., 3, 3] = 1.0
    return M


def forward_kinematics(rot_mats: torch.Tensor, J_rest: torch.Tensor,
                       parents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose joint transforms along the tree.

    ``rot_mats`` (T, J, 3, 3), ``J_rest`` (J, 3), ``parents`` (J,).
    Returns (posed_joints (T, J, 3), G_rel (T, J, 4, 4)) where ``G_rel`` is the
    rest-removed transform used for skinning.
    """
    T, J = rot_mats.shape[0], rot_mats.shape[1]
    dtype, device = rot_mats.dtype, rot_mats.device
    J_rest_b = J_rest.expand(T, J, 3)

    # local translations: root at J_0, others at offset from parent
    rel_t = J_rest_b.clone()
    rel_t[:, 1:, :] = J_rest_b[:, 1:, :] - J_rest_b[:, parents[1:], :]
    locals_ = _rigid(rot_mats, rel_t)                         # (T, J, 4, 4)

    G = [locals_[:, 0]]
    for j in range(1, J):
        G.append(G[parents[j]] @ locals_[:, j])
    G = torch.stack(G, dim=1)                                 # (T, J, 4, 4)

    posed_joints = G[..., :3, 3]                              # translation parts

    # rest removal: G'_j = G_j @ [[I, -J_j],[0,1]]
    J_homo = torch.cat((J_rest, torch.zeros(J, 1, dtype=dtype, device=device)),
                       dim=-1)                                # (J, 4) with 0 last
    # build [[I, -J_j],[0,1]]
    rest_inv = torch.eye(4, dtype=dtype, device=device).expand(J, 4, 4).clone()
    rest_inv[:, :3, 3] = -J_rest
    G_rel = G @ rest_inv                                      # (T, J, 4, 4)
    return posed_joints, G_rel


class SMPLXBodyModel:
    """Callable forward model M(q, beta)."""

    def __init__(self, tensors: BodyModelTensors) -> None:
        self.t = tensors
        self.layout = tensors.layout

    def _rot_features_no_root(self, rot_mats: torch.Tensor) -> torch.Tensor:
        """Pose feature (T, 9*(J-1)) = vec(R_j - I) for j = 1..J-1."""
        T, J = rot_mats.shape[0], rot_mats.shape[1]
        eye = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device)
        feat = (rot_mats[:, 1:] - eye).reshape(T, (J - 1) * 9)
        return feat

    def forward(self, seq: MotionSequence) -> BodyOutput:
        t = self.t
        beta, expr, gamma = seq.beta, seq.expr, seq.gamma
        rot_mats = rotation_6d_to_matrix(seq.rot6d)           # (T, J, 3, 3)
        T = seq.num_frames

        # shaped template (shape is constant over t; expression varies per frame)
        shaped = t.template + torch.einsum("vcs,s->vc", t.shape_dirs, beta)   # (Nv,3)
        expr_off = torch.einsum("vce,te->tvc", t.expr_dirs, expr)            # (T,Nv,3)

        # joints regress from the shaped template (NOT expression)
        J_rest = torch.einsum("jv,vc->jc", t.joint_regressor, shaped)        # (J,3)

        # pose correctives (exclude root)
        pose_feat = self._rot_features_no_root(rot_mats)                     # (T,9(J-1))
        pose_off = torch.einsum("vcp,tp->tvc", t.pose_dirs, pose_feat)       # (T,Nv,3)

        v_posed = shaped[None] + expr_off + pose_off                        # (T,Nv,3)

        posed_joints, G_rel = forward_kinematics(rot_mats, J_rest, t.parents)

        # linear blend skinning
        Tmat = torch.einsum("vj,tjab->tvab", t.weights, G_rel)              # (T,Nv,4,4)
        ones = torch.ones(T, t.num_vertices, 1, dtype=v_posed.dtype, device=v_posed.device)
        v_homo = torch.cat((v_posed, ones), dim=-1)                         # (T,Nv,4)
        v_skinned = torch.einsum("tvab,tvb->tva", Tmat, v_homo)[..., :3]    # (T,Nv,3)

        # global translation
        verts = v_skinned + gamma[:, None, :]
        joints = posed_joints + gamma[:, None, :]
        return BodyOutput(vertices=verts, joints=joints)

    __call__ = forward


# ---------------------------------------------------------------------------
# Controllable toy model (no licensed data)
# ---------------------------------------------------------------------------
def make_toy_model(layout: SMPLXLayout | None = None, num_vertices: int = 60,
                   seed: int = 0, dtype: torch.dtype = torch.float64
                   ) -> BodyModelTensors:
    """A small, fully-controllable SMPL-X-shaped model for exact verification.

    Weights and the joint regressor are row-stochastic (partition of unity); the
    kinematic tree is a valid rooted tree with parents[j] < j; bases are small
    random tensors. All properties that do not depend on *learned* bases are then
    provable on this model.
    """
    layout = layout or SMPLXLayout()
    J = layout.n_joints
    g = torch.Generator().manual_seed(seed)
    Nv = num_vertices

    template = torch.randn(Nv, 3, generator=g, dtype=dtype)
    shape_dirs = 0.1 * torch.randn(Nv, 3, layout.n_shape, generator=g, dtype=dtype)
    expr_dirs = 0.1 * torch.randn(Nv, 3, layout.n_expr, generator=g, dtype=dtype)
    pose_dirs = 0.05 * torch.randn(Nv, 3, 9 * (J - 1), generator=g, dtype=dtype)

    # row-stochastic joint regressor (each joint = convex combo of vertices)
    jr_logits = torch.randn(J, Nv, generator=g, dtype=dtype)
    joint_regressor = torch.softmax(jr_logits, dim=-1)

    # row-stochastic skinning weights (partition of unity)
    w_logits = torch.randn(Nv, J, generator=g, dtype=dtype)
    weights = torch.softmax(w_logits, dim=-1)

    # valid rooted tree: parents[0] = -1, parents[j] in [0, j-1]
    parents = torch.zeros(J, dtype=torch.long)
    parents[0] = -1
    for j in range(1, J):
        parents[j] = int(torch.randint(0, j, (1,), generator=g))

    return BodyModelTensors(
        template=template, shape_dirs=shape_dirs, expr_dirs=expr_dirs,
        pose_dirs=pose_dirs, joint_regressor=joint_regressor, weights=weights,
        parents=parents, layout=layout,
    )


def rest_pose_sequence(layout: SMPLXLayout, T: int = 1,
                       dtype: torch.dtype = torch.float64) -> MotionSequence:
    """A zero-motion sequence: identity rotations (6D), zero gamma/expr/beta."""
    J = layout.n_joints
    # identity rotation in 6D = first two columns of I = [1,0,0, 0,1,0]
    ident6d = torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=dtype)
    rot6d = ident6d.expand(T, J, 6).clone()
    return MotionSequence(
        gamma=torch.zeros(T, 3, dtype=dtype),
        rot6d=rot6d,
        expr=torch.zeros(T, layout.n_expr, dtype=dtype),
        beta=torch.zeros(layout.n_shape, dtype=dtype),
        layout=layout,
    )
