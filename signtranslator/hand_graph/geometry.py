"""Wrist-relative invariant geometry and the contact soft-distance field.

See docs/HAND_GRAPH.md §3-4. Representing hand landmarks relative to the wrist
makes the model invariant to camera/global pose:

    x̃_i = x_i − x_wrist               (translation-invariant)
    x̂_i = R_wristᵀ (x_i − x_wrist)     (translation- AND rotation-invariant)

The wrist frame ``R_wrist`` is built from two hand-spanning vectors by the SAME
Gram-Schmidt map used for 6D rotations in the pose layer, so it is a proper SO(3)
frame that rotates WITH the hand (``R_wrist ↦ R R_wrist`` under a global rotation
``R``), which is exactly what makes ``x̂`` rotation-invariant.

Contact is a symmetric soft-distance field (§4):

    p^contact_ij = σ( w_hᵀ(h_i + h_j) + w_d · d_ij + w_s · s_ij + b ),
    d_ij = ‖x_i − x_j‖,  s_ij = ‖ẋ_i − ẋ_j‖,  w_d = −softplus(θ_d) ≤ 0,

so p ∈ (0,1), is symmetric in (i,j), and is monotonically non-increasing in
distance (∂p/∂d = p(1−p) w_d ≤ 0).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pose.rotations import rotation_6d_to_matrix
from .hetero_graph import WRIST, MIDDLE_MCP


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------
def estimate_velocity(x_seq: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """(T, V, 3) positions -> (T, V, 3) velocities by finite difference.

    Backward difference with the first frame held (v_0 = v_1) so the output has
    the same length and no fabricated boundary motion.
    """
    if x_seq.shape[0] < 2:
        return torch.zeros_like(x_seq)
    v = torch.zeros_like(x_seq)
    v[1:] = (x_seq[1:] - x_seq[:-1]) / dt
    v[0] = v[1]
    return v


# ---------------------------------------------------------------------------
# wrist-relative coordinates
# ---------------------------------------------------------------------------
def wrist_relative(x: torch.Tensor, wrist_of: torch.Tensor) -> torch.Tensor:
    """x̃_i = x_i − x_{wrist_of[i]}. ``x`` (..., V, 3), ``wrist_of`` (V,) long."""
    wrist_pos = x.index_select(-2, wrist_of)                 # (..., V, 3)
    return x - wrist_pos


def wrist_frame_from_landmarks(hand_xyz: torch.Tensor) -> torch.Tensor:
    """(..., 21, 3) hand landmarks -> (..., 3, 3) orthonormal wrist frame.

    Spanning vectors: a = middle_MCP − wrist (finger axis), b = index_MCP −
    pinky_MCP (palm axis). The frame is Gram-Schmidt(a, b), i.e. exactly the
    6D->SO(3) map, so it is a valid rotation that is equivariant to global
    rotation of the hand.
    """
    a = hand_xyz[..., MIDDLE_MCP, :] - hand_xyz[..., WRIST, :]
    b = hand_xyz[..., 5, :] - hand_xyz[..., 17, :]           # index_MCP - pinky_MCP
    six = torch.cat((a, b), dim=-1)                          # (..., 6)
    return rotation_6d_to_matrix(six)                        # (..., 3, 3)


def wrist_frame_relative(x: torch.Tensor, wrist_of: torch.Tensor,
                         frame_of: torch.Tensor) -> torch.Tensor:
    """x̂_i = R_iᵀ (x_i − x_{wrist_of[i]}).

    ``x`` (V, 3), ``wrist_of`` (V,), ``frame_of`` (V, 3, 3) the wrist frame for
    each node. Rotation- and translation-invariant.
    """
    rel = x - x.index_select(0, wrist_of)                    # (V, 3)
    # R^T rel  ==  einsum over the frame's transpose
    return torch.einsum("vab,va->vb", frame_of, rel)         # (frame_of[...,:,k])·rel


# ---------------------------------------------------------------------------
# contact soft-distance field
# ---------------------------------------------------------------------------
class ContactField(nn.Module):
    """Symmetric, distance-monotone contact probability between node pairs."""

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.w_h = nn.Linear(feat_dim, 1, bias=False)        # applied to (h_i + h_j)
        self.theta_d = nn.Parameter(torch.zeros(()))         # distance weight = -softplus
        self.w_s = nn.Parameter(torch.zeros(()))             # relative-speed weight
        self.bias = nn.Parameter(torch.zeros(()))

    def logit(self, h_i, h_j, d, s):
        w_d = -F.softplus(self.theta_d)                      # <= 0 -> monotone in d
        return self.w_h(h_i + h_j).squeeze(-1) + w_d * d + self.w_s * s + self.bias

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor,
                x_i: torch.Tensor, x_j: torch.Tensor,
                v_i: Optional[torch.Tensor] = None,
                v_j: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return p^contact (…,) for aligned node pairs.

        ``h_*`` (…, feat), ``x_*`` (…, 3), optional ``v_*`` (…, 3).
        """
        d = torch.linalg.norm(x_i - x_j, dim=-1)
        if v_i is not None and v_j is not None:
            s = torch.linalg.norm(v_i - v_j, dim=-1)
        else:
            s = torch.zeros_like(d)
        return torch.sigmoid(self.logit(h_i, h_j, d, s))


def hard_contact(x_i: torch.Tensor, x_j: torch.Tensor, rho: float) -> torch.Tensor:
    """Ground-truth contact label 1[‖x_i − x_j‖ < rho] (float)."""
    return (torch.linalg.norm(x_i - x_j, dim=-1) < rho).to(x_i.dtype)
