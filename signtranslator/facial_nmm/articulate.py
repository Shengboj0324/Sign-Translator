"""Marker -> FLAME/SMPL-X articulation (docs/FACIAL_NMM.md §7).

A scoped non-manual marker is articulated into face parameters:

    ψ   = A · (marker_one_hot · value)        (FLAME expression coefficients)
    jaw = R(head_marker · value)              (jaw rotation, Doc-04 6D->SO(3))
    eye = R(gaze)                             (eye rotation)

and rig blendshapes via the Doc-08 linear ``apply_blendshapes``. **Intensity
monotonicity:** the articulation is monotone in the marker ``value`` (a stronger
brow raise yields a larger expression coefficient), so grammatical intensity is
preserved -- never collapsed to a cosmetic constant.

HONEST SCOPE: the FLAME model tensors are licensed; ``A`` is a (here random,
later learned/anatomical) marker->expression matrix and the mesh is produced by the
Doc-04 SMPL-X pipeline. The mapping's algebraic properties are proved regardless.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..pose.rotations import rotation_6d_to_matrix, is_rotation_matrix
from ..avatar_render.rigged import apply_blendshapes
from .channels import Marker


class MarkerArticulator(nn.Module):
    """Maps marker activations (intensity per marker) to FLAME expression coeffs."""

    def __init__(self, num_markers: int, num_expr: int) -> None:
        super().__init__()
        self.A = nn.Linear(num_markers, num_expr, bias=False)   # ψ = A m

    def expression(self, marker_intensity: torch.Tensor) -> torch.Tensor:
        """(N, num_markers) intensities (marker_one_hot · value) -> (N, num_expr) ψ."""
        return self.A(marker_intensity)


def jaw_rotation(marker_value: torch.Tensor, six_d: torch.Tensor) -> torch.Tensor:
    """A jaw rotation whose ANGLE scales with the marker value.

    ``six_d`` (…, 6) is a base 6D rotation direction; the value scales the axis-angle
    so a stronger marker opens the jaw more. Returns (…, 3, 3) in SO(3).
    """
    R = rotation_6d_to_matrix(six_d)
    from ..pose.rotations import matrix_to_axis_angle, axis_angle_to_matrix
    aa = matrix_to_axis_angle(R)
    scaled = marker_value.unsqueeze(-1) * aa
    return axis_angle_to_matrix(scaled)


def eye_rotation(gaze_6d: torch.Tensor) -> torch.Tensor:
    """Eye rotation from a 6D gaze direction (valid SO(3))."""
    return rotation_6d_to_matrix(gaze_6d)


def articulate_blendshapes(mean_face: torch.Tensor, basis: torch.Tensor,
                           expr: torch.Tensor) -> torch.Tensor:
    """Rig blendshapes for the articulated expression (Doc-08 linear blendshapes)."""
    return apply_blendshapes(mean_face, basis, expr)


def marker_one_hot(marker: Marker, num_markers: int, value: float,
                   dtype=torch.get_default_dtype()) -> torch.Tensor:
    """(num_markers,) one-hot of ``marker`` scaled by ``value`` (the intensity)."""
    v = torch.zeros(num_markers, dtype=dtype)
    v[int(marker)] = value
    return v
