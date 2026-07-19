"""Rotation representations and conversions for SMPL-X pose.

We regress rotations in the **continuous 6D** representation of Zhou et al.
(*On the Continuity of Rotation Representations in Neural Networks*,
arXiv:1812.07035) and convert to SO(3). SMPL-X's native encoding is axis-angle,
so we also provide Rodrigues (axis-angle <-> matrix) and quaternion conversions.

Design points (see docs/HUMAN_REPRESENTATION.md §1):

* All functions are batched over leading dims and differentiable.
* 6D -> matrix is Gram-Schmidt; matrix -> 6D drops the third column. The pair is a
  continuous section of SO(3): matrix -> 6D -> matrix is exact.
* Axis-angle uses Taylor expansions near phi = 0 so R(0) = I with a finite,
  correct gradient (the naive sin(phi)/phi is 0/0 at 0).
* Geodesic distance d(R1,R2) = arccos((tr(R1^T R2) - 1)/2) is a bi-invariant
  metric, used for rotation-error evaluation and as a proper SO(3) loss.

Correctness (proved in tests, not asserted): produced matrices are special
orthogonal (R^T R = I, det = +1); every conversion round-trips; the 6D encoding
is continuous where Euler/quaternion jump; gradients flow.
"""

from __future__ import annotations

import torch

# Below this rotation angle (radians) we use Taylor expansions to avoid 0/0.
_ANGLE_EPS = 1e-6
# Guards a division by a vector norm.
_NORM_EPS = 1e-8


# ---------------------------------------------------------------------------
# 6D <-> matrix (Zhou et al.)
# ---------------------------------------------------------------------------
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on the two 3-vectors.

    d6 = [a; b]; r1 = a/|a|, r2 = normalize(b - (r1.b) r1), r3 = r1 x r2.
    The three become the *columns* of R.
    """
    if d6.shape[-1] != 6:
        raise ValueError(f"expected last dim 6, got {d6.shape[-1]}")
    a, b = d6[..., :3], d6[..., 3:]
    r1 = _normalize(a)
    # remove the r1 component of b, then normalise
    b_proj = b - (r1 * b).sum(-1, keepdim=True) * r1
    r2 = _normalize(b_proj)
    r3 = torch.cross(r1, r2, dim=-1)
    # stack as columns: R[..., :, k] = r_{k+1}
    return torch.stack((r1, r2, r3), dim=-1)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): the first two columns, flattened as [col0; col1].

    This is the continuous section: feeding the result back through
    ``rotation_6d_to_matrix`` returns R exactly (the two columns are already
    orthonormal, so Gram-Schmidt is the identity on them).
    """
    _check_matrix(R)
    col0 = R[..., :, 0]
    col1 = R[..., :, 1]
    return torch.cat((col0, col1), dim=-1)


# ---------------------------------------------------------------------------
# axis-angle <-> matrix (Rodrigues / exp-log map on SO(3))
# ---------------------------------------------------------------------------
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle (SMPL-X native) -> (..., 3, 3) rotation.

    R = I + sin(phi) K + (1 - cos phi) K^2, K = [k]_x, phi = |aa|, k = aa/phi.
    Near phi = 0: sin(phi)/phi -> 1 and (1 - cos phi)/phi^2 -> 1/2 (Taylor), so
    R(0) = I with a finite gradient.
    """
    if aa.shape[-1] != 3:
        raise ValueError(f"expected last dim 3, got {aa.shape[-1]}")
    phi = torch.linalg.norm(aa, dim=-1, keepdim=True)          # (..., 1)
    small = phi < _ANGLE_EPS
    # coefficients A = sin(phi)/phi, B = (1 - cos phi)/phi^2, Taylor near 0
    phi_safe = torch.where(small, torch.ones_like(phi), phi)
    A = torch.where(small, 1.0 - phi * phi / 6.0, torch.sin(phi_safe) / phi_safe)
    B = torch.where(small, 0.5 - phi * phi / 24.0,
                    (1.0 - torch.cos(phi_safe)) / (phi_safe * phi_safe))
    K = _skew(aa)                                              # (..., 3, 3)
    KK = K @ K
    A = A[..., None]                                           # (..., 1, 1)
    B = B[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    return eye + A * K + B * KK


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle (the SO(3) log map).

    The angle is computed as ``phi = atan2(||v||/2, (tr R - 1)/2)`` with
    ``v = (R - R^T)`` the skew part (= 2 sin(phi) * axis). ``atan2`` is used
    instead of ``arccos`` because ``arccos`` has an INFINITE derivative at
    cos(phi) = +/-1 (i.e. phi = 0 or pi), which back-propagates NaN gradients;
    ``atan2`` is finite-gradient across the whole range. The norm carries a tiny
    additive floor so ``d||v|| = v/||v||`` stays finite at v = 0. Away from
    phi = pi the axis is ``v/||v||``; at phi ~ pi (where v -> 0 loses the axis
    direction) the axis is recovered from the symmetric part (R + I)/2.
    """
    _check_matrix(R)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_phi = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)

    vx = R[..., 2, 1] - R[..., 1, 2]
    vy = R[..., 0, 2] - R[..., 2, 0]
    vz = R[..., 1, 0] - R[..., 0, 1]
    v = torch.stack((vx, vy, vz), dim=-1)                      # 2 sin(phi) * axis
    # floored norm: finite gradient at v = 0 (plain ||.|| has v/||v|| = 0/0 there)
    vnorm = torch.sqrt((v * v).sum(-1) + 1e-24)                # (...,)
    sin_phi = 0.5 * vnorm                                      # = |sin(phi)| >= 0
    phi = torch.atan2(sin_phi, cos_phi)                        # in [0, pi], finite grad

    # generic + near-0: axis = v/||v||, so out = (v/||v||) * phi. At phi->0 this
    # tends to v/2 (correct) with a finite gradient thanks to the floor.
    out = v / vnorm[..., None] * phi[..., None]

    # near-pi region: v -> 0 loses direction; recover axis from (R + I)/2.
    near_pi = (torch.pi - phi) <= _ANGLE_EPS
    if near_pi.any():
        S = (R + torch.eye(3, dtype=R.dtype, device=R.device)) * 0.5
        diag = torch.stack((S[..., 0, 0], S[..., 1, 1], S[..., 2, 2]), dim=-1)
        # clamp to a tiny positive floor so sqrt has a finite gradient at 0
        axis = torch.sqrt(torch.clamp(diag, min=1e-12))
        # fix signs using the off-diagonal entries relative to the largest comp
        k = torch.argmax(axis, dim=-1)
        axis = _fix_pi_axis_signs(S, axis, k)
        out = torch.where(near_pi[..., None], axis * phi[..., None], out)
    # near-0 region stays the zero vector (out initialised to 0)
    return out


# ---------------------------------------------------------------------------
# quaternion <-> matrix  (q = (w, x, y, z), scalar-first, unit)
# ---------------------------------------------------------------------------
def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) unit quaternion (w, x, y, z) -> (..., 3, 3)."""
    if q.shape[-1] != 4:
        raise ValueError(f"expected last dim 4, got {q.shape[-1]}")
    q = _normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    R = torch.stack((
        1 - (tyy + tzz), txy - twz,       txz + twy,
        txy + twz,       1 - (txx + tzz), tyz - twx,
        txz - twy,       tyz + twx,       1 - (txx + tyy),
    ), dim=-1)
    return R.reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) unit quaternion (w, x, y, z), w >= 0 canonical.

    Uses the numerically stable branch on the largest of {1+tr, diagonal terms}.
    """
    _check_matrix(R)
    m = R
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    # four candidate "squared component times 4" quantities
    c0 = 1.0 + m00 + m11 + m22        # 4 w^2
    c1 = 1.0 + m00 - m11 - m22        # 4 x^2
    c2 = 1.0 - m00 + m11 - m22        # 4 y^2
    c3 = 1.0 - m00 - m11 + m22        # 4 z^2
    cands = torch.stack((c0, c1, c2, c3), dim=-1)
    branch = torch.argmax(cands, dim=-1)                       # (...,)

    def _w_branch():
        s = torch.sqrt(torch.clamp(c0, min=_NORM_EPS)) * 2.0   # 4w
        w = 0.25 * s
        x = (m[..., 2, 1] - m[..., 1, 2]) / s
        y = (m[..., 0, 2] - m[..., 2, 0]) / s
        z = (m[..., 1, 0] - m[..., 0, 1]) / s
        return torch.stack((w, x, y, z), dim=-1)

    def _x_branch():
        s = torch.sqrt(torch.clamp(c1, min=_NORM_EPS)) * 2.0   # 4x
        w = (m[..., 2, 1] - m[..., 1, 2]) / s
        x = 0.25 * s
        y = (m[..., 0, 1] + m[..., 1, 0]) / s
        z = (m[..., 0, 2] + m[..., 2, 0]) / s
        return torch.stack((w, x, y, z), dim=-1)

    def _y_branch():
        s = torch.sqrt(torch.clamp(c2, min=_NORM_EPS)) * 2.0   # 4y
        w = (m[..., 0, 2] - m[..., 2, 0]) / s
        x = (m[..., 0, 1] + m[..., 1, 0]) / s
        y = 0.25 * s
        z = (m[..., 1, 2] + m[..., 2, 1]) / s
        return torch.stack((w, x, y, z), dim=-1)

    def _z_branch():
        s = torch.sqrt(torch.clamp(c3, min=_NORM_EPS)) * 2.0   # 4z
        w = (m[..., 1, 0] - m[..., 0, 1]) / s
        x = (m[..., 0, 2] + m[..., 2, 0]) / s
        y = (m[..., 1, 2] + m[..., 2, 1]) / s
        z = 0.25 * s
        return torch.stack((w, x, y, z), dim=-1)

    q = torch.where((branch == 0)[..., None], _w_branch(),
        torch.where((branch == 1)[..., None], _x_branch(),
        torch.where((branch == 2)[..., None], _y_branch(), _z_branch())))
    q = _normalize(q)
    # canonicalise to w >= 0 (double-cover: q ~ -q)
    q = torch.where((q[..., :1] < 0), -q, q)
    return q


# ---------------------------------------------------------------------------
# composed conversions and metric
# ---------------------------------------------------------------------------
def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def axis_angle_to_rotation_6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) x (..., 3, 3) -> (...,) angle of R1^T R2 in [0, pi]."""
    _check_matrix(R1)
    _check_matrix(R2)
    rel = R1.transpose(-1, -2) @ R2
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    return torch.arccos(cos)


def is_rotation_matrix(R: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
    """Elementwise (...,) bool: R^T R = I and det R = +1 within ``atol``."""
    _check_matrix(R)
    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand(R.shape)
    orth = torch.linalg.norm(R.transpose(-1, -2) @ R - eye, dim=(-1, -2))
    det = torch.linalg.det(R)
    return (orth < atol) & ((det - 1.0).abs() < atol)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True),
                           min=_NORM_EPS)


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric [v]_x with [v]_x u = v x u."""
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack((zero, -z, y), dim=-1)
    row1 = torch.stack((z, zero, -x), dim=-1)
    row2 = torch.stack((-y, x, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _fix_pi_axis_signs(S: torch.Tensor, axis: torch.Tensor,
                       k: torch.Tensor) -> torch.Tensor:
    """For phi ~ pi, set signs of the axis from the row of (R+I)/2 = a a^T that
    corresponds to the largest |a_k| (>0 by convention)."""
    idx = k[..., None, None].expand(S.shape[:-2] + (1, 3))
    row = torch.gather(S, -2, idx).squeeze(-2)                 # (..., 3) = a_k * a
    signs = torch.sign(row)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    # the reference component itself must be positive
    ref_sign = torch.gather(signs, -1, k[..., None])
    signs = signs * ref_sign
    return axis * signs


def _check_matrix(R: torch.Tensor) -> None:
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing shape (3, 3), got {tuple(R.shape[-2:])}")
