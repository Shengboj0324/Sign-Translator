"""Camera projection and robust re-projection for SMPL-X fitting.

See docs/HUMAN_REPRESENTATION.md §4. A pinhole camera projects 3D joints to the
image; the fit compares the projection with 2D detections under a **robust**
error so a few grossly wrong detections cannot dominate (Geman-McClure), weighted
by detection confidence.

    Pi(X) = (fx * Xc/Zc + cx, fy * Yc/Zc + cy),  [Xc;Yc;Zc] = R X + t,  Zc > 0.

Weak perspective (scale s, offset (tx,ty)) is provided as the SMPLify-style
approximation valid when depth variation << distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PerspectiveCamera:
    """Pinhole camera. Intrinsics fx, fy, cx, cy; extrinsics R (3,3), t (3,)."""

    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    R: torch.Tensor
    t: torch.Tensor
    z_eps: float = 1e-6

    def project(self, X: torch.Tensor):
        """(..., 3) world points -> ((..., 2) pixels, (...,) bool in-front mask).

        Points with Zc <= 0 are behind the camera; their pixel is still returned
        (with a clamped Z to avoid inf) but flagged invalid via the mask.
        """
        Xc = torch.einsum("ab,...b->...a", self.R, X) + self.t
        Zc = Xc[..., 2]
        in_front = Zc > self.z_eps
        Z_safe = torch.where(in_front, Zc, torch.full_like(Zc, self.z_eps))
        u = self.fx * Xc[..., 0] / Z_safe + self.cx
        v = self.fy * Xc[..., 1] / Z_safe + self.cy
        return torch.stack((u, v), dim=-1), in_front

    @staticmethod
    def look_at(fx, fy, cx, cy, eye, target, up=(0.0, 1.0, 0.0),
                dtype=torch.float64) -> "PerspectiveCamera":
        """Build a camera at ``eye`` looking at ``target`` (right-handed, +Z forward)."""
        eye = torch.as_tensor(eye, dtype=dtype)
        target = torch.as_tensor(target, dtype=dtype)
        up = torch.as_tensor(up, dtype=dtype)
        f = target - eye
        f = f / torch.linalg.norm(f)                          # forward = +Z cam
        # Right-handed camera frame: x = up x f, y = f x x, z = f, so that
        # x x y = z and det([x;y;z]) = +1 (a proper rotation in SO(3)).
        # (Using r = f x up here would give a LEFT-handed frame with det = -1,
        # i.e. a reflection -- a mirror-imaged, invalid extrinsic.)
        r = torch.cross(up, f, dim=-1)
        r = r / torch.linalg.norm(r)                          # +X cam (right)
        u = torch.cross(f, r, dim=-1)                         # +Y cam
        # world->camera rotation has rows [r; u; f]
        R = torch.stack((r, u, f), dim=0)
        t = -R @ eye
        mk = lambda x: torch.as_tensor(x, dtype=dtype)
        return PerspectiveCamera(mk(fx), mk(fy), mk(cx), mk(cy), R, t)


@dataclass
class WeakPerspectiveCamera:
    """Orthographic-plus-scale: x = s (X, Y) + (tx, ty)."""

    s: torch.Tensor
    tx: torch.Tensor
    ty: torch.Tensor

    def project(self, X: torch.Tensor) -> torch.Tensor:
        u = self.s * X[..., 0] + self.tx
        v = self.s * X[..., 1] + self.ty
        return torch.stack((u, v), dim=-1)


# ---------------------------------------------------------------------------
# robust error
# ---------------------------------------------------------------------------
def geman_mcclure(r: torch.Tensor, sigma: float) -> torch.Tensor:
    """rho(r) = r^2 / (r^2 + sigma^2) in [0, 1). Quadratic near 0, saturates at 1."""
    r2 = r * r
    return r2 / (r2 + sigma * sigma)


def geman_mcclure_influence(r: torch.Tensor, sigma: float) -> torch.Tensor:
    """rho'(r) = 2 sigma^2 r / (r^2 + sigma^2)^2 -- redescending (-> 0 as r->inf)."""
    s2 = sigma * sigma
    denom = (r * r + s2) ** 2
    return 2.0 * s2 * r / denom


def reprojection_loss(projected: torch.Tensor, keypoints: torch.Tensor,
                      confidence: torch.Tensor | None = None,
                      sigma: float = 100.0, robust: bool = True) -> torch.Tensor:
    """Sum_i c_i * rho( ||proj_i - k_i|| ).

    ``projected``/``keypoints`` are (..., 2); ``confidence`` (...,) in [0, 1].
    With ``robust=False`` uses the squared residual (non-robust L2) instead.
    """
    if projected.shape != keypoints.shape:
        raise ValueError("projected and keypoints must have the same shape")
    resid = torch.linalg.norm(projected - keypoints, dim=-1)   # (...,)
    per = geman_mcclure(resid, sigma) if robust else resid * resid
    if confidence is not None:
        per = confidence * per
    return per.sum()
