"""Identity/motion separation and a signer-leakage guard.

Design constraint (source document): *"Keep identity shape separate from
linguistic motion to prevent signer leakage."* We enforce and **prove** this
(docs/HUMAN_REPRESENTATION.md §7), rather than assert it:

1. Per-joint **world orientations** produced by forward kinematics depend only on
   the joint rotations and the kinematic tree -- **not** on identity ``beta``
   (which only sets bone-length offsets / joint translations). So the orientation
   channel of the motion is identity-invariant by construction.

2. A linear **leakage probe** trained to recover ``beta`` from motion-only
   features cannot beat predicting the mean (normalised error ~ 1). The same probe
   *does* recover ``beta`` when it is (wrongly) folded into the features
   (normalised error ~ 0), so the guard has power: it would catch real leakage.
"""

from __future__ import annotations

import torch

from .rotations import rotation_6d_to_matrix


def world_joint_rotations(rot6d: torch.Tensor, parents: torch.Tensor
                          ) -> torch.Tensor:
    """(T, J, 6) rotations + tree -> (T, J, 3, 3) world orientations.

    Composes rotations along the tree: R^world_j = R^world_{parent} @ R_j. This is
    a pure function of the rotations and ``parents`` -- **independent of identity
    ``beta``** (which only affects joint *translations*).
    """
    R = rotation_6d_to_matrix(rot6d)                          # (T, J, 3, 3)
    J = R.shape[1]
    out = [R[:, 0]]
    for j in range(1, J):
        out.append(out[int(parents[j])] @ R[:, j])
    return torch.stack(out, dim=1)


class LinearProbe:
    """Closed-form ridge-regression probe X -> Y, used to test for leakage."""

    def __init__(self, l2: float = 1e-3) -> None:
        self.l2 = l2
        self.W = None
        self.b = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "LinearProbe":
        n, d = X.shape
        Xc = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)  # bias column
        A = Xc.T @ Xc + self.l2 * torch.eye(d + 1, dtype=X.dtype)
        W = torch.linalg.solve(A, Xc.T @ Y)
        self.W, self.b = W[:-1], W[-1]
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.W + self.b


def normalised_recovery_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MSE(pred, target) / MSE(mean, target).

    ~1.0 means the predictor does no better than the mean (no information);
    ~0.0 means near-perfect recovery.
    """
    mse = (pred - target).pow(2).mean()
    baseline = (target - target.mean(0, keepdim=True)).pow(2).mean()
    return (mse / torch.clamp(baseline, min=1e-12)).item()
