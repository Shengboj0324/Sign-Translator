"""The SMPL-X representation contract: the per-frame state and its layout.

Per-frame motion (see docs/HUMAN_REPRESENTATION.md §2):

    q_t = (gamma_t, theta_body, theta_lhand, theta_rhand, theta_jaw, theta_eye, psi_t)

with identity shape ``beta`` **sequence-constant** and held *outside* the motion.
All joint rotations are stored in the continuous 6D representation.

The layout fixes a single canonical joint order used everywhere downstream (the
kinematic tree in ``body_model.py`` matches it):

    index 0            : global orientation (pelvis / root)
    indices 1..21      : body joints (21)
    index 22           : jaw
    indices 23, 24     : left eye, right eye
    indices 25..39     : left hand (15)
    indices 40..54     : right hand (15)

Design intent enforced in code, not just documented: identity (``beta``) is kept
strictly separate from linguistic motion to prevent **signer leakage**. A
``MotionSequence`` stores motion and identity in different fields; its serialized
*motion* vector never contains ``beta``; and ``retarget`` swaps identity while
leaving every motion tensor bit-identical, which is what makes cross-signer
transfer well defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import torch

ROT_DIM = 6  # every joint rotation is a 6D continuous rotation


@dataclass(frozen=True)
class SMPLXLayout:
    """Joint counts and the canonical ordering of the pose parts."""

    n_body: int = 22          # global orient (idx 0) + 21 body joints
    n_jaw: int = 1
    n_eye: int = 2
    n_hand: int = 15          # per hand
    n_expr: int = 10
    n_shape: int = 10

    @property
    def n_joints(self) -> int:
        return self.n_body + self.n_jaw + self.n_eye + 2 * self.n_hand

    # --- canonical slices into the (n_joints, 6) rotation block ---------------
    def part_slices(self) -> Dict[str, slice]:
        b = self.n_body
        j = b + self.n_jaw
        e = j + self.n_eye
        lh = e + self.n_hand
        rh = lh + self.n_hand
        return {
            "global_orient": slice(0, 1),
            "body": slice(1, b),
            "jaw": slice(b, b + self.n_jaw),
            "eye": slice(j, e),
            "lhand": slice(e, lh),
            "rhand": slice(lh, rh),
        }

    @property
    def motion_dim(self) -> int:
        """Length of the flat per-frame motion vector: gamma + rot6d + expr."""
        return 3 + self.n_joints * ROT_DIM + self.n_expr


@dataclass
class MotionSequence:
    """A T-frame motion plus a single, sequence-constant identity.

    ``gamma``  : (T, 3)             root translation
    ``rot6d``  : (T, n_joints, 6)   per-joint 6D rotations in canonical order
    ``expr``   : (T, n_expr)        expression coefficients
    ``beta``   : (n_shape,)         identity shape (constant over the sequence)
    """

    gamma: torch.Tensor
    rot6d: torch.Tensor
    expr: torch.Tensor
    beta: torch.Tensor
    layout: SMPLXLayout = field(default_factory=SMPLXLayout)

    def __post_init__(self) -> None:
        L = self.layout
        T = self.gamma.shape[0]
        if self.gamma.shape != (T, 3):
            raise ValueError(f"gamma must be (T, 3), got {tuple(self.gamma.shape)}")
        if self.rot6d.shape != (T, L.n_joints, ROT_DIM):
            raise ValueError(
                f"rot6d must be (T, {L.n_joints}, {ROT_DIM}), got {tuple(self.rot6d.shape)}")
        if self.expr.shape != (T, L.n_expr):
            raise ValueError(f"expr must be (T, {L.n_expr}), got {tuple(self.expr.shape)}")
        if self.beta.shape != (L.n_shape,):
            raise ValueError(f"beta must be ({L.n_shape},), got {tuple(self.beta.shape)}")

    @property
    def num_frames(self) -> int:
        return self.gamma.shape[0]

    # --- part accessors -------------------------------------------------------
    def part(self, name: str) -> torch.Tensor:
        """(T, n_part, 6) rotations for a named part."""
        return self.rot6d[:, self.layout.part_slices()[name], :]

    # --- motion-only features (NO identity) -----------------------------------
    def motion_features(self) -> torch.Tensor:
        """Flat per-frame motion (T, motion_dim): gamma | rot6d | expr.

        Deliberately excludes ``beta`` -- this is the representation a downstream
        model sees, so identity cannot leak through it.
        """
        T = self.num_frames
        flat_rot = self.rot6d.reshape(T, -1)
        return torch.cat((self.gamma, flat_rot, self.expr), dim=-1)

    @classmethod
    def from_motion_features(cls, feats: torch.Tensor, beta: torch.Tensor,
                             layout: SMPLXLayout | None = None) -> "MotionSequence":
        """Inverse of ``motion_features`` (identity supplied separately)."""
        layout = layout or SMPLXLayout()
        if feats.shape[-1] != layout.motion_dim:
            raise ValueError(
                f"feature dim {feats.shape[-1]} != motion_dim {layout.motion_dim}")
        T = feats.shape[0]
        gamma = feats[:, :3]
        rot_end = 3 + layout.n_joints * ROT_DIM
        rot6d = feats[:, 3:rot_end].reshape(T, layout.n_joints, ROT_DIM)
        expr = feats[:, rot_end:]
        return cls(gamma=gamma, rot6d=rot6d, expr=expr, beta=beta, layout=layout)

    # --- identity handling ----------------------------------------------------
    def retarget(self, new_beta: torch.Tensor) -> "MotionSequence":
        """Return the SAME motion under a different identity.

        Motion tensors are shared unchanged; only ``beta`` differs. This is the
        operational meaning of identity/motion separation: the linguistic content
        is invariant to who signs it.
        """
        if new_beta.shape != (self.layout.n_shape,):
            raise ValueError(f"beta must be ({self.layout.n_shape},)")
        return replace(self, beta=new_beta)

    def motion_equal(self, other: "MotionSequence") -> bool:
        """True iff the motion (not identity) is bit-identical."""
        return (torch.equal(self.gamma, other.gamma)
                and torch.equal(self.rot6d, other.rot6d)
                and torch.equal(self.expr, other.expr))
