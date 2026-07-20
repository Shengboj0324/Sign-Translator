"""Rigged-mesh track: LBS + retargeting + blendshapes (docs/AVATAR_RENDER.md §2).

Linear blend skinning is the Doc-04 equation ``v' = Σ_k w_k G'_k [v;1]``. Retargeting
maps a source (SMPL-X) rig to a production skeleton by a joint correspondence and a
rest-pose alignment rotation. **Handedness-certified retargeting:** the alignment
is a proper rotation (``det = +1``, via the Doc-04 Kabsch reflection guard), so
mirroring errors are impossible by construction. Facial blendshapes are linear in
the expression coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

from ..pose.metrics import kabsch
from ..diffusion_gen.constraints import project_joint_limits


# ---------------------------------------------------------------------------
# linear blend skinning
# ---------------------------------------------------------------------------
def apply_lbs(v_rest: torch.Tensor, weights: torch.Tensor,
              g_rel: torch.Tensor) -> torch.Tensor:
    """v'_i = Σ_j w_{ij} (G'_j [v_i; 1])_{1:3}.

    ``v_rest`` (V, 3), ``weights`` (V, J) partition of unity, ``g_rel`` (J, 4, 4)
    rest-removed transforms (as produced by Doc-04 ``forward_kinematics``).
    """
    V = v_rest.shape[0]
    Tmat = torch.einsum("vj,jab->vab", weights, g_rel)       # (V, 4, 4)
    ones = torch.ones(V, 1, dtype=v_rest.dtype, device=v_rest.device)
    homo = torch.cat((v_rest, ones), dim=-1)                 # (V, 4)
    return torch.einsum("vab,vb->va", Tmat, homo)[..., :3]


# ---------------------------------------------------------------------------
# handedness-certified retargeting
# ---------------------------------------------------------------------------
# Importance tiers: wrist/finger joints take priority in the correspondence.
PRIORITY_TIERS = ("FINGERS", "WRIST", "FACE", "ARMS", "TORSO")


@dataclass
class RetargetMap:
    """A source->target joint correspondence and a proper rest-alignment rotation."""

    correspondence: Dict[int, int]
    align: torch.Tensor                                      # (3, 3), det = +1

    def __post_init__(self) -> None:
        det = float(torch.linalg.det(self.align))
        if abs(det - 1.0) > 1e-4:
            raise ValueError(f"align must be a proper rotation (det=+1), got {det}")

    @property
    def preserves_handedness(self) -> bool:
        return float(torch.linalg.det(self.align)) > 0


def build_retarget(source_rest: torch.Tensor, target_rest: torch.Tensor,
                   correspondence: Dict[int, int]) -> RetargetMap:
    """Fit the rest-alignment rotation between corresponding joints via Kabsch.

    Kabsch's reflection guard forces ``det = +1``, so the alignment can NEVER be a
    reflection -- mirroring errors are ruled out. ``source_rest``/``target_rest``
    are (J, 3) rest joint positions.
    """
    src = torch.stack([source_rest[s] for s in correspondence])
    tgt = torch.stack([target_rest[t] for t in correspondence.values()])
    _, R, _, _ = kabsch(src, tgt)
    return RetargetMap(correspondence=dict(correspondence), align=R)


def retarget_residual(rmap: RetargetMap, source_rest: torch.Tensor,
                      target_rest: torch.Tensor) -> float:
    """Mean alignment error after applying the (proper) rotation. A large residual
    for a mirror-imaged target is the signature of a refused reflection."""
    src = torch.stack([source_rest[s] for s in rmap.correspondence])
    tgt = torch.stack([target_rest[t] for t in rmap.correspondence.values()])
    src_c = src - src.mean(0); tgt_c = tgt - tgt.mean(0)
    aligned = src_c @ rmap.align.T
    return float((aligned - tgt_c).norm(dim=-1).mean())


def prioritized_joints(importance: Dict[int, str]) -> List[int]:
    """Order joints so higher-priority tiers (fingers/wrist/face) come first."""
    rank = {tier: i for i, tier in enumerate(PRIORITY_TIERS)}
    return sorted(importance, key=lambda j: rank.get(importance[j], len(PRIORITY_TIERS)))


# ---------------------------------------------------------------------------
# joint-limit correction (reuse Doc-07)
# ---------------------------------------------------------------------------
def correct_joint_limits(angles: torch.Tensor, theta_max: float) -> torch.Tensor:
    """Project retargeted joint angles onto the production rig's limits."""
    return project_joint_limits(angles, theta_max)


# ---------------------------------------------------------------------------
# facial blendshapes
# ---------------------------------------------------------------------------
def apply_blendshapes(mean_face: torch.Tensor, basis: torch.Tensor,
                      expr: torch.Tensor) -> torch.Tensor:
    """f = f̄ + Σ_j ψ_j B_j. ``mean_face`` (Nv, 3), ``basis`` (Nv, 3, E), ``expr`` (E,)."""
    return mean_face + torch.einsum("vce,e->vc", basis, expr)
