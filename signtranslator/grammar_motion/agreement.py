"""Agreement (indicating) verbs: path and facing toward loci.

Within a verb span [t₀, t₁] with dominant-hand point h(t) (body frame), source and
goal loci depend on the verb class (Padden 1988; Meir 1998): regular agreeing verbs
move subject → object, backwards verbs object → subject, spatial verbs source →
goal; the palm faces the syntactic object. With ``Δh = h(t₁) − h(t₀)`` and
``u = unit(L_goal − L_src)``:

    c_dir = ⟨unit(Δh), u⟩,  e_dir = arccos c_dir,
    e_end = ∠(h(t₁) − o, L_goal − o)     (directions from the chest o; hands rarely reach a locus),
    f     = ⟨n_palm(t₁), unit(L_obj − h(t₁))⟩,
    L_agr = λ_dir (1 − c_dir) + λ_end e_end² + λ_face (1 − f),   λ_dir > λ_end.

Weighting direction above endpoints follows Lu & Huenerfauth (2012): the movement
vector varies far less across signers than the start and end points.
"""
from __future__ import annotations

import numpy as np

from .frames import unit

WEIGHTS = dict(direction=1., endpoint=.5, facing=.25)
SOURCE_GOAL = {"agreeing": ("subject", "object"), "backwards": ("object", "subject"),
               "spatial": ("source", "goal")}


def agreement_errors(hand: np.ndarray, palm_normal: np.ndarray | None, source: np.ndarray, goal: np.ndarray,
                     obj: np.ndarray | None = None) -> dict:
    """Degrees of path-direction and endpoint error and facing cosine for one verb span (numpy)."""
    dh = hand[-1] - hand[0]
    u = unit(goal - source)
    c_dir = float(unit(dh) @ u) if np.linalg.norm(dh) > 1e-9 else 0.
    e_end = float(np.degrees(np.arccos(np.clip(unit(hand[-1]) @ unit(goal), -1, 1))))
    out = dict(direction_deg=float(np.degrees(np.arccos(np.clip(c_dir, -1, 1)))), endpoint_deg=e_end,
               path_length=float(np.linalg.norm(dh)))
    if palm_normal is not None and obj is not None:
        out["facing_cos"] = float(unit(palm_normal) @ unit(obj - hand[-1]))
    return out


def agreement_loss(hand, source, goal, palm_normal=None, obj=None, weights=WEIGHTS):
    """Differentiable L_agr for (B, T, 3) hand paths and (B, 3) loci (torch)."""
    import torch
    F = torch.nn.functional
    dh = hand[:, -1] - hand[:, 0]
    c_dir = (F.normalize(dh, dim=-1) * F.normalize(goal - source, dim=-1)).sum(-1)
    cos_end = (F.normalize(hand[:, -1], dim=-1) * F.normalize(goal, dim=-1)).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    loss = weights["direction"] * (1 - c_dir) + weights["endpoint"] * torch.acos(cos_end) ** 2
    if palm_normal is not None and obj is not None:
        facing = (F.normalize(palm_normal, dim=-1) * F.normalize(obj - hand[:, -1], dim=-1)).sum(-1)
        loss = loss + weights["facing"] * (1 - facing)
    return loss.mean()


def verb_class_from_path(hand: np.ndarray, palm_normal: np.ndarray) -> str:
    """'regular' when path and final facing are parallel, 'backwards' when antiparallel (the sign of ⟨Δh, n⟩)."""
    return "regular" if float(unit(hand[-1] - hand[0]) @ unit(palm_normal)) >= 0 else "backwards"


__all__ = ["WEIGHTS", "SOURCE_GOAL", "agreement_errors", "agreement_loss", "verb_class_from_path"]
