"""Linguistically-aware level of detail + consent (docs/AVATAR_RENDER.md §7).

**Innovation:** LOD decimates by **linguistic importance** and is *proved* to never
drop fingers or facial cues at any level -- only lower tiers (arms/torso/background)
are decimated. This directly implements the document's rule: "LOD that never
removes linguistically important fingers or facial cues."

Also here: appearance-consent gating and an explicit non-impersonation mode, so an
avatar's identity/appearance can only be used with recorded consent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import torch


class ImportanceTier(IntEnum):
    """Higher value = more linguistically important (kept longer under LOD)."""

    BACKGROUND = 0
    TORSO = 1
    ARMS = 2
    HANDS_PALM = 3
    FACE = 4
    FINGERS = 5


# Fingers and face are never decimated -- they carry the linguistic signal.
PROTECTED_TIERS = frozenset({ImportanceTier.FINGERS, ImportanceTier.FACE})


def lod_keep_mask(tiers: torch.Tensor, level: int) -> torch.Tensor:
    """(V,) bool mask of vertices kept at LOD ``level``.

    A vertex is kept iff its tier value ``>= level`` OR its tier is protected
    (fingers/face). ``level`` in [0, 5]; higher = more aggressive decimation. The
    protection clause is what guarantees fingers/face always survive.
    """
    protected = torch.zeros_like(tiers, dtype=torch.bool)
    for pt in PROTECTED_TIERS:
        protected |= (tiers == int(pt))
    return (tiers >= level) | protected


def fingers_face_always_kept(tiers: torch.Tensor, max_level: int = 5) -> bool:
    """Guarantee check: at EVERY LOD level, all finger/face vertices are kept."""
    protected = torch.zeros_like(tiers, dtype=torch.bool)
    for pt in PROTECTED_TIERS:
        protected |= (tiers == int(pt))
    for level in range(max_level + 1):
        keep = lod_keep_mask(tiers, level)
        if not bool(torch.all(keep[protected])):
            return False
    return True


def budget_curve(tiers: torch.Tensor, max_level: int = 5) -> List[int]:
    """Number of kept vertices at each LOD level (monotone non-increasing)."""
    return [int(lod_keep_mask(tiers, lv).sum()) for lv in range(max_level + 1)]


# ---------------------------------------------------------------------------
# consent / non-impersonation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppearanceConsent:
    """Recorded consent for using a specific avatar appearance/identity."""

    subject_id: str
    consented: bool
    non_impersonation_mode: bool = True     # visibly synthetic by default


def can_render_identity(consent: AppearanceConsent) -> bool:
    """Photorealistic identity rendering is allowed only with recorded consent."""
    return consent.consented


def requires_synthetic_marker(consent: AppearanceConsent) -> bool:
    """Whether the render must carry an obvious "synthetic avatar" marker.

    Always true unless the subject consented AND explicitly disabled the
    non-impersonation mode -- so the default is a visibly non-impersonating avatar.
    """
    return not (consent.consented and not consent.non_impersonation_mode)
