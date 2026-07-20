"""Curriculum orchestration + frozen baselines (Doc-11 §7).

The five-stage curriculum is an explicit ordered schedule whose unlocked capacity is
monotone non-decreasing; a `FrozenBaseline` registry snapshots each stage's weights
so a later regression is always detectable (retain frozen baselines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List

import torch
import torch.nn as nn


class Stage(IntEnum):
    UNIMODAL_MASKED = 0        # (1) masked unimodal motion/video
    CROSSVIEW_ALIGN = 1        # (2) RGB<->pose, 2D<->3D alignment
    MOTION_TEXT_CONTRAST = 2   # (3) motion<->text/speech with curated negatives
    SUPERVISED = 3             # (4) supervised sign-plan and production
    LIMITED_E2E = 4            # (5) limited end-to-end tuning


@dataclass(frozen=True)
class CurriculumStage:
    stage: Stage
    objective: str
    unlocked: FrozenSet[str]   # CUMULATIVE set of trainable components


#: cumulative unlock — each stage adds components, never removes.
CURRICULUM: List[CurriculumStage] = [
    CurriculumStage(Stage.UNIMODAL_MASKED, "masked_motion_nll",
                    frozenset({"motion_encoder"})),
    CurriculumStage(Stage.CROSSVIEW_ALIGN, "multiview_infonce",
                    frozenset({"motion_encoder", "pose_encoder", "rgb_encoder"})),
    CurriculumStage(Stage.MOTION_TEXT_CONTRAST, "hard_negative_infonce",
                    frozenset({"motion_encoder", "pose_encoder", "rgb_encoder",
                               "text_encoder", "aligner"})),
    CurriculumStage(Stage.SUPERVISED, "sign_plan_and_production",
                    frozenset({"motion_encoder", "pose_encoder", "rgb_encoder",
                               "text_encoder", "aligner", "planner",
                               "production_head"})),
    CurriculumStage(Stage.LIMITED_E2E, "end_to_end_finetune",
                    frozenset({"motion_encoder", "pose_encoder", "rgb_encoder",
                               "text_encoder", "aligner", "planner",
                               "production_head", "decoder"})),
]


def is_monotone_unlock(curriculum: List[CurriculumStage] = CURRICULUM) -> bool:
    """Unlocked capacity never shrinks across the ordered curriculum."""
    for prev, cur in zip(curriculum, curriculum[1:]):
        if cur.stage <= prev.stage:
            return False
        if not cur.unlocked.issuperset(prev.unlocked):
            return False
    return True


def stage_objective(stage: Stage) -> str:
    return CURRICULUM[int(stage)].objective


# ---------------------------------------------------------------------------
# frozen baselines
# ---------------------------------------------------------------------------
@dataclass
class FrozenBaseline:
    """Snapshots of model weights retained per curriculum stage."""

    _snaps: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def snapshot(self, name: str, module: nn.Module) -> None:
        self._snaps[name] = {k: v.detach().clone()
                             for k, v in module.state_dict().items()}

    def has(self, name: str) -> bool:
        return name in self._snaps

    def max_param_drift(self, name: str, module: nn.Module) -> float:
        """Largest abs weight change vs the snapshot (0.0 == bit-identical)."""
        if name not in self._snaps:
            raise KeyError(f"no baseline {name!r}")
        snap = self._snaps[name]
        cur = module.state_dict()
        drift = 0.0
        for k, v in snap.items():
            drift = max(drift, float((cur[k] - v).abs().max()))
        return drift

    def matches(self, name: str, module: nn.Module) -> bool:
        return self.max_param_drift(name, module) == 0.0
