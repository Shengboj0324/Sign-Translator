"""Adapters mapping external keypoint layouts onto the project skeleton.

Real pipelines produce keypoints in a tracker-specific layout: MediaPipe
Holistic (33 pose + 21 per hand), OpenPose BODY_25 + hands, or an SMPL-X joint
regressor. This module maps those layouts onto the 27-joint skeleton defined in
:mod:`signtranslator.skeleton.graph`, handling the two things that always bite:

  * **index mapping** - a source layout rarely has a 1:1 correspondence, so a
    target joint may be a single source joint or the *midpoint* of several
    (e.g. our "chest" from the two shoulders).
  * **confidence / visibility** - trackers emit per-keypoint confidence. Points
    below threshold are marked missing so the cleaning pass can interpolate them
    rather than training on garbage coordinates.

Adapters are declarative: a mapping table plus a resolver, so adding a new
tracker means adding a table, not code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..skeleton.graph import NUM_DEFAULT_JOINTS

# A target joint is either one source index or the mean of several.
JointSource = Union[int, Tuple[int, ...]]


# Target skeleton (see skeleton/graph.py):
#   0 head, 1 neck, 2 chest, 3 r_shoulder, 4 r_elbow, 5 r_wrist,
#   6 l_shoulder, 7 l_elbow, 8 l_wrist, 9-17 right hand, 18-26 left hand
#
# MediaPipe Holistic pose landmark indices used below:
#   0 nose, 11 l_shoulder, 12 r_shoulder, 13 l_elbow, 14 r_elbow,
#   15 l_wrist, 16 r_wrist, 23 l_hip, 24 r_hip
MEDIAPIPE_POSE_MAP: Dict[int, JointSource] = {
    0: 0,             # head        <- nose
    1: (11, 12),      # neck        <- shoulder midpoint
    2: (11, 12, 23, 24),  # chest   <- torso centroid
    3: 12,            # r_shoulder
    4: 14,            # r_elbow
    5: 16,            # r_wrist
    6: 11,            # l_shoulder
    7: 13,            # l_elbow
    8: 15,            # l_wrist
}

# MediaPipe hand landmarks (21 per hand): 0 wrist, 1-4 thumb, 5-8 index,
# 9-12 middle, 13-16 ring, 17-20 pinky.
# Our per-hand layout (9 joints): palm, thumb1, thumb2, index1, index2,
# mid1, mid2, ring, pinky.
MEDIAPIPE_HAND_MAP: Dict[int, JointSource] = {
    0: 0,    # palm    <- hand wrist
    1: 2,    # thumb1  <- thumb MCP
    2: 4,    # thumb2  <- thumb tip
    3: 5,    # index1  <- index MCP
    4: 8,    # index2  <- index tip
    5: 9,    # mid1    <- middle MCP
    6: 12,   # mid2    <- middle tip
    7: 16,   # ring    <- ring tip
    8: 20,   # pinky   <- pinky tip
}

# OpenPose BODY_25: 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist,
# 5 l_shoulder, 6 l_elbow, 7 l_wrist, 8 mid_hip
OPENPOSE_POSE_MAP: Dict[int, JointSource] = {
    0: 0, 1: 1, 2: (1, 8), 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
}

RIGHT_HAND_OFFSET = 9
LEFT_HAND_OFFSET = 18


@dataclass
class AdapterResult:
    pose: torch.Tensor        # (C, T, V) mapped keypoints
    missing: torch.Tensor     # (C, T, V) bool, True where unreliable/absent

    @property
    def missing_rate(self) -> float:
        return float(self.missing.float().mean())


def _gather(src: torch.Tensor, source: JointSource) -> torch.Tensor:
    """Select one source joint or average several. src: (C, T, V_src)."""
    if isinstance(source, int):
        return src[:, :, source]
    idx = torch.tensor(source, dtype=torch.long, device=src.device)
    return src[:, :, idx].mean(dim=2)


def _gather_conf(conf: Optional[torch.Tensor], source: JointSource) -> Optional[torch.Tensor]:
    """Confidence of a mapped joint = min over contributing source joints."""
    if conf is None:
        return None
    if isinstance(source, int):
        return conf[:, source]
    idx = torch.tensor(source, dtype=torch.long, device=conf.device)
    return conf[:, idx].min(dim=1).values


def apply_mapping(src: torch.Tensor, mapping: Dict[int, JointSource],
                  out: torch.Tensor, missing: torch.Tensor, offset: int,
                  conf: Optional[torch.Tensor] = None,
                  conf_threshold: float = 0.3) -> None:
    """Write mapped joints into ``out``/``missing`` in place at ``offset``.

    A written joint is missing only when its confidence is below threshold; with
    no confidence supplied every written joint is considered valid. Joints never
    written by any mapping keep their initial ``missing=True``.
    """
    for tgt, source in mapping.items():
        j = tgt + offset
        out[:, :, j] = _gather(src, source)
        c = _gather_conf(conf, source)
        if c is None:
            missing[:, :, j] = False
        else:
            missing[:, :, j] = (c < conf_threshold).unsqueeze(0).expand(out.shape[0], -1)


class KeypointAdapter:
    """Maps a tracker's keypoints onto the 27-joint skeleton.

    Args:
        pose_map: source->target mapping for the body.
        hand_map: per-hand mapping (applied to both hands).
        num_joints: size of the target skeleton.
        conf_threshold: keypoints below this confidence are marked missing.
    """

    def __init__(self, pose_map: Dict[int, JointSource],
                 hand_map: Optional[Dict[int, JointSource]] = None,
                 num_joints: int = NUM_DEFAULT_JOINTS,
                 conf_threshold: float = 0.3) -> None:
        self.pose_map = pose_map
        self.hand_map = hand_map
        self.num_joints = num_joints
        self.conf_threshold = conf_threshold

    def __call__(self, body: torch.Tensor,
                 right_hand: Optional[torch.Tensor] = None,
                 left_hand: Optional[torch.Tensor] = None,
                 body_conf: Optional[torch.Tensor] = None,
                 right_conf: Optional[torch.Tensor] = None,
                 left_conf: Optional[torch.Tensor] = None) -> AdapterResult:
        """Map one clip.

        Args:
            body: ``(C, T, V_body)`` source body keypoints.
            right_hand / left_hand: ``(C, T, V_hand)`` optional hand keypoints.
            *_conf: ``(T, V_*)`` optional per-keypoint confidences.
        """
        if body.dim() != 3:
            raise ValueError("body must be (C, T, V_src)")
        c, t, _ = body.shape
        out = torch.zeros(c, t, self.num_joints, dtype=body.dtype)
        # Joints we never write stay marked missing so cleaning can fill them.
        missing = torch.ones(c, t, self.num_joints, dtype=torch.bool)

        apply_mapping(body, self.pose_map, out, missing, 0, body_conf,
                      self.conf_threshold)

        if self.hand_map is not None:
            for hand, offset, hconf in ((right_hand, RIGHT_HAND_OFFSET, right_conf),
                                        (left_hand, LEFT_HAND_OFFSET, left_conf)):
                if hand is None:
                    continue  # an absent hand stays flagged missing for cleaning
                if hand.shape[0] != c or hand.shape[1] != t:
                    raise ValueError("hand keypoints must share (C, T) with body")
                apply_mapping(hand, self.hand_map, out, missing, offset, hconf,
                              self.conf_threshold)

        return AdapterResult(pose=out, missing=missing)


def mediapipe_holistic_adapter(conf_threshold: float = 0.3) -> KeypointAdapter:
    """Adapter for MediaPipe Holistic (33 pose + 21 landmarks per hand)."""
    return KeypointAdapter(MEDIAPIPE_POSE_MAP, MEDIAPIPE_HAND_MAP,
                           conf_threshold=conf_threshold)


def openpose_adapter(conf_threshold: float = 0.3) -> KeypointAdapter:
    """Adapter for OpenPose BODY_25 (+ optional 21-point hands)."""
    return KeypointAdapter(OPENPOSE_POSE_MAP, MEDIAPIPE_HAND_MAP,
                           conf_threshold=conf_threshold)
