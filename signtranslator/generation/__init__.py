"""Canonical-state generation: reviewed SIR and a declared clock to motion.

This package connects the canonical 11-channel multichannel state to the active
governed training path. It does not acquire motion, approve a rig, or establish
that generated motion is intelligible ASL. Generated states are emitted with
``observed=False`` and zero confidence, so they can never be consumed as
supervision by the existing governed admission and objective contracts.
"""

from .canonical_state import (
    CANONICAL_BRANCHES,
    CanonicalChannelBinding,
    CanonicalStateConfig,
    CanonicalStateSIRDecoder,
    GeneratedStateProvenance,
    channel_bindings_from_state,
    channel_loss_per_example,
)
from .representation_comparison import (
    compare_reconstructions,
    hidden_spans,
    hold_out_rest_baseline,
    interpolation_baseline,
)
from .stream_render import (
    encode_timestamped_video,
    presentation_timestamps,
    verify_timestamped_video,
    write_frame_bundle,
)
from .state_rig import (
    BONE_LOCAL_AFTER_REST,
    ChannelBoneMapping,
    PosedStateFrame,
    StateRigBinding,
    pose_state_frame,
    pose_state_sequence,
)

__all__ = [
    "BONE_LOCAL_AFTER_REST",
    "CANONICAL_BRANCHES",
    "CanonicalChannelBinding",
    "CanonicalStateConfig",
    "CanonicalStateSIRDecoder",
    "ChannelBoneMapping",
    "GeneratedStateProvenance",
    "PosedStateFrame",
    "StateRigBinding",
    "channel_bindings_from_state",
    "channel_loss_per_example",
    "compare_reconstructions",
    "encode_timestamped_video",
    "hidden_spans",
    "hold_out_rest_baseline",
    "interpolation_baseline",
    "presentation_timestamps",
    "verify_timestamped_video",
    "write_frame_bundle",
    "pose_state_frame",
    "pose_state_sequence",
]
