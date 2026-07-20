"""Hand-motion graph reasoning layer (Document 05).

Implements `05_hand_motion_graph_reasoning.md` (see docs/HAND_GRAPH.md): a
heterogeneous temporal graph over typed nodes {body, hands, face, locus} with
relational message passing, Graphormer-style structural attention, wrist-relative
invariant geometry, a contact soft-distance field, a multi-scale temporal pyramid,
confidence-aware masking, auxiliary heads, and hand-specific evaluation.
"""

from .hetero_graph import (
    NodeType, EdgeType, NUM_EDGE_TYPES, HAND_LANDMARKS, HAND_BONES,
    WRIST, FINGERTIPS, MIDDLE_MCP, FINGER_MCPS,
    HandGraph, HandGraphBuilder, build_two_hand_graph, temporal_unroll,
    knn_distance_edges, shortest_path_distances, validate_hand_graph,
)
from .relational import (
    group_softmax, RelationalGraphAttention, RelationalGraphNetwork,
)
from .geometry import (
    estimate_velocity, wrist_relative, wrist_frame_from_landmarks,
    wrist_frame_relative, ContactField, hard_contact,
)
from .structural import (
    edge_type_matrix, dense_bias_inputs, CentralityEncoding, GraphormerAttention,
)
from .temporal import (
    TemporalPyramid, masked_normalized_conv1d, MaskedTemporalConv,
    to_time_series, from_time_series,
)
from .heads import (
    masked_cross_entropy, masked_bce_with_logits, HandshapeHead,
    SelectedFingersHead, PalmOrientationHead, contact_loss, mirror_points,
    symmetry_loss,
)
from .metrics import (
    hand_scale, fingertip_error_in_hand_scale, handshape_accuracy, ContactPRF,
    contact_prf, collision_rate, mirror_hand, left_right_consistency,
)
from .model import HandGraphReasoner, hand_embeddings

__all__ = [
    "NodeType", "EdgeType", "NUM_EDGE_TYPES", "HAND_LANDMARKS", "HAND_BONES",
    "WRIST", "FINGERTIPS", "MIDDLE_MCP", "FINGER_MCPS",
    "HandGraph", "HandGraphBuilder", "build_two_hand_graph", "temporal_unroll",
    "knn_distance_edges", "shortest_path_distances", "validate_hand_graph",
    "group_softmax", "RelationalGraphAttention", "RelationalGraphNetwork",
    "estimate_velocity", "wrist_relative", "wrist_frame_from_landmarks",
    "wrist_frame_relative", "ContactField", "hard_contact",
    "edge_type_matrix", "dense_bias_inputs", "CentralityEncoding",
    "GraphormerAttention",
    "TemporalPyramid", "masked_normalized_conv1d", "MaskedTemporalConv",
    "to_time_series", "from_time_series",
    "masked_cross_entropy", "masked_bce_with_logits", "HandshapeHead",
    "SelectedFingersHead", "PalmOrientationHead", "contact_loss", "mirror_points",
    "symmetry_loss",
    "hand_scale", "fingertip_error_in_hand_scale", "handshape_accuracy",
    "ContactPRF", "contact_prf", "collision_rate", "mirror_hand",
    "left_right_consistency",
    "HandGraphReasoner", "hand_embeddings",
]
