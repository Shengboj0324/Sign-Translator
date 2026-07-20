"""Streaming: bounded-right-context attention + SO(3) chunk blending.

See docs/MOTION_TRANSFORMER.md §7.

* ``bounded_right_context_mask`` — an additive attention mask where query frame
  ``t`` may attend to keys ``j ≤ t + R``; output ``t`` then depends only on inputs
  up to ``t + R``, so the latency is ``R`` frames (proved). Full bidirectional
  attention (``R = ∞``) is the offline upper bound, reported separately.

* **Innovation — rotation-space chunk blending.** Overlapping predicted chunks are
  stitched by a SLERP crossfade in SO(3):

    slerp(R_a, R_b, α) = R_a · exp( α · log(R_aᵀ R_b) )   (constant-speed geodesic),

  so `slerp(·,·,0)=R_a`, `slerp(·,·,1)=R_b`, and
  `d_{SO(3)}(R_a, slerp(R_a,R_b,α)) = α · d_{SO(3)}(R_a,R_b)`. A linear ramp `α:0→1`
  over the overlap gives a C0-continuous, boundary-matching transition that stays
  on the manifold (naive averaging of rotation matrices leaves SO(3)).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..pose.rotations import (
    matrix_to_axis_angle, axis_angle_to_matrix, geodesic_distance,
)


# ---------------------------------------------------------------------------
# streaming attention mask + latency
# ---------------------------------------------------------------------------
def bounded_right_context_mask(T: int, right_context: int,
                               left_context: Optional[int] = None,
                               device=None) -> torch.Tensor:
    """(T, T) additive mask. ``mask[t, j] = 0`` iff ``t − L ≤ j ≤ t + R`` else −inf.

    ``right_context = R`` bounds look-ahead (and thus latency); ``left_context = L``
    optionally bounds the past window (``None`` = unbounded past).
    """
    idx = torch.arange(T, device=device)
    j = idx.unsqueeze(0)                                     # keys
    t = idx.unsqueeze(1)                                     # queries
    allowed = j <= t + right_context
    if left_context is not None:
        allowed = allowed & (j >= t - left_context)
    mask = torch.zeros(T, T, device=device)
    return mask.masked_fill(~allowed, float("-inf"))


def streaming_latency_frames(chunk_size: int, right_context: int) -> int:
    """Emission latency in frames: buffer a chunk, then look ahead R frames."""
    return chunk_size + right_context


# ---------------------------------------------------------------------------
# SO(3) SLERP + chunk blending
# ---------------------------------------------------------------------------
def slerp(R_a: torch.Tensor, R_b: torch.Tensor, alpha) -> torch.Tensor:
    """Constant-speed geodesic interpolation between rotations. (..., 3, 3)."""
    R_rel = R_a.transpose(-1, -2) @ R_b
    aa = matrix_to_axis_angle(R_rel)                         # (..., 3)
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=R_a.dtype, device=R_a.device)
    aa_scaled = alpha.unsqueeze(-1) * aa if alpha.dim() else alpha * aa
    return R_a @ axis_angle_to_matrix(aa_scaled)


def crossfade_rotations(Ra_seq: torch.Tensor, Rb_seq: torch.Tensor) -> torch.Tensor:
    """SLERP crossfade over an overlap of length ``O``. Both (O, 3, 3).

    Ramp ``α: 0 → 1`` so the blend equals ``Ra`` at the first overlap frame and
    ``Rb`` at the last -- the boundary constraints for a seamless transition.
    """
    O = Ra_seq.shape[0]
    if O == 0:
        return Ra_seq
    if O == 1:
        alpha = torch.zeros(1, dtype=Ra_seq.dtype, device=Ra_seq.device)
    else:
        alpha = torch.linspace(0.0, 1.0, O, dtype=Ra_seq.dtype, device=Ra_seq.device)
    return slerp(Ra_seq, Rb_seq, alpha)


def overlap_add_rotations(chunks: List[torch.Tensor], overlap: int) -> torch.Tensor:
    """Stitch overlapping rotation chunks into one sequence via SLERP crossfade.

    Each ``chunk`` is ``(L_i, 3, 3)`` and consecutive chunks share ``overlap``
    frames. Output length = ``Σ L_i − overlap·(#chunks−1)``. In each overlap the
    old chunk's tail and the new chunk's head are crossfaded.
    """
    if not chunks:
        raise ValueError("no chunks")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    out = [chunks[0]] if overlap == 0 else [chunks[0][:chunks[0].shape[0] - overlap]]
    for c in range(1, len(chunks)):
        prev, cur = chunks[c - 1], chunks[c]
        if overlap > 0:
            tail = prev[prev.shape[0] - overlap:]
            head = cur[:overlap]
            out.append(crossfade_rotations(tail, head))
            body = cur[overlap: cur.shape[0] - overlap] if c < len(chunks) - 1 else cur[overlap:]
            out.append(body)
        else:
            out.append(cur)
    return torch.cat(out, dim=0)
