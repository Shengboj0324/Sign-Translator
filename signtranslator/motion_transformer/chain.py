"""Full motion-generation chain (docs/MOTION_TRANSFORMER.md §9).

Ties the pieces together: a Doc-04 pose-6D motion sequence is tokenised by the
``MotionVQVAE``; a clause plan (Doc-02/03) is encoded and up-sampled; the motion
decoder cross-attends to the plan; and a recurrent memory carries spatial loci and
prior pose across chunks. A streaming path stitches overlapping predicted chunks in
SO(3).

The chain consumes motion features laid out ``(N, C, T)`` with ``C = J·6`` (6D
rotations), the representation shared with Docs 04-05.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..pose.rotations import rotation_6d_to_matrix
from .autoencoder import MotionVQVAE
from .backbone import ClausePlanner, DurationModel, MotionDecoder, RecurrentMemory
from .decoding import _SinusoidalPositions
from .streaming import overlap_add_rotations


class MotionGenerationChain(nn.Module):
    def __init__(self, num_joints: int, dim: int = 64, num_codes: int = 128,
                 num_downsamples: int = 1, rvq_stages: int = 2,
                 max_duration: int = 32) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = num_joints * 6
        self.dim = dim
        self.tokenizer = MotionVQVAE(self.in_channels, dim, num_codes,
                                     num_downsamples, rvq_stages)
        self.planner = ClausePlanner(dim)
        self.duration = DurationModel(dim, max_duration)
        self.decoder = MotionDecoder(dim)
        self.memory = RecurrentMemory(dim, dim)
        self.query_pos = _SinusoidalPositions(dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_motion = nn.Linear(dim, self.in_channels)

    # -- tokenizer path -----------------------------------------------------
    def reconstruct(self, motion: torch.Tensor):
        """(N, C, T) -> (recon (N, C, T), quantiser output dict)."""
        return self.tokenizer(motion)

    # -- plan-conditioned generation ---------------------------------------
    def decode_from_plan(self, clause_tokens: torch.Tensor,
                         motion_len: int) -> torch.Tensor:
        """Plan events -> motion features (N, C, motion_len) via cross-attention."""
        plan = self.planner(clause_tokens)                   # (N, L, dim)
        N = clause_tokens.shape[0]
        query = self.query_pos(self.query_token.expand(N, motion_len, self.dim))
        latents = self.decoder(query, plan)                  # (N, motion_len, dim)
        return self.to_motion(latents).transpose(1, 2)       # (N, C, motion_len)

    # -- discourse memory across chunks ------------------------------------
    def run_discourse(self, chunk_summaries: List[torch.Tensor]) -> torch.Tensor:
        """Fold a list of per-chunk summaries (each (N, dim)) through the memory,
        returning the final state that carries loci from the earliest chunk."""
        state = self.memory.init_state(chunk_summaries[0].shape[0],
                                       device=chunk_summaries[0].device,
                                       dtype=chunk_summaries[0].dtype)
        for s in chunk_summaries:
            state = self.memory(s, state)
        return state

    # -- streaming blend ----------------------------------------------------
    @staticmethod
    def stitch_rotation_chunks(chunks: List[torch.Tensor], overlap: int) -> torch.Tensor:
        """Overlap-add stitch of predicted rotation chunks (each (L, 3, 3))."""
        return overlap_add_rotations(chunks, overlap)

    def motion_to_rotations(self, motion: torch.Tensor) -> torch.Tensor:
        """(N, C, T) motion feats -> (N, T, J, 3, 3) rotations for blending/eval."""
        N, C, T = motion.shape
        six = motion.permute(0, 2, 1).reshape(N, T, self.num_joints, 6)
        return rotation_6d_to_matrix(six)
