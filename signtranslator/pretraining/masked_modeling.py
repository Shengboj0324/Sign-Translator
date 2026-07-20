"""Masked motion modeling objective (Doc-11 §2).

MAE-style asymmetric encoder/decoder over a flat token sequence: the encoder sees
ONLY visible tokens; a lightweight decoder receives the encoded visibles plus a
learned mask token at hidden positions and predicts the discrete latent tokens
(cross-entropy against Doc-06 VQ indices) or 6D rotations (Doc-04 geodesic). The
loss is computed on masked positions ONLY. Includes the wav2vec-2 diversity term.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pose.rotations import rotation_6d_to_matrix, geodesic_distance


# ---------------------------------------------------------------------------
# objectives (loss functions)
# ---------------------------------------------------------------------------
def masked_token_nll(logits: torch.Tensor, target_idx: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy L_mask over masked positions only.

    ``logits`` (N, K), ``target_idx`` (N,) long, ``mask`` (N,) bool. Returns
    (1/|M|) Σ_{i∈M} −log softmax(logits_i)[target_i].
    """
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")
    if not mask.any():
        raise ValueError("no masked positions")
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)   # (N,)
    return nll[mask].mean()


def masked_rotation_geodesic(pred_6d: torch.Tensor, target_6d: torch.Tensor,
                             mask: torch.Tensor) -> torch.Tensor:
    """Mean geodesic rotation error over masked frames (reuse Doc-04)."""
    if not mask.any():
        raise ValueError("no masked positions")
    Rp = rotation_6d_to_matrix(pred_6d[mask])
    Rt = rotation_6d_to_matrix(target_6d[mask])
    return geodesic_distance(Rp, Rt).mean()


def codebook_diversity_loss(target_idx: torch.Tensor, num_codes: int) -> torch.Tensor:
    """wav2vec-2 diversity: −H(mean code usage), minimised at uniform usage.

    Encourages spreading assignments across the codebook (anti-collapse). Entropy
    in nats; the loss is most negative (best) when usage is uniform.
    """
    usage = torch.bincount(target_idx, minlength=num_codes).float()
    probs = usage / usage.sum()
    entropy = -(probs * (probs + 1e-12).log()).sum()
    return -entropy


def copy_through_logits(tokens: torch.Tensor, mask: torch.Tensor,
                        num_codes: int) -> torch.Tensor:
    """Logits of a decoder that copies its INPUT token (mask token at hidden pos).

    Used to prove masked-only scoring separates prediction from copying: a copier
    scores ~0 on visible positions but chance on masked positions (its input there
    is the mask sentinel, not the true token).
    """
    n = tokens.shape[0]
    logits = torch.zeros(n, num_codes)
    visible = ~mask
    # visible: put all mass on the (correct) input token.
    logits[visible] = F.one_hot(tokens[visible], num_codes).float() * 20.0
    # masked: input is the sentinel -> uniform (no information), i.e. chance.
    return logits


# ---------------------------------------------------------------------------
# asymmetric masked model
# ---------------------------------------------------------------------------
class MaskedMotionModel(nn.Module):
    """MAE-style masked token predictor over a flat (frame, part) sequence.

    Encoder consumes only visible tokens; decoder fills masked positions with a
    learned mask token. The model output is invariant to the *identity* of masked
    input tokens (they are never read) — the MAE asymmetry, verified in tests.
    """

    def __init__(self, num_codes: int, dim: int = 32, max_frames: int = 128,
                 num_parts: int = 4, heads: int = 4, enc_layers: int = 2,
                 dec_layers: int = 1) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.token_embed = nn.Embedding(num_codes, dim)
        self.frame_embed = nn.Embedding(max_frames, dim)
        self.part_embed = nn.Embedding(num_parts, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
        dec = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.decoder = nn.TransformerEncoder(dec, dec_layers)
        self.head = nn.Linear(dim, num_codes)

    def _pos(self, positions: torch.Tensor) -> torch.Tensor:
        return self.frame_embed(positions[:, 0]) + self.part_embed(positions[:, 1])

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """(N,) tokens, (N,2) [frame,part] positions, (N,) bool mask -> (N, K)."""
        pos = self._pos(positions)                            # (N, d)
        visible = ~mask
        # ---- encoder: ONLY visible tokens ----
        vis_in = (self.token_embed(tokens[visible]) + pos[visible]).unsqueeze(0)
        vis_enc = self.encoder(vis_in).squeeze(0)             # (Nv, d)
        # ---- decoder input: scatter encoded visibles, mask token at hidden ----
        dec_in = pos.clone()
        dec_in[visible] = dec_in[visible] + vis_enc           # visible carry content
        dec_in[mask] = dec_in[mask] + self.mask_token         # hidden = sentinel
        dec_out = self.decoder(dec_in.unsqueeze(0)).squeeze(0)
        return self.head(dec_out)                             # (N, K)
