"""Vector quantiser (VQ-VAE) with straight-through gradient, EMA, and Code Reset.

Implements docs/MOTION_TRANSFORMER.md §1 (van den Oord et al., arXiv:1711.00937;
EMA + Code Reset per T2M-GPT, arXiv:2301.06052).

    k*(z) = argmin_k ‖z − e_k‖²,   z_q = e_{k*},
    z_q^{ste} = z_e + (z_q − z_e).detach()   (identity gradient to the encoder),
    L_commit = ‖z_e − sg[z_q]‖²,   L_codebook = ‖sg[z_e] − z_q‖².

With ``ema=True`` the codebook is updated by an exponential moving average of the
assigned encoder outputs (with Laplace smoothing) and ``L_codebook`` is dropped;
with ``ema=False`` the codebook is a parameter trained by ``L_codebook``.
``reset_dead_codes`` re-seeds under-used codes from live encoder outputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, dim: int, beta: float = 0.25,
                 ema: bool = True, decay: float = 0.99, eps: float = 1e-5,
                 reset_threshold: float = 1.0) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold

        codebook = torch.randn(num_codes, dim)
        if ema:
            self.register_buffer("codebook", codebook)
            self.register_buffer("cluster_size", torch.zeros(num_codes))
            self.register_buffer("ema_w", codebook.clone())
        else:
            self.codebook = nn.Parameter(codebook)
            self.register_buffer("cluster_size", torch.zeros(num_codes))

    # -- distances / assignment --------------------------------------------
    def distances(self, z: torch.Tensor) -> torch.Tensor:
        """(M, d) -> (M, K) squared distances via ‖z‖² − 2 z·e + ‖e‖²."""
        cb = self.codebook
        z2 = (z * z).sum(-1, keepdim=True)                   # (M, 1)
        e2 = (cb * cb).sum(-1)                                # (K,)
        cross = z @ cb.t()                                   # (M, K)
        return z2 - 2.0 * cross + e2.unsqueeze(0)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.distances(z).argmin(dim=-1)               # ties -> lowest index
        return idx, F.embedding(idx, self.codebook)

    # -- forward ------------------------------------------------------------
    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        shape = z_e.shape
        flat = z_e.reshape(-1, self.dim)
        idx, z_q = self.quantize(flat)

        commit_loss = F.mse_loss(flat, z_q.detach())
        codebook_loss = F.mse_loss(z_q, flat.detach())
        z_q_ste = flat + (z_q - flat).detach()               # straight-through

        if self.ema and self.training:
            self._ema_update(flat.detach(), idx)

        onehot = F.one_hot(idx, self.num_codes).type_as(flat)
        probs = onehot.mean(0)
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        loss = self.beta * commit_loss
        if not self.ema:
            loss = loss + codebook_loss

        return {
            "z_q": z_q_ste.reshape(shape),
            "indices": idx.reshape(shape[:-1]),
            "loss": loss,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "encodings": flat.detach(),
        }

    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        """Map code indices back to their codebook vectors (…,) -> (…, d)."""
        return F.embedding(idx, self.codebook)

    # -- EMA + reset --------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.num_codes).type_as(flat)   # (M, K)
        n = onehot.sum(0)                                        # (K,)
        dw = onehot.t() @ flat                                   # (K, d)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        # Laplace-smoothed counts so empty codes never divide by zero
        total = self.cluster_size.sum()
        smoothed = ((self.cluster_size + self.eps)
                    / (total + self.num_codes * self.eps) * total)
        self.codebook.copy_(self.ema_w / smoothed.unsqueeze(1).clamp_min(self.eps))

    @torch.no_grad()
    def reset_dead_codes(self, encoder_outputs: torch.Tensor) -> int:
        """Re-seed codes with usage below ``reset_threshold`` from random rows of
        ``encoder_outputs`` (M, d). Returns the number of codes reset."""
        dead = self.cluster_size < self.reset_threshold
        n_dead = int(dead.sum())
        if n_dead == 0 or encoder_outputs.shape[0] == 0:
            return 0
        pick = torch.randint(0, encoder_outputs.shape[0], (n_dead,),
                             device=encoder_outputs.device)
        seeds = encoder_outputs[pick]
        self.codebook[dead] = seeds
        self.cluster_size[dead] = self.reset_threshold
        if self.ema:
            self.ema_w[dead] = seeds
        return n_dead
