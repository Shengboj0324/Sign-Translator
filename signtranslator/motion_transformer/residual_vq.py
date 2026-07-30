"""Residual vector quantisation and part-specific codebooks.

See docs/MOTION_TRANSFORMER.md §2-3.

Residual VQ (SoundStream, arXiv:2107.03312) cascades quantisers over the residual:

    r_1 = z_e;  c_i = Q_i(r_i);  r_{i+1} = r_i − c_i;  z_q = Σ_i c_i.

A single straight-through is applied over the WHOLE cascade
(``z_q^{ste} = z_e + (Σc_i − z_e).detach()``) so the encoder receives an identity
gradient from the full quantised vector -- applying per-stage STE would detach all
stages after the first (a subtle bug this avoids).

Part-specific quantisation (``PartitionedVQ``) splits the latent by channel into
torso/hands/face and quantises each part with its own (residual) codebook, so
low-variance torso motion cannot consume hand capacity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import VectorQuantizer


# ---------------------------------------------------------------------------
# k-means initialisation (deterministic given a generator)
# ---------------------------------------------------------------------------
def kmeans(data: torch.Tensor, k: int, iters: int = 10,
           generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Lloyd's k-means -> (k, d) centroids. Empty clusters keep their seed."""
    M, d = data.shape
    if M < k:                                                # pad by repetition
        reps = (k + M - 1) // M
        data = data.repeat(reps, 1)[:k]
        M = k
    perm = torch.randperm(M, generator=generator)[:k]
    centroids = data[perm].clone()
    for _ in range(iters):
        d2 = torch.cdist(data, centroids) ** 2
        assign = d2.argmin(1)
        for j in range(k):
            members = data[assign == j]
            if members.shape[0] > 0:
                centroids[j] = members.mean(0)
    return centroids


class ResidualVQ(nn.Module):
    def __init__(self, num_stages: int, num_codes: int, dim: int,
                 beta: float = 0.25, ema: bool = True, shared: bool = False) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.dim = dim
        self.beta = beta
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, dim, beta=beta, ema=ema)
            for _ in range(num_stages)
        ])

    @torch.no_grad()
    def init_from_data(self, z_e: torch.Tensor, iters: int = 10, seed: int = 0) -> None:
        """k-means-initialise each stage's codebook on the running residual."""
        g = torch.Generator().manual_seed(seed)
        residual = z_e.reshape(-1, self.dim)
        for vq in self.quantizers:
            centroids = kmeans(residual, vq.num_codes, iters, g).to(residual.dtype)
            vq.codebook.copy_(centroids)
            if vq.ema:
                vq.ema_w.copy_(centroids)
                vq.cluster_size.fill_(1.0)
            _, c = vq.quantize(residual)
            residual = residual - c

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        shape = z_e.shape
        flat = z_e.reshape(-1, self.dim)
        residual = flat
        z_q_hard = torch.zeros_like(flat)
        commit = flat.new_zeros(())
        codebook_loss = flat.new_zeros(())
        indices: List[torch.Tensor] = []
        residual_norms: List[float] = [residual.norm(dim=-1).mean().detach().item()]

        for vq in self.quantizers:
            idx, c = vq.quantize(residual)
            commit = commit + F.mse_loss(residual, c.detach())
            if not vq.ema:
                codebook_loss = codebook_loss + F.mse_loss(c, residual.detach())
            elif self.training:
                vq._ema_update(residual.detach(), idx)
            z_q_hard = z_q_hard + c
            residual = residual - c
            indices.append(idx)
            residual_norms.append(residual.norm(dim=-1).mean().detach().item())

        z_q_ste = flat + (z_q_hard - flat).detach()          # ONE STE for the cascade
        loss = self.beta * commit + codebook_loss
        return {
            "z_q": z_q_ste.reshape(shape),
            "indices": torch.stack(indices, dim=-1).reshape(shape[:-1] + (self.num_stages,)),
            "loss": loss,
            "commit_loss": commit,
            "final_residual": residual.reshape(shape),
            "residual_norms": residual_norms,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """(..., num_stages) code indices -> summed codebook vectors (..., dim)."""
        z = None
        for i, vq in enumerate(self.quantizers):
            c = vq.decode_indices(indices[..., i])
            z = c if z is None else z + c
        return z


class PartitionedVQ(nn.Module):
    """Part-specific (R)VQ: split channels by part and quantise each separately."""

    def __init__(self, part_dims: Dict[str, int], num_stages: int = 1,
                 num_codes: int = 256, beta: float = 0.25, ema: bool = True) -> None:
        super().__init__()
        self.part_names = list(part_dims)
        self.part_dims = part_dims
        self.total_dim = sum(part_dims.values())
        self.quantizers = nn.ModuleDict({
            p: ResidualVQ(num_stages, num_codes, d, beta=beta, ema=ema)
            for p, d in part_dims.items()
        })
        # channel offsets for the split
        self.offsets: Dict[str, slice] = {}
        cur = 0
        for p, d in part_dims.items():
            self.offsets[p] = slice(cur, cur + d)
            cur += d

    @torch.no_grad()
    def init_from_data(self, z_e: torch.Tensor, seed: int = 0) -> None:
        for p in self.part_names:
            self.quantizers[p].init_from_data(z_e[..., self.offsets[p]], seed=seed)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        parts_zq = []
        loss = z_e.new_zeros(())
        per_part: Dict[str, Dict] = {}
        for p in self.part_names:
            out = self.quantizers[p](z_e[..., self.offsets[p]])
            parts_zq.append(out["z_q"])
            loss = loss + out["loss"]
            per_part[p] = out
        return {"z_q": torch.cat(parts_zq, dim=-1), "loss": loss, "parts": per_part}
