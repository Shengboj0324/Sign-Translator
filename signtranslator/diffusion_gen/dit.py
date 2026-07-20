"""Temporal Diffusion Transformer with adaLN-Zero + cross-attention.

Implements docs/DIFFUSION_GEN.md §4 (Peebles & Xie, arXiv:2212.09748). A
conditioning vector ``c`` (timestep embedding + pooled conditions) drives adaptive
LayerNorm modulation; the modulation MLP is **zero-initialised** so every gate is 0
and each block is the identity at init, and the final head is zero-initialised so
the model outputs 0 at init — a stable training start. Rich sequence conditions
(sign plan / duration / prosody / discourse / style / prior motion) enter by
cross-attention to condition tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..models.denoiser import timestep_embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: ``x·(1+scale) + shift`` (broadcast over the token axis)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """A DiT block: adaLN-Zero self-attention, optional cross-attention, adaLN MLP."""

    def __init__(self, dim: int, cond_dim: int, num_heads: int = 4,
                 cross: bool = True, mlp_mult: int = 4) -> None:
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        if cross:
            self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_mult), nn.GELU(),
                                 nn.Linear(dim * mlp_mult, dim))
        # modulation: 3 groups (SA, CA, MLP) x (shift, scale, gate); CA optional
        self.n_groups = 3 if cross else 2
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, self.n_groups * 3 * dim))
        nn.init.zeros_(self.mod[-1].weight)                  # zero-init -> identity block
        nn.init.zeros_(self.mod[-1].bias)
        self.dim = dim

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        params = self.mod(c).chunk(self.n_groups * 3, dim=-1)
        i = 0
        sh_sa, sc_sa, g_sa = params[i], params[i + 1], params[i + 2]; i += 3
        x = x + g_sa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), sh_sa, sc_sa),
            modulate(self.norm1(x), sh_sa, sc_sa),
            modulate(self.norm1(x), sh_sa, sc_sa), need_weights=False)[0]
        if self.cross and cond_tokens is not None:
            sh_ca, sc_ca, g_ca = params[i], params[i + 1], params[i + 2]; i += 3
            q = modulate(self.norm_ca(x), sh_ca, sc_ca)
            x = x + g_ca.unsqueeze(1) * self.cross_attn(
                q, cond_tokens, cond_tokens, need_weights=False)[0]
        elif self.cross:
            i += 3                                            # skip CA params
        sh_m, sc_m, g_m = params[i], params[i + 1], params[i + 2]
        x = x + g_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh_m, sc_m))
        return x


class TemporalDiT(nn.Module):
    """Temporal DiT denoiser: (x_t, t, conditions) -> prediction (eps/x0/v)."""

    def __init__(self, in_dim: int, dim: int = 128, depth: int = 4,
                 num_heads: int = 4, cond_dim: Optional[int] = None,
                 max_len: int = 2048, cross: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim or dim
        self.in_proj = nn.Linear(in_dim, dim)
        self.register_buffer("pos", _sinusoidal(max_len, dim), persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(dim, self.cond_dim), nn.SiLU(),
                                   nn.Linear(self.cond_dim, self.cond_dim))
        self.cond_proj = nn.Linear(self.cond_dim, self.cond_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, self.cond_dim, num_heads, cross) for _ in range(depth)])
        # final adaLN-Zero head -> output 0 at init
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(self.cond_dim, 2 * dim))
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)
        self.final_linear = nn.Linear(dim, in_dim)
        nn.init.zeros_(self.final_linear.weight); nn.init.zeros_(self.final_linear.bias)

    def condition_vector(self, t: torch.Tensor,
                         cond_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.t_mlp(timestep_embedding(t, self.dim).to(self.t_mlp[0].weight.dtype))
        if cond_vec is not None:
            c = c + self.cond_proj(cond_vec)
        return c

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond_vec: Optional[torch.Tensor] = None,
                cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x_t`` (N, T, in_dim), ``t`` (N,). Returns (N, T, in_dim)."""
        h = self.in_proj(x_t) + self.pos[:x_t.shape[1]].unsqueeze(0).to(x_t.dtype)
        c = self.condition_vector(t, cond_vec)
        for blk in self.blocks:
            h = blk(h, c, cond_tokens)
        shift, scale = self.final_mod(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.final_linear(h)


def _sinusoidal(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float()
                    * (-torch.log(torch.tensor(10000.0)) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
