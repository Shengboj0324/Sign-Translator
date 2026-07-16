"""Transformer denoiser epsilon_theta(x_t, t, c) for motion diffusion.

The denoiser predicts the Gaussian noise added to a motion clip. It operates on
a clip flattened to ``(N, T, V*C)`` frame-feature vectors and is conditioned on

    * the diffusion timestep ``t`` via a sinusoidal timestep embedding, and
    * the language latent ``c`` (the shared-manifold conditioning vector).

Both conditioning signals are injected additively into every frame token before
a stack of Transformer encoder layers, an established and stable design for
motion diffusion models (cf. MDM, Tevet et al., 2023).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of integer diffusion timesteps (as in DDPM/Transformers)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (N, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:  # pad odd dims
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class MotionDenoiser(nn.Module):
    """Predicts noise for a motion clip of shape (N, C, T, V)."""

    def __init__(self, num_joints: int, in_channels: int, cond_dim: int,
                 hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, max_frames: int = 512) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.frame_dim = num_joints * in_channels
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.frame_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.frame_dim)
        # Zero-init the output so the model starts near an identity noise-predictor,
        # a common stabiliser for diffusion training.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, C, T, V), t: (N,), cond: (N, cond_dim) -> eps_hat (N, C, T, V)."""
        n, c, T, v = x.shape
        if (c, v) != (self.in_channels, self.num_joints):
            raise ValueError("channel/joint mismatch with denoiser configuration")

        tokens = x.permute(0, 2, 3, 1).reshape(n, T, v * c)  # (N, T, V*C)
        h = self.input_proj(tokens) + self.pos_emb[:, :T]

        t_emb = self.time_mlp(timestep_embedding(t, self.hidden_dim)).unsqueeze(1)  # (N,1,H)
        h = h + t_emb
        if cond is not None:
            h = h + self.cond_proj(cond).unsqueeze(1)

        h = self.transformer(h)
        h = self.output_proj(self.out_norm(h))  # (N, T, V*C)
        return h.reshape(n, T, v, c).permute(0, 3, 1, 2).contiguous()


class CrossModalDenoiser(nn.Module):
    """Denoiser that **cross-attends** the full language token sequence.

    Instead of conditioning on a single pooled latent, each frame token attends
    (via Transformer cross-attention) to every gloss/word token, letting the
    generator align sub-parts of the motion with sub-parts of the sentence.

    For classifier-free guidance a learned ``null_token`` provides the
    unconditional context; it is always present as an extra memory slot so a
    dropped sample still has a valid key to attend to.

    Conditioning interface: ``cond`` is either ``None`` (fully unconditional) or
    a tuple ``(memory, mask)`` with ``memory`` of shape ``(N, L, context_dim)``
    and ``mask`` of shape ``(N, L)`` (``True`` = valid token). ``drop`` is an
    optional ``(N,)`` boolean vector selecting per-sample unconditional context.
    """

    def __init__(self, num_joints: int, in_channels: int, context_dim: int,
                 hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 ff_mult: int = 4, dropout: float = 0.1, max_frames: int = 512) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.frame_dim = num_joints * in_channels
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(self.frame_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.null_token, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.frame_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_memory(self, n: int, device, cond, drop):
        """Return (memory (N, 1+L, H), key_padding_mask (N, 1+L) True=ignore)."""
        null = self.null_token.expand(n, 1, self.hidden_dim)
        if cond is None:
            valid = torch.ones(n, 1, dtype=torch.bool, device=device)
            return null, ~valid
        memory, mask = cond
        if mask is None:
            mask = torch.ones(memory.shape[:2], dtype=torch.bool, device=device)
        mem = torch.cat([null, self.context_proj(memory)], dim=1)       # (N, 1+L, H)
        valid = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=device),
                           mask.bool()], dim=1)                          # (N, 1+L)
        if drop is not None:
            # Dropped samples keep only the null slot (index 0).
            valid = valid.clone()
            valid[drop, 1:] = False
        return mem, ~valid

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond=None,
                drop: torch.Tensor | None = None) -> torch.Tensor:
        n, c, T, v = x.shape
        if (c, v) != (self.in_channels, self.num_joints):
            raise ValueError("channel/joint mismatch with denoiser configuration")

        tokens = x.permute(0, 2, 3, 1).reshape(n, T, v * c)
        h = self.input_proj(tokens) + self.pos_emb[:, :T]
        h = h + self.time_mlp(timestep_embedding(t, self.hidden_dim)).unsqueeze(1)

        mem, key_padding = self._build_memory(n, x.device, cond, drop)
        h = self.decoder(h, mem, memory_key_padding_mask=key_padding)
        h = self.output_proj(self.out_norm(h))
        return h.reshape(n, T, v, c).permute(0, 3, 1, 2).contiguous()
