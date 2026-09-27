"""Spatio-temporal graph convolutional encoder (pose sequence -> motion embedding).

Given a clip of 3D keypoints ``X`` with shape ``(N, C_in, T, V)`` (batch,
channels, frames, joints) the encoder produces a single fixed-size *motion
embedding* per clip. Each ST-GCN block factorises spatio-temporal convolution
into:

    1. a **graph** convolution over joints using the partitioned adjacency
       ``A in R^{K x V x V}``:   f_out(v) = sum_k  (A_k  X W_k)  ,
    2. a **temporal** convolution (1D conv along the frame axis).

Residual connections and batch-norm follow Yan et al. (2018). The final block's
features are globally pooled over time and joints to yield the embedding.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .masked_normalization import masked_batch_norm


class GraphConvolution(nn.Module):
    r"""Partitioned spatial graph convolution.

    Implements :math:`\sum_{k=1}^{K} A_k X W_k`. We realise the ``K`` linear
    maps ``W_k`` with a single ``1x1`` convolution producing ``K * C_out``
    channels, then contract against the adjacency with an ``einsum``. The
    adjacency is registered as a buffer so it moves with ``.to(device)`` but is
    excluded from gradient updates.
    """

    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray,
                 adaptive: bool = False) -> None:
        super().__init__()
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape (K, V, V)")
        self.num_partitions = adjacency.shape[0]
        self.out_channels = out_channels
        self.register_buffer("A", torch.as_tensor(adjacency, dtype=torch.float32))
        # One 1x1 conv emitting K*out_channels, reshaped to (N, K, C_out, T, V).
        self.theta = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)

        # Optional **learnable adjacency refinement** (CTR-GCN / 2s-AGCN idea):
        # the anatomical skeleton is not the only useful topology -- signing
        # couples joints that share no bone (e.g. the two hands during a
        # two-handed sign). A residual, zero-initialised term lets the model
        # learn such edges while starting exactly at the anatomical prior.
        self.adaptive = adaptive
        if adaptive:
            self.A_refine = nn.Parameter(torch.zeros_like(self.A))

    def effective_adjacency(self) -> torch.Tensor:
        """Anatomical adjacency plus (if enabled) the learned refinement."""
        return self.A + self.A_refine if self.adaptive else self.A

    def forward(self, x: torch.Tensor, validity_mask=None) -> torch.Tensor:
        # x: (N, C_in, T, V)
        n, _, t, v = x.shape
        if v != self.A.shape[-1]:
            raise ValueError(f"joint dim {v} != adjacency V {self.A.shape[-1]}")
        if validity_mask is not None:
            x = torch.where(validity_mask.unsqueeze(1), x, 0)
        feat = self.theta(x)  # (N, K*C_out, T, V)
        if validity_mask is not None:
            # Exclude learned bias at unavailable source joints as well as values.
            feat = torch.where(validity_mask.unsqueeze(1), feat, 0)
        feat = feat.view(n, self.num_partitions, self.out_channels, t, v)
        # Contract joints with each partition adjacency and sum over partitions.
        #   out[n,c,t,w] = sum_{k,v} feat[n,k,c,t,v] * A[k,v,w]
        out = torch.einsum("nkctv,kvw->nctw", feat, self.effective_adjacency())
        if validity_mask is not None:
            out = torch.where(validity_mask.unsqueeze(1), out, 0)
        return out.contiguous()


class STGCNBlock(nn.Module):
    """Graph conv + temporal conv with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray,
                 temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.0,
                 residual: bool = True, adaptive: bool = False) -> None:
        super().__init__()
        assert temporal_kernel % 2 == 1, "temporal kernel must be odd for 'same' padding"
        pad = (temporal_kernel - 1) // 2

        self.gcn = GraphConvolution(in_channels, out_channels, adjacency, adaptive=adaptive)
        self.gcn_bn = nn.BatchNorm2d(out_channels)

        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.act = nn.ReLU(inplace=True)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, validity_mask=None) -> torch.Tensor:
        if validity_mask is None:
            res = 0.0 if self.residual is None else self.residual(x)
            y = self.act(self.gcn_bn(self.gcn(x)))
            return self.act(self.tcn(y) + res)
        if self.tcn[0].stride != (1, 1):
            raise ValueError("masked ST-GCN currently requires stride one")
        support = validity_mask.unsqueeze(1)
        x = torch.where(support, x, 0)
        if self.residual is None:
            res = 0.0
        elif isinstance(self.residual, nn.Identity):
            res = x
        else:
            res = self.residual[0](x)
            res = masked_batch_norm(res, support.expand_as(res), self.residual[1])
        y = self.gcn(x, validity_mask)
        y = self.act(masked_batch_norm(y, support.expand_as(y), self.gcn_bn))
        y = self.tcn[0](y)
        y = masked_batch_norm(y, support.expand_as(y), self.tcn[1])
        y = self.tcn[2](y)
        return torch.where(support, self.act(y + res), 0)


class STGCNEncoder(nn.Module):
    """Stack of ST-GCN blocks + global pooling -> motion embedding."""

    def __init__(self, in_channels: int, adjacency: np.ndarray,
                 channels: Sequence[int] = (64, 128, 256),
                 temporal_kernel: int = 9, num_joints: int | None = None,
                 adaptive: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_joints = num_joints if num_joints is not None else adjacency.shape[-1]
        # Normalise raw coordinates across the (C*V) feature dimension.
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_joints)

        blocks = []
        prev = in_channels
        for i, ch in enumerate(channels):
            blocks.append(
                STGCNBlock(prev, ch, adjacency, temporal_kernel=temporal_kernel,
                           residual=(i > 0), adaptive=adaptive)
            )
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor, return_sequence: bool = False, *,
                validity_mask=None, frame_mask=None, confidence=None) -> torch.Tensor:
        """Encode a pose clip.

        Args:
            x: ``(N, C_in, T, V)`` pose sequence.
            return_sequence: if ``True`` return per-frame features
                ``(N, T, out_dim)`` (pooled over joints only) for sequence
                decoding (e.g. CTC recognition); otherwise return the clip
                embedding ``(N, out_dim)`` (pooled over time and joints).
        """
        if x.ndim != 4:
            raise ValueError("expected input of shape (N, C, T, V)")
        n, c, t, v = x.shape
        if c != self.in_channels or v != self.num_joints:
            raise ValueError("pose channels/joints differ from encoder layout")
        valid = None
        if any(mask is not None for mask in (validity_mask, frame_mask, confidence)):
            from .diffusion import GaussianMotionDiffusion
            x, weights = GaussianMotionDiffusion.motion_support(
                x, validity_mask=validity_mask, frame_mask=frame_mask, confidence=confidence)
            valid = weights[:, 0] > 0
            if frame_mask is not None and bool((frame_mask[:, 1:] & ~frame_mask[:, :-1]).any()):
                raise ValueError("frame_mask must be a contiguous valid prefix")
        # Data batch-norm over joint-channels (standard ST-GCN preprocessing).
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        if valid is None:
            x = self.data_bn(x)
        else:
            bn_support = valid.permute(0, 2, 1).unsqueeze(2).expand(n, v, c, t)
            x = masked_batch_norm(x, bn_support.reshape(n, v * c, t), self.data_bn)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()  # back to (N,C,T,V)

        for block in self.blocks:
            x = block(x, valid)  # temporal length preserved (stride 1)

        if valid is not None:
            counts = valid.sum(dim=-1)
            seq = (x.sum(dim=-1) / counts.clamp_min(1).unsqueeze(1)).transpose(1, 2)
            if return_sequence:
                return seq.contiguous()
            frames = counts > 0
            return seq.sum(dim=1) / frames.sum(dim=1).unsqueeze(-1)
        if return_sequence:
            return x.mean(dim=3).transpose(1, 2).contiguous()  # (N, T, out_dim)
        return x.mean(dim=(2, 3))  # (N, out_dim)
