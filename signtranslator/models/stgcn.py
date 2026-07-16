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


class GraphConvolution(nn.Module):
    r"""Partitioned spatial graph convolution.

    Implements :math:`\sum_{k=1}^{K} A_k X W_k`. We realise the ``K`` linear
    maps ``W_k`` with a single ``1x1`` convolution producing ``K * C_out``
    channels, then contract against the adjacency with an ``einsum``. The
    adjacency is registered as a buffer so it moves with ``.to(device)`` but is
    excluded from gradient updates.
    """

    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray) -> None:
        super().__init__()
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape (K, V, V)")
        self.num_partitions = adjacency.shape[0]
        self.out_channels = out_channels
        self.register_buffer("A", torch.as_tensor(adjacency, dtype=torch.float32))
        # One 1x1 conv emitting K*out_channels, reshaped to (N, K, C_out, T, V).
        self.theta = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, T, V)
        n, _, t, v = x.shape
        if v != self.A.shape[-1]:
            raise ValueError(f"joint dim {v} != adjacency V {self.A.shape[-1]}")
        feat = self.theta(x)  # (N, K*C_out, T, V)
        feat = feat.view(n, self.num_partitions, self.out_channels, t, v)
        # Contract joints with each partition adjacency and sum over partitions.
        #   out[n,c,t,w] = sum_{k,v} feat[n,k,c,t,v] * A[k,v,w]
        out = torch.einsum("nkctv,kvw->nctw", feat, self.A)
        return out.contiguous()


class STGCNBlock(nn.Module):
    """Graph conv + temporal conv with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray,
                 temporal_kernel: int = 9, stride: int = 1, dropout: float = 0.0,
                 residual: bool = True) -> None:
        super().__init__()
        assert temporal_kernel % 2 == 1, "temporal kernel must be odd for 'same' padding"
        pad = (temporal_kernel - 1) // 2

        self.gcn = GraphConvolution(in_channels, out_channels, adjacency)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = 0.0 if self.residual is None else self.residual(x)
        y = self.act(self.gcn_bn(self.gcn(x)))
        y = self.tcn(y)
        return self.act(y + res)


class STGCNEncoder(nn.Module):
    """Stack of ST-GCN blocks + global pooling -> motion embedding."""

    def __init__(self, in_channels: int, adjacency: np.ndarray,
                 channels: Sequence[int] = (64, 128, 256),
                 temporal_kernel: int = 9, num_joints: int | None = None) -> None:
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
                           residual=(i > 0))
            )
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C_in, T, V) -> embedding (N, out_dim)."""
        if x.ndim != 4:
            raise ValueError("expected input of shape (N, C, T, V)")
        n, c, t, v = x.shape
        # Data batch-norm over joint-channels (standard ST-GCN preprocessing).
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()  # back to (N,C,T,V)

        for block in self.blocks:
            x = block(x)

        # Global average pool over time and joints.
        x = x.mean(dim=(2, 3))  # (N, out_dim)
        return x
