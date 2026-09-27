"""Observed-support batch normalization using existing BatchNorm parameters.

Forward variance is the supported population variance. Running variance uses the
usual n/(n-1) correction only when at least two observations exist. Unsupported
channels retain their running statistics; singleton channels update mean only.
"""
from __future__ import annotations

import torch
from torch import nn


def masked_batch_norm(x: torch.Tensor, valid: torch.Tensor,
                      module: nn.modules.batchnorm._BatchNorm) -> torch.Tensor:
    """Normalize only supported entries; preserve zero payload off support.

    ``valid`` must already have exactly the input shape. Fixed-momentum tracked
    BatchNorm is required, as used by ST-GCN; this is not a generic SyncBatchNorm.
    """
    if valid.shape != x.shape or valid.dtype != torch.bool or valid.device != x.device:
        raise ValueError("normalization support must be boolean and match input")
    if x.ndim not in (3, 4) or x.shape[1] != module.num_features:
        raise ValueError("normalization input must match BatchNorm features")
    if not module.track_running_stats or module.momentum is None:
        raise ValueError("masked normalization requires fixed-momentum tracked BatchNorm")
    if bool(valid.all()) and (not module.training or x.numel() // x.shape[1] > 1):
        return module(x)
    dims = (0,) + tuple(range(2, x.ndim))
    shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    # Accumulate low-precision input moments in float32; preserve float64 input.
    work = x if x.dtype == torch.float64 else x.float()
    safe = torch.where(valid, work, 0)
    if module.training:
        count = valid.sum(dim=dims)
        denominator = count.clamp_min(1).to(work.dtype)
        mean = safe.sum(dim=dims) / denominator
        centered = torch.where(valid, work - mean.reshape(shape), 0)
        variance = centered.square().sum(dim=dims) / denominator
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            mean_update = torch.lerp(module.running_mean, mean.to(module.running_mean),
                                     module.momentum)
            unbiased = variance * count / (count - 1).clamp_min(1)
            var_update = torch.lerp(module.running_var, unbiased.to(module.running_var),
                                    module.momentum)
            module.running_mean.copy_(torch.where(count > 0, mean_update, module.running_mean))
            module.running_var.copy_(torch.where(count > 1, var_update, module.running_var))
    else:
        mean = module.running_mean.to(work)
        variance = module.running_var.to(work)
    out = (safe - mean.reshape(shape)) * torch.rsqrt(variance.reshape(shape) + module.eps)
    if module.affine:
        out = out * module.weight.to(work).reshape(shape) + module.bias.to(work).reshape(shape)
    return torch.where(valid, out.to(x.dtype), 0)
