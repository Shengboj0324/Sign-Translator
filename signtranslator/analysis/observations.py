"""Unpad canonical batches into individual evaluation observations.

Evaluation crops before convolution/pooling, rather than assuming a CTC length
can undo padded feature computation. Internal missing joints remain masked.
"""
from __future__ import annotations

import torch


def _lengths(batch, key, n, maximum):
    values = batch.get(key)
    if values is None:
        return [maximum] * n
    if (not isinstance(values, torch.Tensor) or values.shape != (n,)
            or values.dtype not in (torch.int32, torch.int64)):
        raise ValueError(f'{key} must contain one integer per sample')
    result = values.tolist()
    if any(x < 1 or x > maximum for x in result):
        raise ValueError(f'{key} exceeds available observation support')
    return result


def observations(batch):
    """Yield single-sample batches, with exact CTC targets and temporal support."""
    pose = batch['pose']
    if pose.ndim != 4 or pose.shape[0] == 0:
        raise ValueError('evaluation requires a nonempty NCTV pose batch')
    n = pose.shape[0]
    lengths = _lengths(batch, 'motion_lengths', n, pose.shape[2])
    speech_lengths = (_lengths(batch, 'speech_input_lengths', n, batch['speech'].shape[1])
                      if 'speech' in batch else None)
    target_rows = {}
    for targets, sizes in [('ctc_targets', 'ctc_lengths'),
                           ('speech_ctc_targets', 'speech_ctc_lengths')]:
        if targets in batch:
            from ..models.recognition import _ctc_target_rows
            target_rows[targets] = _ctc_target_rows(batch[targets], batch[sizes])
            if len(target_rows[targets]) != n:
                raise ValueError('CTC target batch differs from observations')
    for i, length in enumerate(lengths):
        row = {}
        for key, value in batch.items():
            if key in target_rows:
                row[key] = torch.tensor(target_rows[key][i], dtype=value.dtype, device=value.device)
            elif isinstance(value, torch.Tensor):
                row[key] = value[i:i+1]
            elif isinstance(value, (list, tuple)):
                row[key] = value[i:i+1]
            else:
                row[key] = value
        row['pose'] = pose[i:i+1, :, :length]
        for key in ('frame_mask', 'validity_mask', 'confidence', 'frame_timestamps'):
            if key in row:
                row[key] = row[key][:, :length]
        for key in ('src', 'gloss_tokens', 'gloss_seq'):
            if key in row:
                tokens = row[key]
                nonpad = tokens[0] != 0
                width = int(nonpad.sum())
                if width == 0 or not bool(nonpad[:width].all()) or bool(nonpad[width:].any()):
                    raise ValueError(f'{key} requires nonempty content followed only by padding')
                row[key] = tokens[:, :width]
        if speech_lengths is not None:
            row['speech'] = row['speech'][:, :speech_lengths[i]]
        yield row
