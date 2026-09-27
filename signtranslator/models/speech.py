"""Speech -> spoken-token recognition (the acoustic front-end).

Completes the speech-to-sign path. Acoustic features (mel filterbanks, or the
hidden states of a speech foundation model such as Whisper / wav2vec 2.0) are
encoded and decoded to a spoken token sequence with CTC, exactly mirroring the
sign-recognition branch. The recognised tokens then feed the ``GlossPlanner``,
which reorders them into gloss, which conditions motion generation:

    audio features -> [SpeechRecognizer/CTC] -> spoken tokens
                   -> [GlossPlanner]         -> gloss tokens
                   -> [GuidedMotionDiffusion]-> 3D signing motion

A convolutional stack subsamples the acoustic frame rate before the Transformer
(standard practice: audio frame rates are far higher than token rates, and
striding cuts attention cost quadratically) while keeping the CTC input length
comfortably above the target length.

Convention: class index ``0`` is the CTC blank; spoken ids occupy ``1..V``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import _SinusoidalPositionalEncoding
from .recognition import assert_ctc_feasible, ctc_greedy_decode


class SpeechRecognizer(nn.Module):
    """Conv subsampling + Transformer encoder + CTC head over audio features."""

    def __init__(self, input_dim: int, num_tokens: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, ff_mult: int = 4,
                 dropout: float = 0.1, subsample: int = 2) -> None:
        super().__init__()
        if subsample not in (1, 2, 4):
            raise ValueError("subsample must be 1, 2 or 4")
        self.subsample = subsample
        self.num_tokens = num_tokens
        self.num_classes = num_tokens + 1          # +1 for blank (index 0)

        layers: List[nn.Module] = []
        in_ch = input_dim
        stride_left = subsample
        while stride_left > 1:
            layers += [nn.Conv1d(in_ch, hidden_dim, kernel_size=3, stride=2, padding=1),
                       nn.GELU()]
            in_ch = hidden_dim
            stride_left //= 2
        if not layers:
            layers = [nn.Conv1d(in_ch, hidden_dim, kernel_size=3, stride=1, padding=1),
                      nn.GELU()]
        self.subsampler = nn.Sequential(*layers)

        self.pos = _SinusoidalPositionalEncoding(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.num_classes)
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=False)

    def _input_lengths(self, features: torch.Tensor,
                       input_lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if (features.ndim != 3 or not features.is_floating_point()
                or any(d < 1 for d in features.shape)
                or features.shape[2] != self.subsampler[0].in_channels):
            raise ValueError("features must be nonempty floating (N,T,F) matching input_dim")
        n, frames, _ = features.shape
        if input_lengths is None:
            return torch.full((n,), frames, dtype=torch.long, device=features.device)
        if (not torch.is_tensor(input_lengths)
                or input_lengths.dtype not in (torch.int32, torch.int64)
                or input_lengths.shape != (n,)):
            raise ValueError("speech input_lengths must be an integer (N,) tensor")
        lengths = input_lengths.to(features.device)
        if bool(((lengths < 1) | (lengths > frames)).any()):
            raise ValueError("speech input_lengths must be within the supplied frame range")
        return lengths

    def encode(self, features: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode valid acoustic prefixes; output length is ceil(T/subsample).

        Exclude padded inputs before convolution and padded hidden values after
        every strided convolution, before they can enter another receptive field.
        """
        lengths = self._input_lengths(features, input_lengths)
        valid = torch.arange(features.shape[1], device=features.device)[None] < lengths[:, None]
        if not bool(torch.isfinite(features[valid]).all()):
            raise ValueError("valid acoustic features must be finite")
        h = torch.where(valid.unsqueeze(-1), features, 0).transpose(1, 2)
        for layer in self.subsampler:
            h = layer(h)
            if isinstance(layer, nn.Conv1d):
                stride = layer.stride[0]
                lengths = torch.div(lengths - 1, stride, rounding_mode="floor") + 1
                valid = torch.arange(h.shape[2], device=h.device)[None] < lengths[:, None]
            h = torch.where(valid.unsqueeze(1), h, 0)
        h = h.transpose(1, 2)
        if h.shape[1] > self.pos.pe.shape[1]:
            raise ValueError("speech exceeds encoder positional capacity")
        h = self.encoder(self.pos(h), src_key_padding_mask=~valid)
        return torch.where(valid.unsqueeze(-1), self.norm(h), 0)

    def forward(self, features: torch.Tensor,
                input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return log-probabilities; consumers must respect output_lengths."""
        return F.log_softmax(self.classifier(self.encode(features, input_lengths)), dim=-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Positive frame counts after same-padded convolutional subsampling."""
        if (not torch.is_tensor(input_lengths)
                or input_lengths.dtype not in (torch.int32, torch.int64)
                or input_lengths.ndim != 1 or input_lengths.numel() < 1
                or bool((input_lengths < 1).any())):
            raise ValueError("speech lengths must be a nonempty positive integer vector")
        lengths = input_lengths
        stride_left = self.subsample
        while stride_left > 1:
            # Equivalent to ceil(L/2), without overflowing L+1 at integer max.
            lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
            stride_left //= 2
        return lengths

    def loss(self, features: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor,
             input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        lengths = self._input_lengths(features, input_lengths)
        log_probs = self.forward(features, lengths)
        return self.loss_from_log_probs(
            log_probs, targets, target_lengths, self.output_lengths(lengths))

    def loss_from_log_probs(self, log_probs: torch.Tensor, targets: torch.Tensor,
                            target_lengths: torch.Tensor,
                            output_lengths: torch.Tensor) -> torch.Tensor:
        if log_probs.ndim != 3 or log_probs.shape[2] != self.num_classes:
            raise ValueError("speech CTC log-probabilities have an invalid shape")
        if not torch.isfinite(log_probs).all():
            raise FloatingPointError("speech CTC log-probabilities are non-finite")
        assert_ctc_feasible(
            targets, target_lengths, output_lengths,
            num_classes=self.num_classes,
            maximum_input_length=log_probs.shape[1],
            expected_batch_size=log_probs.shape[0],
        )
        loss = self.ctc(log_probs.permute(1, 0, 2), targets, output_lengths,
                        target_lengths)
        if not torch.isfinite(loss):
            raise FloatingPointError("speech CTC loss is non-finite")
        return loss

    @torch.no_grad()
    def decode(self, features: torch.Tensor,
               input_lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        self.eval()
        lengths = self._input_lengths(features, input_lengths)
        log_probs = self.forward(features, lengths)
        return [ctc_greedy_decode(log_probs[i:i + 1, :length])[0]
                for i, length in enumerate(self.output_lengths(lengths).tolist())]
