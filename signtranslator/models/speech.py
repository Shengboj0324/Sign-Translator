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
from .recognition import ctc_greedy_decode


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
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """features (N, T, F) -> hidden (N, T', H) with T' = T / subsample."""
        if features.dim() != 3:
            raise ValueError("features must be (N, T, F)")
        h = self.subsampler(features.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(self.pos(h))
        return self.norm(h)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (N, T, F) -> per-frame log-probs (N, T', num_classes)."""
        return F.log_softmax(self.classifier(self.encode(features)), dim=-1)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Frame counts after convolutional subsampling (stride-2, padding-1)."""
        lengths = input_lengths
        stride_left = self.subsample
        while stride_left > 1:
            lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
            stride_left //= 2
        return lengths

    def loss(self, features: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor,
             input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        log_probs = self.forward(features)
        n, t_out, _ = log_probs.shape
        if input_lengths is None:
            out_lengths = torch.full((n,), t_out, dtype=torch.long,
                                     device=log_probs.device)
        else:
            out_lengths = self.output_lengths(input_lengths).clamp(max=t_out)
        return self.ctc(log_probs.permute(1, 0, 2), targets, out_lengths,
                        target_lengths)

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> List[List[int]]:
        self.eval()
        return ctc_greedy_decode(self.forward(features))
