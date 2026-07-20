"""Decoding strategies over motion tokens + the experiment harness.

See docs/MOTION_TRANSFORMER.md §8. Three ways to model the discrete motion-token
prior:

* ``MotionTokenGPT`` — autoregressive (causal) next-token prediction (T2M-GPT).
* ``MaskedMotionModel`` — masked-span prediction with bidirectional context
  (MotionGPT / MaskGIT style).
* diffusion decoding reuses the Doc-01/04 Gaussian motion diffusion over the
  continuous latents (imported where needed; no re-implementation here).

``compare_reconstruction`` is the honest experiment scaffold for the document's
"raw vs VQ vs residual-VQ" and "shared vs part-specific" comparisons.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import causal_mask
from .residual_vq import ResidualVQ, PartitionedVQ


class _SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class MotionTokenGPT(nn.Module):
    """Autoregressive prior over motion code indices (causal Transformer)."""

    def __init__(self, num_codes: int, dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.bos = num_codes                                 # extra start token
        self.emb = nn.Embedding(num_codes + 1, dim)
        self.pos = _SinusoidalPositions(dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(dim, num_codes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, L) code indices -> (N, L, num_codes) next-token logits (causal)."""
        h = self.pos(self.emb(tokens))
        mask = causal_mask(tokens.shape[1], tokens.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next-token CE: predict tokens[:,1:] from tokens[:,:-1]."""
        N, L = tokens.shape
        bos = torch.full((N, 1), self.bos, dtype=torch.long, device=tokens.device)
        inp = torch.cat([bos, tokens[:, :-1]], dim=1)        # shift right
        logits = self.forward(inp)
        return F.cross_entropy(logits.reshape(-1, self.num_codes), tokens.reshape(-1))

    @torch.no_grad()
    def generate(self, length: int, batch: int = 1,
                 generator: Optional[torch.Generator] = None,
                 device=None) -> torch.Tensor:
        self.eval()
        tokens = torch.full((batch, 1), self.bos, dtype=torch.long, device=device)
        for _ in range(length):
            logits = self.forward(tokens)[:, -1]             # (N, num_codes)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[:, 1:]                                 # drop BOS


class MaskedMotionModel(nn.Module):
    """Masked-span prediction over motion tokens (bidirectional context)."""

    def __init__(self, num_codes: int, dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.mask_token = num_codes
        self.emb = nn.Embedding(num_codes + 1, dim)
        self.pos = _SinusoidalPositions(dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Linear(dim, num_codes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.emb(tokens))
        return self.head(self.encoder(h))                    # no causal mask (bidirectional)

    @staticmethod
    def make_masked_span(tokens: torch.Tensor, mask_token: int,
                         span_frac: float = 0.3,
                         generator: Optional[torch.Generator] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace one contiguous span per sequence with ``mask_token``.

        Returns (masked_input, mask) where ``mask`` (N, L) bool marks the masked
        positions (the ones supervised)."""
        N, L = tokens.shape
        span = max(1, int(round(span_frac * L)))
        masked = tokens.clone()
        mask = torch.zeros(N, L, dtype=torch.bool, device=tokens.device)
        for i in range(N):
            start = int(torch.randint(0, L - span + 1, (1,), generator=generator))
            masked[i, start:start + span] = mask_token
            mask[i, start:start + span] = True
        return masked, mask

    def loss(self, tokens: torch.Tensor, span_frac: float = 0.3,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """CE on the masked positions ONLY (unmasked positions are not supervised)."""
        masked, mask = self.make_masked_span(tokens, self.mask_token, span_frac, generator)
        logits = self.forward(masked)
        per = F.cross_entropy(logits.reshape(-1, self.num_codes),
                              tokens.reshape(-1), reduction="none")
        m = mask.reshape(-1).to(per.dtype)
        return (per * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# experiment harness
# ---------------------------------------------------------------------------
def compare_reconstruction(data: torch.Tensor, dim: int, num_codes: int = 64,
                           rvq_stages: int = 4, seed: int = 0) -> Dict[str, float]:
    """Relative reconstruction error of raw vs VQ vs residual-VQ, at a fixed
    per-stage codebook size. Raw (no quantisation) is the 0-error, full-bitrate
    reference; more RVQ stages should not do worse than a single VQ stage."""
    data = data.reshape(-1, dim)
    denom = float(data.norm()) + 1e-9

    vq = ResidualVQ(1, num_codes, dim, ema=True).double()
    vq.init_from_data(data, seed=seed); vq.eval()
    err_vq = float((vq(data)["z_q"] - data).norm()) / denom

    rvq = ResidualVQ(rvq_stages, num_codes, dim, ema=True).double()
    rvq.init_from_data(data, seed=seed); rvq.eval()
    err_rvq = float((rvq(data)["z_q"] - data).norm()) / denom

    return {"raw": 0.0, "vq": err_vq, "rvq": err_rvq}


def compare_shared_vs_part(data: torch.Tensor, part_dims: Dict[str, int],
                           num_codes: int = 64, seed: int = 0) -> Dict[str, float]:
    """Relative reconstruction error of a shared codebook vs part-specific
    codebooks at the same per-part stage count."""
    total_dim = sum(part_dims.values())
    data = data.reshape(-1, total_dim)
    denom = float(data.norm()) + 1e-9

    shared = ResidualVQ(1, num_codes, total_dim, ema=True).double()
    shared.init_from_data(data, seed=seed); shared.eval()
    err_shared = float((shared(data)["z_q"] - data).norm()) / denom

    part = PartitionedVQ(part_dims, num_stages=1, num_codes=num_codes, ema=True).double()
    part.init_from_data(data, seed=seed); part.eval()
    err_part = float((part(data)["z_q"] - data).norm()) / denom

    return {"shared": err_shared, "part_specific": err_part}
