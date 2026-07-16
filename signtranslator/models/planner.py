"""Semantic planner: English tokens -> sign-language gloss sequence.

Sign languages have their own grammar and word order (e.g. topic-comment
structure), so translation is not a word-for-word mapping. This module is a
sequence-to-sequence Transformer that reorders/rewrites a spoken-language token
sequence into a gloss sequence -- the role the spec assigns to an LLM semantic
planner, implemented here as a compact, trainable encoder-decoder that can be
swapped for a large model behind the same interface.

Token convention (gloss/target vocabulary): ``PAD=0``, ``BOS=1``, ``EOS=2``,
content tokens ``>= 3``.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

PAD, BOS, EOS = 0, 1, 2


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def causal_mask(size: int, device=None) -> torch.Tensor:
    """Boolean causal self-attention mask, ``True`` where attention is disallowed.

    Position ``i`` cannot attend to any future position ``j > i`` (upper
    triangle). A boolean mask matches the dtype of the key-padding masks, which
    is the convention modern PyTorch attention expects.
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device),
                      diagonal=1)


class GlossPlanner(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, ff_mult: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = _SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        # Disable the prototype nested-tensor fast path (emits a UserWarning and
        # is unnecessary here). ``use_nested_tensor`` is the attribute the forward
        # pass actually checks.
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False
        self.out = nn.Linear(d_model, tgt_vocab)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    def _embed(self, tokens: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        return self.pos(emb(tokens) * math.sqrt(self.d_model))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """src (N, S), tgt_in (N, U) -> logits (N, U, tgt_vocab)."""
        src_pad = src == PAD
        tgt_pad = tgt_in == PAD
        tmask = causal_mask(tgt_in.size(1), device=tgt_in.device)
        h = self.transformer(
            self._embed(src, self.src_emb),
            self._embed(tgt_in, self.tgt_emb),
            tgt_mask=tmask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(h)

    def loss(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced cross-entropy.

        ``tgt`` includes BOS ... EOS. Input is ``tgt[:, :-1]``; the target to
        predict is ``tgt[:, 1:]`` (shifted left).
        """
        logits = self.forward(src, tgt[:, :-1])
        gold = tgt[:, 1:]
        return self.loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 32) -> List[List[int]]:
        """Autoregressive greedy decoding starting from BOS until EOS/max_len."""
        self.eval()
        n = src.size(0)
        device = src.device
        ys = torch.full((n, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(max_len):
            logits = self.forward(src, ys)
            nxt = logits[:, -1].argmax(-1)          # (N,)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished = finished | (nxt == EOS)
            if bool(finished.all()):
                break
        # Strip BOS and truncate each sequence at its first EOS.
        results: List[List[int]] = []
        for row in ys[:, 1:].tolist():
            seq: List[int] = []
            for tok in row:
                if tok == EOS:
                    break
                if tok != PAD:
                    seq.append(tok)
            results.append(seq)
        return results
