"""The semantic planner: evidence -> typed sign plan.

A compact encoder-decoder. Evidence enters two ways, both named in the
specification: resampled **acoustic states projected as prefix tokens** (each
acoustic frame -> one encoder token), and optional **transcript token ids**. The
decoder autoregressively emits the plan serialization and is constrained at
generation time by the schema automaton, so it can only ever produce a
well-formed skeleton.

Training minimises the plan NLL

    L_plan = - sum_i log p_theta(s_i | s_{<i}, x)

teacher-forced, with padding masked out.

Vocabulary layout of the decoder:
    ids [0, V)         plan tokens (what the automaton emits; the output head)
    id  V              SOS  (decoder start; input only, never predicted)
    id  V + 1          PAD  (batch padding; ignored by the loss)

Note (carried over from the gloss planner): token embeddings are **not** scaled
by sqrt(d_model). With fixed sinusoidal positions, that scaling drowns the
positional signal and cripples the alignment an autoregressive copy needs.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import PlanVocabulary, DEFAULT_VOCAB
from .automaton import SchemaAutomaton
from .constrained import ConstrainedDecoder


class _SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SemanticPlanner(nn.Module):
    def __init__(self, vocab: PlanVocabulary = DEFAULT_VOCAB,
                 acoustic_dim: Optional[int] = 64, src_vocab: Optional[int] = None,
                 d_model: int = 128, nhead: int = 4, num_encoder_layers: int = 2,
                 num_decoder_layers: int = 3, ff_mult: int = 4,
                 dropout: float = 0.1, max_plan_len: int = 256) -> None:
        super().__init__()
        if acoustic_dim is None and src_vocab is None:
            raise ValueError("the planner needs at least one evidence source "
                             "(acoustic_dim or src_vocab)")
        self.vocab = vocab
        self.plan_vocab_size = vocab.size
        self.sos_id = vocab.size
        self.pad_id = vocab.size + 1
        self.d_model = d_model
        self.max_plan_len = max_plan_len

        self.acoustic_proj = (nn.Linear(acoustic_dim, d_model)
                              if acoustic_dim is not None else None)
        self.src_embed = (nn.Embedding(src_vocab, d_model)
                          if src_vocab is not None else None)
        self.enc_pos = _SinusoidalPositions(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers,
                                             enable_nested_tensor=False)

        # +2 for SOS and PAD input tokens.
        self.tgt_embed = nn.Embedding(vocab.size + 2, d_model, padding_idx=self.pad_id)
        self.dec_pos = _SinusoidalPositions(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * ff_mult,
            dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab.size)      # output over plan tokens only

        self.automaton = SchemaAutomaton(vocab)
        self.constrained = ConstrainedDecoder(self.automaton, vocab)

    # -- encoding -----------------------------------------------------------
    def encode(self, acoustic: Optional[torch.Tensor] = None,
               src_tokens: Optional[torch.Tensor] = None):
        """Build encoder memory and its padding mask from the evidence.

        Returns ``(memory, memory_key_padding_mask)``; the mask is ``None`` when
        no source padding is supplied (acoustic frames are assumed all valid).
        """
        parts: List[torch.Tensor] = []
        if acoustic is not None:
            if self.acoustic_proj is None:
                raise ValueError("planner has no acoustic pathway")
            if acoustic.dim() != 3:
                raise ValueError("acoustic must be (N, T_a, d_acoustic)")
            parts.append(self.acoustic_proj(acoustic))     # (N, T_a, d_model)
        if src_tokens is not None:
            if self.src_embed is None:
                raise ValueError("planner has no source-token pathway")
            parts.append(self.src_embed(src_tokens))       # (N, S, d_model)
        if not parts:
            raise ValueError("no evidence supplied to encode")
        x = torch.cat(parts, dim=1)
        memory = self.encoder(self.enc_pos(x))
        return memory, None

    # -- teacher-forced forward --------------------------------------------
    def forward(self, plan_tokens: torch.Tensor,
                acoustic: Optional[torch.Tensor] = None,
                src_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits ``(N, L, V)`` for each target position under teacher forcing.

        ``plan_tokens`` is the target serialization ``(N, L)`` (BOP..EOP), padded
        with ``pad_id``. The decoder input is ``[SOS] + plan_tokens[:, :-1]``.
        """
        if plan_tokens.dim() != 2:
            raise ValueError("plan_tokens must be (N, L)")
        n, L = plan_tokens.shape
        memory, mem_pad = self.encode(acoustic, src_tokens)

        sos = torch.full((n, 1), self.sos_id, dtype=torch.long,
                         device=plan_tokens.device)
        dec_in = torch.cat([sos, plan_tokens[:, :-1]], dim=1)   # (N, L)
        tgt_pad = dec_in == self.pad_id
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                       device=plan_tokens.device), diagonal=1)
        h = self.dec_pos(self.tgt_embed(dec_in))
        h = self.decoder(h, memory, tgt_mask=causal,
                         tgt_key_padding_mask=tgt_pad,
                         memory_key_padding_mask=mem_pad)
        return self.out(h)

    def plan_nll(self, plan_tokens: torch.Tensor,
                 acoustic: Optional[torch.Tensor] = None,
                 src_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Teacher-forced plan NLL, averaging over non-pad target positions."""
        logits = self.forward(plan_tokens, acoustic, src_tokens)
        return F.cross_entropy(logits.reshape(-1, self.plan_vocab_size),
                               plan_tokens.reshape(-1), ignore_index=self.pad_id)

    # -- constrained generation --------------------------------------------
    @torch.no_grad()
    def generate(self, acoustic: Optional[torch.Tensor] = None,
                 src_tokens: Optional[torch.Tensor] = None,
                 sample: bool = False, temperature: float = 1.0,
                 generator: Optional[torch.Generator] = None) -> List[int]:
        """Constrained-decode a single plan (batch size must be 1).

        Returns the plan token ids (BOP..EOP). The schema automaton guarantees a
        well-formed skeleton regardless of the model's raw logits.
        """
        self.eval()
        memory, mem_pad = self.encode(acoustic, src_tokens)
        if memory.shape[0] != 1:
            raise ValueError("generate() decodes one example at a time")
        device = memory.device

        def logits_fn(prefix: List[int]) -> torch.Tensor:
            dec_in = torch.tensor([[self.sos_id] + list(prefix)], dtype=torch.long,
                                  device=device)
            L = dec_in.shape[1]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device),
                                diagonal=1)
            h = self.dec_pos(self.tgt_embed(dec_in))
            h = self.decoder(h, memory, tgt_mask=causal,
                             memory_key_padding_mask=mem_pad)
            return self.out(h)[0, -1].double()

        if sample:
            return self.constrained.sample_decode(
                logits_fn, max_length=self.max_plan_len,
                temperature=temperature, generator=generator)
        return self.constrained.greedy_decode(logits_fn, max_length=self.max_plan_len)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pad_plan_batch(token_lists: List[List[int]], pad_id: int
                   ) -> torch.Tensor:
    """Right-pad a list of serialized plans into a ``(N, L_max)`` tensor."""
    if not token_lists:
        raise ValueError("empty batch")
    L = max(len(t) for t in token_lists)
    out = torch.full((len(token_lists), L), pad_id, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        out[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
    return out
