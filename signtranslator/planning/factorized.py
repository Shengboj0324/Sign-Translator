"""Factorized training and the LLM-dominance hypothesis test.

FLa-LLM (arXiv:2403.12556) finds that *directly* introducing an LLM into gloss-
free SLT lets the LLM dominate the learning curve while the visual representation
stays weak, and fixes it by factorizing training: pre-train the representation
with a lightweight head, then freeze it and connect the LLM. Their task is
sign->text. The specification asks us to treat the warning as a **testable
hypothesis in the inverse (evidence->sign) direction**.

This module supplies both:

* a **factorized schedule** -- Stage A trains the evidence encoder + a
  lightweight content head; Stage B freezes the encoder and trains only the
  heavy autoregressive decoder ("the LLM") on the plan objective; and

* a **dominance probe** -- after each training regime, a frozen-representation
  linear probe measures how much task-relevant content the encoder retained.
  The hypothesis predicts ``probe(joint) < probe(factorized)``.

The direction is **measured, not assumed**. If the inverse direction does not
reproduce the effect, that is a real, reportable negative result -- exactly as
the document frames it. Nothing here hard-codes the expected outcome.

The components are deliberately small and self-contained so the encoder
representation is cleanly accessible; this is an *experimental probe of an
optimization phenomenon*, not the production planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceEncoder(nn.Module):
    """Source tokens -> (sequence memory, pooled representation).

    Pooling is a **learned attention** over the sequence (a single query), not a
    mean. Mean pooling dilutes a content-bearing token among distractors, so the
    pooled representation -- and any probe of it -- would understate what the
    encoder actually learned. An attention pool lets a trained encoder route the
    relevant token, which is both more realistic and a fairer probe target.
    """

    def __init__(self, src_vocab: int, d_model: int = 48, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed = nn.Embedding(src_vocab, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers,
                                             enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(d_model) / (d_model ** 0.5))
        self.d_model = d_model

    def forward(self, src_tokens: torch.Tensor):
        memory = self.encoder(self.embed(src_tokens))            # (N, S, D)
        scores = (memory @ self.query) / (self.d_model ** 0.5)   # (N, S)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # (N, S, 1)
        pooled = (weights * memory).sum(dim=1)                   # (N, D)
        return memory, pooled


class ContentHead(nn.Module):
    """Lightweight head: pooled representation -> content class logits.

    This is the "lightweight translation model" of the factorized recipe. Being
    lightweight and reading *only* the pooled representation, it can only lower
    its loss by making the representation informative -- which is the point.
    """

    def __init__(self, d_model: int, num_content: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, num_content)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class HeavyDecoder(nn.Module):
    """A comparatively high-capacity autoregressive plan decoder ("the LLM")."""

    def __init__(self, plan_vocab: int, d_model: int = 48, nhead: int = 4,
                 num_layers: int = 4, dropout: float = 0.1,
                 max_len: int = 128) -> None:
        super().__init__()
        self.pad_id = plan_vocab + 1
        self.sos_id = plan_vocab
        self.embed = nn.Embedding(plan_vocab + 2, d_model, padding_idx=self.pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, plan_vocab)
        self.plan_vocab = plan_vocab

    def forward(self, plan_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        n, L = plan_tokens.shape
        sos = torch.full((n, 1), self.sos_id, dtype=torch.long,
                         device=plan_tokens.device)
        dec_in = torch.cat([sos, plan_tokens[:, :-1]], dim=1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool,
                                       device=plan_tokens.device), diagonal=1)
        h = self.embed(dec_in) + self.pos[:, :L]
        h = self.decoder(h, memory, tgt_mask=causal,
                         tgt_key_padding_mask=(dec_in == self.pad_id))
        return self.out(h)

    def nll(self, plan_tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        logits = self.forward(plan_tokens, memory)
        return F.cross_entropy(logits.reshape(-1, self.plan_vocab),
                               plan_tokens.reshape(-1), ignore_index=self.pad_id)


# ---------------------------------------------------------------------------
# Representation probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def _encode_pooled(encoder: EvidenceEncoder, src: torch.Tensor) -> torch.Tensor:
    """Mean-pool the encoder MEMORY for the probe.

    Deliberately the mean of ``memory``, not the attention-pooled representation:
    the attention ``query`` is trained only through the content head (Stage A of
    the factorized regime) and gets no gradient in joint training, so probing the
    attention pool would compare a trained pool against a random one -- a
    confound. ``memory`` is trained by *both* regimes, so its mean is a fair,
    regime-independent probe target.
    """
    encoder.eval()
    return encoder(src)[0].mean(dim=1)


def representation_probe_accuracy(encoder: EvidenceEncoder,
                                  train_src: torch.Tensor, train_y: torch.Tensor,
                                  eval_src: torch.Tensor, eval_y: torch.Tensor,
                                  epochs: int = 200, lr: float = 0.05,
                                  seed: int = 0) -> float:
    """Held-out accuracy of a linear probe on the FROZEN pooled representation.

    Higher means the encoder retained more task-relevant content. The encoder is
    never updated here -- only the probe is trained -- so this measures the
    representation, not a fresh fit of the whole model.
    """
    torch.manual_seed(seed)
    train_repr = _encode_pooled(encoder, train_src)
    eval_repr = _encode_pooled(encoder, eval_src)
    n_classes = int(max(int(train_y.max()), int(eval_y.max()))) + 1
    probe = nn.Linear(encoder.d_model, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(train_repr), train_y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((probe(eval_repr).argmax(-1) == eval_y).double().mean())


# ---------------------------------------------------------------------------
# Training regimes
# ---------------------------------------------------------------------------
@dataclass
class TrainingRegimeResult:
    regime: str
    encoder: EvidenceEncoder
    decoder: HeavyDecoder
    final_plan_nll: float
    encoder_frozen_in_stage_b: bool = False


def factorized_train(encoder: EvidenceEncoder, content_head: ContentHead,
                     decoder: HeavyDecoder, src: torch.Tensor,
                     content_y: torch.Tensor, plan_tokens: torch.Tensor,
                     stage_a_steps: int = 200, stage_b_steps: int = 200,
                     lr: float = 3e-3) -> TrainingRegimeResult:
    """Stage A: encoder + content head. Stage B: freeze encoder, train decoder."""
    # Stage A -- representation init.
    opt_a = torch.optim.Adam(list(encoder.parameters()) + list(content_head.parameters()),
                             lr=lr)
    encoder.train(); content_head.train()
    for _ in range(stage_a_steps):
        _, pooled = encoder(src)
        loss = F.cross_entropy(content_head(pooled), content_y)
        opt_a.zero_grad(); loss.backward(); opt_a.step()

    # Stage B -- freeze the encoder, train only the decoder ("the LLM").
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    opt_b = torch.optim.Adam(decoder.parameters(), lr=lr)
    decoder.train()
    final = 0.0
    for _ in range(stage_b_steps):
        with torch.no_grad():
            memory, _ = encoder(src)
        loss = decoder.nll(plan_tokens, memory)
        opt_b.zero_grad(); loss.backward(); opt_b.step()
        final = float(loss)
    return TrainingRegimeResult("factorized", encoder, decoder, final,
                                encoder_frozen_in_stage_b=True)


def joint_train(encoder: EvidenceEncoder, decoder: HeavyDecoder,
                src: torch.Tensor, plan_tokens: torch.Tensor,
                steps: int = 400, lr: float = 3e-3) -> TrainingRegimeResult:
    """Train encoder + decoder together on the plan objective (no content head)."""
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=lr)
    encoder.train(); decoder.train()
    final = 0.0
    for _ in range(steps):
        memory, _ = encoder(src)
        loss = decoder.nll(plan_tokens, memory)
        opt.zero_grad(); loss.backward(); opt.step()
        final = float(loss)
    return TrainingRegimeResult("joint", encoder, decoder, final)


@dataclass
class DominanceReport:
    """Comparison of representation informativeness across regimes."""

    factorized_probe: float
    joint_probe: float
    factorized_plan_nll: float
    joint_plan_nll: float

    @property
    def dominance_gap(self) -> float:
        """factorized - joint. Positive supports the FLa-LLM hypothesis."""
        return self.factorized_probe - self.joint_probe

    @property
    def hypothesis_supported(self) -> bool:
        return self.joint_probe < self.factorized_probe

    def summary(self) -> str:
        verdict = ("supports" if self.hypothesis_supported else "does NOT support")
        return (f"representation probe -- factorized {self.factorized_probe:.3f} "
                f"vs joint {self.joint_probe:.3f} (gap {self.dominance_gap:+.3f}); "
                f"this run {verdict} the LLM-dominance hypothesis")


# ---------------------------------------------------------------------------
# End-to-end dominance experiment (a tool, not a claim)
# ---------------------------------------------------------------------------
def run_dominance_experiment(src_train: torch.Tensor, content_train: torch.Tensor,
                             plans_train: torch.Tensor, src_eval: torch.Tensor,
                             content_eval: torch.Tensor, src_vocab: int,
                             plan_vocab: int, num_content: int,
                             d_model: int = 32, stage_a_steps: int = 120,
                             stage_b_steps: int = 120, joint_steps: int = 240,
                             seed: int = 0) -> DominanceReport:
    """Train both regimes on identical data and probe each representation.

    Returns the comparison. It does **not** decide the hypothesis: as documented
    in docs/SEMANTIC_PLANNER.md, a linear representation probe does not reliably
    measure representation quality at this synthetic scale (random features probe
    well), so this is a reporting tool, and its direction is not asserted.
    """
    torch.manual_seed(seed)
    enc_f = EvidenceEncoder(src_vocab, d_model=d_model, num_layers=1)
    head = ContentHead(d_model, num_content)
    dec_f = HeavyDecoder(plan_vocab, d_model=d_model, num_layers=3)
    rf = factorized_train(enc_f, head, dec_f, src_train, content_train,
                          plans_train, stage_a_steps, stage_b_steps)
    pf = representation_probe_accuracy(enc_f, src_train, content_train,
                                       src_eval, content_eval, seed=seed)

    torch.manual_seed(seed)
    enc_j = EvidenceEncoder(src_vocab, d_model=d_model, num_layers=1)
    dec_j = HeavyDecoder(plan_vocab, d_model=d_model, num_layers=3)
    rj = joint_train(enc_j, dec_j, src_train, plans_train, joint_steps)
    pj = representation_probe_accuracy(enc_j, src_train, content_train,
                                       src_eval, content_eval, seed=seed)

    return DominanceReport(factorized_probe=pf, joint_probe=pj,
                           factorized_plan_nll=rf.final_plan_nll,
                           joint_plan_nll=rj.final_plan_nll)
