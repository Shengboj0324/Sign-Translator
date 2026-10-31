"""Stage S2's probe P1 (docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design received"): how much text information can a
decoder with no history path read from stage-1 codes? The pure parts: reveal order, head, chain-rule likelihood, fit,
sampling, the leakage check and the verdict. ``scripts/s2_text_info_probe.py`` applies them.

**Reveal order.** A clip's stage-1 codes are read coarse to fine: step t has valuation v(t), the power of 2 dividing t
(step 0 highest), and groups run from the highest valuation down, part by part in the tokenizer's order. Group (v, p)
predicts part p of the steps with valuation v, given exactly: every part of the steps with a higher valuation, and the
earlier parts of its own steps. Nothing else enters: the input holds the mask index everywhere else.

**Chain-rule gain.** A group's codes are predicted in parallel from the same context, so each branch (with the text,
and with the empty text) is a proper joint distribution over the clip's stage-1 codes, and
gain = Σ_groups Σ_targets [log p(c | context, text) − log p(c | context, empty)]. Information that repeats along a clip is
counted once: a code that an earlier code determines gains nothing in either branch (the planted-persistence test
checks that a clip-wide class gains ln C, not steps × ln C). This is the same quantity as ``s1_text_kl.py``'s gain,
which is chained over time.

**Length.** Both branches read the clip's clock and phase, so the text gets no credit for its correlation with duration.
Both branches share one network; the empty text is trained on half of the rows, so the empty branch is not worse fitted
than the text branch.

**Known bias.** The gain estimates I(text; codes) − KL(true conditional ‖ text branch) + KL(true marginal ‖ empty branch). An
underfitted empty branch therefore overstates it: on the planted test, 13% high at 900 steps and 2% at 4,000. The AR
priors' stored figures share this form, with the empty text trained on only 10% of their rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..generation.text_to_motion import sinusoid

TOP_VALUATION = 40


def valuations(steps: int, device=None) -> torch.Tensor:
    """(S,) the power of 2 dividing each step index; step 0 has ``TOP_VALUATION`` (it is read first)."""
    t = torch.arange(steps, device=device)
    v = torch.full((steps,), TOP_VALUATION, device=device, dtype=torch.long)
    if steps > 1:
        rest = t[1:]
        v[1:] = torch.log2((rest & -rest).double()).round().long()
    return v


def chain_groups(steps: int, parts: int) -> list[tuple[int, int]]:
    """The reveal order of a clip of up to ``steps`` steps: (valuation, part), highest valuation first. The order depends
    only on the step index, so a clip's own groups are the same in any batch."""
    return [(int(v), p) for v in sorted(set(valuations(steps).tolist()), reverse=True) for p in range(parts)]


def reveal(codes: torch.Tensor, lengths: torch.Tensor, group: tuple[int, int], mask_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(inputs, targets) for one group. ``codes`` (B, S, P); inputs hold ``mask_index`` wherever a code is not revealed;
    targets (B, S, P) mark the codes the group predicts. Padding is neither revealed nor a target."""
    v, p = group
    B, S, P = codes.shape
    val = valuations(S, codes.device)
    part = torch.arange(P, device=codes.device)
    seen = (val > v)[:, None] | ((val == v)[:, None] & (part < p)[None])
    target = (val == v)[:, None] & (part == p)[None]
    valid = (torch.arange(S, device=codes.device)[None] < lengths[:, None])[..., None]
    seen, target = seen[None] & valid, target[None] & valid
    return torch.where(seen, codes, torch.full_like(codes, mask_index)), target


@dataclass(frozen=True)
class HeadConfig:
    codebook: int = 512
    parts: int = 4
    d_model: int = 192
    heads: int = 4
    layers: int = 4
    feedforward: int = 768
    dropout: float = .1
    text_dim: int = 768
    fps: float = 24.


class TextInfoHead(nn.Module):
    """A small bidirectional transformer over steps: summed embeddings of the revealed codes (mask index elsewhere) plus
    the clip's clock, cross-attending to frozen text tokens; one output head per part. About 3.3M parameters at the
    default size."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        d, K, P = config.d_model, config.codebook, config.parts
        self.code_embed = nn.Parameter(torch.randn(P, K + 1, d) * .02)              # index K: masked
        self.clock = nn.Linear(2 * d, d)
        self.text_in = nn.Linear(config.text_dim, d)
        self.blocks = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, config.heads, config.feedforward, config.dropout,
                                                                       batch_first=True, norm_first=True), config.layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Parameter(torch.randn(P, d, K) * .02)

    @property
    def mask_index(self) -> int:
        return self.config.codebook

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """Logits (B, S, P, K) from ``inputs`` (B, S, P) (``reveal``), each clip's length in steps, and text tokens
        (B, L, text_dim) with their mask (B, L)."""
        d = self.config.d_model
        B, S, P = inputs.shape
        steps = torch.arange(S, device=inputs.device)[None].expand(B, S)
        seconds = steps.float() / self.config.fps
        phase = steps.float() / lengths[:, None].clamp_min(1).float()
        clock = self.clock(torch.cat((sinusoid(seconds, d, max_period=60.), sinusoid(phase, d, max_period=4.)), -1))
        x = self.code_embed[torch.arange(P, device=inputs.device), inputs].sum(2) + clock
        positions = sinusoid(torch.arange(text.shape[1], device=text.device)[None].expand(B, -1), d, max_period=1024.)
        memory = self.text_in(text) + positions
        padding = steps >= lengths[:, None]
        h = self.norm(self.blocks(x, memory, tgt_key_padding_mask=padding, memory_key_padding_mask=~text_mask))
        return torch.einsum("bsd,pdk->bspk", h, self.out)


def padded_codes(rows, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes (B, S, P) long, zero-padded; lengths (B,)) of a list of (S_i, P) integer arrays."""
    lengths = torch.tensor([len(r) for r in rows], device=device)
    out = torch.zeros(len(rows), int(lengths.max()), rows[0].shape[1], dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.as_tensor(np.asarray(r, np.int64), device=device)
    return out, lengths


def padded_text(rows, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(tokens (B, L, D), mask (B, L)) of a list of (L_i, D) arrays."""
    sizes = [len(r) for r in rows]
    out = torch.zeros(len(rows), max(sizes), rows[0].shape[1], dtype=torch.float32, device=device)
    mask = torch.zeros(len(rows), max(sizes), dtype=torch.bool, device=device)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.as_tensor(np.asarray(r, np.float32), device=device)
        mask[i, :len(r)] = True
    return out, mask


@torch.no_grad()
def chain_log_probs(head: TextInfoHead, codes, lengths, text, text_mask, *, reveal_fn=reveal):
    """(per-clip total log-probability (B,), per group (B, G), the groups). Eval mode only: dropout would make the two
    branches of a gain differ by noise."""
    if head.training:
        raise ValueError("chain_log_probs reads a head in eval mode")
    groups = chain_groups(codes.shape[1], head.config.parts)
    per = torch.zeros(len(codes), len(groups), dtype=torch.float64, device=codes.device)
    for g, group in enumerate(groups):
        inputs, target = reveal_fn(codes, lengths, group, head.mask_index)
        if not target.any():
            continue
        lp = head(inputs, lengths, text, text_mask).log_softmax(-1).double()
        taken = lp.gather(-1, codes[..., None])[..., 0]
        per[:, g] = (taken * target).sum((1, 2))
    return per.sum(1), per, groups


def group_totals(per: np.ndarray, groups, parts) -> dict:
    """Mean nats per clip by part and by valuation, from (clips, groups) gains."""
    by_part = {str(parts[p]): round(float(per[:, [g for g, x in enumerate(groups) if x[1] == p]].sum(1).mean()), 4)
               for p in range(len(parts))}
    values = sorted({x[0] for x in groups}, reverse=True)
    by_val = {("step0" if v == TOP_VALUATION else f"every{2 ** (v + 1)}"):
              round(float(per[:, [g for g, x in enumerate(groups) if x[0] == v]].sum(1).mean()), 4) for v in values}
    return dict(by_part=by_part, by_valuation=by_val)


def training_batch_loss(head: TextInfoHead, codes, lengths, text, text_mask, empty_text, generator: torch.Generator,
                        *, drop_text: float = .5, reveal_fn=reveal) -> torch.Tensor:
    """Mean cross entropy over one group's targets. Each conditional is its own function of its context, so any positive
    weighting of groups has the same optimum; a uniform draw over scales gives the sparse anchors their updates. With
    probability ``drop_text`` per row the text is replaced by the empty text."""
    vals = sorted({int(x) for x in valuations(codes.shape[1]).tolist()}, reverse=True)
    v = vals[int(torch.randint(0, len(vals), (), generator=generator))]
    p = int(torch.randint(0, head.config.parts, (), generator=generator))
    inputs, target = reveal_fn(codes, lengths, (v, p), head.mask_index)
    keep = torch.rand(len(codes), generator=generator) >= drop_text
    empty = empty_text.to(text).expand(len(codes), -1, -1)
    width = max(text.shape[1], empty.shape[1])
    tokens = torch.zeros(len(codes), width, text.shape[2], dtype=text.dtype, device=text.device)
    mask = torch.zeros(len(codes), width, dtype=torch.bool, device=text.device)
    tokens[:, :text.shape[1]] = text * text_mask[..., None]
    mask[:, :text.shape[1]] = text_mask
    drop = (~keep).to(text.device)
    tokens[drop] = 0.
    mask[drop] = False
    tokens[drop, :empty.shape[1]] = empty[drop]
    mask[drop, :empty.shape[1]] = True
    logits = head(inputs, lengths, tokens, mask)
    if not target.any():
        return logits.sum() * 0.
    return F.cross_entropy(logits[target], codes[target])


def fit_head(head: TextInfoHead, rows, *, empty_text, steps: int, batch: int = 16, lr: float = 3e-4, seed: int = 0,
             stop_read=None, every: int = 500, patience: int = 4, pool: int = 8, log=None, reveal_fn=reveal) -> dict:
    """Fit on ``rows`` (list of (codes (S, P), text tokens (L, D))); keep the weights with the lowest ``stop_read()``
    (text-branch chain NLL on the declared stop clips); stop after ``patience`` reads without a new best. Batches are
    drawn from a sorted pool of ``pool`` × ``batch`` rows, to limit padding."""
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=.01)
    device = next(head.parameters()).device
    best, best_step, best_value, history, stale = None, 0, math.inf, [], 0
    done = 0
    for step in range(1, steps + 1):
        head.train()
        pick = rng.integers(0, len(rows), min(pool * batch, max(len(rows), batch)))
        pick = pick[np.argsort([len(rows[i][0]) for i in pick])]
        at = int(rng.integers(0, max(1, len(pick) - batch + 1)))
        chosen = [rows[i] for i in pick[at:at + batch]]
        codes, lengths = padded_codes([c for c, _ in chosen], device)
        text, mask = padded_text([t for _, t in chosen], device)
        loss = training_batch_loss(head, codes, lengths, text, mask, empty_text, generator, reveal_fn=reveal_fn)
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError("nonfinite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        done = step
        if stop_read is not None and step % every == 0:
            head.eval()
            value = float(stop_read())
            history.append(dict(step=step, stop_nll_nats_per_second=round(value, 4), loss=round(float(loss.detach()), 4)))
            if log:
                log(history[-1])
            if value < best_value:
                best_value, best_step, stale = value, step, 0
                best = copy.deepcopy(head.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
    if best is not None:
        head.load_state_dict(best)
    head.eval()
    return dict(steps_run=done, best_step=best_step or done, best_stop_nll=None if best is None else round(best_value, 4),
                history=history, config=asdict(head.config), parameters=sum(p.numel() for p in head.parameters()))


@torch.no_grad()
def sample_chain(head: TextInfoHead, lengths, text, text_mask, *, seed: int = 0, temperature: float = 1.) -> torch.Tensor:
    """Codes (B, S, P) sampled group by group in the reveal order, from the text alone (no reference code enters)."""
    if head.training:
        raise ValueError("sample_chain reads a head in eval mode")
    generator = torch.Generator().manual_seed(seed)
    S = int(lengths.max())
    codes = torch.full((len(lengths), S, head.config.parts), head.mask_index, dtype=torch.long, device=lengths.device)
    for group in chain_groups(S, head.config.parts):
        inputs, target = reveal(codes, lengths, group, head.mask_index)
        if not target.any():
            continue
        logits = head(inputs, lengths, text, text_mask)[target] / temperature
        drawn = torch.multinomial(logits.softmax(-1).cpu(), 1, generator=generator)[:, 0].to(codes.device)
        codes[target] = drawn
    return codes


@torch.no_grad()
def check_no_leak(head: TextInfoHead, codes, lengths, text, text_mask, *, reveal_fn=reveal, tolerance: float = 1e-5) -> int:
    """For every group, replace the group's targets and every code not yet revealed by other codes: the logits at the
    targets must not move. Raises ValueError on a leak; returns the number of groups checked. Also raises if changing an
    already revealed code never moves anything, which would make the check vacuous."""
    if head.training:
        raise ValueError("check_no_leak reads a head in eval mode")
    rng = torch.Generator().manual_seed(0)
    K, checked, moved = head.config.codebook, 0, False
    for group in chain_groups(codes.shape[1], head.config.parts):
        inputs, target = reveal_fn(codes, lengths, group, head.mask_index)
        if not target.any():
            continue
        base = head(inputs, lengths, text, text_mask)
        hidden = reveal(codes, lengths, group, head.mask_index)[0] == head.mask_index
        other = torch.where(hidden, torch.randint(0, K, codes.shape, generator=rng).to(codes), codes)
        changed = head(reveal_fn(other, lengths, group, head.mask_index)[0], lengths, text, text_mask)
        if (base[target] - changed[target]).abs().max() > tolerance:
            raise ValueError(f"group {group}: the logits read a code that is not revealed")
        seen = reveal(codes, lengths, group, head.mask_index)[0] != head.mask_index
        if seen.any():
            flipped = torch.where(seen, (codes + 1) % K, codes)
            if (head(reveal_fn(flipped, lengths, group, head.mask_index)[0], lengths, text, text_mask)[target]
                    - base[target]).abs().max() > tolerance:
                moved = True
        checked += 1
    if not moved:
        raise ValueError("the revealed codes never moved the logits: the check would be vacuous")
    return checked


__all__ = ["HeadConfig", "TextInfoHead", "TOP_VALUATION", "valuations", "chain_groups", "reveal", "padded_codes", "padded_text",
           "chain_log_probs", "group_totals", "training_batch_loss", "fit_head", "sample_chain", "check_no_leak"]
