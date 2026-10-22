"""English -> ASL gloss planner with locus slots and non-manual markers (plan item 2, trained on ASLLRP / RIT).

A transformer encoder-decoder: English words (``WordVocabulary`` of the back-translator, byte fallback) in, the
dominant-hand surface gloss sequence (``GlossVocabulary``, byte fallback) out. On every token that closes a sign the
decoder's state also predicts that sign's attributes:

- ``source`` and ``target`` locus slots (0 = none, 1..``MAX_LOCI`` in order of first placement), softmax each;
- the grammatical non-manual markers spanning the sign (``MARKERS``), independent Bernoulli per marker - markers nest
  and overlap (a wh question inside a topic), so a softmax over markers would be wrong.

Loss: label-smoothed cross-entropy on the gloss tokens + cross-entropy on the two locus slots + binary cross-entropy
on the markers, both only at sign-closing positions. Generation decodes the gloss tokens (greedy or beam) and reads the
attributes from the decoder state at each sign-closing token of the chosen sequence.

Evaluation (``evaluate_plans``): gloss word error rate and BLEU-4 over base glosses, locus and marker scores on the
signs aligned to the reference by the edit-distance path (marker F1 per tier and micro-averaged; locus accuracy on the
aligned signs that carry a locus in either sequence), and utterance-level marker presence F1 (does the plan carry a
wh question when the reference does).
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..generation.motion_to_text import PAD as WPAD, WordVocabulary, corpus_bleu
from .asllrp_gloss import BOS, EOS, MARKERS, MAX_LOCI, PAD, GlossVocabulary, is_unit_end


@dataclass(frozen=True)
class EnglishGlossConfig:
    source_vocabulary: int
    target_vocabulary: int
    d_model: int = 256
    heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 3
    feedforward: int = 1024
    dropout: float = .3
    max_source: int = 96
    max_target: int = 160
    label_smoothing: float = .1
    locus_weight: float = 1.
    marker_weight: float = 1.

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "EnglishGlossConfig":
        return cls(**data)


def _positions(n: int, d: int, device) -> torch.Tensor:
    p = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    f = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(1e4) / d))
    out = torch.zeros(n, d, device=device)
    out[:, 0::2], out[:, 1::2] = torch.sin(p * f), torch.cos(p * f)
    return out


class EnglishGlossPlanner(nn.Module):
    def __init__(self, config: EnglishGlossConfig) -> None:
        super().__init__()
        self.config = c = config
        self.source = nn.Embedding(c.source_vocabulary, c.d_model, padding_idx=WPAD)
        self.target = nn.Embedding(c.target_vocabulary, c.d_model, padding_idx=PAD)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            c.d_model, c.heads, c.feedforward, c.dropout, batch_first=True, norm_first=True), c.encoder_layers,
            enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            c.d_model, c.heads, c.feedforward, c.dropout, batch_first=True, norm_first=True), c.decoder_layers)
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.source_locus = nn.Linear(c.d_model, MAX_LOCI + 1)
        self.target_locus = nn.Linear(c.d_model, MAX_LOCI + 1)
        self.markers = nn.Linear(c.d_model, len(MARKERS))

    def encode(self, source: torch.Tensor):
        mask = source == WPAD
        h = self.source(source) * math.sqrt(self.config.d_model) + _positions(source.shape[1], self.config.d_model, source.device)
        return self.encoder_norm(self.encoder(h, src_key_padding_mask=mask)), mask

    def decode(self, memory, memory_mask, tokens: torch.Tensor) -> torch.Tensor:
        """Decoder states (B, L, d) for ``tokens`` (B, L), causal."""
        L = tokens.shape[1]
        h = self.target(tokens) * math.sqrt(self.config.d_model) + _positions(L, self.config.d_model, tokens.device)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 1)
        h = self.decoder(h, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == PAD, memory_key_padding_mask=memory_mask)
        return self.decoder_norm(h)

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        return states @ self.target.weight.t()

    def loss(self, source, tokens, src_loci, dst_loci, markers, unit_end):
        """Total loss and its terms. ``tokens`` (B, L) start with BOS; attribute tensors are aligned with ``tokens``."""
        memory, mmask = self.encode(source)
        target = tokens[:, 1:]
        states = self.decode(memory, mmask, tokens)                       # causal: state t has seen tokens <= t
        logits = self.logits(states[:, :-1])                              # state t predicts token t+1
        token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD,
                                     label_smoothing=self.config.label_smoothing)
        end = unit_end[:, 1:]                       # a sign's attributes are read at the token that closes it
        out = dict(token_loss=token_loss.detach())
        total = token_loss
        if end.any():
            sel = states[:, 1:][end]
            locus = (F.cross_entropy(self.source_locus(sel), src_loci[:, 1:][end])
                     + F.cross_entropy(self.target_locus(sel), dst_loci[:, 1:][end]))
            marker = F.binary_cross_entropy_with_logits(self.markers(sel), markers[:, 1:][end].float())
            total = total + self.config.locus_weight * locus + self.config.marker_weight * marker
            out.update(locus_loss=locus.detach(), marker_loss=marker.detach())
        keep = target != PAD
        out["token_accuracy"] = ((logits.argmax(-1) == target) & keep).sum() / keep.sum().clamp_min(1)
        return total, out

    @torch.no_grad()
    def generate(self, source: torch.Tensor, *, beam: int = 1, max_tokens: int | None = None,
                 length_penalty: float = .6) -> list[dict]:
        """Per input: {"tokens": ids without BOS, "loci": [(src, dst)] per sign, "markers": [[0/1]*7] per sign}."""
        max_tokens = max_tokens or self.config.max_target
        memory, mmask = self.encode(source)
        results = []
        for b in range(source.shape[0]):
            mem, mm = memory[b:b + 1], mmask[b:b + 1]
            beams, finished = [([BOS], 0.)], []
            for _ in range(max_tokens - 1):
                toks = torch.tensor([s for s, _ in beams], device=source.device)
                logp = F.log_softmax(self.logits(self.decode(mem.expand(len(beams), -1, -1), mm.expand(len(beams), -1),
                                                             toks)[:, -1]), -1)
                logp[:, PAD] = -math.inf; logp[:, BOS] = -math.inf
                cand = []
                for (seq, score), row in zip(beams, logp):
                    top = torch.topk(row, beam)
                    cand += [(seq + [int(i)], score + float(v)) for v, i in zip(top.values, top.indices)]
                cand.sort(key=lambda c: -c[1])
                beams = []
                for seq, score in cand:
                    (finished if seq[-1] == EOS else beams).append((seq, score))
                    if len(beams) == beam:
                        break
                if not beams or (len(finished) >= beam and max(s for _, s in finished) >= beams[0][1]):
                    break
            finished += beams
            norm = lambda seq, score: score / (((5 + len(seq)) / 6) ** length_penalty)  # noqa: E731
            seq = max(finished, key=lambda c: norm(*c))[0]
            seq = seq if seq[-1] == EOS else seq + [EOS]
            states = self.decode(mem, mm, torch.tensor([seq], device=source.device))[0]
            ends = [i for i, t in enumerate(seq) if is_unit_end(t)]
            if ends:
                sel = states[ends]
                loci = list(zip(self.source_locus(sel).argmax(-1).tolist(), self.target_locus(sel).argmax(-1).tolist()))
                marks = (torch.sigmoid(self.markers(sel)) > .5).int().tolist()
            else:
                loci, marks = [], []
            results.append(dict(tokens=seq[1:], loci=loci, markers=marks))
        return results


def encode_sources(vocab: WordVocabulary, texts, *, max_tokens: int) -> torch.Tensor:
    rows = [vocab.encode(t, max_tokens=max_tokens) for t in texts]
    out = torch.full((len(rows), max(len(r) for r in rows)), WPAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


def _align(ref: list[str], hyp: list[str]):
    """Edit-distance alignment: (errors, [(i_ref, j_hyp) matched-or-substituted pairs])."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[n][m], pairs[::-1]


def evaluate_plans(references: list[dict], hypotheses: list[dict]) -> dict:
    """``references``/``hypotheses``: {"glosses": [...], "loci": [(s, t)], "markers": [[0/1]*7]} per utterance."""
    errors = words = 0
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    loc_hit = loc_n = 0
    utt = collections.Counter()
    for ref, hyp in zip(references, hypotheses):
        e, pairs = _align(ref["glosses"], hyp["glosses"])
        errors += e; words += len(ref["glosses"])
        for i, j in pairs:
            rm, hm = ref["markers"][i], hyp["markers"][j]
            for k in range(len(MARKERS)):
                tp[k] += rm[k] and hm[k]; fp[k] += hm[k] and not rm[k]; fn[k] += rm[k] and not hm[k]
            if any(ref["loci"][i]) or any(hyp["loci"][j]):
                loc_n += 1; loc_hit += tuple(ref["loci"][i]) == tuple(hyp["loci"][j])
        for k in range(len(MARKERS)):
            r = any(m[k] for m in ref["markers"]); h = any(m[k] for m in hyp["markers"])
            utt[("tp", k)] += r and h; utt[("fp", k)] += h and not r; utt[("fn", k)] += r and not h

    def f1(t, p, n):
        return 2 * t / max(2 * t + p + n, 1)
    per = {MARKERS[k]: round(f1(tp[k], fp[k], fn[k]), 4) for k in range(len(MARKERS))}
    bleu = corpus_bleu([" ".join(h["glosses"]) for h in hypotheses], [" ".join(r["glosses"]) for r in references])
    return dict(gloss_wer=errors / max(words, 1), gloss_bleu=bleu["bleu"],
                marker_f1_micro=f1(sum(tp.values()), sum(fp.values()), sum(fn.values())), marker_f1=per,
                utterance_marker_f1={MARKERS[k]: round(f1(utt[("tp", k)], utt[("fp", k)], utt[("fn", k)]), 4)
                                     for k in range(len(MARKERS))},
                locus_accuracy=loc_hit / max(loc_n, 1), locus_signs=loc_n)


def plan_from_generation(vocab: GlossVocabulary, out: dict) -> dict:
    """Decoded glosses with their attributes (one per sign-closing token)."""
    glosses = vocab.decode(out["tokens"])
    n = min(len(glosses), len(out["loci"]))
    return dict(glosses=glosses[:n], loci=[tuple(x) for x in out["loci"][:n]], markers=out["markers"][:n])


__all__ = ["EnglishGlossConfig", "EnglishGlossPlanner", "encode_sources", "evaluate_plans", "plan_from_generation"]
