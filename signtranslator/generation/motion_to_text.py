"""Motion -> English back-translator on the canonical motion representation (plan item 3a, docs/NEXT_PHASE_PLAN.md Q1).

It reads the same per-frame features the generators and the retrieval evaluator see (``diffusion_targets``), so
generated motion is scored directly, with no rendering step whose errors would count against the generator. It is
trained on the training partition only and frozen before it scores anything; it is never a training loss for a
generator. Its BLEU is relative (generated vs real motion through the same translator), never an absolute claim.

- **Vocabulary** (``WordVocabulary``): lower-cased NFC words and punctuation from the training sentences, with byte
  fallback: a word outside the vocabulary is spelled as UTF-8 byte tokens closed by an end-of-word token, so every
  sentence is encodable and decodable exactly (up to case and spacing).
- **Model** (``MotionToText``): per-frame features -> linear -> strided 1D convolution (time / ``downsample``) ->
  transformer encoder; an autoregressive transformer decoder with tied input/output embeddings; label-smoothed
  cross-entropy.
- **Decoding**: greedy or beam search with a length penalty.
- **Metric** (``corpus_bleu``): BLEU-4 over lower-cased word/punctuation tokens, with the brevity penalty.
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.multichannel import MotionBatch
from .text_to_motion import TextToMotionConfig, diffusion_layout, diffusion_targets, sinusoid

PAD, BOS, EOS, WORD_END = 0, 1, 2, 259                 # 3..258 are the 256 byte tokens
BYTE0, FIRST_WORD = 3, 260
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    """Lower-cased NFC word and punctuation tokens (the unit of the vocabulary and of BLEU)."""
    return _TOKEN.findall(unicodedata.normalize("NFC", text).lower())


class WordVocabulary:
    def __init__(self, vocabulary: list[str]):
        self.words = list(vocabulary)
        self.index = {w: FIRST_WORD + i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, texts, *, max_words: int = 8000, min_count: int = 2) -> "WordVocabulary":
        counts = collections.Counter(w for text in texts for w in words(text))
        ranked = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
        return cls(ranked[:max_words])

    def __len__(self) -> int:
        return FIRST_WORD + len(self.words)

    def encode(self, text: str, *, max_tokens: int | None = None) -> list[int]:
        ids = [BOS]
        for w in words(text):
            if w in self.index:
                ids.append(self.index[w])
            else:
                ids += [BYTE0 + b for b in w.encode("utf-8")] + [WORD_END]
        ids.append(EOS)
        if max_tokens is not None and len(ids) > max_tokens:
            ids = ids[:max_tokens - 1] + [EOS]
        return ids

    def decode(self, ids) -> str:
        out, spelled = [], bytearray()
        for i in ids:
            i = int(i)
            if i in (PAD, BOS):
                continue
            if i == EOS:
                break
            if BYTE0 <= i < BYTE0 + 256:
                spelled.append(i - BYTE0)
            elif i == WORD_END:
                out.append(spelled.decode("utf-8", errors="replace")); spelled = bytearray()
            elif i >= FIRST_WORD and i - FIRST_WORD < len(self.words):
                out.append(self.words[i - FIRST_WORD])
        if spelled:
            out.append(spelled.decode("utf-8", errors="replace"))
        text = " ".join(out)
        return re.sub(r" ([.,!?;:%)\]'])", r"\1", text)

    def to_dict(self) -> dict:
        return dict(words=self.words)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.words).encode()).hexdigest()


def encode_batch(vocabulary: WordVocabulary, texts, *, max_tokens: int = 64) -> torch.Tensor:
    rows = [vocabulary.encode(t, max_tokens=max_tokens) for t in texts]
    out = torch.full((len(rows), max(len(r) for r in rows)), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
    return out


@dataclass(frozen=True)
class MotionToTextConfig:
    motion: TextToMotionConfig
    vocabulary_size: int
    d_model: int = 384
    heads: int = 6
    encoder_layers: int = 4
    decoder_layers: int = 4
    feedforward: int = 1536
    dropout: float = .1
    downsample: int = 2
    max_tokens: int = 64
    label_smoothing: float = .1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["motion"] = self.motion.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "MotionToTextConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)


class MotionToText(nn.Module):
    def __init__(self, config: MotionToTextConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        width = sum(flat for _, _, _, flat in diffusion_layout(config.motion))
        self.motion_in = nn.Linear(width, d)
        self.downsample = nn.Conv1d(d, d, kernel_size=config.downsample * 2 - 1, stride=config.downsample,
                                    padding=config.downsample - 1) if config.downsample > 1 else nn.Identity()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.encoder_layers, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(d)
        self.embed = nn.Embedding(config.vocabulary_size, d, padding_idx=PAD)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.decoder_layers)
        self.decoder_norm = nn.LayerNorm(d)

    def encode(self, motion: MotionBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """(memory (B, T', d), memory mask (B, T') True = valid) of a batch of (real or generated) motion."""
        x = diffusion_targets(self.config.motion, motion)                   # (B, T, W)
        valid = motion.frame_valid.to(x.device)
        h = self.motion_in(x) * valid[..., None]
        if self.config.downsample > 1:
            h = self.downsample(h.transpose(1, 2)).transpose(1, 2)
            k = self.config.downsample
            valid = F.max_pool1d(valid.float()[:, None], k, k, ceil_mode=True)[:, 0, :h.shape[1]] > 0
        positions = torch.arange(h.shape[1], device=h.device)[None].expand(len(h), -1)
        h = h + sinusoid(positions, self.config.d_model, max_period=1e3)
        return self.encoder_norm(self.encoder(h, src_key_padding_mask=~valid)), valid

    def decode(self, memory, memory_valid, tokens: torch.Tensor) -> torch.Tensor:
        """Logits (B, L, V) for next-token prediction given ``tokens`` (B, L)."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens)
        h = self.embed(tokens) * math.sqrt(self.config.d_model) + sinusoid(positions, self.config.d_model, max_period=1e4)
        causal = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device), 1)
        h = self.decoder(h, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == PAD,
                         memory_key_padding_mask=~memory_valid)
        return self.decoder_norm(h) @ self.embed.weight.t()

    def loss(self, motion: MotionBatch, tokens: torch.Tensor):
        memory, valid = self.encode(motion)
        logits = self.decode(memory, valid, tokens[:, :-1])
        target = tokens[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=PAD,
                               label_smoothing=self.config.label_smoothing)
        keep = target != PAD
        accuracy = ((logits.argmax(-1) == target) & keep).sum() / keep.sum().clamp_min(1)
        return loss, dict(token_accuracy=accuracy.detach())

    @torch.no_grad()
    def generate(self, motion: MotionBatch, *, beam: int = 4, max_tokens: int | None = None,
                 length_penalty: float = .6) -> list[list[int]]:
        """Token sequences (without BOS) per clip; ``beam`` 1 is greedy."""
        max_tokens = max_tokens or self.config.max_tokens
        memory, valid = self.encode(motion)
        out = []
        for b in range(len(memory)):
            mem, val = memory[b:b + 1], valid[b:b + 1]
            beams = [([BOS], 0.)]
            finished = []
            for _ in range(max_tokens - 1):
                tokens = torch.tensor([s for s, _ in beams], device=mem.device)
                logp = F.log_softmax(self.decode(mem.expand(len(beams), -1, -1), val.expand(len(beams), -1),
                                                 tokens)[:, -1], -1)
                logp[:, PAD] = -math.inf
                candidates = []
                for (seq, score), row in zip(beams, logp):
                    top = torch.topk(row, beam)
                    candidates += [(seq + [int(i)], score + float(v)) for v, i in zip(top.values, top.indices)]
                candidates.sort(key=lambda c: -c[1])
                beams = []
                for seq, score in candidates:
                    (finished if seq[-1] == EOS else beams).append((seq, score))
                    if len(beams) == beam:
                        break
                if not beams or (len(finished) >= beam and max(s for _, s in finished) >= beams[0][1]):
                    break
            finished += beams
            norm = lambda seq, score: score / (((5 + len(seq)) / 6) ** length_penalty)   # noqa: E731
            best = max(finished, key=lambda c: norm(*c))[0]
            out.append(best[1:])
        return out


def corpus_bleu(hypotheses: list[str], references: list[str], max_n: int = 4) -> dict:
    """Corpus BLEU (one reference per segment) over ``words`` tokens; ``bleu`` is 0-100."""
    matches, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hypotheses, references):
        h, r = words(hyp), words(ref)
        hyp_len += len(h); ref_len += len(r)
        for n in range(1, max_n + 1):
            hc = collections.Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = collections.Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            matches[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    precisions = [m / t if t else 0. for m, t in zip(matches, totals)]
    if hyp_len == 0 or min(precisions) == 0:
        bleu = 0.
    else:
        bp = 1. if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
        bleu = 100 * bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return dict(bleu=bleu, precisions=[100 * p for p in precisions], hypothesis_length=hyp_len,
                reference_length=ref_len)


__all__ = ["WordVocabulary", "words", "encode_batch", "MotionToTextConfig", "MotionToText", "corpus_bleu",
           "PAD", "BOS", "EOS"]
