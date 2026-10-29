"""The sentence refiner (docs/NEXT_PHASE_PLAN.md, "Next lever after the oracle finding", option 1, 2026-10-29 17:38
UTC). Built and tested only; it trains and runs once declared.

**Task.** A composition of isolated signs (round 15's stage-1 words at sign-fitted timing) reads like isolated signs
joined by bridges; the back-translator was trained on How2Sign's continuous signing. The refiner maps the composition's
tokenizer codes to the codes of a real How2Sign sentence: trained on pairs (``scripts/build_refiner_pairs.py``) whose
input is a composition of *real* isolated train clips of the sentence's stage-1 words, fitted to the sentence's own
duration, and whose target is that sentence's real codes.

**Model** (:class:`SentenceRefiner`). The RVQ prior (``motion_rvq.RVQPrior``), warm-started from F12, with its text
conditioning *replaced* by an encoder over the composed input's codes:
- the source encoder's input at position l is the sum of the prior's own per-position code embeddings of the
  composition's codes at l (``RVQPrior.code_embed``, the same tables the temporal decoder's history uses, all
  residual stages) plus clock features (seconds and phase, as the prior's ``time`` projection; ``source_time`` starts as
  a copy of it);
- the source encoder has the text encoder's layer shape and starts from its weights (layers beyond the text encoder's
  count start fresh);
- the temporal decoder, depth transformer and output heads are the prior's; the decoder cross-attends to the encoded
  source where it attended to the text. Text, grammar, locus and duration modules are removed: the refiner never
  sees the English (its controls, another sentence's words and rest frames, would otherwise be read through the text);
- the output has the input's length (one target step per source step), on the clock the prior conditions on.

**Training** (``refiner_loss``): teacher-forced cross entropy of the target codes, summed over the code positions of a
step (the prior's factorisation), where the target's part was observed when ``part_valid_loss`` (F12's option);
``history_dropout`` (F12's 0.5: every history step before a random cut replaced by the start code, so the decoder must
use the source); ``source_dropout``: a row's source replaced by the empty source (one start-code position), so
``refine_codes`` can guide (``guidance`` != 1) against it, as the prior guides against the empty text.

**Pairs** (``pack_pair`` / ``unpack_pair``): one deterministic binary file per sentence (a JSON header, then the raw
arrays in name order), so a rebuild writes identical bytes and the index can pin each file's SHA-256.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .motion_rvq import MotionRVQ, RVQPrior, RVQPriorConfig
from .text_to_motion import diffusion_targets, diffusion_valid, sinusoid

FPS = 24.
KIND = "sentence_refiner"
PAIR_SCHEMA = "refiner-pairs-v1"
PAIR_MAGIC = b"RFPAIR1\n"
# the prior's text-side modules, removed from the refiner (it is conditioned on the composition only)
TEXT_MODULES = ("embed", "grammar", "loci", "text_encoder", "duration")


@dataclass(frozen=True)
class RefinerConfig:
    prior: RVQPriorConfig
    source_layers: int = 4
    source_dropout: float = .1
    history_dropout: float = .5
    part_valid_loss: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["prior"] = self.prior.to_dict()
        return data

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "RefinerConfig":
        data = dict(data)
        data["prior"] = RVQPriorConfig.from_dict(data["prior"])
        return cls(**data)

    @classmethod
    def from_prior(cls, prior: RVQPriorConfig, **overrides) -> "RefinerConfig":
        """The refiner of ``prior``: the source encoder as deep as its text encoder; history dropout and the
        part-valid loss as the prior trained (F12: 0.5 and on), unless overridden (None: the prior's)."""
        kw = dict(source_layers=prior.text_layers, history_dropout=prior.history_dropout,
                  part_valid_loss=prior.part_valid_loss)
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(prior, **kw)


class SentenceRefiner(nn.Module):
    def __init__(self, config: RefinerConfig) -> None:
        super().__init__()
        if config.prior.text_model:
            raise ValueError("the refiner replaces the text encoder: a pretrained text model has no place in it")
        if config.source_layers < 1:
            raise ValueError("source_layers must be >= 1")
        self.config = config
        prior = RVQPrior(config.prior)
        for name in TEXT_MODULES:
            if hasattr(prior, name):
                delattr(prior, name)
        self.prior = prior
        self.J, self.K = prior.J, prior.K
        p = config.prior
        layer = nn.TransformerEncoderLayer(p.d_model, p.heads, p.feedforward, p.dropout, batch_first=True,
                                           norm_first=True)
        self.source_encoder = nn.TransformerEncoder(layer, config.source_layers, enable_nested_tensor=False)
        self.source_time = nn.Linear(2 * p.d_model, p.d_model)

    @property
    def step(self) -> int:
        return self.config.prior.tokenizer.step

    def warm_start(self, prior_state: dict) -> dict:
        """Load a prior's weights (``RVQPrior.state_dict``): every kept prior module exactly (missing keys refused),
        the source encoder from the text encoder where the layer exists, ``source_time`` from ``time``. Returns the
        counts and the prior keys left unused (the text side)."""
        own = self.prior.state_dict()
        missing = sorted(k for k in own if k not in prior_state)
        if missing:
            raise ValueError(f"the prior state lacks {missing[:5]}")
        bad = sorted(k for k in own if tuple(prior_state[k].shape) != tuple(own[k].shape))
        if bad:
            raise ValueError(f"the prior state differs in shape at {bad[:5]}")
        self.prior.load_state_dict({k: prior_state[k] for k in own})
        text = {k[len("text_encoder."):]: v for k, v in prior_state.items() if k.startswith("text_encoder.")}
        encoder = self.source_encoder.state_dict()
        copied = sorted(k for k in encoder if k in text and tuple(text[k].shape) == tuple(encoder[k].shape))
        encoder.update({k: text[k] for k in copied})
        self.source_encoder.load_state_dict(encoder)
        self.source_time.load_state_dict({"weight": prior_state["time.weight"], "bias": prior_state["time.bias"]})
        unused = sorted(k for k in prior_state if k not in own)
        return dict(prior_tensors=len(own), source_encoder_copied=len(copied), source_encoder_tensors=len(encoder),
                    unused_prior_tensors=len(unused), unused_prefixes=sorted({k.split(".")[0] for k in unused}))

    def clock(self, length: int, device) -> torch.Tensor:
        """(1, length) seconds of each step: the first frame it covers."""
        return (torch.arange(length, device=device, dtype=torch.float32) * self.step / FPS)[None]

    def encode_source(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Memory (B, L, d) of source codes (B, L, J) (index K: the start code, the empty source and padding) under
        their valid positions ``mask`` (B, L): summed per-position code embeddings plus clock and phase features."""
        d = self.config.prior.d_model
        lengths = mask.sum(1).clamp_min(1).float()
        seconds = self.clock(codes.shape[1], codes.device).expand(codes.shape[0], -1)
        phase = seconds / (lengths * self.step / FPS)[:, None]
        x = self.prior._embed_codes(codes).sum(2) + self.source_time(
            torch.cat((sinusoid(seconds, d, max_period=60.), sinusoid(phase, d, max_period=4.)), -1))
        return self.source_encoder(x, src_key_padding_mask=~mask)

    def empty_source(self, rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(codes (rows, 1, J) of start codes, mask (rows, 1) True): the source a dropped row trains with."""
        return (torch.full((rows, 1, self.J), self.K, dtype=torch.long, device=device),
                torch.ones((rows, 1), dtype=torch.bool, device=device))


def refiner_loss(model: SentenceRefiner, source, source_mask, target, target_valid, target_part=None, *,
                 train: bool = True):
    """(loss, terms) of target codes (B, S, J) at valid steps ``target_valid`` (B, S) given source codes (B, L, J)
    with ``source_mask`` (B, L). ``target_part`` (B, S, P): the target's observed parts (the part-valid loss).
    Training only: source dropout and history dropout (the config's rates)."""
    config, prior = model.config, model.prior
    B, S, J = target.shape
    device = target.device
    # the clock's duration is the source's length, as in refine_codes (a pair's input has its target's length)
    duration = source_mask.sum(1).float() * model.step / FPS
    if train and config.source_dropout > 0:
        drop = torch.rand(B, device=device) < config.source_dropout
        if bool(drop.any()):
            source, source_mask = source.clone(), source_mask.clone()
            source[drop] = model.K
            source_mask[drop] = False
            source_mask[drop, 0] = True
    memory = model.encode_source(source, source_mask)
    start = torch.full_like(target[:, :1], model.K)
    previous = torch.cat((start, target[:, :-1]), 1)
    if train and config.history_dropout > 0:                # the prior's round-9 history dropout, unchanged
        drop = torch.rand(B, device=device) < config.history_dropout
        cut = (torch.rand(B, device=device) * S).long() + 1
        hidden = (torch.arange(S, device=device)[None] < cut[:, None]) & drop[:, None]
        previous = torch.where(hidden[..., None], torch.full_like(previous, model.K), previous)
    seconds = model.clock(S, device).expand(B, -1)
    h = prior.temporal_states(memory, source_mask, previous, seconds, duration)
    gold = target[target_valid]                                                               # (N, J)
    logits = prior.depth_logits(h[target_valid], gold[:, :-1])                                # (N, J, K)
    per_code = F.cross_entropy(logits.reshape(-1, model.K), gold.reshape(-1), reduction="none").reshape(gold.shape)
    if config.part_valid_loss and target_part is not None:
        P = target_part.shape[-1]
        weight = target_part[target_valid][:, torch.arange(J, device=device) % P].float()
        cross = (per_code * weight).sum() / weight.sum().clamp_min(1.)
    else:
        cross = per_code.mean()
    accuracy = (logits.argmax(-1) == gold).float()
    P = len(model.config.prior.tokenizer.part_slices())
    return cross, {"tokens": float(cross.detach()), "token_accuracy": float(accuracy.mean()),
                   "stage1_accuracy": float(accuracy[:, :P].mean())}


def _per_code(temperature, names, J) -> list[float]:
    if isinstance(temperature, dict):
        unknown = set(temperature) - set(names)
        if unknown:
            raise ValueError(f"unknown parts in temperature: {sorted(unknown)}")
        return [float(temperature.get(names[j % len(names)], 1.)) for j in range(J)]
    return [float(temperature)] * J


def pad_sources(model: SentenceRefiner, sources, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes (B, L, J) padded with the start code, mask (B, L)) of (L_i, J) code sequences."""
    L = max(int(s.shape[0]) for s in sources)
    codes = torch.full((len(sources), L, model.J), model.K, dtype=torch.long, device=device)
    mask = torch.zeros((len(sources), L), dtype=torch.bool, device=device)
    for r, s in enumerate(sources):
        if not len(s):
            raise ValueError("an empty source")
        codes[r, :len(s)] = s.to(device).long()
        mask[r, :len(s)] = True
    return codes, mask


@torch.no_grad()
def refine_codes(model: SentenceRefiner, sources, *, guidance: float = 1., temperature: float | dict = 1.,
                 top_k: int = 0, seed: int = 0) -> list[torch.Tensor]:
    """Sampled target codes, one (L_i, J) sequence per source (L_i, J): the input's length, every step and depth
    position one batched call, as ``generate_rvq_motion_batch`` samples (guidance against the empty source,
    per-part temperatures, top-k; rows shorter than the longest are sampled to the common length and cut back)."""
    model.eval()
    prior = model.prior
    device = next(model.parameters()).device
    codes, mask = pad_sources(model, sources, device)
    B, L = mask.shape
    memory = model.encode_source(codes, mask)
    if guidance != 1.:
        e_codes, e_mask = model.empty_source(1, device)
        empty = model.encode_source(e_codes, e_mask).expand(B, -1, -1)
        empty_mask = e_mask.expand(B, -1)
    lengths = mask.sum(1).float()
    duration = lengths * model.step / FPS
    clock = model.clock(L, device).expand(B, -1)
    names = [part for part, _ in model.config.prior.tokenizer.part_slices()]
    per_code = _per_code(temperature, names, model.J)
    generator = torch.Generator().manual_seed(seed)
    sequence = torch.full((B, 1, model.J), model.K, dtype=torch.long, device=device)
    for k in range(L):
        h = prior.temporal_states(memory, mask, sequence, clock[:, :k + 1], duration)[:, -1]
        h_free = (prior.temporal_states(empty, empty_mask, sequence, clock[:, :k + 1], duration)[:, -1]
                  if guidance != 1. else None)
        step_codes = torch.zeros((B, 0), dtype=torch.long, device=device)
        for j in range(model.J):
            logits = prior.depth_logits(h, step_codes)[:, -1]
            if h_free is not None:
                free = prior.depth_logits(h_free, step_codes)[:, -1]
                logits = free + guidance * (logits - free)
            logits = logits / max(per_code[j], 1e-6)
            if top_k:
                threshold = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, -math.inf)
            choice = torch.multinomial(torch.softmax(logits.float(), -1).cpu(), 1, generator=generator).to(device)
            step_codes = torch.cat((step_codes, choice), 1)
        sequence = torch.cat((sequence, step_codes[:, None]), 1)
    return [sequence[r, 1:int(lengths[r]) + 1].cpu() for r in range(B)]


# ---- tokenizer round trips ---------------------------------------------------------------------------------------

@torch.no_grad()
def tokenize_state(tokenizer: MotionRVQ, state, device="cpu") -> dict:
    """The tokenizer's codes of one state (all stages): ``codes`` (S, J), ``valid`` (S,) token validity, ``part``
    (S, P) whether each part was observed in the step's frames (at least half of its unit columns valid: the
    prior's part-valid rule, ``motion_rvq.part_code_weights``), all on the CPU."""
    from ..data.multichannel import collate_multichannel
    tokenizer.eval()
    config = tokenizer.config.motion
    motion = collate_multichannel([state])
    unit = diffusion_targets(config, motion)
    observed = diffusion_valid(config, motion)
    _, codes, _ = tokenizer.quantize(tokenizer.encode(unit.to(device)))
    frame_valid = motion.frame_valid.to(device)
    valid = tokenizer.token_valid(frame_valid)
    part = torch.stack([tokenizer.token_valid((observed[..., getattr(tokenizer, f"columns_{p}").cpu()].float()
                                               .mean(-1) >= .5).to(device)) for p, _ in tokenizer.parts], -1)
    return dict(codes=codes[0].cpu(), valid=valid[0].cpu(), part=part[0].cpu())


@torch.no_grad()
def refine_states(model: SentenceRefiner, tokenizer: MotionRVQ, states, *, seed: int, guidance: float = 1.,
                  temperature: float | dict = 1., top_k: int = 0, post=None, device="cpu", recon: bool = False,
                  return_codes: bool = False):
    """Each state tokenized (``tokenize_state``), refined (``refine_codes``, one batch, ``seed``) and decoded as a
    generation at its own frame count (``compositional.make_decode``, then ``post``: the generator's
    post-processing). ``recon``: decoded from the input's own codes, no refinement (the tokenizer round trip).
    ``return_codes``: ``(states, output codes, input codes)``, codes on the CPU."""
    from .compositional import make_decode
    decode = make_decode(tokenizer, model.config.prior.tokenizer.motion, post or (lambda s: s))
    inputs = [tokenize_state(tokenizer, s, device)["codes"] for s in states]
    codes = inputs if recon else refine_codes(model, inputs, guidance=guidance, temperature=temperature, top_k=top_k,
                                              seed=seed)
    out = [decode(c[None].to(device), len(s.timestamps), "refined") for c, s in zip(codes, states)]
    return (out, codes, inputs) if return_codes else out


def state_dict_digest(state: dict) -> str:
    """SHA-256 over a state dict's names, dtypes, shapes and bytes (sorted names; CPU tensors)."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


# ---- pair files --------------------------------------------------------------------------------------------------

def pack_pair(arrays: dict, meta: dict) -> bytes:
    """Deterministic bytes of named arrays and a JSON-able ``meta``: magic, header length, sorted-key JSON header
    (schema, meta, each array's name, dtype with byte order, shape, offset, size), then the arrays' C-order bytes in
    name order."""
    header, blobs, offset = dict(schema=PAIR_SCHEMA, meta=meta, arrays=[]), [], 0
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        if a.dtype.kind not in "iub":
            raise ValueError(f"{name}: pair arrays are integer or boolean, not {a.dtype}")
        b = a.tobytes()
        header["arrays"].append(dict(name=name, dtype=a.dtype.str, shape=list(a.shape), offset=offset, nbytes=len(b)))
        blobs.append(b)
        offset += len(b)
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return PAIR_MAGIC + len(head).to_bytes(8, "big") + head + b"".join(blobs)


def unpack_pair(payload: bytes) -> tuple[dict, dict]:
    """(arrays, meta) of ``pack_pair`` bytes; refuses another schema or a truncated file."""
    if not payload.startswith(PAIR_MAGIC):
        raise ValueError("not a refiner pair file")
    n = int.from_bytes(payload[len(PAIR_MAGIC):len(PAIR_MAGIC) + 8], "big")
    start = len(PAIR_MAGIC) + 8
    header = json.loads(payload[start:start + n])
    if header.get("schema") != PAIR_SCHEMA:
        raise ValueError(f"pair schema {header.get('schema')!r}, expected {PAIR_SCHEMA!r}")
    body = memoryview(payload)[start + n:]
    if sum(a["nbytes"] for a in header["arrays"]) != len(body):
        raise ValueError("a truncated pair file")
    arrays = {a["name"]: np.frombuffer(body[a["offset"]:a["offset"] + a["nbytes"]], dtype=np.dtype(a["dtype"]))
              .reshape(a["shape"]).copy() for a in header["arrays"]}
    return arrays, header["meta"]


def pair_arrays(target: dict, sources: list[dict]) -> dict:
    """The arrays of one sentence's pair file from ``tokenize_state`` outputs: the target's codes, validity and
    observed parts, and each composed input's codes."""
    out = dict(target_codes=target["codes"].numpy().astype(np.int16), target_valid=target["valid"].numpy(),
               target_part=target["part"].numpy())
    for k, s in enumerate(sources):
        out[f"source_codes_{k}"] = s["codes"].numpy().astype(np.int16)
    return out


# ---- checkpoints -------------------------------------------------------------------------------------------------

def load_target_index(store, key: str, sha256: str):
    """The training targets' code n-gram index (``memorisation.CodeIndex``) a refiner checkpoint names, hash-checked;
    with its targets' sentence ids (``signs``)."""
    from .memorisation import CodeIndex
    payload = store.get(key)
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise SystemExit(f"{key} differs from the SHA-256 its refiner checkpoint records")
    arrays, meta = unpack_pair(payload)
    return CodeIndex.from_arrays(arrays, meta), meta


def load_refiner(store, key: str, sha256: str | None = None, device="cpu"):
    """(refiner with its EMA weights, its tokenizer, the checkpoint's metadata) of a refiner checkpoint;
    ``sha256`` pins the file."""
    payload = store.get(key)
    if sha256 and hashlib.sha256(payload).hexdigest() != sha256:
        raise SystemExit(f"{key} differs from the pinned SHA-256")
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if saved.get("kind") != KIND:
        raise SystemExit(f"{key} is not a {KIND} checkpoint ({saved.get('kind')})")
    model = SentenceRefiner(RefinerConfig.from_dict(saved["config"]))
    model.load_state_dict(saved.get("ema") or saved["model"])
    tokenizer = MotionRVQ(model.config.prior.tokenizer)
    tokenizer.load_state_dict(saved["tokenizer_state"])
    if state_dict_digest(tokenizer.state_dict()) != saved["tokenizer_digest"]:
        raise SystemExit(f"{key}: its tokenizer weights differ from their recorded digest")
    meta = {k: saved.get(k) for k in ("step", "tokenizer_key", "tokenizer_digest", "pairs_index_key",
                                      "pairs_index_sha256", "init_key", "init_sha256", "config_sha256",
                                      "target_index_key", "target_index_sha256")}
    meta["sha256"] = hashlib.sha256(payload).hexdigest()
    return model.to(device).eval(), tokenizer.to(device).eval().requires_grad_(False), meta


__all__ = ["FPS", "KIND", "PAIR_SCHEMA", "RefinerConfig", "SentenceRefiner", "refiner_loss", "refine_codes",
           "pad_sources", "tokenize_state", "refine_states", "state_dict_digest", "pack_pair", "unpack_pair",
           "pair_arrays", "load_refiner", "load_target_index"]
