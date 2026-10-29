"""Isolated-sign recogniser on the canonical motion representation (plan item 4a, ASL Citizen).

It classifies a clip of canonical motion into one of the training partition's glosses, reading the same per-frame
features the generators emit (``diffusion_targets``), so generated citation-form signs can be scored at the sign
level beside the sentence-level back-translator. Encoder: per-frame features -> linear -> strided 1D convolution ->
transformer -> masked mean pool -> linear classifier; label-smoothed cross-entropy. ``GlossMotionDataset`` serves
one partition of a gloss-labelled manifest (ASL Citizen: the publisher's signer-disjoint splits).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..data.multichannel import MotionBatch, collate_multichannel
from ..pose.resample import resample_motion
from ..data_engineering.corpus_manifest import ContentCache, load_sample, usable_channels
from .recognizer_corpora import check_loadable
from .text_to_motion import TextToMotionConfig, diffusion_layout, diffusion_targets, sinusoid


class GlossMotionDataset(Dataset):
    """(label index, state) for one partition; records whose gloss is not in ``labels`` are left out.

    Recogniser v3 options (all off by default, so recogniser v2's data are unchanged):
    - ``template``: every state (real or decoded) is conformed to this state's layouts (``layout_adapter.conform``)
      before augmentation, so corpora and decoded copies batch together;
    - ``recon_p``: a record with a decoded copy (``_recon``, ``recognizer_corpora.attach_recon``) is served as that
      copy with this probability (one draw per load);
    - ``decoded``: only records with a decoded copy, always served decoded (the decoded validation read);
    - ``hand_fraction_fallback``: a record whose quality has no per-hand frame counts is filtered on its
      ``quality.hand_fraction`` (Sem-Lex) instead of counting as no hand seen.
    Every record is checked by ``recognizer_corpora.check_loadable`` when the dataset is made and on every load."""

    def __init__(self, manifest: Mapping, *, partition: str, labels: Mapping[str, int], cache: ContentCache,
                 fetch: Callable[[str], bytes], min_hand_fraction: float = 0., fps: float | None = None,
                 augment: Callable | None = None, template=None, recon_p: float = 0., decoded: bool = False,
                 hand_fraction_fallback: bool = False) -> None:
        def hand_share(q):
            if not q:
                return 1.
            if hand_fraction_fallback and q.get("left_hand_frames") is None and q.get("right_hand_frames") is None:
                return float(q["hand_fraction"]) if q.get("hand_fraction") is not None else 1.
            return max(q.get("left_hand_frames") or 0, q.get("right_hand_frames") or 0) / max(q.get("frames") or 1, 1)

        def keep(r):
            # the hand-visibility filter needs the record's quality counts; a corpus without them (ASLLVD) is not
            # filtered here (ASLLVD sign clips: dominant hand tracked in a median 100%, p10 97% of frames)
            return (r.get("partition") == partition and (r.get("gloss") or {}).get("label") in labels
                    and usable_channels(r, "continuous") and hand_share(r.get("quality")) >= min_hand_fraction
                    and (not decoded or bool(r.get("_recon"))))
        if not 0. <= recon_p <= 1.:
            raise ValueError("recon_p must lie in [0, 1]")
        self.records = [r for r in manifest["records"] if keep(r)]
        for r in self.records:
            check_loadable(r)                        # no test record, no How2Sign signer 9, ever
        self.labels, self.cache, self.fetch = labels, cache, fetch
        self.fps, self.augment = fps, augment        # corpora differ in frame rate: resample before the model sees frames
        self.template, self.recon_p, self.decoded = template, float(recon_p), bool(decoded)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        check_loadable(r)
        if self.decoded or (self.recon_p > 0 and r.get("_recon") and _recon_rng().random() < self.recon_p):
            state = load_state(r["_recon"], cache=self.cache, fetch=self.fetch)
        else:
            state = load_sample(r, "continuous", cache=self.cache, fetch=self.fetch)
        if self.template is not None:
            from ..data.layout_adapter import conform
            state = conform(state, self.template)
        if self.augment is not None:
            state = self.augment(state)
        if self.fps:
            state = resample_motion(state, self.fps)
        return self.labels[r["gloss"]["label"]], state


def augment_sign(state, rng, *, speed=(.8, 1.25), trim=.1, face_dropout=.2, validity_dropout=0.):
    """Training-time variation a recogniser must be invariant to: signing speed (the clock is rescaled, so a later
    resample sees the sign faster or slower), a little lead-in/lead-out trimmed (clip boundaries differ by corpus) and,
    sometimes, the face removed (so the hands carry the decision). Resample after this, never before.

    ``validity_dropout`` (recogniser v3; at 0 nothing more is drawn, so v2's random stream is unchanged): with this
    probability the body joints of one group (``VALIDITY_GROUPS``: the left arm, the right arm, both arms, the torso
    and legs, or arms and torso, uniformly) are made invalid, on every frame (half the time) or on each frame with a
    probability drawn from [0.3, 1). Webcam corpora rarely track the arms while the generator always emits them, so the
    recogniser must read a sign either way. Finger channels are never touched."""
    from dataclasses import replace
    import numpy as np
    from ..data_engineering.corpus_manifest import mask_channels
    t = state.timestamps
    n = len(t)
    if n > 8 and trim > 0:
        a = int(rng.random() * trim * n); b = n - int(rng.random() * trim * n)
        if b - a >= 4:
            state = replace(state, timestamps=t[a:b], channels={k: replace(c, values=c.values[a:b], valid=c.valid[a:b],
                            observed=c.observed[a:b], confidence=c.confidence[a:b]) for k, c in state.channels.items()})
    factor = speed[0] + rng.random() * (speed[1] - speed[0])
    state = replace(state, timestamps=state.timestamps[0] + (state.timestamps - state.timestamps[0]) / factor)
    if rng.random() < face_dropout:
        state = mask_channels(state, [k for k in state.channels if k not in ("face", "blink")])
    if validity_dropout > 0 and rng.random() < validity_dropout:
        group = VALIDITY_GROUPS[min(int(rng.random() * len(VALIDITY_GROUPS)), len(VALIDITY_GROUPS) - 1)]
        frames = len(state.timestamps)
        if rng.random() < .5:
            rows = np.ones(frames, dtype=bool)
        else:
            rate = .3 + .7 * rng.random()
            rows = np.array([rng.random() < rate for _ in range(frames)], dtype=bool)
        state = invalidate_joints(state, VALIDITY_GROUPS_JOINTS[group], rows)
    return state


_ARMS = ("LeftUpperArm", "LeftForearm", "LeftHand", "RightUpperArm", "RightForearm", "RightHand")
_TORSO = ("Pelvis", "Chest", "LeftThigh", "LeftShin", "RightThigh", "RightShin")
VALIDITY_GROUPS_JOINTS = {"left_arm": _ARMS[:3], "right_arm": _ARMS[3:], "arms": _ARMS, "torso": _TORSO,
                          "arms_and_torso": _ARMS + _TORSO}
VALIDITY_GROUPS = tuple(VALIDITY_GROUPS_JOINTS)


def invalidate_joints(state, joints, rows):
    """``state`` with the ``body`` joints named in ``joints`` (those present) invalid on the frames ``rows``: validity,
    observation and confidence cleared, payload zero (the canonical absent payload)."""
    from dataclasses import replace
    import numpy as np
    body = state.channels.get("body")
    rows = np.asarray(rows, dtype=bool)
    if body is None or not rows.any():
        return state
    wanted = set(joints)
    cols = [i for i, label in enumerate(body.labels) if label in wanted]
    if not cols:
        return state
    cells = np.zeros(body.valid.shape, dtype=bool)
    cells[np.ix_(rows, cols)] = True
    values = body.values.copy()
    values[cells] = 0.
    channel = replace(body, values=values, valid=body.valid & ~cells, observed=body.observed & ~cells,
                      confidence=np.where(cells, 0., body.confidence).astype(body.confidence.dtype))
    return replace(state, channels=dict(state.channels, body=channel))


def load_state(part: Mapping, *, cache: ContentCache, fetch: Callable[[str], bytes]):
    """The state at ``part["state_key"]``, checked against ``part["state_sha256"]``, as stored: no masking, so a
    decoded copy keeps the generator's form (every trained channel valid)."""
    import tempfile
    from pathlib import Path
    from ..pose.multichannel import MultichannelMotion
    data = cache.get(part["state_key"], part["state_sha256"], fetch)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "state.npz"
        path.write_bytes(data)
        return MultichannelMotion.load(path, expected_sha256=part["state_sha256"])


_RECON_RNG = {}


def _recon_rng():
    """One generator per loader process for the decoded-copy draw, seeded from torch (workers get distinct seeds)."""
    import os
    import random
    pid = os.getpid()
    if pid not in _RECON_RNG:
        _RECON_RNG[pid] = random.Random(torch.initial_seed() ^ 0x5EC0)
    return _RECON_RNG[pid]


def collate_gloss_motion(items):
    labels, states = zip(*items)
    return torch.tensor(labels, dtype=torch.long), collate_multichannel(states)




@dataclass(frozen=True)
class SignRecognizerConfig:
    motion: TextToMotionConfig
    classes: int
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    feedforward: int = 1024
    dropout: float = .1
    downsample: int = 2
    label_smoothing: float = .1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["motion"] = self.motion.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "SignRecognizerConfig":
        data = dict(data)
        data["motion"] = TextToMotionConfig.from_dict(data["motion"])
        return cls(**data)


class SignRecognizer(nn.Module):
    def __init__(self, config: SignRecognizerConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        width = sum(flat for _, _, _, flat in diffusion_layout(config.motion))
        self.motion_in = nn.Linear(width, d)
        k = config.downsample
        self.downsample = nn.Conv1d(d, d, kernel_size=2 * k - 1, stride=k, padding=k - 1) if k > 1 else nn.Identity()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True),
            config.layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(config.dropout), nn.Linear(d, config.classes))

    def forward(self, motion: MotionBatch) -> torch.Tensor:
        """Logits (B, classes)."""
        x = diffusion_targets(self.config.motion, motion)
        valid = motion.frame_valid.to(x.device)
        h = self.motion_in(x) * valid[..., None]
        if self.config.downsample > 1:
            h = self.downsample(h.transpose(1, 2)).transpose(1, 2)
            k = self.config.downsample
            valid = F.max_pool1d(valid.float()[:, None], k, k, ceil_mode=True)[:, 0, :h.shape[1]] > 0
        positions = torch.arange(h.shape[1], device=h.device)[None].expand(len(h), -1)
        h = self.encoder(h + sinusoid(positions, self.config.d_model, max_period=1e3), src_key_padding_mask=~valid)
        w = valid[..., None].to(h.dtype)
        return self.head((h * w).sum(1) / w.sum(1).clamp_min(1.))

    def loss(self, motion: MotionBatch, labels: torch.Tensor):
        logits = self(motion)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)
        return loss, dict(top1=(logits.argmax(1) == labels).float().mean().detach())


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, ks=(1, 5)) -> dict:
    ranked = logits.topk(min(max(ks), logits.shape[1]), dim=1).indices      # k beyond the class count = all classes
    return {f"top{k}": float((ranked[:, :k] == labels[:, None]).any(1).float().mean()) for k in ks}


__all__ = ["GlossMotionDataset", "augment_sign", "invalidate_joints", "load_state", "VALIDITY_GROUPS",
           "collate_gloss_motion", "SignRecognizerConfig", "SignRecognizer", "top_k_accuracy"]
