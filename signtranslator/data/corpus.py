"""On-disk sign-language corpus: schema, synthetic generator, ingestion.

A corpus is a directory containing

    manifest.json     metadata (dims, vocab sizes, split counts, spec)
    <split>.npz       arrays for each split (``train``, ``val``, ...)

Each ``.npz`` holds
    pose      float32 (N, C, T, V)   3D keypoint clips
    concepts  int64   (N, L)         padded concept ids (0 = PAD, content 1..K)
    lengths   int64   (N,)           true concept-sequence length per sample

Each ``.npz`` also stores ``src_concepts`` (N, L): the spoken-language concept
ids, produced by applying a fixed vocabulary bijection (a "cipher") to
``concepts``. The planner must invert this substitution — a monotonic
spoken→gloss translation that generalises from few examples (unlike a positional
permutation, which memorises on a small corpus). The remaining branch tensors are
derived deterministically from ``concepts`` so every modality stays consistent:

    spoken tokens (planner source) = perm(concepts) + offset   (stored)
    gloss tokens  (planner target / generation conditioning / alignment) = concepts
    CTC targets   (recognition)    = concepts mapped to 1..K
    motion        = per-concept trajectory signatures concatenated over time
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Reserved target-vocabulary tokens (shared with planner.py conventions).
PAD, BOS, EOS = 0, 1, 2
CONTENT_OFFSET = 3  # content ids start at 3 in src/gloss token vocabularies


@dataclass
class CorpusSpec:
    """Immutable description of a corpus's structure and vocabularies."""

    num_concepts: int          # K distinct signs/concepts
    seq_len: int               # concepts per sample (L)
    num_joints: int
    in_channels: int
    num_frames: int
    # Derived vocab sizes (kept explicit for model construction).
    src_vocab: int             # spoken-language token vocabulary
    gloss_vocab: int           # gloss token vocabulary (planner target + conditioning)
    num_glosses: int           # CTC classes (== num_concepts)
    speech_frames: int = 64    # frames of acoustic features per utterance
    speech_dim: int = 40       # mel-filterbank channels

    @staticmethod
    def build(num_concepts: int, seq_len: int, num_joints: int,
              in_channels: int, num_frames: int, speech_frames: int = 64,
              speech_dim: int = 40) -> "CorpusSpec":
        vocab = num_concepts + CONTENT_OFFSET
        return CorpusSpec(
            num_concepts=num_concepts, seq_len=seq_len, num_joints=num_joints,
            in_channels=in_channels, num_frames=num_frames,
            src_vocab=vocab, gloss_vocab=vocab, num_glosses=num_concepts,
            speech_frames=speech_frames, speech_dim=speech_dim,
        )


def _concept_trajectory(freqs, phases, amps, num_frames: int) -> np.ndarray:
    """Deterministic per-concept motion signature -> (C, t, V)."""
    t = np.linspace(0, 1, num_frames)[None, None, :]
    traj = amps[..., None] * np.sin(2 * np.pi * freqs[..., None] * t + phases[..., None])
    return np.transpose(traj, (1, 2, 0))  # (C, t, V)


def generate_corpus(out_dir: str, spec: Optional[CorpusSpec] = None,
                    counts: Optional[Dict[str, int]] = None,
                    noise: float = 0.02, seed: int = 0) -> CorpusSpec:
    """Generate and write a synthetic corpus to ``out_dir``.

    Returns the :class:`CorpusSpec` (also persisted in ``manifest.json``).
    """
    spec = spec or CorpusSpec.build(num_concepts=12, seq_len=4, num_joints=27,
                                    in_channels=3, num_frames=32)
    counts = counts or {"train": 256, "val": 64}
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    K, V, C, T = spec.num_concepts, spec.num_joints, spec.in_channels, spec.num_frames
    # Per-concept motion parameters (shared across splits).
    freqs = rng.uniform(0.5, 3.0, size=(K, V, C))
    phases = rng.uniform(0, 2 * np.pi, size=(K, V, C))
    amps = rng.uniform(0.5, 1.5, size=(K, V, C))
    # Per-concept acoustic signature: a mel-filterbank profile plus a temporal
    # modulation, standing in for the spectrogram of the spoken word.
    mel_profile = rng.normal(0.0, 1.0, size=(K, spec.speech_dim))
    mel_mod = rng.uniform(1.0, 4.0, size=(K,))

    seg_bounds = np.linspace(0, T, spec.seq_len + 1).astype(int)
    sp_bounds = np.linspace(0, spec.speech_frames, spec.seq_len + 1).astype(int)
    # Fixed spoken->gloss vocabulary bijection (shared across splits).
    perm = np.random.default_rng(seed + 999).permutation(K)

    def make_split(n: int, split_seed: int):
        r = np.random.default_rng(split_seed)
        concepts = r.integers(0, K, size=(n, spec.seq_len))  # 0..K-1
        src_concepts = perm[concepts]                        # ciphered spoken ids
        pose = np.zeros((n, C, T, V), dtype=np.float32)
        speech = np.zeros((n, spec.speech_frames, spec.speech_dim), dtype=np.float32)
        for i in range(n):
            for s in range(spec.seq_len):
                k = concepts[i, s]
                lo, hi = seg_bounds[s], seg_bounds[s + 1]
                seg = _concept_trajectory(freqs[k], phases[k], amps[k], hi - lo)
                seg = seg + r.normal(0, noise, size=(C, 1, V))
                pose[i, :, lo:hi, :] = seg

                # Acoustic segment for the *spoken* word (indexed by the
                # ciphered id, since speech carries spoken-language identity).
                slo, shi = sp_bounds[s], sp_bounds[s + 1]
                sk = src_concepts[i, s]
                tt = np.linspace(0, 1, shi - slo)[:, None]
                seg_sp = (mel_profile[sk][None, :]
                          * (1.0 + 0.3 * np.sin(2 * np.pi * mel_mod[sk] * tt)))
                speech[i, slo:shi] = seg_sp + r.normal(0, noise, size=seg_sp.shape)
        lengths = np.full(n, spec.seq_len, dtype=np.int64)
        return (pose, concepts.astype(np.int64), src_concepts.astype(np.int64),
                speech, lengths)

    split_counts = {}
    train_pose = None
    for j, (split, n) in enumerate(counts.items()):
        pose, concepts, src_concepts, speech, lengths = make_split(n, seed + 1 + j)
        np.savez_compressed(os.path.join(out_dir, f"{split}.npz"),
                            pose=pose, concepts=concepts,
                            src_concepts=src_concepts, speech=speech,
                            lengths=lengths)
        split_counts[split] = int(n)
        if split == "train":
            train_pose = pose

    # Normalisation statistics are computed from the TRAIN split only (never
    # from val/test) and applied to every split -- standard practice that avoids
    # leaking held-out statistics into evaluation. Shape (C, 1, V): per channel
    # and joint, pooled over samples and time.
    if train_pose is None:
        train_pose = pose
    mean = train_pose.mean(axis=(0, 2), keepdims=True)[0]      # (C, 1, V)
    std = train_pose.std(axis=(0, 2), keepdims=True)[0]        # (C, 1, V)
    std = np.maximum(std, 1e-3)

    manifest = {"spec": asdict(spec), "splits": split_counts, "seed": seed,
                "perm": perm.tolist(),
                "pose_mean": mean.tolist(), "pose_std": std.tolist()}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return spec


def load_manifest(corpus_dir: str) -> dict:
    with open(os.path.join(corpus_dir, "manifest.json")) as f:
        return json.load(f)


def validate_corpus(corpus_dir: str) -> CorpusSpec:
    """Check the corpus on disk is well-formed; raise on any inconsistency."""
    manifest = load_manifest(corpus_dir)
    spec = CorpusSpec(**manifest["spec"])
    for split, n in manifest["splits"].items():
        path = os.path.join(corpus_dir, f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing split file: {path}")
        with np.load(path) as z:
            pose, concepts, lengths = z["pose"], z["concepts"], z["lengths"]
            src_concepts = z["src_concepts"]
        if pose.shape != (n, spec.in_channels, spec.num_frames, spec.num_joints):
            raise ValueError(f"{split}: pose shape {pose.shape} != expected")
        if concepts.shape != (n, spec.seq_len):
            raise ValueError(f"{split}: concepts shape {concepts.shape} != expected")
        if src_concepts.shape != (n, spec.seq_len):
            raise ValueError(f"{split}: src_concepts shape {src_concepts.shape} != expected")
        if lengths.shape != (n,):
            raise ValueError(f"{split}: lengths shape {lengths.shape} != expected")
        for name, arr in (("concepts", concepts), ("src_concepts", src_concepts)):
            if arr.min() < 0 or arr.max() >= spec.num_concepts:
                raise ValueError(f"{split}: {name} out of range [0, {spec.num_concepts})")
        if not np.isfinite(pose).all():
            raise ValueError(f"{split}: non-finite pose values")
        with np.load(path) as z:
            if "speech" in z.files:
                speech = z["speech"]
                if speech.shape != (n, spec.speech_frames, spec.speech_dim):
                    raise ValueError(f"{split}: speech shape {speech.shape} != expected")
                if not np.isfinite(speech).all():
                    raise ValueError(f"{split}: non-finite speech values")
    return spec


class PoseStandardizer:
    """Applies corpus normalisation statistics: ``z = (x - mean) / std``.

    Diffusion models assume roughly unit-scale data; standardising the pose
    channels materially improves generative fidelity. Statistics come from the
    train split only.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean          # (C, 1, V)
        self.std = std            # (C, 1, V)

    @staticmethod
    def from_manifest(manifest: dict) -> "PoseStandardizer":
        mean = torch.tensor(manifest["pose_mean"], dtype=torch.float32)
        std = torch.tensor(manifest["pose_std"], dtype=torch.float32)
        return PoseStandardizer(mean, std)

    def _broadcast(self, x: torch.Tensor):
        # Accept (C, T, V) or (N, C, T, V).
        if x.dim() == 4:
            return self.mean.unsqueeze(0).to(x.device), self.std.unsqueeze(0).to(x.device)
        return self.mean.to(x.device), self.std.to(x.device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._broadcast(x)
        return (x - m) / s

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        m, s = self._broadcast(z)
        return z * s + m


class SignDataset(Dataset):
    """Reads one split of an on-disk corpus (pose standardised by default)."""

    def __init__(self, corpus_dir: str, split: str = "train",
                 normalize: bool = True) -> None:
        manifest = load_manifest(corpus_dir)
        self.spec = CorpusSpec(**manifest["spec"])
        self.standardizer = PoseStandardizer.from_manifest(manifest)
        with np.load(os.path.join(corpus_dir, f"{split}.npz")) as z:
            self.pose = torch.from_numpy(z["pose"])
            self.concepts = torch.from_numpy(z["concepts"])
            self.src_concepts = torch.from_numpy(z["src_concepts"])
            self.lengths = torch.from_numpy(z["lengths"])
            self.speech = (torch.from_numpy(z["speech"]) if "speech" in z.files
                           else None)
        if normalize:
            self.pose = self.standardizer.normalize(self.pose)

    def __len__(self) -> int:
        return self.pose.shape[0]

    def __getitem__(self, idx: int) -> dict:
        n = int(self.lengths[idx])
        item = {"pose": self.pose[idx],
                "concepts": self.concepts[idx, :n],
                "src_concepts": self.src_concepts[idx, :n]}
        if self.speech is not None:
            item["speech"] = self.speech[idx]
        return item


def collate_corpus(batch: List[dict]) -> dict:
    """Derive every branch's tensors from the concept sequences.

    Produces a batch dict with keys consumed by ``BidirectionalSignTranslator``:
    ``pose``, ``gloss_tokens``, ``src``, ``gloss_seq``, ``ctc_targets``,
    ``ctc_lengths`` (and ``concepts`` for reference).
    """
    pose = torch.stack([b["pose"] for b in batch], dim=0)
    concept_lists = [b["concepts"] for b in batch]
    src_lists = [b["src_concepts"] for b in batch]
    lengths = torch.tensor([len(c) for c in concept_lists], dtype=torch.long)
    max_len = int(lengths.max())

    n = len(batch)
    gloss_tokens = torch.zeros(n, max_len, dtype=torch.long)       # content+3, PAD=0
    src = torch.zeros(n, max_len, dtype=torch.long)                 # ciphered content+3
    gloss_seq = torch.full((n, max_len + 2), PAD, dtype=torch.long)  # BOS..EOS
    ctc_parts: List[torch.Tensor] = []

    for i, (c, sc) in enumerate(zip(concept_lists, src_lists)):
        L = len(c)
        content = c + CONTENT_OFFSET                    # gloss token ids
        gloss_tokens[i, :L] = content
        src[i, :L] = sc + CONTENT_OFFSET                # ciphered spoken tokens
        gloss_seq[i, 0] = BOS
        gloss_seq[i, 1:1 + L] = content
        gloss_seq[i, 1 + L] = EOS
        ctc_parts.append(c + 1)                          # CTC ids in 1..K

    ctc_targets = torch.cat(ctc_parts, dim=0)
    ctc_lengths = lengths.clone()

    out_speech = None
    if "speech" in batch[0]:
        out_speech = torch.stack([b["speech"] for b in batch], dim=0)

    result = {
        "pose": pose,
        "concepts": concept_lists,
        "gloss_tokens": gloss_tokens,
        "src": src,
        "gloss_seq": gloss_seq,
        "ctc_targets": ctc_targets,
        "ctc_lengths": ctc_lengths,
    }
    if out_speech is not None:
        result["speech"] = out_speech
        # The speech branch predicts the *spoken* token sequence, so its CTC
        # targets come from src_concepts (1..K), not the gloss order.
        result["speech_ctc_targets"] = torch.cat(
            [sc + 1 for sc in src_lists], dim=0)
        result["speech_ctc_lengths"] = lengths.clone()
    return result
