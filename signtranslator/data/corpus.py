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
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Reserved target-vocabulary tokens (shared with planner.py conventions).
PAD, BOS, EOS = 0, 1, 2
CONTENT_OFFSET = 3  # content ids start at 3 in src/gloss token vocabularies


def ctc_min_input_length(labels) -> int:
    """Exact minimum CTC frames: labels plus blanks between adjacent repeats."""
    seq = list(labels)
    return len(seq) + sum(a == b for a, b in zip(seq, seq[1:]))


def subsampled_length(length: int, factor: int) -> int:
    """Exact output length of the recognizer's stride-2 convolution stack."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if factor not in (1, 2, 4):
        raise ValueError("factor must be 1, 2, or 4")
    out = int(length)
    stride_left = factor
    while stride_left > 1:
        out = (out + 1) // 2
        stride_left //= 2
    return out


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
    num_source_tokens: int = 0 # 0 means same content cardinality as num_concepts

    @property
    def source_token_count(self) -> int:
        return self.num_source_tokens or self.num_concepts

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
            num_source_tokens=num_concepts,
        )


def _concept_trajectory(freqs, phases, amps, num_frames: int) -> np.ndarray:
    """Deterministic per-concept motion signature -> (C, t, V)."""
    t = np.linspace(0, 1, num_frames)[None, None, :]
    traj = amps[..., None] * np.sin(2 * np.pi * freqs[..., None] * t + phases[..., None])
    return np.transpose(traj, (1, 2, 0))  # (C, t, V)


def generate_corpus(out_dir: str, spec: Optional[CorpusSpec] = None,
                    counts: Optional[Dict[str, int]] = None,
                    noise: float = 0.02, seed: int = 0,
                    overwrite: bool = False) -> CorpusSpec:
    """Generate and write a synthetic corpus to ``out_dir``.

    Returns the :class:`CorpusSpec` (also persisted in ``manifest.json``).
    """
    if os.path.isdir(out_dir):
        entries = os.listdir(out_dir)
        if entries and not overwrite:
            raise FileExistsError(
                f"refusing to generate synthetic data in non-empty directory "
                f"{os.path.abspath(out_dir)!r}; choose an empty directory or pass "
                "overwrite=True explicitly")
    elif os.path.exists(out_dir):
        raise NotADirectoryError(f"corpus path is not a directory: {out_dir!r}")

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
                            lengths=lengths,
                            source_lengths=lengths,
                            motion_lengths=np.full(n, T, dtype=np.int64),
                            speech_lengths=np.full(n, spec.speech_frames, dtype=np.int64),
                            validity_mask=np.ones((n, T, V), dtype=np.bool_),
                            confidence=np.ones((n, T, V), dtype=np.float32),
                            frame_timestamps=np.broadcast_to(
                                np.arange(T, dtype=np.float64), (n, T)))
        split_counts[split] = int(n)
        if split == "train":
            train_pose = pose

    # Normalisation statistics are computed from the TRAIN split only (never
    # from val/test) and applied to every split -- standard practice that avoids
    # leaking held-out statistics into evaluation. Shape (C, 1, V): per channel
    # and joint, pooled over samples and time.
    if train_pose is None:
        # Fail loudly rather than silently normalising with val/test statistics
        # (a held-out leak). A corpus must have a 'train' split to fit stats.
        raise ValueError(
            "no 'train' split in `counts`; cannot compute normalisation "
            f"statistics without leaking held-out data (got splits: {list(counts)})")
    mean = train_pose.mean(axis=(0, 2), keepdims=True)[0]      # (C, 1, V)
    std = train_pose.std(axis=(0, 2), keepdims=True)[0]        # (C, 1, V)
    std = np.maximum(std, 1e-3)

    manifest = {"format_version": 1, "spec": asdict(spec),
                "splits": split_counts, "seed": seed,
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
    format_version = int(manifest.get("format_version", 1))
    if format_version not in (1, 2):
        raise ValueError(f"unsupported corpus format_version {format_version}")
    if format_version == 2:
        required_manifest = {
            "gloss_vocabulary", "source_vocabulary", "joint_names", "records",
            "coordinate_system", "extractor_id", "language", "normalization_fit_split",
            "landmark_parts",
        }
        missing_manifest = required_manifest - set(manifest)
        if missing_manifest:
            raise ValueError(f"manifest missing v2 fields {sorted(missing_manifest)}")
        if manifest["normalization_fit_split"] != "train":
            raise ValueError("v2 normalization must be fitted on the train split")
        gloss_vocabulary = manifest["gloss_vocabulary"]
        source_vocabulary = manifest["source_vocabulary"]
        joint_names = manifest["joint_names"]
        for name, values, expected in (
                ("gloss_vocabulary", gloss_vocabulary, spec.num_concepts),
                ("source_vocabulary", source_vocabulary, spec.source_token_count),
                ("joint_names", joint_names, spec.num_joints)):
            if (not isinstance(values, list) or len(values) != expected
                    or len(set(values)) != expected
                    or any(not isinstance(value, str) or not value for value in values)):
                raise ValueError(f"{name} must contain {expected} unique non-empty strings")
        landmark_parts = manifest["landmark_parts"]
        required_parts = {"body", "left_hand", "right_hand", "face"}
        if not isinstance(landmark_parts, dict) or set(landmark_parts) != required_parts:
            raise ValueError(f"landmark_parts must contain exactly {sorted(required_parts)}")
        flattened_parts = [name for part in required_parts for name in landmark_parts[part]]
        if (any(not landmark_parts[part] for part in required_parts)
                or len(flattened_parts) != spec.num_joints
                or set(flattened_parts) != set(joint_names)):
            raise ValueError("landmark_parts must partition joint_names exactly once")
        records = manifest["records"]
        if not isinstance(records, list) or len(records) != sum(manifest["splits"].values()):
            raise ValueError("manifest records must contain one entry per sample")
        record_ids = [record.get("sample_id") for record in records]
        if any(not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
            raise ValueError("manifest sample_id values must be unique and non-empty")
        for record in records:
            digest = record.get("media_sha256", "")
            if (record.get("split") not in manifest["splits"] or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest.lower())):
                raise ValueError("manifest record has invalid split or media SHA-256")
            if (not record.get("license") or not record.get("intended_use")
                    or not record.get("provenance")):
                raise ValueError("manifest record lacks governed usage or provenance")
            try:
                # Local import avoids a data <-> data_engineering initialization cycle.
                from ..data_engineering.schema import (
                    ConsentState, DataAuthorization, validate_authorization,
                )
                consent = ConsentState[record.get("consent", "")]
                authorization = DataAuthorization.from_manifest(record.get("authorization"))
                violations = validate_authorization(
                    authorization, consent, record["intended_use"],
                    requested_actions=("download", "create_derivatives", "model_training"))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("manifest record has invalid authorization") from error
            if authorization.license_identifier != record["license"] or violations:
                raise ValueError(
                    f"manifest record authorization is inconsistent: {violations}")
        record_split_counts = {
            split: sum(record["split"] == split for record in records)
            for split in manifest["splits"]
        }
        if record_split_counts != manifest["splits"]:
            raise ValueError("manifest record split counts disagree with splits")
    for split, n in manifest["splits"].items():
        path = os.path.join(corpus_dir, f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing split file: {path}")
        if format_version == 2:
            expected_hash = manifest.get("shard_sha256", {}).get(f"{split}.npz")
            if not expected_hash:
                raise ValueError(f"{split}: missing shard SHA-256")
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_hash:
                raise ValueError(f"{split}: shard SHA-256 mismatch")
        with np.load(path, allow_pickle=False) as z:
            pose, concepts, lengths = z["pose"], z["concepts"], z["lengths"]
            src_concepts = z["src_concepts"]
            files = set(z.files)
        if pose.shape != (n, spec.in_channels, spec.num_frames, spec.num_joints):
            raise ValueError(f"{split}: pose shape {pose.shape} != expected")
        if concepts.shape != (n, spec.seq_len):
            raise ValueError(f"{split}: concepts shape {concepts.shape} != expected")
        if src_concepts.ndim != 2 or src_concepts.shape[0] != n:
            raise ValueError(f"{split}: src_concepts must be (N, L_source)")
        if lengths.shape != (n,):
            raise ValueError(f"{split}: lengths shape {lengths.shape} != expected")
        with np.load(path, allow_pickle=False) as z:
            source_lengths = z["source_lengths"] if "source_lengths" in z.files else lengths
        if source_lengths.shape != (n,):
            raise ValueError(f"{split}: source_lengths shape {source_lengths.shape} != expected")
        if np.any(lengths < 1) or np.any(lengths > concepts.shape[1]):
            raise ValueError(f"{split}: invalid gloss lengths")
        if np.any(source_lengths < 1) or np.any(source_lengths > src_concepts.shape[1]):
            raise ValueError(f"{split}: invalid source lengths")
        limits = (("concepts", concepts, spec.num_concepts),
                  ("src_concepts", src_concepts, spec.source_token_count))
        for name, arr, limit in limits:
            if arr.min() < 0 or arr.max() >= limit:
                raise ValueError(f"{split}: {name} out of range [0, {limit})")
        if not np.isfinite(pose).all():
            raise ValueError(f"{split}: non-finite pose values")
        with np.load(path, allow_pickle=False) as z:
            if "speech" in z.files:
                speech = z["speech"]
                if speech.shape != (n, spec.speech_frames, spec.speech_dim):
                    raise ValueError(f"{split}: speech shape {speech.shape} != expected")
                if not np.isfinite(speech).all():
                    raise ValueError(f"{split}: non-finite speech values")
            if format_version == 2:
                required = {"motion_lengths", "validity_mask", "confidence",
                            "frame_timestamps", "sample_ids", "source_lengths"}
                missing = required - files
                if missing:
                    raise ValueError(f"{split}: missing v2 arrays {sorted(missing)}")
                motion_lengths = z["motion_lengths"]
                validity = z["validity_mask"]
                confidence = z["confidence"]
                timestamps = z["frame_timestamps"]
                sample_ids = z["sample_ids"]
                if motion_lengths.shape != (n,):
                    raise ValueError(f"{split}: motion_lengths must be (N,)")
                if validity.shape != (n, spec.num_frames, spec.num_joints):
                    raise ValueError(f"{split}: validity_mask shape mismatch")
                if confidence.shape != validity.shape or timestamps.shape != (n, spec.num_frames):
                    raise ValueError(f"{split}: confidence/timestamp shape mismatch")
                if validity.dtype != np.bool_:
                    raise TypeError(f"{split}: validity_mask must be boolean")
                if sample_ids.shape != (n,) or len(set(sample_ids.tolist())) != n:
                    raise ValueError(f"{split}: sample_ids must be unique with shape (N,)")
                expected_ids = {record["sample_id"] for record in manifest["records"]
                                if record["split"] == split}
                if set(sample_ids.tolist()) != expected_ids:
                    raise ValueError(f"{split}: shard sample_ids disagree with manifest records")
                if not np.isfinite(confidence).all() or np.any(
                        (confidence < 0) | (confidence > 1)):
                    raise ValueError(f"{split}: confidence outside [0, 1]")
                if np.any(confidence[~validity] != 0):
                    raise ValueError(f"{split}: invalid observations have confidence")
                for row, frame_count in enumerate(motion_lengths.tolist()):
                    if not 1 <= frame_count <= spec.num_frames:
                        raise ValueError(f"{split}: invalid motion length at row {row}")
                    ts = timestamps[row, :frame_count]
                    if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                        raise ValueError(f"{split}: invalid timestamps at row {row}")
                    if np.any(timestamps[row, frame_count:] != -1):
                        raise ValueError(f"{split}: timestamp padding must be -1")
                    if (np.any(validity[row, frame_count:])
                            or np.any(confidence[row, frame_count:] != 0)
                            or np.any(pose[row, :, frame_count:, :] != 0)):
                        raise ValueError(f"{split}: motion padding must be zero and invalid")
                    gloss_length = int(lengths[row])
                    source_length = int(source_lengths[row])
                    if (np.any(concepts[row, gloss_length:] != 0)
                            or np.any(src_concepts[row, source_length:] != 0)):
                        raise ValueError(f"{split}: token padding must be zero")
                    motion_minimum = ctc_min_input_length(
                        concepts[row, :gloss_length].tolist())
                    if frame_count < motion_minimum:
                        raise ValueError(
                            f"{split}: row {row} motion is not exactly CTC-feasible")
                has_speech = "speech" in files
                speech_aux = {"speech_lengths", "speech_timestamps"}
                present_speech_aux = speech_aux & files
                if ((has_speech and present_speech_aux != speech_aux)
                        or (not has_speech and present_speech_aux)):
                    raise ValueError(f"{split}: speech arrays must appear together")
                if has_speech:
                    speech = z["speech"]
                    speech_lengths = z["speech_lengths"]
                    speech_timestamps = z["speech_timestamps"]
                    if speech_lengths.shape != (n,) or speech_timestamps.shape != (
                            n, spec.speech_frames):
                        raise ValueError(f"{split}: speech length/timestamp shape mismatch")
                    for row, speech_length in enumerate(speech_lengths.tolist()):
                        if not 1 <= speech_length <= spec.speech_frames:
                            raise ValueError(f"{split}: invalid speech length at row {row}")
                        speech_ts = speech_timestamps[row, :speech_length]
                        if (not np.isfinite(speech_ts).all()
                                or np.any(np.diff(speech_ts) <= 0)):
                            raise ValueError(f"{split}: invalid speech timestamps at row {row}")
                        if (np.any(speech_timestamps[row, speech_length:] != -1)
                                or np.any(speech[row, speech_length:] != 0)):
                            raise ValueError(f"{split}: speech padding must be zero/-1")
                        speech_minimum = ctc_min_input_length(
                            src_concepts[row, :int(source_lengths[row])].tolist())
                        speech_usable = subsampled_length(
                            speech_length, int(manifest.get("speech_subsample", 2)))
                        if speech_usable < speech_minimum:
                            raise ValueError(
                                f"{split}: row {row} speech is not exactly CTC-feasible")
    mean = np.asarray(manifest.get("pose_mean"), dtype=np.float64)
    std = np.asarray(manifest.get("pose_std"), dtype=np.float64)
    expected_stats = (spec.in_channels, 1, spec.num_joints)
    if mean.shape != expected_stats or std.shape != expected_stats:
        raise ValueError("pose normalization statistic shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("pose normalization statistics must be finite with positive std")
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
                 normalize: bool = True, validate: bool = True) -> None:
        if validate:
            validate_corpus(corpus_dir)
        manifest = load_manifest(corpus_dir)
        self.spec = CorpusSpec(**manifest["spec"])
        self.standardizer = PoseStandardizer.from_manifest(manifest)
        with np.load(os.path.join(corpus_dir, f"{split}.npz")) as z:
            self.pose = torch.from_numpy(z["pose"])
            self.concepts = torch.from_numpy(z["concepts"])
            self.src_concepts = torch.from_numpy(z["src_concepts"])
            self.lengths = torch.from_numpy(z["lengths"])
            self.source_lengths = (torch.from_numpy(z["source_lengths"])
                                   if "source_lengths" in z.files else self.lengths)
            self.motion_lengths = (torch.from_numpy(z["motion_lengths"])
                                   if "motion_lengths" in z.files else
                                   torch.full((len(self.lengths),), self.spec.num_frames,
                                              dtype=torch.long))
            self.validity_mask = (torch.from_numpy(z["validity_mask"])
                                  if "validity_mask" in z.files else None)
            self.confidence = (torch.from_numpy(z["confidence"])
                               if "confidence" in z.files else None)
            self.frame_timestamps = (torch.from_numpy(z["frame_timestamps"])
                                     if "frame_timestamps" in z.files else None)
            self.sample_ids = (np.array(z["sample_ids"], copy=True)
                               if "sample_ids" in z.files else None)
            self.speech = (torch.from_numpy(z["speech"]) if "speech" in z.files
                           else None)
            self.speech_lengths = (torch.from_numpy(z["speech_lengths"])
                                   if "speech_lengths" in z.files else None)
            self.speech_timestamps = (torch.from_numpy(z["speech_timestamps"])
                                      if "speech_timestamps" in z.files else None)
        if normalize:
            self.pose = self.standardizer.normalize(self.pose)
            if self.validity_mask is not None:
                # Invalid/padded observations must remain a neutral numeric value after
                # normalization; their false mask and zero confidence retain missingness.
                valid = self.validity_mask.unsqueeze(1).expand_as(self.pose)
                self.pose = torch.where(valid, self.pose, torch.zeros_like(self.pose))

    def __len__(self) -> int:
        return self.pose.shape[0]

    def __getitem__(self, idx: int) -> dict:
        n = int(self.lengths[idx])
        source_n = int(self.source_lengths[idx])
        item = {"pose": self.pose[idx],
                "concepts": self.concepts[idx, :n],
                "src_concepts": self.src_concepts[idx, :source_n],
                "motion_length": int(self.motion_lengths[idx])}
        motion_length = item["motion_length"]
        item["pose"] = item["pose"][:, :motion_length]
        if self.validity_mask is not None:
            item["validity_mask"] = self.validity_mask[idx, :motion_length]
        if self.confidence is not None:
            item["confidence"] = self.confidence[idx, :motion_length]
        if self.frame_timestamps is not None:
            item["frame_timestamps"] = self.frame_timestamps[idx, :motion_length]
        if self.sample_ids is not None:
            item["sample_id"] = str(self.sample_ids[idx])
        if self.speech is not None:
            speech_length = (int(self.speech_lengths[idx])
                             if self.speech_lengths is not None else self.speech.shape[1])
            item["speech"] = self.speech[idx, :speech_length]
            item["speech_length"] = speech_length
            if self.speech_timestamps is not None:
                item["speech_timestamps"] = self.speech_timestamps[idx, :speech_length]
        return item


def collate_corpus(batch: List[dict], speech_subsample: int = 2) -> dict:
    """Derive every branch's tensors from the concept sequences.

    Produces a batch dict with keys consumed by ``BidirectionalSignTranslator``:
    ``pose``, ``gloss_tokens``, ``src``, ``gloss_seq``, ``ctc_targets``,
    ``ctc_lengths`` (and ``concepts`` for reference).
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")
    motion_lengths = torch.tensor([int(b["motion_length"]) for b in batch],
                                  dtype=torch.long)
    pose = pad_sequence([b["pose"].permute(1, 0, 2) for b in batch],
                        batch_first=True).permute(0, 2, 1, 3)
    frame_mask = (torch.arange(pose.shape[2]).unsqueeze(0)
                  < motion_lengths.unsqueeze(1))
    concept_lists = [b["concepts"] for b in batch]
    src_lists = [b["src_concepts"] for b in batch]
    lengths = torch.tensor([len(c) for c in concept_lists], dtype=torch.long)
    source_lengths = torch.tensor([len(c) for c in src_lists], dtype=torch.long)
    max_len = int(lengths.max())
    max_source_len = int(source_lengths.max())

    n = len(batch)
    gloss_tokens = torch.zeros(n, max_len, dtype=torch.long)       # content+3, PAD=0
    src = torch.zeros(n, max_source_len, dtype=torch.long)          # source content+3
    gloss_seq = torch.full((n, max_len + 2), PAD, dtype=torch.long)  # BOS..EOS
    ctc_parts: List[torch.Tensor] = []

    for i, (c, sc) in enumerate(zip(concept_lists, src_lists)):
        L = len(c)
        min_motion = ctc_min_input_length(c.tolist())
        if int(motion_lengths[i]) < min_motion:
            raise ValueError(
                f"sample {i}: motion CTC requires {min_motion} frames for target "
                f"{c.tolist()}, got {int(motion_lengths[i])}")
        content = c + CONTENT_OFFSET                    # gloss token ids
        gloss_tokens[i, :L] = content
        src[i, :len(sc)] = sc + CONTENT_OFFSET          # source-language tokens
        gloss_seq[i, 0] = BOS
        gloss_seq[i, 1:1 + L] = content
        gloss_seq[i, 1 + L] = EOS
        ctc_parts.append(c + 1)                          # CTC ids in 1..K

    ctc_targets = torch.cat(ctc_parts, dim=0)
    ctc_lengths = lengths.clone()

    has_speech = all("speech" in b for b in batch)
    if any("speech" in b for b in batch) and not has_speech:
        raise ValueError("speech modality must be present for every sample or none")

    result = {
        "pose": pose,
        "motion_lengths": motion_lengths,
        "frame_mask": frame_mask,
        "concepts": concept_lists,
        "gloss_tokens": gloss_tokens,
        "src": src,
        "gloss_seq": gloss_seq,
        "ctc_targets": ctc_targets,
        "ctc_lengths": ctc_lengths,
        "ctc_input_lengths": motion_lengths.clone(),
    }
    if all("validity_mask" in b for b in batch):
        result["validity_mask"] = pad_sequence(
            [b["validity_mask"] for b in batch], batch_first=True,
            padding_value=False)
    if all("confidence" in b for b in batch):
        result["confidence"] = pad_sequence(
            [b["confidence"] for b in batch], batch_first=True,
            padding_value=0.0)
    if all("frame_timestamps" in b for b in batch):
        result["frame_timestamps"] = pad_sequence(
            [b["frame_timestamps"] for b in batch], batch_first=True,
            padding_value=-1.0)
    if all("sample_id" in b for b in batch):
        result["sample_ids"] = [b["sample_id"] for b in batch]
    if has_speech:
        speech_lengths = torch.tensor([int(b["speech_length"]) for b in batch],
                                      dtype=torch.long)
        result["speech"] = pad_sequence([b["speech"] for b in batch], batch_first=True)
        result["speech_input_lengths"] = speech_lengths
        # The speech branch predicts the *spoken* token sequence, so its CTC
        # targets come from src_concepts (1..K), not the gloss order.
        result["speech_ctc_targets"] = torch.cat(
            [sc + 1 for sc in src_lists], dim=0)
        result["speech_ctc_lengths"] = source_lengths.clone()
        if all("speech_timestamps" in b for b in batch):
            result["speech_timestamps"] = pad_sequence(
                [b["speech_timestamps"] for b in batch], batch_first=True,
                padding_value=-1.0)
        for i, sc in enumerate(src_lists):
            usable = subsampled_length(int(speech_lengths[i]), speech_subsample)
            minimum = ctc_min_input_length(sc.tolist())
            if usable < minimum:
                raise ValueError(
                    f"sample {i}: speech CTC requires {minimum} post-subsample "
                    f"frames for target {sc.tolist()}, got {usable}")
    return result
