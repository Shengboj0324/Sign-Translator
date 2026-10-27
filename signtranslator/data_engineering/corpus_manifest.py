"""Content-addressed corpus manifest and verified, size-bounded state loading.

The manifest is the single integration point between acquisition, conversion,
cross-modal validation and training. Each sign gets one record that joins its
evidence, and every downstream choice is a function of that record:

* the **skeleton sample** is usable only if marker validation passed; its hand
  channels are usable only if the glove validation of that hand passed;
* the **face sample** lives on its own device clock and is a separate sample;
  it is never merged frame-wise with the skeleton sample;
* gloss candidates are carried for review and never select a vocabulary entry.

States live in object storage. :class:`ContentCache` keeps at most a declared
number of bytes locally, named by SHA-256, and every byte returned to a caller
has been hashed against the manifest, whether it came from the network or disk.
:func:`load_sample` then masks every channel the manifest does not admit, so a
consumer cannot silently train on an unvalidated hand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping

import numpy as np

from ..pose.multichannel import CHANNELS, MotionChannel, MultichannelMotion

MANIFEST_SCHEMA_VERSION = 1
SKELETON_CHANNELS = ("root_translation", "body", "head_rotation")
FACE_CHANNELS = ("face", "left_gaze", "right_gaze", "blink")
HAND_CHANNELS = {"Left": "left_hand", "Right": "right_hand"}
KINDS = ("skeleton", "face", "continuous")
_SHA = set("0123456789abcdef")


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA


def usable_channels(record: Mapping, kind: str) -> tuple[str, ...]:
    """Channels the evidence admits for one sample kind, in canonical order."""
    if kind == "skeleton":
        skeleton = record.get("skeleton") or {}
        if not skeleton.get("state_sha256") or skeleton.get("marker_validation") != "passed":
            return ()
        names = set(skeleton.get("available_channels", ())) & set(SKELETON_CHANNELS)
        for side, channel in HAND_CHANNELS.items():
            if (record.get("hands") or {}).get(side, {}).get("status") == "validated":
                names.add(channel)
        return tuple(name for name in CHANNELS if name in names)
    if kind == "face":
        face = record.get("face") or {}
        if face.get("status") != "converted" or not face.get("state_sha256"):
            return ()
        return tuple(name for name in CHANNELS if name in FACE_CHANNELS)
    if kind == "continuous":
        # Video-estimated states: every channel the estimator produced shares one
        # clock; per-frame validity masks inside the state carry the rest.
        part = record.get("continuous") or {}
        if part.get("status") != "converted" or not part.get("state_sha256"):
            return ()
        names = set(part.get("available_channels", ()))
        return tuple(name for name in CHANNELS if name in names)
    raise ValueError("kind must be 'skeleton', 'face' or 'continuous'")


def build_corpus_manifest(*, corpus_id: str, inputs: Mapping[str, str], records: list[dict],
                          split: Mapping) -> dict:
    """Assemble and self-check a manifest; the result is deterministic in its inputs."""
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        raise ValueError("corpus_id must be named")
    if not inputs or not all(_is_sha(v) for v in inputs.values()):
        raise ValueError("inputs must map names to SHA-256 digests")
    signs = [r.get("sign") for r in records]
    if len(set(signs)) != len(signs) or any(not isinstance(s, str) or not s for s in signs):
        raise ValueError("records need unique sign identifiers")
    ordered = sorted((dict(r) for r in records), key=lambda r: r["sign"])
    for record in ordered:
        for kind in KINDS:
            part = record.get(kind) or {}
            if part.get("state_sha256") is not None and not _is_sha(part["state_sha256"]):
                raise ValueError(f"{record['sign']}: {kind} state hash is malformed")
        record["usable"] = {kind: list(usable_channels(record, kind)) for kind in KINDS}
    counts = collections.Counter()
    for record in ordered:
        for kind in KINDS:
            if record["usable"][kind]:
                counts[f"{kind}_samples"] += 1
        for side, channel in HAND_CHANNELS.items():
            if channel in record["usable"]["skeleton"]:
                counts[f"{channel}_validated"] += 1
        counts[f"gloss_{(record.get('gloss') or {}).get('status', 'absent')}"] += 1
    manifest = dict(schema_version=MANIFEST_SCHEMA_VERSION, corpus_id=corpus_id,
                    inputs=dict(sorted(inputs.items())), split=dict(split), counts=dict(sorted(counts.items())),
                    records=ordered, phase_exit_approved=False,
                    rules=["skeleton sample requires passed marker validation",
                           "a hand channel requires its own validated glove comparison",
                           "the face sample keeps its own clock and is never merged with the skeleton",
                           "gloss candidates require review and select nothing"])
    canonical_json(manifest)  # refuses NaN or non-JSON values
    return manifest


def manifest_sha256(manifest: Mapping) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def write_manifest(manifest: Mapping, path: Path) -> str:
    """Write the canonical bytes once; never overwrite."""
    payload = canonical_json(manifest)
    with Path(path).open("xb") as stream:
        stream.write(payload + b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path, *, expected_sha256: str) -> dict:
    data = Path(path).read_bytes()
    if not data.endswith(b"\n") or hashlib.sha256(data[:-1]).hexdigest() != expected_sha256:
        raise ValueError("manifest content hash mismatch")
    manifest = json.loads(data[:-1])
    if canonical_json(manifest) != data[:-1]:
        raise ValueError("manifest bytes are not canonical")
    for record in manifest["records"]:
        for kind in KINDS:
            if record["usable"].get(kind, []) != list(usable_channels(record, kind)):
                raise ValueError(f"{record['sign']}: stored usable channels disagree with the rule")
    return manifest


class ContentCache:
    """At most ``max_bytes`` of verified payloads on disk, evicted least recently used."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.fetches = 0
        self._total = None                       # running estimate of the bytes on disk (scanned once, then tracked)

    def _path(self, sha256: str) -> Path:
        if not _is_sha(sha256):
            raise ValueError("a lowercase SHA-256 is required")
        return self.directory / sha256

    def size(self) -> int:
        return sum(p.stat().st_size for p in self.directory.iterdir() if _is_sha(p.name))

    def get(self, key: str, sha256: str, fetch: Callable[[str], bytes]) -> bytes:
        path = self._path(sha256)
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == sha256:
                os.utime(path)
                return data
            path.unlink()  # corrupted at rest: refetch rather than trust it
        data = fetch(key)
        self.fetches += 1
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"fetched bytes for {key} do not match the manifest hash")
        if len(data) <= self.max_bytes:
            with tempfile.NamedTemporaryFile(dir=self.directory, delete=False, prefix=".partial-") as tmp:
                tmp.write(data)
            os.replace(tmp.name, path)
            if self._total is None:
                self._total = self.size()
            else:
                self._total += len(data)
            if self._total > self.max_bytes:     # scan and evict only when over budget, not on every insert
                self._evict(keep=path)
        return data

    def _evict(self, *, keep: Path) -> None:
        entries = sorted((p for p in self.directory.iterdir() if _is_sha(p.name) and p != keep),
                         key=lambda p: p.stat().st_mtime_ns)
        total = self.size()                      # the true size: other processes may share the directory
        for entry in entries:
            if total <= self.max_bytes:
                break
            total -= entry.stat().st_size
            entry.unlink()
        self._total = total


def mask_channels(state: MultichannelMotion, keep) -> MultichannelMotion:
    """Return a copy whose channels outside ``keep`` carry no valid value."""
    keep = set(keep)
    if keep - set(CHANNELS):
        raise ValueError("unknown channel")
    channels = {}
    for name, channel in state.channels.items():
        if name in keep:
            channels[name] = channel
            continue
        shape = channel.valid.shape
        channels[name] = MotionChannel(
            values=np.zeros_like(channel.values), valid=np.zeros(shape, dtype=bool),
            observed=np.zeros(shape, dtype=bool), confidence=np.zeros_like(channel.confidence),
            labels=channel.labels, coordinate_frame=channel.coordinate_frame, units=channel.units,
            source_id=channel.source_id, source_sha256=channel.source_sha256,
            convention=channel.convention, inference_method=None)
    return replace(state, channels=channels)


def load_sample(record: Mapping, kind: str, *, cache: ContentCache,
                fetch: Callable[[str], bytes]) -> MultichannelMotion:
    """Fetch, verify and decode one sample with every unadmitted channel masked.

    A record with ``"mirror": true`` is returned left-right mirrored (``signtranslator.pose.mirror``, exact), after
    masking: right-dominant canonical form without rewriting the stored state (round 5 amendment 4)."""
    usable = usable_channels(record, kind)
    if not usable:
        raise ValueError(f"{record.get('sign')}: no usable {kind} channels")
    part = record[kind]
    data = cache.get(part["state_key"], part["state_sha256"], fetch)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "state.npz"
        path.write_bytes(data)
        state = MultichannelMotion.load(path, expected_sha256=part["state_sha256"])
    state = mask_channels(state, usable)
    if record.get("mirror") is True:
        from ..pose.mirror import mirror_state
        state = mirror_state(state, sample_id=state.sample_id)
    return state


def performer_split(records: list[dict], kind: str = "skeleton") -> dict:
    """Descriptive split: train on the largest performer group, test on the rest.

    With fewer than three independent groups no leave-one-group-out plan exists,
    and the record says so instead of inventing confirmatory folds.
    """
    groups = collections.Counter(r["performer_sha256"] for r in records
                                 if r.get("performer_sha256") and usable_channels(r, kind))
    if len(groups) < 2:
        raise ValueError("a performer-held-out split needs at least two performer groups")
    ordered = sorted(groups.items(), key=lambda item: (-item[1], item[0]))
    return dict(kind=kind, train_performer_sha256=ordered[0][0],
                test_performer_sha256=[p for p, _ in ordered[1:]],
                samples_by_performer=dict(sorted(groups.items())), independent_units=len(groups),
                confirmatory=len(groups) >= 3,
                note=("descriptive only: fewer than three independent performer groups"
                      if len(groups) < 3 else "leave-one-group-out folds are available"))


__all__ = ["MANIFEST_SCHEMA_VERSION", "canonical_json", "usable_channels", "build_corpus_manifest",
           "manifest_sha256", "write_manifest", "load_manifest", "ContentCache", "mask_channels",
           "load_sample", "performer_split"]
