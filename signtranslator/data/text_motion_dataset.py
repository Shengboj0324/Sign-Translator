"""Sentence-aligned text and canonical motion from a frozen manifest.

Each record carries its English text, its ``continuous`` state reference and a
``partition`` assigned when the manifest was built from a frozen signer-disjoint
fold plan. The dataset serves one partition only, so a validation or test signer
can never reach training through this path. States are fetched through
:class:`ContentCache` (hash-checked) with unadmitted channels masked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping
import hashlib

from torch.utils.data import Dataset

from ..data.multichannel import collate_multichannel
from ..data_engineering.corpus_manifest import ContentCache, load_sample, usable_channels
from ..generation.text_to_motion import encode_texts

PARTITIONS = ("train", "val", "test")


class TextMotionDataset(Dataset):
    def __init__(self, manifest: Mapping, *, partition: str, cache: ContentCache,
                 fetch: Callable[[str], bytes], max_seconds: float | None = None, fps: float | None = None,
                 augment: Callable | None = None, template=None, word_spans: bool = False,
                 max_text_bytes: int = 256) -> None:
        """``fps``: resample every state whose median rate differs from it by more than 1%
        (``pose.resample.resample_motion``) - corpora are recorded at 24, 25-30 and 30 fps, and the tokenizers and
        priors count time in frames. ``None`` keeps every state at its own rate.

        ``word_spans`` (opt-in, alignment supervision): items become (text, state, spans), spans the (T, 2) per-frame
        memory positions of the word being signed (``motion_rvq.frame_word_spans``) for records that carry
        ``sign_spans`` — a list of ``{"gloss": label, "seconds": [start, end]}`` relative to the state's first frame
        (docs/research/alignment-supervision-design.md) — and None for the others."""
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        records = [r for r in manifest["records"]
                   if r.get("partition") == partition and r.get("text") and usable_channels(r, "continuous")]
        if max_seconds is not None:
            records = [r for r in records if r["continuous"].get("seconds", 0.) <= max_seconds]
        self.records, self.cache, self.fetch, self.fps, self.augment = records, cache, fetch, fps, augment
        self.template = template
        self.word_spans, self.max_text_bytes = word_spans, max_text_bytes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        state = load_sample(record, "continuous", cache=self.cache, fetch=self.fetch)
        if self.template is not None:
            from .layout_adapter import conform
            state = conform(state, self.template)
        if self.augment is not None:
            state = self.augment(state)
        if self.fps is not None and len(state.timestamps) > 1:
            import numpy as np
            rate = 1. / float(np.median(np.diff(state.timestamps)))
            if abs(rate - self.fps) > .01 * self.fps:
                from ..pose.resample import resample_motion
                state = resample_motion(state, self.fps)
        if self.word_spans:
            spans = None
            if record.get("sign_spans"):
                from ..generation.motion_rvq import frame_word_spans
                spans = frame_word_spans(record["text"], [(s["gloss"], *s["seconds"]) for s in record["sign_spans"]],
                                         state.timestamps, max_bytes=self.max_text_bytes)
            return record["text"], state, spans
        return record["text"], state


def collate_text_motion(items, *, max_text_bytes: int = 256, loci: bool = False, word_spans: bool = False):
    """(tokens, mask, motion, texts) — plus (B, SLOT_WIDTH) referent-locus slots measured in each
    state's own pointing (``grammar_motion.loci_conditioning.state_loci``) when ``loci``, and, last, when
    ``word_spans`` (items (text, state, spans or None)), the (B, T, 2) int64 per-frame word spans padded with −1
    (rows without spans are all −1)."""
    if word_spans:
        texts, states, spans = zip(*items)
    else:
        texts, states = zip(*items)
    tokens, mask = encode_texts(texts, max_bytes=max_text_bytes)
    motion = collate_multichannel(states)
    batch = (tokens, mask, motion, list(texts))
    if loci:
        import numpy as np
        import torch
        from ..grammar_motion.loci_conditioning import state_loci
        batch = batch + (torch.as_tensor(np.stack([state_loci(s) for s in states]), dtype=torch.float32),)
    if word_spans:
        import torch
        out = torch.full((len(states), motion.frame_valid.shape[1], 2), -1, dtype=torch.long)
        for row, span in enumerate(spans):
            if span is not None:
                span = torch.as_tensor(span, dtype=torch.long)[:out.shape[1]]
                out[row, :len(span)] = span
        batch = batch + (out,)
    return batch


class DirectoryStore:
    """Object store backed by a local directory (tests, or a Runpod volume mirror)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def put(self, key: str, payload: bytes) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())


def _stream_errors() -> tuple:
    """botocore/urllib3 exception types raised when a response body stream breaks.

    Includes TLS failures mid-body (urllib3 ``SSLError``, e.g. "[SSL] record layer failure"),
    which are not ``ProtocolError``s: one stopped a v3 tokenizer run in its first minutes
    (2026-10-21, pod in Australia).
    """
    out = []
    try:
        from botocore.exceptions import (ConnectionClosedError, IncompleteReadError, ReadTimeoutError,
                                         ResponseStreamingError, SSLError as BotoSSLError)
        out += [ConnectionClosedError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError, BotoSSLError]
    except ImportError:
        pass
    try:
        from urllib3.exceptions import ProtocolError, SSLError as Urllib3SSLError
        out += [ProtocolError, Urllib3SSLError]
    except ImportError:
        pass
    return tuple(out)


class R2Store:
    """Object store backed by Cloudflare R2; credentials come from the environment.

    Reads ``R2_ENDPOINT_URL``, ``R2_BUCKET``, ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY`` (for example Runpod secrets); nothing is logged.
    """

    _SETTINGS = ("R2_ENDPOINT_URL", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")

    def __init__(self, env: Mapping[str, str]) -> None:
        missing = [k for k in self._SETTINGS if not env.get(k)]
        if missing:
            raise ValueError(f"missing R2 settings: {missing}")
        self._settings = {k: env[k] for k in self._SETTINGS}
        self.bucket = env["R2_BUCKET"]
        self._client = None

    @property
    def s3(self):
        """The boto3 client, created on first use in each process (clients cannot be pickled)."""
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client("s3", endpoint_url=self._settings["R2_ENDPOINT_URL"], region_name="auto",
                                        aws_access_key_id=self._settings["AWS_ACCESS_KEY_ID"],
                                        aws_secret_access_key=self._settings["AWS_SECRET_ACCESS_KEY"],
                                        # R2 throttles (429) under many readers: adaptive retries. read_timeout bounds a
                                        # stalled response (each socket read, not a whole transfer); at 300 s one stalled
                                        # log upload held a training run for 6 minutes (2026-10-21), and puts and gets are
                                        # idempotent, so a shorter wait and a retry cost nothing.
                                        config=Config(connect_timeout=20, read_timeout=60,
                                                      retries={"mode": "adaptive", "max_attempts": 20}))
        return self._client

    def __getstate__(self):
        # Data-loader workers started by "spawn" (macOS, Windows) receive the store by
        # pickling; they rebuild their own client. The settings stay in process memory.
        state = dict(self.__dict__)
        state["_client"] = None
        return state

    def __repr__(self) -> str:
        return f"R2Store(bucket={self.bucket!r})"              # never shows credentials

    @staticmethod
    def _throttled(error) -> bool:
        """R2 throttling or a transient outage. A 429 on HEAD has no body, hence no error code,
        so botocore's adaptive retries miss it; the status code is checked as well."""
        response = getattr(error, "response", None) or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return status in (429, 500, 502, 503, 504) or code in (
            "429", "503", "SlowDown", "TooManyRequests", "ServiceUnavailable", "InternalError")

    def _call(self, function, attempts: int = 10):
        """``function()`` with throttled responses retried under jittered exponential backoff."""
        import random
        import time
        from botocore.exceptions import ClientError
        for attempt in range(attempts):
            try:
                return function()
            except ClientError as error:
                if not self._throttled(error) or attempt == attempts - 1:
                    raise
                time.sleep(min(60., .5 * 2 ** attempt) * (.5 + random.random()))

    def get(self, key: str, attempts: int = 8) -> bytes:
        """The object's bytes; a body read cut off mid-stream (broken pipe, reset, short read) is retried.

        botocore retries the request but not the streaming of its body, which R2 can drop
        under load; missing keys and other client errors are raised at once.
        """
        import random
        import time
        from botocore.exceptions import ClientError
        for attempt in range(attempts):
            try:
                return self._call(lambda: self.s3.get_object(Bucket=self.bucket, Key=key))["Body"].read()
            except ClientError:
                raise
            except (OSError, ConnectionError) + _stream_errors() as error:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(30., .5 * 2 ** attempt) * (.5 + random.random()))

    def put(self, key: str, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        self._call(lambda: self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload, Metadata={"sha256": digest}))
        if hashlib.sha256(self.get(key)).hexdigest() != digest:
            raise IOError(f"R2 re-read mismatch for {key}")
        return digest

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._call(lambda: self.s3.head_object(Bucket=self.bucket, Key=key))
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            kwargs = dict(Bucket=self.bucket, Prefix=prefix)
            if token:
                kwargs["ContinuationToken"] = token
            page = self._call(lambda: self.s3.list_objects_v2(**kwargs))
            keys += [o["Key"] for o in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return sorted(keys)
            token = page["NextContinuationToken"]


__all__ = ["TextMotionDataset", "collate_text_motion", "DirectoryStore", "R2Store", "PARTITIONS"]
