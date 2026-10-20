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
                 fetch: Callable[[str], bytes], max_seconds: float | None = None) -> None:
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}")
        records = [r for r in manifest["records"]
                   if r.get("partition") == partition and r.get("text") and usable_channels(r, "continuous")]
        if max_seconds is not None:
            records = [r for r in records if r["continuous"].get("seconds", 0.) <= max_seconds]
        self.records, self.cache, self.fetch = records, cache, fetch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return record["text"], load_sample(record, "continuous", cache=self.cache, fetch=self.fetch)


def collate_text_motion(items, *, max_text_bytes: int = 256, loci: bool = False):
    """(tokens, mask, motion, texts) — plus (B, SLOT_WIDTH) referent-locus slots measured in each
    state's own pointing (``grammar_motion.loci_conditioning.state_loci``) when ``loci``."""
    texts, states = zip(*items)
    tokens, mask = encode_texts(texts, max_bytes=max_text_bytes)
    batch = (tokens, mask, collate_multichannel(states), list(texts))
    if loci:
        import numpy as np
        import torch
        from ..grammar_motion.loci_conditioning import state_loci
        batch = batch + (torch.as_tensor(np.stack([state_loci(s) for s in states]), dtype=torch.float32),)
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
    """botocore/urllib3 exception types raised when a response body stream breaks."""
    out = []
    try:
        from botocore.exceptions import ConnectionClosedError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError
        out += [ConnectionClosedError, IncompleteReadError, ReadTimeoutError, ResponseStreamingError]
    except ImportError:
        pass
    try:
        from urllib3.exceptions import ProtocolError
        out.append(ProtocolError)
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
                                        config=Config(connect_timeout=20, read_timeout=300,
                                                      retries={"mode": "adaptive", "max_attempts": 20}))  # R2 throttles (429) under many readers
        return self._client

    def __getstate__(self):
        # Data-loader workers started by "spawn" (macOS, Windows) receive the store by
        # pickling; they rebuild their own client. The settings stay in process memory.
        state = dict(self.__dict__)
        state["_client"] = None
        return state

    def __repr__(self) -> str:
        return f"R2Store(bucket={self.bucket!r})"              # never shows credentials

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
                return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except ClientError:
                raise
            except (OSError, ConnectionError) + _stream_errors() as error:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(30., .5 * 2 ** attempt) * (.5 + random.random()))

    def put(self, key: str, payload: bytes) -> str:
        digest = hashlib.sha256(payload).hexdigest()
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload, Metadata={"sha256": digest})
        if hashlib.sha256(self.get(key)).hexdigest() != digest:
            raise IOError(f"R2 re-read mismatch for {key}")
        return digest

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
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
            page = self.s3.list_objects_v2(**kwargs)
            keys += [o["Key"] for o in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return sorted(keys)
            token = page["NextContinuationToken"]


__all__ = ["TextMotionDataset", "collate_text_motion", "DirectoryStore", "R2Store", "PARTITIONS"]
