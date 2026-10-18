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


def collate_text_motion(items, *, max_text_bytes: int = 256):
    texts, states = zip(*items)
    tokens, mask = encode_texts(texts, max_bytes=max_text_bytes)
    return tokens, mask, collate_multichannel(states), list(texts)


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


class R2Store:
    """Object store backed by Cloudflare R2; credentials come from the environment.

    Reads ``R2_ENDPOINT_URL``, ``R2_BUCKET``, ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY`` (for example Runpod secrets); nothing is logged.
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        import boto3
        from botocore.config import Config
        missing = [k for k in ("R2_ENDPOINT_URL", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") if not env.get(k)]
        if missing:
            raise ValueError(f"missing R2 settings: {missing}")
        self.bucket = env["R2_BUCKET"]
        self.s3 = boto3.client("s3", endpoint_url=env["R2_ENDPOINT_URL"], region_name="auto",
                               aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                               aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                               config=Config(connect_timeout=20, read_timeout=300,
                                             retries={"mode": "standard", "max_attempts": 8}))

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

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
