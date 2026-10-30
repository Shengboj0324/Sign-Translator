"""Stream one large file from a URL straight into the store (R2) as a multipart upload, with no local copy.

The SHA-256 and byte count are computed while streaming; the byte count must equal the server's Content-Length.
``<key>.INDEX.json`` records the source, size, SHA-256, the server's Last-Modified and ETag, the licence and a note
(for example who accepted the licence and when). Data only: nothing is unpacked or executed.

    python scripts/acquire_url_to_store.py --url https://download.microsoft.com/.../ASL_STEM_Wiki.zip \
        --key acquisition/2026-10-30-v1/asl-stem-wiki/ASL_STEM_Wiki.zip --licence "Microsoft Research License"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Hashing:
    """A read-only stream that hashes and counts what passes through it."""

    def __init__(self, raw, report_every: int = 2 * 2 ** 30):
        self.raw, self.digest, self.bytes = raw, hashlib.sha256(), 0
        self.report_every, self.next_report, self.started = report_every, report_every, time.monotonic()

    def read(self, n: int = -1) -> bytes:
        chunk = self.raw.read(n)
        self.digest.update(chunk)
        self.bytes += len(chunk)
        if self.bytes >= self.next_report:
            rate = self.bytes / max(time.monotonic() - self.started, 1e-6) / 1e6
            print(f"{self.bytes / 1e9:.1f} GB ({rate:.0f} MB/s)", flush=True)
            self.next_report += self.report_every
        return chunk


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--licence", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--part-mb", type=int, default=256)
    args = p.parse_args(argv)
    from boto3.s3.transfer import TransferConfig
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    request = urllib.request.Request(args.url, headers={"User-Agent": "sign-translator"})
    with urllib.request.urlopen(request, timeout=300) as response:
        length = int(response.headers["Content-Length"])
        meta = dict(last_modified=response.headers.get("Last-Modified"), etag=response.headers.get("ETag"))
        print(f"{args.url}: {length / 1e9:.2f} GB -> {args.key}", flush=True)
        stream = Hashing(response)
        part = args.part_mb * 2 ** 20
        store.s3.upload_fileobj(stream, store.bucket, args.key,
                                Config=TransferConfig(multipart_chunksize=part, multipart_threshold=part,
                                                      max_concurrency=8))
    if stream.bytes != length:
        raise SystemExit(f"streamed {stream.bytes} bytes, the server announced {length}")
    head = store.s3.head_object(Bucket=store.bucket, Key=args.key)
    if int(head["ContentLength"]) != length:
        raise SystemExit(f"the stored object has {head['ContentLength']} bytes, expected {length}")
    index = dict(source=args.url, key=args.key, bytes=length, sha256=stream.digest.hexdigest(), licence=args.licence,
                 note=args.note, acquired_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta)
    store.put(f"{args.key}.INDEX.json", json.dumps(index, indent=1).encode())
    print(f"INDEX {args.key}.INDEX.json: {length / 1e9:.2f} GB, sha256 {index['sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
