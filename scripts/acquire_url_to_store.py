"""Stream one large file from a URL straight into the store (R2) as a multipart upload, with no local copy.

The SHA-256 and byte count are computed while streaming; the byte count must equal the server's Content-Length.
With ``--parallel n`` (servers that accept byte ranges), n ranged requests fetch the parts concurrently, each part
uploaded as it arrives (memory: about n x ``--part-mb``); the file's SHA-256 is then computed by streaming the stored
object back, and every part's own SHA-256 is recorded too.
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


def _range(url: str, start: int, end: int, retries: int = 6) -> bytes:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "sign-translator", "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=300) as r:
                if r.status != 206:
                    raise IOError(f"range request answered {r.status}")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"range {start}-{end}: {len(data)} bytes")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def parallel(args, store) -> int:
    from concurrent.futures import ThreadPoolExecutor
    head = urllib.request.urlopen(urllib.request.Request(args.url, method="HEAD",
                                                         headers={"User-Agent": "sign-translator"}), timeout=120)
    length = int(head.headers["Content-Length"])
    if head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise SystemExit("the server does not accept byte ranges: use --parallel 1")
    meta = dict(last_modified=head.headers.get("Last-Modified"), etag=head.headers.get("ETag"))
    part = args.part_mb * 2 ** 20
    ranges = [(i, a, min(a + part, length) - 1) for i, a in enumerate(range(0, length, part))]
    if len(ranges) > 10000:
        raise SystemExit(f"{len(ranges)} parts: raise --part-mb")
    print(f"{args.url}: {length / 1e9:.2f} GB in {len(ranges)} parts, {args.parallel} at a time -> {args.key}", flush=True)
    upload = store.s3.create_multipart_upload(Bucket=store.bucket, Key=args.key)["UploadId"]
    started, done = time.monotonic(), [0]

    def one(item):
        i, a, b = item
        data = _range(args.url, a, b)
        etag = store.s3.upload_part(Bucket=store.bucket, Key=args.key, UploadId=upload, PartNumber=i + 1,
                                    Body=data)["ETag"]
        done[0] += len(data)
        if (i + 1) % 20 == 0:
            rate = done[0] / max(time.monotonic() - started, 1e-6) / 1e6
            print(f"{done[0] / 1e9:.1f} GB ({rate:.0f} MB/s)", flush=True)
        return dict(PartNumber=i + 1, ETag=etag, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    try:
        with ThreadPoolExecutor(args.parallel) as pool:
            parts = list(pool.map(one, ranges))
        store.s3.complete_multipart_upload(Bucket=store.bucket, Key=args.key, UploadId=upload, MultipartUpload=dict(
            Parts=[dict(PartNumber=x["PartNumber"], ETag=x["ETag"]) for x in parts]))
    except BaseException:
        store.s3.abort_multipart_upload(Bucket=store.bucket, Key=args.key, UploadId=upload)
        raise
    stored = int(store.s3.head_object(Bucket=store.bucket, Key=args.key)["ContentLength"])
    if stored != length:
        raise SystemExit(f"the stored object has {stored} bytes, expected {length}")
    digest, body = hashlib.sha256(), store.s3.get_object(Bucket=store.bucket, Key=args.key)["Body"]
    while chunk := body.read(8 * 2 ** 20):
        digest.update(chunk)
    index = dict(source=args.url, key=args.key, bytes=length, sha256=digest.hexdigest(), licence=args.licence,
                 note=args.note, acquired_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), part_bytes=part,
                 parts=[dict(number=x["PartNumber"], bytes=x["bytes"], sha256=x["sha256"]) for x in parts], **meta)
    store.put(f"{args.key}.INDEX.json", json.dumps(index, indent=1).encode())
    print(f"INDEX {args.key}.INDEX.json: {length / 1e9:.2f} GB, sha256 {index['sha256']}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--licence", required=True)
    p.add_argument("--note", default="")
    p.add_argument("--part-mb", type=int, default=256)
    p.add_argument("--parallel", type=int, default=1, help="concurrent ranged requests (1: one stream)")
    args = p.parse_args(argv)
    from boto3.s3.transfer import TransferConfig
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    if args.parallel > 1:
        return parallel(args, store)
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
