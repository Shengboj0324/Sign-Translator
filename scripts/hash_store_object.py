"""SHA-256 of one stored object, read back in ranged chunks (each retried on its own, so a dropped connection costs one
chunk, not the whole read), and optionally the ``<key>.INDEX.json`` that ``acquire_url_to_store.py`` writes.

    python scripts/hash_store_object.py --key acquisition/.../ASL_STEM_Wiki.zip --index-source <url> \
        --index-licence "..." --index-note "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ranged_sha256(store, key: str, chunk: int, retries: int = 8) -> tuple[str, int]:
    size = int(store.s3.head_object(Bucket=store.bucket, Key=key)["ContentLength"])
    digest, done, started = hashlib.sha256(), 0, time.monotonic()
    while done < size:
        end = min(done + chunk, size) - 1
        for attempt in range(retries):
            try:
                data = store.s3.get_object(Bucket=store.bucket, Key=key, Range=f"bytes={done}-{end}")["Body"].read()
                if len(data) != end - done + 1:
                    raise IOError(f"range {done}-{end}: {len(data)} bytes")
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
        digest.update(data)
        done = end + 1
        if done // chunk % 20 == 0 or done == size:
            print(f"{done / 1e9:.1f} / {size / 1e9:.1f} GB ({done / max(time.monotonic() - started, 1e-6) / 1e6:.0f} MB/s)",
                  flush=True)
    return digest.hexdigest(), size


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--key", required=True)
    p.add_argument("--chunk-mb", type=int, default=512)
    p.add_argument("--index-source", help="write <key>.INDEX.json with this source URL")
    p.add_argument("--index-licence", default="")
    p.add_argument("--index-note", default="")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    sha, size = ranged_sha256(store, args.key, args.chunk_mb * 2 ** 20)
    print(f"SHA256 {args.key} {size} {sha}", flush=True)
    if args.index_source:
        index = dict(source=args.index_source, key=args.key, bytes=size, sha256=sha, licence=args.index_licence,
                     note=args.index_note, hashed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        store.put(f"{args.key}.INDEX.json", json.dumps(index, indent=1).encode())
        print(f"INDEX {args.key}.INDEX.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
