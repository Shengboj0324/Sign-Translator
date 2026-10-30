"""Upload every file under a local directory to the store (R2) under a prefix, with an INDEX.json (relative path, key,
size, SHA-256, source, licence, note). Files already stored at their size are not uploaded again. Data only: nothing
is opened beyond hashing (archives are not unpacked, pickles never loaded).

    python scripts/store_local_files.py --dir /root/sa --prefix acquisition/2026-10-30-v1/signavatars \
        --source "Google Drive links from the SignAvatars team (email 2026-10-30)" --licence "CC BY-NC-SA 4.0 (research)"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 2 ** 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--prefix", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--licence", required=True)
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    prefix = args.prefix.rstrip("/")
    rows = []
    for path in sorted(x for x in args.dir.rglob("*") if x.is_file()):
        rel = path.relative_to(args.dir).as_posix()
        key = f"{prefix}/files/{rel}"
        size, digest = path.stat().st_size, sha256(path)
        try:
            head = store.s3.head_object(Bucket=store.bucket, Key=key)
            same = int(head["ContentLength"]) == size and head.get("Metadata", {}).get("sha256") == digest
        except Exception:
            same = False
        if not same:
            store.s3.upload_file(str(path), store.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        rows.append(dict(path=rel, key=key, size=size, sha256=digest))
        print(f"{'kept' if same else 'stored'} {rel} {size / 1e6:.1f} MB", flush=True)
    index = dict(source=args.source, licence=args.licence, note=args.note, files=rows,
                 total_bytes=sum(r["size"] for r in rows),
                 acquired_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    store.put(f"{prefix}/INDEX.json", json.dumps(index, indent=1).encode())
    print(f"INDEX {prefix}/INDEX.json: {len(rows)} files, {index['total_bytes'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
