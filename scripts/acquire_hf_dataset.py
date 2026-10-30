"""Copy a Hugging Face dataset repository, file by file, into the store (R2), pinned and checked.

The revision is resolved to its commit first and every file is fetched from that commit, so a rerun copies the same
bytes. Each file is streamed to a temporary file while its SHA-256 is computed; a file that Hugging Face stores in LFS
must match the SHA-256 the repository records for it, or the run stops. It is then uploaded to ``<prefix>/files/<path>``
and the temporary file deleted, so the local disk only ever holds ``--workers`` files. A file already in the store at
its size is not fetched again (resumable). ``<prefix>/INDEX.json`` records the repository, commit, licence (from the
dataset card), and every file's path, key, size and SHA-256.

Data only: no file is executed or deserialised here.

    python scripts/acquire_hf_dataset.py --repo facebook/2M-Flores-ASL --prefix acquisition/2026-10-30-v1/2m-flores-asl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HUB = "https://huggingface.co"
CHUNK = 8 * 2 ** 20


def _get_json(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sign-translator"}),
                                        timeout=60) as r:
                return json.loads(r.read()), r.headers
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def resolve(repo: str, revision: str) -> dict:
    info, _ = _get_json(f"{HUB}/api/datasets/{repo}/revision/{urllib.parse.quote(revision, safe='')}")
    return info


def list_files(repo: str, commit: str) -> list[dict]:
    """Every file of the commit: path, size, and the LFS SHA-256 where the file is in LFS."""
    out, url = [], f"{HUB}/api/datasets/{repo}/tree/{commit}?recursive=true&expand=false"
    while url:
        page, headers = _get_json(url)
        out += [dict(path=e["path"], size=int(e.get("size") or 0),
                     sha256=(e.get("lfs") or {}).get("oid")) for e in page if e.get("type") == "file"]
        link = headers.get("Link") or ""
        url = next((p.split(";")[0].strip(" <>") for p in link.split(",") if 'rel="next"' in p), None)
    return out


def fetch(repo: str, commit: str, path: str, dest: Path, retries: int = 5) -> str:
    url = f"{HUB}/datasets/{repo}/resolve/{commit}/{urllib.parse.quote(path)}"
    for attempt in range(retries):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sign-translator"}),
                                        timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(CHUNK):
                    digest.update(chunk)
                    f.write(chunk)
            return digest.hexdigest()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--prefix", required=True, help="store prefix: files go to <prefix>/files/<path>")
    p.add_argument("--include", action="append", help="glob of paths to copy (repeatable; default: all)")
    p.add_argument("--limit", type=int, help="smoke runs: only the first n files")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tmp", default=tempfile.gettempdir())
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    info = resolve(args.repo, args.revision)
    commit = info["sha"]
    licence = (info.get("cardData") or {}).get("license")
    files = sorted(list_files(args.repo, commit), key=lambda e: e["path"])
    if args.include:
        files = [e for e in files if any(fnmatch.fnmatch(e["path"], g) for g in args.include)]
    if args.limit:
        files = files[:args.limit]
    prefix = args.prefix.rstrip("/")
    print(f"{args.repo}@{commit}: {len(files)} files, {sum(e['size'] for e in files) / 1e9:.2f} GB, licence {licence}",
          flush=True)

    def one(e: dict) -> dict:
        key = f"{prefix}/files/{e['path']}"
        head = None
        try:
            head = store.s3.head_object(Bucket=store.bucket, Key=key)
        except Exception:
            pass
        if head is not None and int(head["ContentLength"]) == e["size"] and head.get("Metadata", {}).get("sha256"):
            return dict(e, key=key, sha256=head["Metadata"]["sha256"], skipped=True)
        with tempfile.NamedTemporaryFile(dir=args.tmp, delete=False) as t:
            local = Path(t.name)
        try:
            digest = fetch(args.repo, commit, e["path"], local)
            if local.stat().st_size != e["size"]:
                raise SystemExit(f"{e['path']}: {local.stat().st_size} bytes, the repository lists {e['size']}")
            if e["sha256"] and digest != e["sha256"]:
                raise SystemExit(f"{e['path']}: SHA-256 {digest} differs from the repository's {e['sha256']}")
            store.s3.upload_file(str(local), store.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        finally:
            local.unlink(missing_ok=True)
        return dict(e, key=key, sha256=digest, skipped=False)

    done, started, size = [], time.monotonic(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(one, e) for e in files]):
            row = future.result()
            done.append(row)
            size += row["size"]
            if len(done) % 25 == 0 or len(done) == len(files):
                rate = size / max(time.monotonic() - started, 1e-6) / 1e6
                print(f"{len(done)}/{len(files)} files, {size / 1e9:.1f} GB ({rate:.0f} MB/s)", flush=True)
    rows = sorted(({k: v for k, v in r.items() if k != "skipped"} for r in done), key=lambda r: r["path"])
    index = dict(repo=args.repo, revision=args.revision, commit=commit, licence=licence,
                 source=f"{HUB}/datasets/{args.repo}", files=rows, total_bytes=sum(r["size"] for r in rows),
                 acquired_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), complete=args.limit is None
                 and not args.include)
    store.put(f"{prefix}/INDEX.json", json.dumps(index, indent=1, sort_keys=True).encode())
    print(f"INDEX {prefix}/INDEX.json: {len(rows)} files, {index['total_bytes'] / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
