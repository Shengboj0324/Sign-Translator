"""PopSign ASL v1.0 (Georgia Tech, CC BY 4.0): per-sign training tars into R2, and a per-sign selection for extraction.

For each sign of ``--signs`` (one per line), from ``https://signdata.cc.gatech.edu/data/popsign_v1_0/<category>/<split>/<sign>.tar``:
1. download the tar (resumable with curl) and verify its size against the server's Content-Length;
2. upload it unchanged to R2 under ``--prefix/raw/<category>/<split>/<sign>.tar``, with a provenance file (source URL,
   SHA-256, bytes, licence, attribution, re-read check);
3. select up to ``--per-sign`` clips: distinct participants first (the clip name begins with the participant id), in a
   deterministic hash order, then any;
4. pack the selection into a stored (uncompressed) zip, ``--prefix/selection/<category>/<split>/<sign>.zip``, also
   with provenance, for ``extract_video_states.py``.

A sign whose selection zip and raw tar are both in R2 is skipped, so the job resumes. Writes one JSON line per sign.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "https://signdata.cc.gatech.edu/data/popsign_v1_0"
LICENCE = "PopSign ASL v1.0 (Georgia Tech): CC BY 4.0 (signdata.cc.gatech.edu/view/datasets/popsign_v1_0)"
ATTRIBUTION = "PopSign ASL v1.0, Thad Starner, Rohit Sridhar, Matthew So, Gururaj Deshpande et al., Georgia Institute of Technology"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def upload(store, key: str, path: Path, url: str) -> str:
    """Multipart upload, then a size and checksum re-read (streamed); provenance beside it."""
    digest = sha256_file(path)
    store.s3.upload_file(str(path), store.bucket, key)
    h = hashlib.sha256()
    body = store.s3.get_object(Bucket=store.bucket, Key=key)["Body"]
    for block in iter(lambda: body.read(1 << 20), b""):
        h.update(block)
    if h.hexdigest() != digest:
        raise RuntimeError(f"{key}: R2 copy does not match")
    prov = dict(source_url=url, sha256=digest, bytes=path.stat().st_size, verified_by_r2_reread=True, license=LICENCE,
                attribution=ATTRIBUTION, acquired_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    store.put(key + ".provenance.json", json.dumps(prov, indent=2).encode())
    return digest


def select(names: list[str], k: int) -> list[str]:
    ordered = sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    chosen, seen = [], set()
    for n in ordered:
        who = Path(n).name.split("-")[0]
        if who not in seen:
            chosen.append(n); seen.add(who)
    chosen += [n for n in ordered if n not in chosen]
    return chosen[:k]


def one(store, args, sign: str) -> dict:
    url = f"{BASE}/{args.category}/{args.split}/{sign}.tar"
    raw = f"{args.prefix}/raw/{args.category}/{args.split}/{sign}.tar"
    sel = f"{args.prefix}/selection/{args.category}/{args.split}/{sign}.zip"
    if store.exists(raw + ".provenance.json") and store.exists(sel + ".provenance.json"):
        return dict(sign=sign, status="already_present", selection_key=sel,
                    selection_sha256=json.loads(store.get(sel + ".provenance.json"))["sha256"])
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        tar_path = Path(tmp) / f"{sign}.tar"
        head = subprocess.run(["curl", "-sSIL", url], capture_output=True, text=True, check=True).stdout.lower()
        length = int([l.split(":")[1] for l in head.splitlines() if l.startswith("content-length")][-1])
        for _ in range(4):
            subprocess.run(["curl", "-sSL", "-C", "-", "--retry", "5", "-o", str(tar_path), url], check=False)
            if tar_path.exists() and tar_path.stat().st_size == length:
                break
        if not tar_path.exists() or tar_path.stat().st_size != length:
            return dict(sign=sign, status="error", error=f"download incomplete ({tar_path.stat().st_size if tar_path.exists() else 0}/{length})")
        raw_sha = upload(store, raw, tar_path, url)
        with tarfile.open(tar_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".mp4")}
            chosen = select(list(members), args.per_sign)
            zip_path = Path(tmp) / f"{sign}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
                for n in chosen:
                    z.writestr(f"{sign}/{Path(n).name}", tar.extractfile(members[n]).read())
        sel_sha = upload(store, sel, zip_path, url + " (selection)")
        return dict(sign=sign, status="converted", clips_in_tar=len(members), selected=len(chosen),
                    participants=len({Path(n).name.split("-")[0] for n in chosen}), raw_key=raw, raw_sha256=raw_sha,
                    selection_key=sel, selection_sha256=sel_sha, tar_bytes=length)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--signs", required=True, help="file with one sign name per line (as in the PopSign URLs)")
    p.add_argument("--category", default="game")
    p.add_argument("--split", default="train")
    p.add_argument("--per-sign", type=int, default=60)
    p.add_argument("--prefix", default="acquisition/2026-10-17-v1/popsign-v1")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--workdir", default="/root/ps")
    args = p.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    signs = [s.strip() for s in Path(args.signs).read_text().splitlines() if s.strip()]
    errors = 0
    with ThreadPoolExecutor(args.parallel) as pool:
        for result in pool.map(lambda s: _safe(store, args, s), signs):
            errors += result["status"] == "error"
            print(json.dumps(result), flush=True)
    return 2 if errors else 0


def _safe(store, args, sign):
    try:
        return one(store, args, sign)
    except Exception as error:        # noqa: BLE001 - reported per sign; a rerun resumes
        return dict(sign=sign, status="error", error=f"{type(error).__name__}: {error}"[:300])


if __name__ == "__main__":
    raise SystemExit(main())
