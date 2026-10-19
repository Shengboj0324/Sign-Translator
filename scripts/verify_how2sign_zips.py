"""Verify acquired How2Sign clip zips in the object store, member by member.

The transfer already proved each zip's bytes (SHA-256 against the source, then an R2
re-read). This checks what the bytes contain:

* every member decompresses and matches its CRC-32 (``zipfile`` raises otherwise);
* the members are exactly the clips the official translation CSV names for that split
  and view (missing and unexpected names are reported, never silently accepted);
* optionally, a deterministic sample of clips is byte-compared (SHA-256) with an
  independently uploaded mirror of individual clips.

A manifest of every member (name, size, CRC-32, SHA-256) is written write-once beside
the zip, so later stages can prove which clip bytes they consumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_video_states import StoreRangeFile  # noqa: E402

PREFIX = "acquisition/2026-10-17-v1/how2sign-public"
SPLITS, VIEWS = ("val", "test", "train"), ("front", "side")
CROSS_MIRROR = "WayenVan/how2sign-front-clips"      # individual front clips, uploaded independently


def expected_clips(csv_text: str, view: str) -> set[str]:
    """Clip file names the translation CSV declares for one view."""
    rows = csv.DictReader(io.StringIO(csv_text), delimiter="\t")
    return {row["SENTENCE_NAME"].replace("-rgb_front", f"-rgb_{view}") + ".mp4" for row in rows}


def verify_archive(fileobj, expected: set[str]) -> dict:
    """Read every member (CRC-32 checked by ``zipfile``); hash it; compare names with ``expected``."""
    members, bad = [], []
    with zipfile.ZipFile(fileobj) as archive:
        infos = sorted((i for i in archive.infolist() if not i.is_dir()), key=lambda i: i.header_offset)
        for info in infos:
            digest = hashlib.sha256()
            try:
                with archive.open(info) as member:
                    for chunk in iter(lambda: member.read(1 << 20), b""):
                        digest.update(chunk)
            except (zipfile.BadZipFile, OSError) as error:   # CRC mismatch or truncated data
                bad.append(dict(name=info.filename, error=str(error)))
                continue
            members.append(dict(name=info.filename, size=info.file_size, crc32=f"{info.CRC:08x}",
                                sha256=digest.hexdigest()))
    names = {Path(m["name"]).name for m in members if m["name"].endswith(".mp4")}
    return dict(members=members, corrupt=bad, clips=len(names), missing=sorted(expected - names),
                unexpected=sorted(names - expected), passed=not bad and len(names) > 0)


def cross_mirror_paths(split: str) -> dict[str, str]:
    """Clip name -> repository path in the independent per-clip mirror."""
    folder = {"val": "validation"}.get(split, split)
    url = f"https://huggingface.co/api/datasets/{CROSS_MIRROR}/tree/main/{folder}?recursive=true"
    paths, cursor = {}, url
    while cursor:
        with urllib.request.urlopen(cursor, timeout=120) as response:
            for item in json.load(response):
                if item["type"] == "file" and item["path"].endswith(".mp4"):
                    paths[Path(item["path"]).name] = item["path"]
            link = response.headers.get("Link", "")
            cursor = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return paths


def cross_mirror_check(members: list[dict], split: str, every: int) -> dict:
    """SHA-256 of every ``every``-th clip (sorted by name) against the independent mirror."""
    paths = cross_mirror_paths(split)
    sample = sorted((m for m in members if m["name"].endswith(".mp4")), key=lambda m: m["name"])[::every]
    matched, differing, absent = 0, [], 0
    for member in sample:
        path = paths.get(Path(member["name"]).name)
        if path is None:
            absent += 1
            continue
        url = f"https://huggingface.co/datasets/{CROSS_MIRROR}/resolve/main/{path}"
        with urllib.request.urlopen(url, timeout=120) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest == member["sha256"]:
            matched += 1
        else:
            differing.append(member["name"])
    return dict(mirror=CROSS_MIRROR, sampled=len(sample), identical=matched, differing=differing,
                absent_from_mirror=absent, passed=not differing and matched > 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--cross-mirror-every", type=int, default=50, help="0 disables the cross-mirror sample")
    args = parser.parse_args(argv)
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    summary = {}
    for split in args.splits.split(","):
        csv_text = store.get(f"{PREFIX}/how2sign_realigned_{split}.csv").decode()
        for view in args.views.split(","):
            key = f"{PREFIX}/{split}_rgb_{view}_clips.zip"
            manifest_key = key + ".manifest.json"
            if not store.exists(key):
                summary[key] = dict(status="not acquired")
                continue
            if store.exists(manifest_key):
                summary[key] = dict(status="already verified")
                continue
            if not store.exists(key + ".provenance.json"):         # still being re-read after transfer
                summary[key] = dict(status="transfer not yet verified")
                continue
            provenance = json.loads(store.get(key + ".provenance.json"))
            if not provenance.get("verified_by_r2_reread"):
                raise SystemExit(f"{key}: transfer was not verified")
            stream = io.BufferedReader(StoreRangeFile(store.s3, store.bucket, key), 8 << 20)
            report = verify_archive(stream, expected_clips(csv_text, view))
            if view == "front" and args.cross_mirror_every:
                report["cross_mirror"] = cross_mirror_check(report["members"], split, args.cross_mirror_every)
                report["passed"] = report["passed"] and report["cross_mirror"]["passed"]
            report.update(key=key, zip_sha256=provenance["sha256"], schema_version=1)
            if report["passed"]:
                store.put(manifest_key, json.dumps(report, sort_keys=True).encode())
            summary[key] = {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()
                            if k not in ("members",)}
            print(json.dumps({key: summary[key]}, default=str), flush=True)
    print(json.dumps(summary, indent=2, default=str))
    return 0 if all(v.get("passed", True) for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
