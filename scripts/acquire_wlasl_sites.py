"""WLASL v0.3 dictionary-site videos (C-UDA 1.0; owner-approved 2026-10-27): polite download, then per-host zips in R2.

WLASL lists 21,083 sign instances, and three quarters of them are files on dictionary sites, not YouTube. This
downloads those files under WLASL's Computational Use of Data Agreement, politely and without circumvention:
- **robots.txt:** each host's file is read first, and a URL it disallows for our user agent is skipped
  (``robots_disallowed``).
- **One request at a time per host,** at least ``--interval`` seconds apart (hosts run in parallel, one thread each),
  with an honest user agent and no cookies, sign-in, proxies or address changes.
- **A refusing host is left alone:** after ``--max-refusals`` consecutive refusals (401, 403, 429, 503) it is stopped.
  The other statuses are recorded per instance. Flash files (``.swf``) are not fetched.
- **Output:** files are kept locally under ``--workdir/<host>/<video_id>.<ext>``. ``--pack`` then writes one stored
  zip per host to ``<prefix>/zips/<host>.zip`` with provenance (licence, attribution, SHA-256, R2 re-read) and a
  ``<prefix>/instances.json`` that joins every downloaded video with its WLASL gloss, signer, split and source URL.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UA = "SignTranslatorResearch/1.0 (WLASL C-UDA computational use; polite, one request at a time)"
LICENCE = ("WLASL v0.3 (Li et al., WACV 2020): Computational Use of Data Agreement (C-UDA 1.0); videos remain their "
           "sites' property; private store, computational use only; trained models are C-UDA Results")
REFUSALS = (401, 403, 429, 503)


def host_of(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/")[0]


def instances(path: Path) -> list[dict]:
    out = []
    for g in json.loads(path.read_text()):
        for i in g["instances"]:
            if "youtu" not in i["url"]:
                out.append(dict(i, gloss=g["gloss"]))
    return out


def fetch_host(host: str, items: list[dict], args) -> list[dict]:
    robots = urllib.robotparser.RobotFileParser()
    scheme = "https" if items[0]["url"].startswith("https") else "http"
    try:
        req = urllib.request.Request(f"{scheme}://{host}/robots.txt", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            robots.parse(r.read().decode("utf-8", "replace").splitlines())
    except Exception:            # no robots.txt (or unreachable): nothing is disallowed by it
        robots.parse([])
    out, refusals, last = [], 0, 0.
    folder = Path(args.workdir) / host
    folder.mkdir(parents=True, exist_ok=True)
    for i in items:
        url = i["url"]; ext = re.sub(r"\?.*", "", url).rsplit(".", 1)[-1].lower()[:4]
        row = dict(video_id=i["video_id"], gloss=i["gloss"], url=url, host=host)
        target = folder / f"{i['video_id']}.{ext}"
        if target.exists() and target.stat().st_size > 0:
            out.append(dict(row, status="downloaded", file=str(target), bytes=target.stat().st_size)); continue
        if ext == "swf":
            out.append(dict(row, status="flash_skipped")); continue
        if not robots.can_fetch(UA, url):
            out.append(dict(row, status="robots_disallowed")); continue
        if refusals >= args.max_refusals:
            out.append(dict(row, status="host_stopped")); continue
        wait = last + args.interval - time.time()
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(args.max_bytes + 1)
            if len(data) > args.max_bytes:
                out.append(dict(row, status="too_large")); continue
            target.write_bytes(data); refusals = 0
            out.append(dict(row, status="downloaded", file=str(target), bytes=len(data)))
        except urllib.error.HTTPError as e:
            refusals = refusals + 1 if e.code in REFUSALS else 0
            out.append(dict(row, status=f"http_{e.code}"))
        except Exception as e:  # noqa: BLE001 - network errors are recorded, not retried aggressively
            out.append(dict(row, status="error", error=f"{type(e).__name__}: {e}"[:160]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wlasl-json", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--interval", type=float, default=3.)
    p.add_argument("--max-refusals", type=int, default=5)
    p.add_argument("--max-bytes", type=int, default=60 << 20)
    p.add_argument("--hosts", help="comma-separated hosts to fetch (default: all non-YouTube)")
    p.add_argument("--limit-per-host", type=int)
    p.add_argument("--pack", action="store_true", help="zip each host's downloads into R2 and write instances.json")
    p.add_argument("--prefix", default="acquisition/2026-10-17-v1/wlasl-v0.3")
    args = p.parse_args(argv)
    items = instances(Path(args.wlasl_json))
    by_host = collections.defaultdict(list)
    for i in items:
        by_host[host_of(i["url"])].append(i)
    if args.hosts:
        by_host = {h: v for h, v in by_host.items() if h in args.hosts.split(",")}
    report = Path(args.workdir) / "download-report.jsonl"
    if not args.pack:
        Path(args.workdir).mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(len(by_host)) as pool, report.open("a") as log:
            for rows in pool.map(lambda h: fetch_host(h, by_host[h][:args.limit_per_host], args), list(by_host)):
                for r in rows:
                    log.write(json.dumps(r) + "\n")
                c = collections.Counter(r["status"] for r in rows)
                print(json.dumps(dict(host=rows[0]["host"] if rows else None, **c)), flush=True)
        return 0
    from signtranslator.data.text_motion_dataset import R2Store
    store = R2Store(os.environ)
    meta = {i["video_id"]: i for i in items}
    latest = {}
    for line in report.read_text().splitlines():
        r = json.loads(line); latest[(r["host"], r["video_id"])] = r
    joined = []
    for host in sorted(by_host):
        rows = [r for (h, _), r in latest.items() if h == host and r["status"] == "downloaded"]
        if not rows:
            continue
        zpath = Path(args.workdir) / f"{host}.zip"
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
            for r in rows:
                z.write(r["file"], f"{host}/{Path(r['file']).name}")
        key = f"{args.prefix}/zips/{host}.zip"
        digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
        store.s3.upload_file(str(zpath), store.bucket, key)
        assert hashlib.sha256(store.get(key)).hexdigest() == digest
        store.put(key + ".provenance.json", json.dumps(dict(
            source=f"WLASL v0.3 instances hosted on {host}", sha256=digest, bytes=zpath.stat().st_size, videos=len(rows),
            verified_by_r2_reread=True, license=LICENCE, attribution="Dongxu Li et al., WLASL (github.com/dxli94/WLASL)",
            acquired_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")), indent=2).encode())
        for r in rows:
            m = meta[r["video_id"]]
            joined.append(dict(video_id=r["video_id"], member=f"{host}/{Path(r['file']).name}", zip_key=key, host=host,
                               gloss=m["gloss"], signer_id=m.get("signer_id"), split=m.get("split"), url=m["url"],
                               variation_id=m.get("variation_id"), bbox=m.get("bbox"), fps=m.get("fps")))
        print(json.dumps(dict(host=host, videos=len(rows), zip_key=key, sha256=digest[:12])), flush=True)
    store.put(f"{args.prefix}/instances.json", json.dumps(dict(license=LICENCE, instances=joined), indent=1).encode())
    print(json.dumps(dict(instances=len(joined))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
