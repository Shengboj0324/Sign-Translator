"""Keep this host's live extraction claims fresh (YouTube-ASL staging area).

A claim (``acquire_youtube_asl.claim``) older than ``CLAIM_TTL`` (3 h) is taken over, on the assumption that its
extractor died. The highest-yield videos take 5-8 h to extract, so live claims went stale and a second worker
converted the same video again. Run every 20 min on each extraction pod: every claim this host holds whose process
is alive gets ``at`` = now. The rewrite is conditional on the claim's ETag, so a claim released or taken over in
between is never recreated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args(argv)
    from botocore.exceptions import ClientError
    from train_text_to_motion import make_store
    store = make_store(args.store)
    host, refreshed, skipped = socket.gethostname(), 0, 0
    for key in store.list(f"{args.prefix}/staged/"):
        if not key.endswith(".claim"):
            continue
        try:
            head = store.s3.get_object(Bucket=store.bucket, Key=key)
            held = json.loads(head["Body"].read())
        except ClientError:
            continue                                    # released meanwhile
        if held.get("host") != host or not alive(int(held.get("pid", -1))):
            continue
        body = json.dumps(dict(held, at=time.time())).encode()
        try:
            store.s3.put_object(Bucket=store.bucket, Key=key, Body=body, IfMatch=head["ETag"])
            refreshed += 1
        except ClientError:
            skipped += 1                                # changed or deleted meanwhile: never recreated
    print(json.dumps(dict(host=host, refreshed=refreshed, skipped=skipped)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
