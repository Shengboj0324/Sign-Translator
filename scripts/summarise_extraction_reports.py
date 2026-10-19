"""Summarise extraction shard reports: clips by status and frame-level detection rates.

Reads every ``reports/*.json`` under a prefix (MediaPipe extraction or Sapiens jobs),
groups by zip, and prints one JSON summary: clip counts by status, total video
seconds, and the fraction of frames in which pose, face and each hand were found
(frame-weighted over converted clips). Read-only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402

DETECTIONS = ("pose_frames", "face_frames", "left_hand_frames", "right_hand_frames")


def summarise(reports: list[dict]) -> dict:
    by_zip = collections.defaultdict(lambda: dict(status=collections.Counter(), seconds=0., frames=0,
                                                  found=collections.Counter(), shards=0))
    for report in reports:
        entry = by_zip[report.get("zip_key", "?")]
        entry["shards"] += 1
        for result in report.get("results", []):
            entry["status"][result["status"]] += 1
            entry["seconds"] += result.get("seconds", 0.)
            if result["status"] == "converted" and "frames" in result:
                entry["frames"] += result["frames"]
                for key in DETECTIONS:
                    entry["found"][key] += result.get(key, 0)
    out = {}
    for zip_key, entry in sorted(by_zip.items()):
        frames = entry["frames"]
        out[zip_key] = dict(shards=entry["shards"], clips=dict(entry["status"]), hours=round(entry["seconds"] / 3600, 2),
                            frames=frames, detection_rate={k.replace("_frames", ""): round(entry["found"][k] / frames, 4)
                                                           for k in DETECTIONS} if frames else {})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path> for a local mirror")
    parser.add_argument("--prefix", required=True, help="job prefix holding reports/")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    keys = [k for k in store.list(args.prefix.rstrip("/") + "/reports/") if k.endswith(".json")]
    print(json.dumps(summarise([json.loads(store.get(k)) for k in keys]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
