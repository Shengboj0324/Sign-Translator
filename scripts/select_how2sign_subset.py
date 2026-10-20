"""Choose a budget-sized, signer-covering subset of How2Sign recording sessions.

Whole sessions only (synchronisation and calibration are per session). Every
non-take-5 session of the small fold-plan components is kept — they hold the
validation and test signers and four of the six training signers — and sessions of
the dominant component (c02, signers 5 and 8) are added in a fixed pseudo-random
order (SHA-256 of the session name) until ``--c02-hours`` of sentence time is
reached; the validation component (c03) can be capped the same way (``--c03-hours``). Take-5 sessions are left out: their lateral clips need the raw re-cut
(audit B80), which costs more per hour. Writes the session list, the hours per
component and a clip-name regular expression for ``--match``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data.text_motion_dataset import R2Store  # noqa: E402

CSV_KEY = "acquisition/2026-10-17-v1/how2sign-public/how2sign_realigned_train.csv"
PLAN = ROOT / "Sign Translator Stage Documentation/evidence/w1-fold-plan-2026-10-17/fold-plan.json"


def sessions(rows, components):
    out = collections.defaultdict(lambda: dict(hours=0., clips=0, component=None))
    for r in rows:
        video_name = r["VIDEO_NAME"].removesuffix("-rgb_front")          # VIDEOID-TAKE
        s = out[video_name]
        s["hours"] += (float(r["END_REALIGNED"]) - float(r["START_REALIGNED"])) / 3600
        s["clips"] += 1
        s["component"] = components.get(r["SENTENCE_NAME"], s["component"])
    return out


def select(table, c02_hours: float, c03_hours: float | None = None):
    chosen = [n for n, s in table.items() if not n.endswith("-5") and s["component"] not in (None, "c02", "c03")]
    validation = sorted((n for n, s in table.items() if not n.endswith("-5") and s["component"] == "c03"),
                        key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    kept = 0.
    for name in validation:                    # the validation signer, optionally capped
        if c03_hours is not None and kept >= c03_hours:
            break
        chosen.append(name)
        kept += table[name]["hours"]
    pool = sorted((n for n, s in table.items() if not n.endswith("-5") and s["component"] == "c02"),
                  key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    total = 0.
    for name in pool:
        if total >= c02_hours:
            break
        chosen.append(name)
        total += table[name]["hours"]
    return sorted(chosen)


def clip_regex(names) -> str:
    """Matches '<VIDEOID>_<n>-<TAKE>-rgb_(front|side)' file names of the chosen sessions."""
    parts = []
    for name in names:
        video, take = name.rsplit("-", 1)
        parts.append(re.escape(video) + r"_\d+-" + re.escape(take))
    return r"^(?:" + "|".join(parts) + r")-rgb_(?:front|side)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c02-hours", type=float, required=True)
    parser.add_argument("--c03-hours", type=float, help="cap on the validation component's hours (default: all)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = list(csv.DictReader(io.StringIO(R2Store(os.environ).get(CSV_KEY).decode()), delimiter="\t"))
    plan = json.loads(PLAN.read_text())
    table = sessions(rows, plan["sample_components"])
    chosen = select(table, args.c02_hours, args.c03_hours)
    hours = collections.Counter()
    for n in chosen:
        hours[table[n]["component"]] += table[n]["hours"]
    record = dict(schema_version=1, csv_key=CSV_KEY, fold_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                  c02_hours=args.c02_hours, c03_hours=args.c03_hours, sessions=chosen, hours={k: round(v, 3) for k, v in sorted(hours.items())},
                  total_hours=round(sum(hours.values()), 3), clips=sum(table[n]["clips"] for n in chosen),
                  match=clip_regex(chosen))
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("hours", "total_hours", "clips")}), len(chosen), "sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
