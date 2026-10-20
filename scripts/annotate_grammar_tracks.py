"""Write the grammar record of every state in a manifest (a derived, write-once dataset).

For each record of a text-motion manifest the canonical state is loaded (hash
pinned), non-manual tracks are z-scored against its recording session's pooled
neutral face, and ``grammar_motion.tracks.grammar_record`` is stored as JSON beside
the English features, under ``<prefix>/<sign>.json``, plus an index. These are
measurements, not labels: they condition and evaluate, and are never treated as
human annotation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_grammar_on_how2sign import session_of  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.grammar_motion.nmm import pooled_baselines  # noqa: E402
from signtranslator.grammar_motion.text import sentence_features  # noqa: E402
from signtranslator.grammar_motion.tracks import grammar_record  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    records = [r for r in manifest["records"] if r.get("continuous")]

    def load(r):
        payload = store.get(r["continuous"]["state_key"])
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "s.npz"; p.write_bytes(payload)
            return MultichannelMotion.load(p, expected_sha256=r["continuous"]["state_sha256"])

    sessions = {}
    for r in records:
        sessions.setdefault(session_of(r["sign"]), []).append(r)
    index = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for session, rows in sorted(sessions.items()):
            states = list(pool.map(load, rows))
            faces = [s.channels["face"] for s in states if s.channels["face"].valid.any()
                     and "browInnerUp" in s.channels["face"].labels]
            baselines = pooled_baselines(faces) if faces else None
            for r, state in zip(rows, states):
                record = dict(grammar_record(state, baselines), english=sentence_features(r["text"]) if r.get("text") else None,
                              state_sha256=r["continuous"]["state_sha256"], manifest_sha256=args.manifest_sha256)
                payload = json.dumps(record, sort_keys=True, default=float).encode()
                key = f"{args.prefix.rstrip('/')}/{r['sign']}.json"
                store.put(key, payload)
                index.append(dict(sign=r["sign"], key=key, sha256=hashlib.sha256(payload).hexdigest()))
    store.put(f"{args.prefix.rstrip('/')}/index.json", json.dumps(dict(schema_version=1, manifest_sha256=args.manifest_sha256,
                                                                       records=index), sort_keys=True).encode())
    print(json.dumps(dict(records=len(index), sessions=len(sessions))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
