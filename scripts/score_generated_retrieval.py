"""Re-score stored generated states with a frozen retrieval evaluator (no regeneration).

``evaluate_text_to_motion.py --save-generated PREFIX`` stores ``PREFIX/<guidance>/<sign>.npz``.
This reads them and the manifest's texts, and writes R-precision of the generated motion, of
the real motion of the same clips, and of the duration-only control (the real clips with every
channel removed), so a meaning score can be read against both bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import load_manifest  # noqa: E402
from signtranslator.pose.multichannel import MultichannelMotion  # noqa: E402
from evaluate_text_to_motion import blanked, load_retrieval, load_state, retrieval_scores  # noqa: E402
from train_text_to_motion import make_store  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--retrieval-key", required=True)
    parser.add_argument("--generated", required=True, nargs="+", help="prefixes written by --save-generated")
    parser.add_argument("--output-key")
    args = parser.parse_args(argv)
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"; path.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(path, expected_sha256=args.manifest_sha256)
    texts = {r["sign"]: r["text"] for r in manifest["records"] if r.get("text")}
    by_sign = {r["sign"]: r for r in manifest["records"]}
    model = load_retrieval(store, args.retrieval_key, args.manifest_sha256)
    out = {}
    for prefix in args.generated:
        keys = sorted(k for k in store.list(prefix.rstrip("/") + "/") if k.endswith(".npz"))
        groups = {}
        for key in keys:
            groups.setdefault(key.rsplit("/", 2)[-2], []).append(key)
        out[prefix] = {}
        for tag, members in sorted(groups.items()):
            signs = [k.rsplit("/", 1)[-1][:-4] for k in members]
            generated = []
            for key in members:
                payload = store.get(key)
                with tempfile.TemporaryDirectory() as t:
                    p = Path(t) / "s.npz"; p.write_bytes(payload)
                    generated.append(MultichannelMotion.load(p, expected_sha256=hashlib.sha256(payload).hexdigest()))
            references = [load_state(store, by_sign[s]["continuous"]["state_key"], by_sign[s]["continuous"]["state_sha256"])
                          for s in signs]
            sentence = [texts[s] for s in signs]
            out[prefix][tag] = dict(generated=retrieval_scores(model, generated, sentence),
                                    real=retrieval_scores(model, references, sentence),
                                    duration_only=retrieval_scores(model, blanked(references), sentence))
            print(json.dumps({prefix: {tag: {k: {m: round(v[m], 3) for m in ("r_at_1", "r_at_3")}
                                             for k, v in out[prefix][tag].items()}}}), flush=True)
    if args.output_key:
        store.put(args.output_key, json.dumps(dict(retrieval_key=args.retrieval_key, results=out), indent=2,
                                              sort_keys=True).encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
