"""ASL Citizen isolated signs → one-word English text → motion records (round 3: teach the generator single signs).

The full prior failed R5: one-word prompts were recognisable 0% of the time, against 99% for real. It had only seen
How2Sign sentences. ASL Citizen (Microsoft Research; research use, private store) has 83k isolated-sign clips of
2,731 glosses by 52 signers, with the publisher's signer-disjoint splits. This writes them as text records whose text
is the gloss's English word:
- the variant index is dropped (``CANDY2`` → "candy"), so the variants of one word are all valid ways to sign it;
- for slash alternatives the first is kept (``THIS/IT`` → "this");
- dotted spellings are joined (``W.H.A.T`` → "what"), and hyphens become spaces (``STAND-UP`` → "stand up");
- labels that are not dictionary words are left out: classifier or constructed-action forms (``-CA``, ``-CL``) and
  number compounds (``9OCLOCK``).

Partitions are the publisher's, unchanged. Every record keeps its continuous state, gloss, performer and quality.
Only ``train`` trains; R5 draws its real reference from ``val``.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def english(label: str) -> str | None:
    """The English word of an ASL Citizen gloss label, or None when it is not a dictionary word."""
    if re.search(r"-(CA|CL)\b", label) or re.match(r"\d", label):
        return None
    base = label.split("/")[0]
    base = re.sub(r"\d+$", "", base).replace(".", "")
    if not re.fullmatch(r"[A-Z][A-Z-]*", base):
        return None
    return base.replace("-", " ").lower()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2")
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--corpus-id", default="asl-citizen-text-v1")
    parser.add_argument("--output-key", required=True)
    args = parser.parse_args(argv)
    from signtranslator.data_engineering.corpus_manifest import build_corpus_manifest, canonical_json, load_manifest
    from train_text_to_motion import make_store
    store = make_store(args.store)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "m.json"; p.write_bytes(store.get(args.manifest_key) + b"\n")
        manifest = load_manifest(p, expected_sha256=args.manifest_sha256)
    records, why = [], collections.Counter()
    for r in manifest["records"]:
        if (r.get("continuous") or {}).get("status") != "converted":
            why["no_state"] += 1; continue
        word = english(r["gloss"]["label"])
        if word is None:
            why["not_a_word"] += 1; continue
        records.append(dict(r, text=word))
        why[f"kept_{r['partition']}"] += 1
    counts = dict(collections.Counter(r["partition"] for r in records))
    split = dict(manifest["split"], counts=counts, note="publisher splits, unchanged; text = the gloss's English word")
    out = build_corpus_manifest(corpus_id=args.corpus_id, records=records,
                                inputs={args.manifest_key: args.manifest_sha256}, split=split)
    out["license"] = ("ASL Citizen (Microsoft Research): research use, private store, no redistribution; "
                      "text derived from the publisher's gloss labels")
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite an existing manifest")
    digest = store.put(args.output_key, canonical_json(out))
    print(json.dumps(dict(sha256=digest, counts=counts, why=dict(why),
                          words=len({r["text"] for r in records}))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
