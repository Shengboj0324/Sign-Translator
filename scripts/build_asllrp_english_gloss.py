"""English -> ASL gloss pairs from the ASLLRP / RIT continuous corpora (DAI 2): the planner's training data.

Joins the DAI 2 utterance search export (``asllrp-dai2-v1/asllrp-continuous/utterance-search-*.json``: per utterance
its English translation and the SignStream rough-gloss tiers) with ``derived/signbank-v1/utterances.json`` (the
signs in time order with their main-entry glosses). Each pair carries:

- ``english``: the corpus translation;
- ``glosses``: the main-entry gloss of every sign in time order (the planner's target), and ``surface``: the
  dominant-hand gloss tokens as annotated (with agreement indices such as ``IX-3p:i`` and fingerspelling);
- ``markers``: the grammatical non-manual markers (topic/focus, wh / yes-no / rhetorical question, negation,
  conditional/when, relative clause), each with the surface tokens it spans. The tiers are column-aligned with the
  dominant-hand line, so a marker covers the tokens whose first character lies inside its underline.

Partitions are signer-disjoint and identical to ``build_asllrp_gloss_manifest.py`` (val Jonathan, test Ben).
Research use only (ASLLRP Data Terms of Use); written once, never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data.text_motion_dataset import DirectoryStore, R2Store  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import canonical_json  # noqa: E402
from build_asllrp_gloss_manifest import PARTITION, signer  # noqa: E402

HAND_TIERS = {"dh", "dh shape", "ndh", "ndh shape", "engl trans"}
_LABEL = re.compile(r"^[\t ]*([a-z][a-z /\-]*?):(.*)$")


def tiers(text: str) -> dict[str, str]:
    """Tier name -> content, columns relative to the character after the tier's colon (labels are right-aligned)."""
    out = {}
    for line in text.splitlines():
        m = _LABEL.match(line)
        if m:
            out.setdefault(m.group(1).strip(), m.group(2))
    return out


def tokens(content: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group()) for m in re.finditer(r"\S+", content)]


def markers(tier_map: dict[str, str]) -> list[dict]:
    """Every underline span of the non-hand tiers, with the dominant-hand tokens it covers."""
    surface = tokens(tier_map.get("dh", ""))
    out = []
    for tier, content in tier_map.items():
        if tier in HAND_TIERS:
            continue
        for m in re.finditer(r"_+(\S*)", content):
            start, end = m.start(), m.end()
            covered = [i for i, (col, _) in enumerate(surface) if start <= col < end]
            out.append(dict(tier=tier, label=m.group(1) or tier, token_start=covered[0] if covered else None,
                            token_end=covered[-1] + 1 if covered else None))
    return sorted(out, key=lambda x: (x["token_start"] if x["token_start"] is not None else -1, x["tier"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="r2", help="r2, or dir:<path>")
    parser.add_argument("--search-key", default="acquisition/2026-10-17-v1/asllrp-dai2-v1/asllrp-continuous/utterance-search-2026-10-22.json")
    parser.add_argument("--utterances-key", default="acquisition/2026-10-17-v1/derived/signbank-v1/utterances.json")
    parser.add_argument("--output-key", default="acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/english-gloss-v1/pairs.json")
    args = parser.parse_args(argv)
    store = R2Store(os.environ) if args.store == "r2" else DirectoryStore(Path(args.store[4:]))
    search_payload, utt_payload = store.get(args.search_key), store.get(args.utterances_key)
    inputs = {args.search_key: hashlib.sha256(search_payload).hexdigest(),
              args.utterances_key: hashlib.sha256(utt_payload).hexdigest()}
    by_utterance = {(u["source"], u["utterance"]): u for u in json.loads(utt_payload)["items"]}
    pairs, skipped, marker_counts = [], collections.Counter(), collections.Counter()
    for row in json.loads(search_payload)["utterances"]:
        m = re.match(r"(.+)-U-(\d+)$", row["name"])
        u = by_utterance.get((m.group(1), int(m.group(2)))) if m else None
        if u is None:
            skipped["no utterance record"] += 1; continue
        who = signer(u["source"])
        if who not in PARTITION:
            skipped["unknown signer"] += 1; continue
        tier_map = tiers(row["rough_gloss_text"])
        english = " ".join(tier_map.get("engl trans", "").split())
        glosses = [s["gloss"] for s in u["signs"] if s["gloss"]]
        if not english or not glosses:
            skipped["no english" if not english else "no glosses"] += 1; continue
        found = markers(tier_map)
        marker_counts.update(x["tier"] for x in found)
        pairs.append(dict(id=f"{u['collection']}:{row['name']}", collection=u["collection"], source=u["source"],
                          utterance=u["utterance"], utterance_video=u["utterance_video"], partition=PARTITION[who],
                          english=english, glosses=glosses, surface=[t for _, t in tokens(tier_map.get("dh", ""))],
                          markers=found, dai_resource_id=row.get("resource_id")))
    counts = {p: sum(x["partition"] == p for x in pairs) for p in ("train", "val", "test")}
    if store.exists(args.output_key):
        raise SystemExit("refusing to overwrite " + args.output_key)
    digest = store.put(args.output_key, canonical_json(dict(
        schema_version=1, inputs=dict(sorted(inputs.items())), partitions=counts, skipped=dict(skipped),
        marker_counts=dict(sorted(marker_counts.items())), split="signer-disjoint, fixed: val Jonathan, test Ben",
        license="ASLLRP Sign Bank / DAI Data Terms of Use: research use only, no redistribution", pairs=pairs)))
    print(json.dumps(dict(output_key=args.output_key, sha256=digest, pairs=len(pairs), partitions=counts,
                          skipped=dict(skipped), markers=dict(marker_counts))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
