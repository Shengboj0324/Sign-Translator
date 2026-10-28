"""Inventory of the Sem-Lex archives against the metadata: the known gaps the converter reports but does not fail on.

Streams each verified ``<split>.tar.gz`` once (the bytes are read through and discarded) and writes, per split: the
archive SHA-256, the count of file members by extension, the member count in the converter's ``<split>/<id>.npy``
form, the metadata ids of that split with no such member (``missing``), and the members with no metadata row or only
another split's. The output is the pinned ``semlex_known_missing.json`` next to the converter; it is rewritten after
every split, so a long run keeps what it has.

As found on 2026-10-28: ``test.tar.gz`` holds Holistic ``.npy`` pose files, but ``val.tar.gz`` and ``train.tar.gz``
hold ``.webm`` *videos* (``./val/<id>.webm``), which the pose-file converter cannot read.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

NAME = re.compile(r"^(?:\./)?(test|val|train)/([A-Za-z0-9_-]{1,64})\.([A-Za-z0-9]{1,8})$")


def inventory(store, split: str, metadata: dict) -> dict:
    from convert_semlex_poses import BASE, MEMBER, open_stream
    key = f"{BASE}/data/{split}.tar.gz"
    provenance = json.loads(store.get(key + ".provenance.json"))
    if not provenance.get("verified_by_r2_reread"):
        raise SystemExit(f"{key} is not a verified acquisition")
    extensions, pose_ids, other_ids, odd = collections.Counter(), set(), set(), 0
    with open_stream(store, key) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = NAME.match(member.name)
            if match is None or match.group(1) != split:
                odd += 1
                continue
            extensions[match.group(3).lower()] += 1
            if MEMBER.match(member.name):
                pose_ids.add(match.group(2))
            else:
                other_ids.add(match.group(2))
    expected = {v for v, rows in metadata.items() if any(r["split"] == split for r in rows)}
    form = "npy" if pose_ids and not other_ids else ("+".join(sorted(extensions)) or "empty")
    return dict(archive_key=key, archive_sha256=provenance["sha256"], format=form,
                extensions=dict(sorted(extensions.items())), other_names=odd, pose_members=len(pose_ids),
                missing=sorted(expected - pose_ids) if form == "npy" else None,
                expected_ids=len(expected), ids_with_other_files=len(other_ids),
                ids_with_other_files_missing_from_metadata=len(other_ids - set(metadata)),
                metadata_ids_without_other_file=len(expected - other_ids) if other_ids else None,
                no_metadata_row=sorted(v for v in pose_ids if v not in metadata),
                other_split_row=sorted(v for v in pose_ids if v in metadata and v not in expected))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--splits", default="test,val,train")
    p.add_argument("--store", default="r2")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    from convert_semlex_poses import METADATA_KEY, METADATA_SHA256, load_metadata, make_store
    store = make_store(args.store)
    metadata = load_metadata(store, METADATA_KEY, METADATA_SHA256)
    out = dict(metadata_sha256=METADATA_SHA256, splits={})
    for split in args.splits.split(","):
        s = out["splits"][split] = inventory(store, split, metadata)
        print(split, s["format"], s["extensions"], s["pose_members"], "pose files;",
              None if s["missing"] is None else len(s["missing"]), "missing", flush=True)
        Path(args.output).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
