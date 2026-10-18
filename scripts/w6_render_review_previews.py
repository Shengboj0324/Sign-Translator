"""Render a skeleton preview video for every animated 3D-LEX sign, for gloss reviewers.

The GLB archive is read once into memory under its publisher-verified hash. Each
animated file is drawn in four panels on its own clock (front, side, each hand) and
uploaded write-once to R2 with a re-read check. An index binds every preview to its
source GLB hash; a zip of all previews is uploaded for distribution to reviewers.
Files without animation have no preview, and the index says so.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.gltf_motion import adapt_glb, joint_world_positions  # noqa: E402
from signtranslator.generation.skeleton_preview import (  # noqa: E402
    encode_preview, mixamo_bones, preview_joints, render_preview_frames,
)
from w4_convert_3dlex import INFERENCE, SOURCE, client, open_zip_in_memory  # noqa: E402
from w4_ingest_3dlex import put_once  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    s3, bucket = client()
    prefix = args.prefix.rstrip("/")
    archive = open_zip_in_memory(s3, bucket, f"{SOURCE}/Animation/ASL_ANIM_GLB.zip")
    bones = mixamo_bones()
    joints = preview_joints(bones)
    entries, bundle = [], io.BytesIO()
    members = sorted(n for n in archive.namelist() if n.endswith(".glb"))[:args.limit]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as distribution, tempfile.TemporaryDirectory() as tmp:
        for position, member in enumerate(members):
            sign = member.rsplit("/", 1)[-1].removesuffix(".glb")
            data = archive.read(member)
            entry = dict(sign=sign, glb_sha256=hashlib.sha256(data).hexdigest())
            try:
                times, positions = joint_world_positions(data, joints)
            except ValueError as error:
                entries.append(dict(entry, status="no_preview", reason=str(error)))
                continue
            # Show exactly the frames the canonical state keeps; files the adapter
            # refuses are shown whole with outlier-trimmed scaling, and marked.
            state, _ = adapt_glb(data, sample_id=sign, source_id="preview", clock_id="preview",
                                 inference_method=INFERENCE, salvage_reoriented_root=True)
            if state is not None:
                keep = state.channels["body"].valid.any(axis=1)
                frames = render_preview_frames(positions[keep], joints, bones)
                times, basis = times[keep], "adapter_valid_frames"
            else:
                frames = render_preview_frames(positions, joints, bones, bounds_quantile=.02)
                basis = "all_frames_adapter_refused"
            path = Path(tmp) / f"{sign}.mp4"
            info = encode_preview(frames, times, path)
            payload = path.read_bytes()
            path.unlink()
            key = f"{prefix}/{sign}.mp4"
            entries.append(dict(entry, status="rendered", frames_basis=basis, key=key, sha256=put_once(s3, bucket, key, payload),
                                bytes=len(payload), frames=info["frames"],
                                seconds=round(float(times[-1] - times[0]), 4)))
            distribution.writestr(f"previews/{sign}.mp4", payload)
            if position % 100 == 0:
                print(f"{position}/{len(members)}", flush=True)
    rendered = [e for e in entries if e["status"] == "rendered"]
    zip_bytes = bundle.getvalue()
    zip_key = f"{prefix}/review-previews.zip"
    index = dict(schema_version=1, prefix=prefix, rendered=len(rendered), without_preview=len(entries) - len(rendered),
                 zip_key=zip_key, zip_sha256=put_once(s3, bucket, zip_key, zip_bytes), zip_bytes=len(zip_bytes),
                 note="skeleton inspection previews; panels: front, side, left hand, right hand (left blue, right red)",
                 entries=entries)
    body = json.dumps(index, indent=2, sort_keys=True).encode() + b"\n"
    put_once(s3, bucket, f"{prefix}/index.json", body)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("xb") as stream:
        stream.write(body)
    print(json.dumps({k: index[k] for k in ("rendered", "without_preview", "zip_key", "zip_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
