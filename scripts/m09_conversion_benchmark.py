"""M09: measured cost of the real 3D-LEX ingestion path on this host.

The 2026-10-17 benchmark profiled synthetic kernels; M09 also asked for a real
source conversion to be measured. Here a fixed, content-ordered sample of validated
signs is fetched from R2 into memory first, so the network is excluded, and every
stage of the real path is timed per file on a monotonic clock after a warm-up pass:
GLB conversion, marker validation, glove validation, face conversion, verified
manifest serving with collation, and interpolation with a conformal bound.
Measurements only: no target is declared here, and linear extrapolations are
labelled as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m09_benchmark import environment, peak_rss_bytes, timings  # noqa: E402
from signtranslator.data.multichannel import collate_multichannel  # noqa: E402
from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample  # noqa: E402
from signtranslator.data_engineering.gltf_motion import (  # noqa: E402
    THREE_D_LEX_MIXAMO68, adapt_glb, skeleton_elbow_flexion,
)
from signtranslator.data_engineering.glove_validation import parse_glove_csv, validate_hands  # noqa: E402
from signtranslator.data_engineering.llf_face import adapt_llf  # noqa: E402
from signtranslator.data_engineering.marker_validation import (  # noqa: E402
    align_joint, marker_elbow_flexion, parse_v3d_markers,
)
from signtranslator.generation.conformal_abstention import span_error_and_difficulty  # noqa: E402
from signtranslator.generation.representation_comparison import hidden_spans, interpolation_baseline  # noqa: E402
from w4_convert_3dlex import INFERENCE, SOURCE, client, open_zip  # noqa: E402

ROTATIONS = ("body", "left_hand", "right_hand", "head_rotation")


def seed(sign):
    return int(hashlib.sha256(sign.encode()).hexdigest()[:8], 16)


def timed(function, items, rounds):
    function(items[0])  # warm-up
    per_file = []
    for _ in range(rounds):
        for item in items:
            start = time.perf_counter()
            function(item)
            per_file.append((time.perf_counter() - start) * 1e3)
    summary = timings(per_file)
    summary["files_per_second"] = 1e3 / summary["mean"]
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--signs", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest, expected_sha256=args.manifest_sha256)
    eligible = sorted((r for r in manifest["records"]
                       if {"left_hand", "right_hand"} <= set(r["usable"]["skeleton"]) and r["usable"]["face"]),
                      key=lambda r: hashlib.sha256(r["sign"].encode()).hexdigest())[:args.signs]
    signs = [r["sign"] for r in eligible]
    s3, bucket = client()
    fetch = lambda key: s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    glb = open_zip(s3, bucket, f"{SOURCE}/Animation/ASL_ANIM_GLB.zip")
    vicon = open_zip(s3, bucket, f"{SOURCE}/BodyPose/ASL_Vicon_MarkerData.zip")
    llf = open_zip(s3, bucket, f"{SOURCE}/FacePose/ASL_LLF.zip")
    vicon_names = {n.rsplit("/", 1)[-1].removesuffix("_V3D_MarkerData.csv"): n for n in vicon.namelist()}
    data = {s: dict(glb=glb.read(f"ANIM_GLB/{s}.glb"), vicon=vicon.read(vicon_names[s]).decode("utf-8", "replace"),
                    llf=llf.read(f"LLF_ASL/{s}.csv")) for s in signs}
    wanted = set(signs)
    for part in (1, 2, 3):
        archive = open_zip(s3, bucket, f"{SOURCE}/Handshape/Handshape Data Part {part}.zip")
        for name in archive.namelist():
            pieces = name.split("/")
            if len(pieces) == 3 and pieces[1] in wanted and pieces[2][:2] in ("L_", "R_") and pieces[2].endswith(".csv"):
                data[pieces[1]]["Left" if pieces[2][0] == "L" else "Right"] = archive.read(name)
    print(f"fetched {len(signs)} signs into memory", flush=True)

    def convert(sign):
        state, report = adapt_glb(data[sign]["glb"], sample_id=sign, source_id="bench", clock_id="bench",
                                  mapping=THREE_D_LEX_MIXAMO68, inference_method=INFERENCE,
                                  salvage_reoriented_root=True)
        return state

    states = {s: convert(s) for s in signs}

    def markers(sign):
        names, values = parse_v3d_markers(data[sign]["vicon"])
        times, left = skeleton_elbow_flexion(data[sign]["glb"], "Left")
        _, right = skeleton_elbow_flexion(data[sign]["glb"], "Right")
        valid = states[sign].channels["body"].valid[:, states[sign].channels["body"].labels.index("LeftArm")]
        return align_joint(times[valid], [left[valid], right[valid]],
                           [marker_elbow_flexion(names, values, "L"), marker_elbow_flexion(names, values, "R")])

    def gloves(sign):
        parsed = {side: parse_glove_csv(data[sign][side], side=side) for side in ("Left", "Right") if side in data[sign]}
        return validate_hands(states[sign], parsed)

    def face(sign):
        return adapt_llf(data[sign]["llf"], sample_id=sign, source_id="bench", clock_id="bench",
                         inference_method="bench")

    def bound(sign):
        state = states[sign]
        hidden = hidden_spans(state, channels=ROTATIONS, span_seconds=.25, spans=1, seed=seed(sign))
        recon = interpolation_baseline(state, hidden, allow_inferred=True)
        return [span_error_and_difficulty(state, n, hidden[n], *recon[n], allow_inferred=True) for n in ROTATIONS]

    results = dict(convert_glb=timed(convert, signs, args.rounds), marker_validation=timed(markers, signs, args.rounds),
                   glove_validation=timed(gloves, signs, args.rounds), face_conversion=timed(face, signs, args.rounds),
                   interpolate_and_bound=timed(bound, signs, args.rounds))
    with tempfile.TemporaryDirectory() as temporary:
        cache = ContentCache(Path(temporary) / "cache", 256 * 2 ** 20)
        for record in eligible:
            load_sample(record, "skeleton", cache=cache, fetch=fetch)  # populate: the network is excluded below
        results["serve_from_verified_cache"] = timed(
            lambda r: load_sample(r, "skeleton", cache=cache, fetch=fetch), eligible, args.rounds)
        batches = [eligible[i:i + 8] for i in range(0, len(eligible) - 7, 8)]
        results["serve_and_collate_batch_of_8"] = timed(
            lambda b: collate_multichannel([load_sample(r, "skeleton", cache=cache, fetch=fetch) for r in b]),
            batches, args.rounds)
    per_file = sum(results[k]["mean"] for k in ("convert_glb", "marker_validation", "glove_validation", "face_conversion"))
    record = dict(schema_version=1, environment=environment(), signs=len(signs), rounds=args.rounds,
                  manifest_sha256=args.manifest_sha256, results=results, peak_rss_bytes=peak_rss_bytes(),
                  linear_extrapolation=dict(
                      note="single process, sum of per-file means, network excluded; linear in measured cost only",
                      full_release_938_files_seconds=per_file * 938 / 1e3),
                  limitations=["one host; no target device or deployment mode is declared",
                               "R2 transfer time is excluded and depends on the link",
                               "per-file cost varies with take length; the sample is content-ordered, not stratified"],
                  phase_exit_approved=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: round(v["p50"], 2) for k, v in results.items()}, indent=1))
    print("extrapolated full conversion seconds", round(record["linear_extrapolation"]["full_release_938_files_seconds"], 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
