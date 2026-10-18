"""Ingest every remaining 3D-LEX modality and freeze one corpus manifest in R2.

Inputs, all read from R2 by ranged GETs (nothing staged beyond one small file):

* the frozen skeleton conversion report (v3) and its canonical states;
* ``ASL_LLF.zip`` -> face-only canonical states on each file's own device clock;
* ``Handshape Data Part 1..3`` -> glove validation of both hands of every
  marker-validated skeleton state;
* the ASL Signbank ECV snapshot -> English-label gloss candidates for review.

Outputs: face states under a new versioned prefix, a manifest that admits each
channel only on its evidence, and reports in the evidence directory. Performer
identity never leaves this process except as the existing salted hash. Every R2
prefix written here must be new; nothing is overwritten.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signtranslator.data_engineering.corpus_manifest import (  # noqa: E402
    build_corpus_manifest, canonical_json, manifest_sha256, performer_split,
)
from signtranslator.data_engineering.glove_validation import (  # noqa: E402
    MAX_CENTRED_MAE_DEGREES, MIN_CORRELATION, STATIC_RANGE_DEGREES, parse_glove_csv,
    parse_glove_metadata, validate_hands,
)
from signtranslator.data_engineering.llf_face import adapt_llf  # noqa: E402
from signtranslator.data_engineering.signbank_ecv import map_labels, parse_ecv  # noqa: E402
from signtranslator.pose.multichannel import CHANNELS, MultichannelMotion  # noqa: E402
from w4_convert_3dlex import SOURCE, client, open_zip  # noqa: E402

GLOVE_FILE = re.compile(r"([LR])_([A-Za-z0-9]+)\.csv")
ECV_KEY = "acquisition/2026-10-17-v1/asl-signbank/ecv-snapshot-2026-10-18T040017Z/asl.ecv"
ECV_SHA256 = "7f51c84d6e45029bf4464bceff3f4e53504d253746e8c350af4af9f650eb1044"
LLF_INFERENCE = ("ARKit TrueDepth face solve exported by Epic Live Link Face; values are fitted "
                 "coefficients and eye angles, not direct observations")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_once(s3, bucket, key, payload: bytes) -> str:
    """Write once. An existing object is accepted only if its bytes are identical."""
    from botocore.exceptions import ClientError
    digest = sha256(payload)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        if sha256(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != digest:
            raise SystemExit(f"refusing to overwrite {key} with different content")
        return digest
    s3.put_object(Bucket=bucket, Key=key, Body=payload, Metadata={"sha256": digest})
    if sha256(s3.get_object(Bucket=bucket, Key=key)["Body"].read()) != digest:
        raise SystemExit(f"R2 re-read mismatch for {key}")
    return digest


def state_bytes(state: MultichannelMotion) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        digest = state.save(path)
        return path.read_bytes(), digest


def load_state(s3, bucket, key, digest) -> MultichannelMotion:
    payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "s.npz"
        path.write_bytes(payload)
        return MultichannelMotion.load(path, expected_sha256=digest)


def skeleton_entry(record: dict) -> dict | None:
    if "state_key" not in record:
        return None
    report = record["report"]
    unavailable = set(report.get("unavailable_channels", []))
    salvage = (report.get("unrepresented") or {}).get("salvaged_global_orientation")
    if salvage:
        unavailable |= {item for item in salvage["unavailable"] if item in CHANNELS}
    return dict(state_key=record["state_key"], state_sha256=record["state_sha256"],
                adapter_status=report["status"],
                marker_validation=(record.get("marker_validation") or {}).get("status"),
                available_channels=[name for name in CHANNELS if name not in unavailable],
                valid_frames=report["frames"] - len(report.get("invalid_frames", [])))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-report", type=Path, required=True)
    parser.add_argument("--face-prefix", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    args.evidence.mkdir(parents=True, exist_ok=False)
    s3, bucket = client()
    face_prefix = args.face_prefix.rstrip("/")
    report_bytes = args.conversion_report.read_bytes()
    conversion = {r["sign"]: r for r in json.loads(report_bytes)["records"]}
    archive_hashes = {}
    for name in ("FacePose/ASL_LLF.zip", *(f"Handshape/Handshape Data Part {k}.zip" for k in (1, 2, 3))):
        provenance = json.loads(s3.get_object(Bucket=bucket, Key=f"{SOURCE}/{name}.provenance.json")["Body"].read())
        if not provenance.get("verified_by_r2_reread") or provenance["sha256"] != provenance["publisher_sha256"]:
            raise SystemExit(f"{name}: archive is not publisher-verified")
        archive_hashes[name] = provenance["sha256"]

    # --- gloss candidates -------------------------------------------------
    ecv_bytes = s3.get_object(Bucket=bucket, Key=ECV_KEY)["Body"].read()
    if sha256(ecv_bytes) != ECV_SHA256:
        raise SystemExit("ECV snapshot hash mismatch")
    lexicon = parse_ecv(ecv_bytes)

    # --- face states --------------------------------------------------------
    llf = open_zip(s3, bucket, f"{SOURCE}/FacePose/ASL_LLF.zip")
    face, face_reports = {}, []
    members = sorted(n for n in llf.namelist() if n.endswith(".csv"))
    for member in members[:args.limit]:
        sign = member.rsplit("/", 1)[-1].removesuffix(".csv")
        payload = llf.read(member)
        state, report = adapt_llf(payload, sample_id=f"3dlex-asl-{sign}-llf", source_id="three_d_lex_asl_llf",
                                  clock_id=f"llf-device-timecode:{sign}", inference_method=LLF_INFERENCE)
        entry = dict(status=report.status, reason=report.reason, source_sha256=report.source_sha256,
                     member=member)
        if state is not None:
            body, digest = state_bytes(state)
            key = f"{face_prefix}/states/{sign}.npz"
            if put_once(s3, bucket, key, body) != digest:
                raise SystemExit("face state digest drift")
            entry.update(state_key=key, state_sha256=digest, frames=report.frames,
                         span_seconds=round(report.span_seconds, 6), dropped_tail=report.dropped_tail,
                         dropped_frames=report.dropped_frames)
        face[sign] = entry
        face_reports.append(dict(sign=sign, **report.to_dict()))
    print(f"face: {collections.Counter(e['status'] for e in face.values())}", flush=True)

    # --- glove validation -------------------------------------------------
    gloves = collections.defaultdict(dict)
    duplicates = set()
    for part in (1, 2, 3):
        archive = open_zip(s3, bucket, f"{SOURCE}/Handshape/Handshape Data Part {part}.zip")
        names = set(archive.namelist())
        for name in sorted(names):
            pieces = name.split("/")
            match = GLOVE_FILE.fullmatch(pieces[-1]) if len(pieces) == 3 else None
            if match is None:
                continue
            sign, prefix, code = pieces[1], match.group(1), match.group(2)
            meta_name = f"{pieces[0]}/{sign}/{prefix}_{code}Meta.json"
            if meta_name not in names:
                continue
            side = {"L": "Left", "R": "Right"}[prefix]
            if side in gloves[sign]:
                duplicates.add(sign)
            gloves[sign][side] = (archive, name, meta_name)
    hand_results, glove_reports = {}, []
    eligible = sorted(s for s, r in conversion.items()
                      if "state_key" in r and (r.get("marker_validation") or {}).get("status") == "passed")
    for sign in eligible[:args.limit]:
        record = conversion[sign]
        if sign in duplicates:
            hand_results[sign] = {side: dict(status="duplicate_glove_export") for side in ("Left", "Right")}
            continue
        parsed, problems = {}, {}
        for side, (archive, name, meta_name) in gloves.get(sign, {}).items():
            try:
                meta = parse_glove_metadata(archive.read(meta_name))
                if meta["side"] != side:
                    raise ValueError("metadata side contradicts the file name")
                parsed[side] = parse_glove_csv(archive.read(name), side=side)
            except ValueError as error:
                problems[side] = str(error)
        state = load_state(s3, bucket, record["state_key"], record["state_sha256"])
        result = {side: v.to_dict() for side, v in validate_hands(state, parsed).items()}
        for side, reason in problems.items():
            result[side] = dict(status="glove_refused", reason=reason)
        for side, glove in parsed.items():
            result[side].update(glove_ambiguous_frames=glove.ambiguous_frames,
                                glove_collapsed_duplicates=glove.collapsed_duplicates)
        hand_results[sign] = result
        glove_reports.append(dict(sign=sign, **result))
    print("hands:", {side: collections.Counter(r[side]["status"] for r in hand_results.values())
                     for side in ("Left", "Right")}, flush=True)

    # --- manifest -----------------------------------------------------------
    signs = sorted(set(conversion) | set(face))
    gloss = map_labels(signs, lexicon)
    records = []
    for sign in signs:
        conv = conversion.get(sign, {})
        records.append(dict(
            sign=sign, performer_sha256=conv.get("performer_sha256"),
            skeleton=skeleton_entry(conv) if conv else None,
            hands={side: {k: v for k, v in hand_results.get(sign, {}).get(side, {"status": "not_evaluated"}).items()
                          if k in ("status", "offset_frames", "reason")} for side in ("Left", "Right")},
            face={k: v for k, v in face.get(sign, {"status": "absent"}).items() if k != "member"},
            gloss=gloss[sign].to_dict()))
    split = performer_split(records, "skeleton")
    inputs = dict(conversion_report=sha256(report_bytes), signbank_ecv=ECV_SHA256,
                  **{f"archive:{k}": v for k, v in archive_hashes.items()})
    manifest = build_corpus_manifest(corpus_id="3d-lex-v1.0-asl-corpus-v1", inputs=inputs, records=records,
                                     split=split)
    payload = canonical_json(manifest)
    digest = put_once(s3, bucket, args.manifest_key, payload)
    assert digest == manifest_sha256(manifest)

    def dump(name, value):
        with (args.evidence / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")

    (args.evidence / "corpus-manifest.json").write_bytes(payload + b"\n")
    dump("face-conversion-report.json", dict(records=face_reports,
                                              summary=collections.Counter(r["status"] for r in face_reports)))
    dump("glove-validation-report.json", dict(
        thresholds=dict(min_correlation=MIN_CORRELATION, max_centred_mae_degrees=MAX_CENTRED_MAE_DEGREES,
                        static_range_degrees=STATIC_RANGE_DEGREES, rate_hz=120),
        duplicate_signs=sorted(duplicates), records=glove_reports))
    dump("gloss-candidates.json", dict(
        licence="Derived from ASL Signbank (CC BY-NC-SA 4.0); share alike; cite Hochgesang, Crasborn and "
                "Lillo-Martin, ASL Signbank, doi:10.6084/m9.figshare.9741788.v2",
        ecv_sha256=ECV_SHA256, entries=len(lexicon.entries),
        candidates={s: gloss[s].to_dict() for s in signs}))
    summary = dict(manifest_key=args.manifest_key, manifest_sha256=digest, counts=manifest["counts"],
                   split=split, face_prefix=face_prefix, inputs=inputs, phase_exit_approved=False)
    dump("summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
