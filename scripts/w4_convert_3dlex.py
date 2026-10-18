"""Convert every 3D-LEX v1.0 ASL GLB in R2 into canonical states stored back in R2.

Run with an interpreter that has numpy/torch and boto3 (the project environment
plus the isolated R2 environment's packages). Archives are read by ranged GETs,
so nothing is staged locally beyond one small state at a time.

Performer grouping comes from the first line of each sign's Vicon marker CSV. The
raw label is never stored: only a salted SHA-256 enters the outputs, and it is a
grouping key, not an identity. Every refused file and every invalidated frame is
reported. The derived prefix is versioned; nothing existing is overwritten.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from signtranslator.data_engineering.gltf_motion import (  # noqa: E402
    THREE_D_LEX_MIXAMO68, adapt_glb, skeleton_elbow_flexion,
)
from signtranslator.data_engineering.marker_validation import (  # noqa: E402
    align_joint, marker_elbow_flexion, parse_v3d_markers,
)

# Fixed before the full run from a 20-file pilot (clean-performer median r 0.994,
# centred error 2.6 degrees). Pilot revision, recorded: one clock per take for
# both elbows, and correlation is required only where an elbow moves at least
# STATIC_RANGE_DEGREES, because Pearson r carries no information at near-zero
# variance. Magnitude agreement is always required.
MIN_CORRELATION = .90
MAX_CENTRED_MAE_DEGREES = 10.
STATIC_RANGE_DEGREES = 5.

SOURCE = "acquisition/2026-10-17-v1/3d-lex-v1.0-asl"
INFERENCE = ("vicon-optical-markers+stretchsense-gloves solved in Shogun Post, exported "
             "FBX then FBX2glTF v0.9.7; values are fitted, not directly observed")
SALT = "3d-lex-v1.0-asl-performer:"


class RangeFile(io.RawIOBase):
    def __init__(self, s3, bucket, key):
        self.s3, self.bucket, self.key, self.position = s3, bucket, key, 0
        self.size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        data = self.s3.get_object(Bucket=self.bucket, Key=self.key,
                                  Range=f"bytes={self.position}-{self.position + size - 1}")["Body"].read()
        if len(data) != size:
            raise IOError("short ranged read")
        self.position += size
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def client():
    import boto3
    from botocore.config import Config
    env = dict(line.split("=", 1) for line in (ROOT / ".env.r2.local").read_text().splitlines() if line)
    return boto3.client("s3", endpoint_url=env["R2_ENDPOINT_URL"], region_name="auto",
                        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
                        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
                        config=Config(connect_timeout=20, read_timeout=300,
                                      retries={"mode": "standard", "max_attempts": 8})), env["R2_BUCKET"]


def open_zip(s3, bucket, key):
    return zipfile.ZipFile(io.BufferedReader(RangeFile(s3, bucket, key), buffer_size=1 << 20))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-prefix", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--salvage", action="store_true",
                        help="convert keyed-axis files with root rotation/translation unavailable")
    parser.add_argument("--validate-markers", action="store_true",
                        help="admit only files whose elbows agree with the raw Vicon markers")
    args = parser.parse_args(argv)
    s3, bucket = client()
    prefix = args.derived_prefix.rstrip("/")
    if s3.list_objects_v2(Bucket=bucket, Prefix=prefix + "/", MaxKeys=1).get("KeyCount"):
        raise SystemExit("derived prefix already holds objects; choose a new version")

    performers, vicon_members = {}, {}
    vicon = open_zip(s3, bucket, f"{SOURCE}/BodyPose/ASL_Vicon_MarkerData.zip")
    for info in vicon.infolist():
        if info.is_dir():
            continue
        sign = info.filename.rsplit("/", 1)[-1].removesuffix("_V3D_MarkerData.csv")
        with vicon.open(info) as stream:
            label = stream.readline().decode("utf-8", "replace").strip()
        performers[sign] = hashlib.sha256((SALT + label).encode()).hexdigest()
        vicon_members[sign] = info.filename

    def validate(sign, data, state):
        thresholds = dict(min_correlation=MIN_CORRELATION,
                          max_centred_mae_degrees=MAX_CENTRED_MAE_DEGREES,
                          static_range_degrees=STATIC_RANGE_DEGREES)
        if sign not in vicon_members:
            return dict(status="unavailable", reason="no Vicon marker file for this sign",
                        thresholds=thresholds)
        names, markers = parse_v3d_markers(vicon.read(vicon_members[sign]).decode("utf-8", "replace"))
        valid = state.channels["body"].valid[:, state.channels["body"].labels.index("LeftArm")]
        times, left = skeleton_elbow_flexion(data, "Left")
        _, right = skeleton_elbow_flexion(data, "Right")
        result = align_joint(times[valid], [left[valid], right[valid]],
                             [marker_elbow_flexion(names, markers, "L"),
                              marker_elbow_flexion(names, markers, "R")])
        if result is None:
            return dict(status="failed", reason="no rate/offset aligns the take", thresholds=thresholds)
        passed = result.passes(**thresholds)
        return dict(status="passed" if all(passed) else "failed", rate_hz=result.rate_hz,
                    offset_frames=result.offset_frames,
                    elbows={side: dict(correlation=round(r, 6) if r == r else None,
                                       centred_mae_degrees=round(mae, 4),
                                       reference_range_degrees=round(span, 3), passed=ok)
                            for side, r, mae, span, ok in zip(
                                ("Left", "Right"), result.correlations,
                                result.centred_mae_degrees, result.reference_ranges_degrees, passed)},
                    thresholds=thresholds)

    archive = open_zip(s3, bucket, f"{SOURCE}/Animation/ASL_ANIM_GLB.zip")
    members = [info for info in archive.infolist() if not info.is_dir()]
    if args.limit:
        members = members[:args.limit]
    records, refusals = [], collections.Counter()
    with tempfile.TemporaryDirectory() as temporary:
        for position, info in enumerate(members):
            sign = info.filename.rsplit("/", 1)[-1].removesuffix(".glb")
            data = archive.read(info)
            sample = f"3dlex-asl-{sign}"
            state, report = adapt_glb(data, sample_id=sample, source_id="3d-lex-v1.0-asl",
                                      clock_id=f"{sample}-glb-clock", inference_method=INFERENCE,
                                      salvage_reoriented_root=args.salvage)
            record = dict(sign=sign, member=info.filename, glb_sha256=report.source_sha256,
                          glb_bytes=len(data), performer_sha256=performers.get(sign),
                          report=report.to_dict())
            if state is not None and args.validate_markers:
                record["marker_validation"] = validate(sign, data, state)
                if record["marker_validation"]["status"] != "passed":
                    refusals[f"marker validation {record['marker_validation']['status']}"
                             + (" (salvaged file)" if report.status == "converted_partial" else "")] += 1
                    records.append(record)
                    continue
            if state is None:
                refusals[report.reason.split(":")[0]] += 1
            else:
                path = Path(temporary) / f"{sign}.npz"
                digest = state.save(path)
                payload = path.read_bytes()
                path.unlink()
                key = f"{prefix}/states/{sign}.npz"
                s3.put_object(Bucket=bucket, Key=key, Body=payload, Metadata={"sha256": digest})
                stored = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                if hashlib.sha256(stored).hexdigest() != digest:
                    raise SystemExit(f"R2 re-read mismatch for {key}")
                record.update(state_key=key, state_sha256=digest, state_bytes=len(payload),
                              frames=len(state.timestamps),
                              valid_frames=int(state.channels["body"].valid.any(axis=1).sum()))
            records.append(record)
            if position % 100 == 0:
                print(f"{position}/{len(members)} processed", flush=True)

    converted = [r for r in records if "state_key" in r]
    invalid = collections.Counter(len(r["report"]["invalid_frames"]) for r in converted)
    trailing = sum(1 for r in converted if any(f["frame"] == r["frames"] - 1
                                               for f in r["report"]["invalid_frames"]))
    summary = dict(
        schema_version=1, source_prefix=SOURCE, derived_prefix=prefix,
        mapping_id=THREE_D_LEX_MIXAMO68.mapping_id, mapping_sha256=THREE_D_LEX_MIXAMO68.sha256(),
        glb_members=len(members), converted=len(converted), refused=len(records) - len(converted),
        refusal_reasons=dict(refusals),
        invalid_frame_count_histogram=dict(sorted(invalid.items())),
        files_with_invalid_trailing_frame=trailing,
        salvaged_partial=sum(1 for r in converted if r["report"]["status"] == "converted_partial"),
        marker_validation=dict(
            enabled=args.validate_markers, min_correlation=MIN_CORRELATION,
            max_centred_mae_degrees=MAX_CENTRED_MAE_DEGREES,
            static_range_degrees=STATIC_RANGE_DEGREES,
            recovered_rates_hz=dict(collections.Counter(
                str(r["marker_validation"]["rate_hz"]) for r in converted
                if "rate_hz" in r.get("marker_validation", {})))),
        frames_total=sum(r["frames"] for r in converted),
        valid_frames_total=sum(r["valid_frames"] for r in converted),
        max_quaternion_norm_error=max((r["report"]["max_quaternion_norm_error"] for r in converted), default=None),
        performer_groups=dict(collections.Counter(r["performer_sha256"] for r in converted)),
        converted_without_performer=sum(1 for r in converted if r["performer_sha256"] is None),
        inference_method=INFERENCE, phase_exit_approved=False)
    body = json.dumps(dict(summary=summary, records=records), indent=1, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=f"{prefix}/conversion-report.json", Body=body)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("xb") as stream:
        stream.write(body)
    print(json.dumps({k: v for k, v in summary.items() if k != "performer_groups"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
