"""Third hand path, format check: does SignAvatars hold a hand source we can read safely, for D20 by exact member, that is not MediaPipe-derived? Report only. **This commit is the format check alone, and that check stopped the conversion:** no D20 member is opened, nothing is converted and no instrument loads here.

**The check, on 5 c02 train clips** (``s2_lib.smoke_records``), by exact member, never from the zip's names: the restricted unpickler, which allows numpy reconstruction and plain types and refuses all else, must load each clip, and the format must be what the notes claim: SMPL-X hand pose as named fields of 45 values per hand, at 24 fps, with its frame stated. Anything else stops with status 4. A pickle that the unpickler refuses is described by ``pickletools`` alone, which executes nothing.

**Found 2026-11-02 on those 5 clips** (the job reproduces it):
- **The restricted unpickler refuses every clip.** Tensors are stored through ``torch._utils._rebuild_tensor_v2`` and ``torch.storage._load_from_bytes``, and the second runs ``torch.load`` on embedded bytes.
- **Fields:** ``left_valid``, ``right_valid``, ``total_valid_index``, ``2d``, ``pred_2d``, ``smplx``, ``unsmooth_smplx``, ``bb2img_trans``, ``focal``, ``princpt``, ``height``, ``width``.
- **Hands are not named.** ``smplx`` looks like a flat (T, 182) float32 vector and ``unsmooth_smplx`` (T, 169), from blob sizes and the shape integers. 3 + 63 + 45 + 45 + 3 + 10 = 169 and + 10 + 3 = 182 are arithmetic only: 45 values per hand and the order are unverified.
- **Frame rate: not determined.** Frames over the two-view clip's stored seconds read 24.05, 25.0, 24.13, 27.09 and 24.24, and that clip is a slice of the source, so this is no measure of the pickle's own rate, and one clip misses. **Frame:** ``bb2img_trans``, ``focal`` and ``princpt`` point to camera space per clip; the pickle does not say.

**How a full read would be read, stated now, used only if the conversion is approved and built (new code, new header).** Forms: its hands, then its hands with wrist orientation, on two-view's and on monocular's body, through v1 and the parent, real forms only, D20 by exact member, control first. Against each path's own hands on the same body, the path read's thresholds, 0.5 BLEU and 0.052 nats on v1, and the parent's margin share: a rise on monocular's body beyond the threshold says the MediaPipe-derived hands cost that instrument its reading; no rise on two-view's body but a rise on monocular's would point at those hands, not at the instrument's fit; no rise on either says no hand source here helps these instruments, which is not evidence on signing. SignAvatars' fit starts from its own keypoints, and how far they share a detector with MediaPipe is not known here.

    python scripts/third_path.py format --out-file W/format.json

The job is ``runpod/fleet/third_path.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402

PROBE = "third-path"
JOB_FILE = "runpod/fleet/third_path.sh"
ZIP_KEY = "acquisition/2026-10-30-v1/signavatars/files/asl-language-default/how2sign_pkls_default_shape.zip"
ZIP_SIZE = 14851588777                                     # INDEX.json; its 14.9 GB hash is not re-read
INNER = "how2sign_pkls_cropTrue_shapeFalse"
FORMAT_CLIPS = 5
MEMBER_CEILING = 64 * 2 ** 20
EXIT_FORMAT = 4
FPS, FPS_TOLERANCE = 24., .5
ALLOWED = frozenset({("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")})
NAMED_HANDS = ("left_hand_pose", "right_hand_pose")
HAND_VALUES = 45


class Refused(pickle.UnpicklingError):
    pass


class RestrictedUnpickler(pickle.Unpickler):
    """Numpy reconstruction and plain types; every other global refuses before it is imported."""

    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise Refused(f"{module}.{name}")
        return getattr(__import__(module, fromlist=[name]), name)

    def persistent_load(self, pid):
        raise Refused("a persistent id")


def safe_load(raw: bytes):
    return RestrictedUnpickler(io.BytesIO(raw)).load()


def member_of(sign: str) -> str:
    """A clip's exact member. A name that could reach another folder refuses."""
    if not sign or "/" in sign or "\\" in sign or ".." in sign or not sign.endswith("-rgb_front"):
        raise SystemExit(f"{sign!r}: not a How2Sign front clip's name")
    return f"{INNER}/{sign}.pkl"


class RangeFile(io.RawIOBase):
    """The zip on the store, read by ranges: its 14.9 GB is never fetched."""

    def __init__(self, store, key: str) -> None:
        self.store, self.key, self.pos = store, key, 0
        self.size = store.s3.head_object(Bucket=store.bucket, Key=key)["ContentLength"]

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or not len(buffer):
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        data = self.store.s3.get_object(Bucket=self.store.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}")["Body"].read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_zip(store) -> zipfile.ZipFile:
    """The zip, checked by size. Its names are held by ``zipfile`` and never read from here: every member is asked for by name."""
    if hasattr(store, "s3"):
        handle = RangeFile(store, ZIP_KEY)
        have = handle.size
        stream = io.BufferedReader(handle, buffer_size=2 ** 20)
    else:
        path = Path(store.root) / ZIP_KEY
        have = path.stat().st_size
        stream = path.open("rb")
    if have != ZIP_SIZE:
        stream.close()
        raise SystemExit(f"{ZIP_KEY}: {have} bytes, its index says {ZIP_SIZE}")
    archive = zipfile.ZipFile(stream)
    archive._filePassed = 0  # closing the archive closes the stream it was given
    return archive


def read_member(archive, sign: str) -> bytes | None:
    """A clip's bytes, by exact member; None if the zip has no such member. A member over the ceiling refuses unread."""
    try:
        info = archive.getinfo(member_of(sign))
    except KeyError:
        return None
    if info.file_size > MEMBER_CEILING:
        raise SystemExit(f"{sign}: {info.file_size} bytes, over {MEMBER_CEILING}")
    return archive.read(info)


def disassemble(raw: bytes) -> dict:
    """What a pickle names and holds, by ``pickletools`` alone: nothing is loaded or run. Field names are the short strings outside the globals' module and name."""
    strings, globals_, blobs, ints, recent = [], Counter(), [], Counter(), []
    for op, arg, _ in pickletools.genops(raw):
        if op.name == "GLOBAL":
            globals_[str(arg).replace(" ", ".")] += 1
        elif op.name == "STACK_GLOBAL" and len(recent) >= 2:
            globals_[f"{recent[-2]}.{recent[-1]}"] += 1
            strings = [s for s in strings if s not in recent[-2:]]
        elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            recent.append(arg)
            if len(arg) < 40 and arg not in strings:
                strings.append(arg)
        elif op.name in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8"):
            blobs.append(len(arg))
        elif op.name in ("BININT1", "BININT2", "BININT") and arg > 6:
            ints[arg] += 1
    return dict(fields=[s for s in strings if s not in ("<", "f4", "f8", "i8", "ascii", "latin1", "b")], globals=dict(sorted(globals_.items())), byte_blobs=blobs, shape_integers=dict(ints.most_common(8)))


def frames_of(info: dict) -> int | None:
    """The clip's frames: the integer the pickle repeats most, every per-frame array's first dimension. A disassembly gives no better."""
    return max(info["shape_integers"], key=info["shape_integers"].get) if info["shape_integers"] else None


def clip_row(sign: str, raw: bytes | None, seconds) -> dict:
    if raw is None:
        return dict(sign=sign, member=False)
    row = dict(sign=sign, member=True, bytes=len(raw))
    try:
        obj = safe_load(raw)
    except pickle.UnpicklingError as error:
        row.update(restricted_load=f"refused: {error}", structure=disassemble(raw))
    else:
        row.update(restricted_load="loaded", fields=sorted(str(k) for k in obj) if isinstance(obj, dict) else type(obj).__name__, structure=disassemble(raw))
        if isinstance(obj, dict):
            row["hands"] = {k: list(getattr(obj[k], "shape", ())) for k in NAMED_HANDS if k in obj}
    frames = frames_of(row["structure"])
    row["frames"] = frames
    row["seconds"] = seconds
    row["fps"] = round(frames / seconds, 2) if frames and seconds else None
    return row


def verdict(rows: list[dict]) -> dict:
    """What is claimed against what the clips hold. ``stop`` is true on any miss: no conversion may rest on this format."""
    misses = []
    if len(rows) < FORMAT_CLIPS or not all(r["member"] for r in rows):
        misses.append("a format clip has no member")
    present = [r for r in rows if r["member"]]
    if any(r["restricted_load"] != "loaded" for r in present):
        misses.append("the restricted unpickler refuses the pickles: " + ", ".join(sorted({r["restricted_load"].removeprefix("refused: ") for r in present if r["restricted_load"] != "loaded"})))
    if any(any(r.get("hands", {}).get(k, [None, None])[-1] != HAND_VALUES for k in NAMED_HANDS) for r in present):
        misses.append(f"no named hand pose of {HAND_VALUES} values per hand")
    if any(r["fps"] is None or abs(r["fps"] - FPS) > FPS_TOLERANCE * 2 for r in present):
        misses.append(f"frames over seconds is not {FPS} fps")
    return dict(stop=bool(misses), misses=misses, camera_or_world="not stated in the pickle: bb2img_trans, focal and princpt point to a per-clip camera")


def run_format(args) -> int:
    """5 c02 train clips by exact member; the verdict. Exit 4 when the format is not what is claimed: nothing further may run."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    records = L.smoke_records(manifest, FORMAT_CLIPS)
    L.check_train_records(records, ("c02",))
    with open_zip(store) as archive:
        rows = [clip_row(r["sign"], read_member(archive, r["sign"]), r["continuous"].get("seconds")) for r in records]
    result = dict(kind=f"{PROBE}-format", report_only=True, list="smoke:c02-train", zip=ZIP_KEY, zip_size=ZIP_SIZE, allowed=sorted(".".join(x) for x in ALLOWED), clips=rows, verdict=verdict(rows), note="pickles the unpickler refuses are described by pickletools alone: nothing refused was run")
    out = Path(args.out_file)
    if out.exists():
        raise SystemExit(f"{out} exists (write-once)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(dict(verdict=result["verdict"], fields=rows[0].get("structure", {}).get("fields"), globals=rows[0].get("structure", {}).get("globals"), fps=[r.get("fps") for r in rows], restricted_load=[r.get("restricted_load") for r in rows])), flush=True)
    if result["verdict"]["stop"]:
        print(f"ALERT the format is not what is claimed: {result['verdict']['misses']}. No D20 member is opened and nothing is converted", flush=True)
        return EXIT_FORMAT
    return 0


def run_put(args) -> int:
    from train_text_to_motion import make_store
    store = make_store(args.store)
    L.refuse_existing(store, args.key, None)
    store.put(args.key, Path(args.file).read_bytes())
    print(json.dumps(dict(stored=args.key)), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("format", help="5 c02 train clips by exact member; exit 4 when the format is not what is claimed")
    f.add_argument("--store", default="r2")
    f.add_argument("--out-file", required=True)
    put = sub.add_parser("put")
    put.add_argument("--store", default="r2")
    put.add_argument("--file", required=True)
    put.add_argument("--key", required=True)
    args = p.parse_args(argv)
    return dict(format=run_format, put=run_put)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
