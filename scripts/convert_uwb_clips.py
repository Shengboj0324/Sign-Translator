"""UWB YouTube-ASL clip keypoints (CC BY 4.0, in R2) -> canonical motion states through the z-free world regressors.

The clips were acquired by ``runpod/fleet/acquire_uwb_keypoints.sh``; this reads them where they are, by ranged GETs
(``build_uwb_keypoint_shards.open_zip``: nothing is staged on disk) and converts each with
``signtranslator.data_engineering.uwb_keypoints`` (adapter ``uwb-holistic-xy-learned-world-v1``).

**Work units.** Each zip's members, in archive order (sequential reads), are cut into chunks of ``--chunk-size``;
unit ``k`` of the sorted list belongs to shard ``k % --shards``. A unit is finished when its report
``<out>/reports/<zip stem>/chunk-<nnnnn>.json`` exists. A rerun skips a finished unit **only if its recorded settings
equal this run's** (regressor SHA, fps-table SHA, overlap-table SHA, evaluation-index SHA, visibility rule, chunk
size, default fps, ...); otherwise the run refuses (exit 2) and names the differing settings. States are written
before the report, so an interrupted unit is redone whole (the conversion is deterministic). Each run writes only
its own unit reports and a per-shard summary (``<out>/shards/shard-<i>-of-<n>.json``); ``--merge-index`` (one
process, after every shard) checks that all unit reports share one settings fingerprint and writes
``<out>/index.json``, so parallel pods never race on a shared file.

**Per clip.**
- Provenance: zip, member, the member's SHA-256 (the state's ``source_sha256``), video id, first and last source frame,
  frame count and its difference from the range, frame size, the English translation, the UWB partition
  (``YT.translations.train.json`` -> ``train``, ``.dev.json`` -> ``monitor``) and the licence.
- **Evaluation leakage** (``--eval-manifest KEY@SHA``, required, repeatable: the How2Sign and ASLLRP manifests): the
  text is checked with ``build_youtube_asl_manifest.EvalIndex`` against every val/test/heldout sentence (exact or >= 0.8
  5-gram Jaccard); ``eval_leak`` and the index's SHA-256 are stored per clip and counted per unit. States are still
  written: ``uwb_manifest.select_training_records`` refuses unchecked records and drops the leaks.
- **Overlap with our own YouTube states** (``--overlap-table``, required: ``uwb_lifter_gate.py --overlap-table-out``):
  a clip paired with one of our cues (IoU >= 0.5) carries ``duplicate_of_our_cue`` (our sample id; the manifest keeps
  ours), and every clip of a video we extracted takes our channel-hashed ``partition`` (``our_partition``), so a
  channel never straddles train and monitor; ``uwb_partition`` keeps UWB's own.
- Clock: ``(first + j) / fps``. The fps is the video's entry in ``--fps-table`` (``fps_source`` = ``estimated``) or
  ``--fps`` (default 30, ``fps_source`` = ``assumed``: 27% of the videos we could estimate are not 30/29.97 fps).
- State: ``<out>/states/<clip id>.npz``, solver flags ``generator`` (upright chest and root, no elbow gate),
  blendshapes from the stored mesh, head rotation unavailable unless ``--head affine``.
- **Visibility** (``--visibility``, required: a declared rule, ``uwb_keypoints.visibility_margin``): ``learned``,
  ``inside`` or ``margin:<m>``. The first real smoke showed the How2Sign-crop visibility head dropping the right forearm
  (agreement 0.41) and the review showed ``inside`` dropping 44% / 31% of forearm frames (elbow below the frame);
  the gate decides.
- Refusals (malformed JSON, frame count off the range by more than 2, fewer than ``--min-frames`` frames, no pose)
  are listed with their reason; transient store errors are ``error`` and the unit is retried by the next run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

PREFIX = "acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1"
TRANSLATIONS = (("YT.translations.train.json", "train"), ("YT.translations.dev.json", "monitor"))
EVAL_PARTITIONS = ("val", "test", "heldout_official")
RUN_ONLY = ("limit", "video", "max_units")          # settings that restrict a run but never change a converted clip
_WORKER: dict = {}


def translations(store, prefix: str) -> dict:
    """clip id -> (text, partition), from the UWB train and dev translation files (whitespace collapsed)."""
    out = {}
    for name, partition in TRANSLATIONS:
        key = f"{prefix}/{name}"
        if not store.exists(key):
            continue
        for _, entry in json.loads(store.get(key)).items():
            for clip in entry.get("clip_order", []):
                text = (entry.get(clip) or {}).get("translation")
                if text:
                    out[clip] = (" ".join(text.split()), partition)
    return out


def load_spec(spec: str | None, store, name: str) -> tuple[bytes | None, str | None]:
    """(bytes, SHA-256) of a local path or ``KEY@SHA256`` in the store (the SHA is checked)."""
    if not spec:
        return None, None
    if Path(spec).exists():
        payload = Path(spec).read_bytes()
    else:
        key, _, sha = spec.rpartition("@")
        if not key:
            raise SystemExit(f"{name}: a local path or KEY@SHA256")
        payload = store.get(key)
        if hashlib.sha256(payload).hexdigest() != sha:
            raise SystemExit(f"{name}: content SHA-256 differs from the given one")
    return payload, hashlib.sha256(payload).hexdigest()


def load_fps_table(spec: str | None, store) -> tuple[dict, str | None]:
    """{video id: fps}; plain mappings and the gate's ``{"schema": "uwb-fps-table-v1", "fps": {...}}`` are accepted."""
    payload, sha = load_spec(spec, store, "--fps-table")
    if payload is None:
        return {}, None
    data = json.loads(payload)
    table = data["fps"] if isinstance(data, dict) and isinstance(data.get("fps"), dict) else data
    return {str(k): float(v) for k, v in table.items() if v is not None}, sha


def load_overlap_table(spec: str, store) -> tuple[dict, str]:
    """The gate's ``uwb-overlap-table-v1``: {"clips": {clip id: {our_sample_id, iou}}, "videos": {id: {partition}}}."""
    payload, sha = load_spec(spec, store, "--overlap-table")
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("schema") != "uwb-overlap-table-v1":
        raise SystemExit("--overlap-table: not a uwb-overlap-table-v1 file")
    return dict(clips=data["clips"], videos=data["videos"]), sha


def eval_index(specs: list[str], store):
    """``EvalIndex`` over the val/test/heldout sentences of every manifest, and its SHA-256 (sentences + rule)."""
    from build_youtube_asl_manifest import EvalIndex
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    sentences = []
    for spec in specs:
        key, _, sha = spec.rpartition("@")
        if not key or not sha:
            raise SystemExit("--eval-manifest: KEY@SHA256")
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "m.json"
            path.write_bytes(store.get(key) + b"\n")
            manifest = load_manifest(path, expected_sha256=sha)
        sentences += [r.get("english") or r.get("text") for r in manifest["records"]
                      if r.get("partition") in EVAL_PARTITIONS and (r.get("english") or r.get("text"))]
    if not sentences:
        raise SystemExit("--eval-manifest: no val/test sentences found")
    index = EvalIndex(sentences)
    digest = hashlib.sha256(json.dumps(dict(sentences=sorted(set(sentences)), jaccard=index.jaccard),
                                       sort_keys=True).encode()).hexdigest()
    return index, digest, len(set(sentences))


def load_regressors(spec: str, sha256: str, store):
    from signtranslator.data_engineering.world_regressor import WorldRegressors
    payload = Path(spec).read_bytes() if Path(spec).exists() else store.get(spec)
    return WorldRegressors.load(payload, expected_sha256=sha256)


def units_of(names: list[str], zip_name: str, chunk: int) -> list[dict]:
    return [dict(zip=zip_name, chunk=k // chunk, members=names[k:k + chunk]) for k in range(0, len(names), chunk)]


def report_key(out_prefix: str, unit: dict, *, partial: bool = False) -> str:
    """The unit's commit marker; a run restricted by ``--limit`` or ``--video`` writes a ``.partial.json`` report
    instead, which never marks the unit finished."""
    suffix = ".partial.json" if partial else ".json"
    return f"{out_prefix}/reports/{unit['zip'].rsplit('.', 1)[0]}/chunk-{unit['chunk']:05d}{suffix}"


def settings(args) -> dict:
    """Everything that changes a converted clip or its record (the resume and merge fingerprint)."""
    from signtranslator.data_engineering.uwb_keypoints import ADAPTER
    return dict(adapter=ADAPTER, world=args.world, visibility=args.visibility, face=args.face, head=args.head,
                flags=args.flags, default_fps=args.fps, fps_table_sha256=args.fps_table_sha256,
                overlap_table_sha256=args.overlap_table_sha256, eval_index_sha256=args.eval_index_sha256,
                eval_manifests=sorted(args.eval_manifest), regressors_sha256=args.regressors_sha256,
                min_frames=args.min_frames, chunk_size=args.chunk_size, prefix=args.prefix, out_prefix=args.out_prefix)


def fingerprint(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def settings_difference(recorded: dict | None, current: dict) -> list[str]:
    recorded = recorded or {}
    return sorted(k for k in set(recorded) | set(current) if k not in RUN_ONLY and recorded.get(k) != current.get(k))


def _init(args, texts, fps_table, overlap, index):
    import torch
    torch.set_num_threads(1)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    _WORKER.update(args=args, store=store, texts=texts, fps=fps_table, overlap=overlap, eval_index=index, archives={},
                   settings=settings(args))
    _WORKER["regressors"] = (load_regressors(args.regressors, args.regressors_sha256, store)
                             if args.world == "learned" else None)
    _WORKER["blendshapes"] = _WORKER["resources"] = None
    if args.face == "blendshapes" or args.head == "affine":
        from signtranslator.data_engineering.semlex_pose import BlendshapeModel, load_face_resources
        resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
        _WORKER["resources"] = resources
        _WORKER["blendshapes"] = BlendshapeModel(resources) if args.face == "blendshapes" else None


def _archive(zip_name: str):
    from build_uwb_keypoint_shards import open_zip
    args, store = _WORKER["args"], _WORKER["store"]
    archive = _WORKER["archives"].get(zip_name)
    if archive is None:
        archive = _WORKER["archives"][zip_name] = open_zip(store, f"{args.prefix}/{zip_name}")
    return archive


def clip_context(clip_id: str, video_id: str) -> dict:
    """Text, partitions, leak and overlap fields of one clip (also for refused clips)."""
    from signtranslator.data_engineering.uwb_keypoints import LICENCE
    text, uwb_partition = _WORKER["texts"].get(clip_id, (None, None))
    ours = _WORKER["overlap"]["videos"].get(video_id)
    pair = _WORKER["overlap"]["clips"].get(clip_id)
    return dict(text=text, uwb_partition=uwb_partition, our_video=ours is not None,
                our_partition=ours and ours.get("partition"),
                partition=(ours.get("partition") if ours else uwb_partition),
                duplicate_of_our_cue=pair and pair.get("our_sample_id"), our_cue_iou=pair and pair.get("iou"),
                eval_leak=None if not text else bool(_WORKER["eval_index"].leaks(text)),
                eval_index_sha256=_WORKER["args"].eval_index_sha256, licence=LICENCE)


def convert_member(name: str, payload: bytes, zip_name: str) -> tuple[dict, bytes | None]:
    """(provenance record, state bytes or None) for one member's bytes."""
    from signtranslator.data_engineering import uwb_keypoints as uk
    args = _WORKER["args"]
    record = dict(zip=zip_name, member=name, member_sha256=hashlib.sha256(payload).hexdigest())
    try:
        clip_id, video_id, first, last = uk.parse_clip_name(name)
    except uk.Refusal as error:
        return dict(record, status="refused", reason=str(error)), None
    fps = _WORKER["fps"].get(video_id)
    fps_source = "estimated" if fps is not None else "assumed"
    fps = float(fps if fps is not None else args.fps)
    record.update(clip_id=clip_id, video_id=video_id, first_frame=first, last_frame=last, fps=fps,
                  fps_source=fps_source, **clip_context(clip_id, video_id))
    try:
        clip = uk.parse_clip(payload, name)
        record.update(frames=clip.frames, frame_count_difference=clip.frame_count_difference,
                      image_size=[clip.width, clip.height])
        if clip.frames < args.min_frames:
            raise uk.Refusal(f"{clip.frames} frames, fewer than {args.min_frames}")
        state, report = uk.convert_clip(
            clip, fps, source_sha256=record["member_sha256"], fps_source=fps_source,
            regressors=_WORKER["regressors"], world=args.world, visibility=args.visibility,
            blendshape_model=_WORKER["blendshapes"], head=args.head, face_resources=_WORKER["resources"],
            flags=args.flags)
    except uk.Refusal as error:
        return dict(record, status="refused", reason=str(error)[:300]), None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.npz"
        digest = state.save(path)
        data = path.read_bytes()
    body = state.channels["body"]
    arms = {s: round(float(body.valid[:, body.labels.index(f"{s}Forearm")].mean()), 4) for s in ("Left", "Right")}
    hands = {s: round(float(state.channels[f"{s.lower()}_hand"].valid.any(axis=1).mean()), 4) for s in ("Left", "Right")}
    record.update(status="converted", state_key=f"{args.out_prefix}/states/{clip_id}.npz", state_sha256=digest,
                  seconds=round(float(state.timestamps[-1] - state.timestamps[0]), 4), forearm_fraction=arms,
                  hand_fraction=hands, face_fraction=round(float(state.channels["face"].valid.any(axis=1).mean()), 4),
                  available_channels=[n for n, c in state.channels.items() if c.valid.any()],
                  hand_assignment=report["hand_assignment"], adapter=report["adapter"],
                  regressors_sha256=report.get("regressors_sha256"), visibility=args.visibility)
    return record, data


def convert_unit(unit: dict) -> dict:
    """Convert one unit; its report is written last (the commit marker) unless a member hit a store error. A finished
    unit whose recorded settings differ from this run's is never skipped silently: ``settings-mismatch``."""
    args, store, current = _WORKER["args"], _WORKER["store"], _WORKER["settings"]
    final = report_key(args.out_prefix, unit)
    if store.exists(final):
        done = json.loads(store.get(final))
        differ = settings_difference(done.get("settings"), current)
        if differ:
            return dict(key=final, status="settings-mismatch", differ=differ)
        return dict(key=final, status="present", counts=done["counts"])
    key = report_key(args.out_prefix, unit, partial=bool(args.limit or args.video))
    started = time.time()
    records, errors = [], 0
    archive = _archive(unit["zip"])
    infos = sorted((archive.getinfo(n) for n in unit["members"]), key=lambda i: i.header_offset)
    for info in infos[:args.limit or None]:
        if args.video and info.filename.rsplit(".", 2)[0] not in args.video:
            continue
        try:
            payload = archive.read(info)
        except Exception as error:  # noqa: BLE001 - a transient store failure: retried by the next run
            errors += 1
            records.append(dict(zip=unit["zip"], member=info.filename, status="error",
                                reason=f"{type(error).__name__}: {error}"[:300]))
            _WORKER["archives"].pop(unit["zip"], None)
            archive = _archive(unit["zip"])
            continue
        record, data = convert_member(info.filename, payload, unit["zip"])
        if data is not None:
            store.put(record["state_key"], data)
        records.append(record)
    counts = collections.Counter(r["status"] for r in records)
    reasons = collections.Counter(re.sub(r"\d+", "N", r["reason"].split(": ", 1)[-1])[:80]
                                  for r in records if r["status"] == "refused")
    converted = [r for r in records if r["status"] == "converted"]
    fps_sources = collections.Counter(r.get("fps_source") for r in converted)
    checks = dict(eval_leak=sum(bool(r.get("eval_leak")) for r in converted),
                  no_text=sum(not r.get("text") for r in converted),
                  duplicate_of_our_cue=sum(bool(r.get("duplicate_of_our_cue")) for r in converted),
                  our_video=sum(bool(r.get("our_video")) for r in converted),
                  partitions=dict(collections.Counter(r.get("partition") for r in converted)))
    report = dict(zip=unit["zip"], chunk=unit["chunk"], members=len(unit["members"]), counts=dict(counts),
                  refusal_reasons=dict(reasons), fps_sources=dict(fps_sources), checks=checks,
                  seconds=round(time.time() - started, 1), settings=current,
                  settings_fingerprint=fingerprint(current),
                  run=dict(limit=args.limit or None, video=sorted(args.video) if args.video else None), clips=records)
    if errors:
        return dict(key=key, status="incomplete", counts=dict(counts))   # no report: the whole unit is redone
    store.put(key, json.dumps(report, indent=1, sort_keys=True).encode())
    return dict(key=key, status="written", counts=dict(counts), checks=checks)


def merge_index(store, out_prefix: str, units_total: int | None = None) -> dict:
    """``<out>/index.json`` from every finished unit report; refuses when the units disagree on settings."""
    from signtranslator.data_engineering.uwb_keypoints import ADAPTER, LICENCE
    keys = sorted(k for k in store.list(f"{out_prefix}/reports/") if re.search(r"chunk-\d+\.json$", k))
    counts, fps_sources, checks = collections.Counter(), collections.Counter(), collections.Counter()
    partitions, fingerprints, units = collections.Counter(), collections.defaultdict(list), []
    for key in keys:
        part = json.loads(store.get(key))
        counts.update(part["counts"])
        fps_sources.update(part.get("fps_sources", {}))
        c = part.get("checks", {})
        checks.update({k: v for k, v in c.items() if k != "partitions"})
        partitions.update(c.get("partitions", {}))
        fp = part.get("settings_fingerprint") or fingerprint(part.get("settings") or {})
        fingerprints[fp].append(key)
        units.append(dict(key=key, settings_fingerprint=fp, counts=part["counts"]))
    if len(fingerprints) > 1:
        raise SystemExit("unit reports disagree on settings: "
                         + json.dumps({fp: v[:3] + (["..."] if len(v) > 3 else []) for fp, v in fingerprints.items()}))
    settings_used = json.loads(store.get(keys[0]))["settings"] if keys else None
    index = dict(corpus_id="youtube-asl-uwb-keypoints-v1", adapter=ADAPTER, licence=LICENCE,
                 units_total=units_total, units_finished=len(keys), counts=dict(counts), fps_sources=dict(fps_sources),
                 checks=dict(checks), partitions=dict(partitions), settings=settings_used,
                 settings_fingerprint=next(iter(fingerprints), None), units=units)
    store.put(f"{out_prefix}/index.json", json.dumps(index, indent=1, sort_keys=True).encode())
    return index


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="r2")
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--out-prefix", required=True, help="where states, reports and the index go")
    p.add_argument("--merge-index", action="store_true", help="only merge the finished unit reports into index.json")
    p.add_argument("--zip", action="append", default=[], help="zip names (default: every raw_keypoints_*.zip)")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--fps", type=float, default=30., help="assumed rate for videos without an fps-table entry")
    p.add_argument("--fps-table", help="local path or KEY@SHA256 of per-video rates (uwb_lifter_gate.py)")
    p.add_argument("--overlap-table", help="required: local path or KEY@SHA256 (uwb_lifter_gate.py --overlap-table-out)")
    p.add_argument("--eval-manifest", action="append", default=[],
                   help="required, repeatable: KEY@SHA256 of a manifest whose val/test sentences must not train")
    p.add_argument("--regressors", help="z-free bundle: local path or store key")
    p.add_argument("--regressors-sha256")
    p.add_argument("--world", choices=("learned", "pseudo"), default="learned")
    p.add_argument("--visibility", help="required, declared: learned | inside | margin:<m>")
    p.add_argument("--face", choices=("blendshapes", "none"), default="blendshapes")
    p.add_argument("--head", choices=("none", "affine"), default="none")
    p.add_argument("--flags", choices=("generator", "upright-chest"), default="generator")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="members per unit (testing)")
    p.add_argument("--max-units", type=int, default=0, help="units of this shard (testing)")
    p.add_argument("--video", action="append", default=[], help="only these video ids (testing)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if args.merge_index:
        index = merge_index(store, args.out_prefix)
        print(json.dumps({k: index[k] for k in ("units_finished", "counts", "checks", "settings_fingerprint")}))
        return 0
    if not 0 <= args.shard < args.shards:
        raise SystemExit("--shard must lie in [0, --shards)")
    from signtranslator.data_engineering.uwb_keypoints import visibility_margin
    if args.visibility is None or args.visibility == "all":
        raise SystemExit("--visibility is a declared rule: learned | inside | margin:<m>")
    try:
        visibility_margin(args.visibility)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.world == "learned" and not (args.regressors and args.regressors_sha256):
        raise SystemExit("--world learned needs --regressors and --regressors-sha256")
    if args.world == "pseudo" and args.visibility == "learned":
        raise SystemExit("--world pseudo has no visibility head: pass --visibility inside")
    if args.out_prefix.rstrip("/") == args.prefix.rstrip("/"):
        raise SystemExit("--out-prefix must differ from the source prefix")
    if not args.eval_manifest:
        raise SystemExit("--eval-manifest is required (How2Sign and ASLLRP val/test sentences must never train)")
    if not args.overlap_table:
        raise SystemExit("--overlap-table is required (cues we already hold come from our own states)")
    args.video = set(args.video)
    from build_uwb_keypoint_shards import open_zip
    fps_table, args.fps_table_sha256 = load_fps_table(args.fps_table, store)
    overlap, args.overlap_table_sha256 = load_overlap_table(args.overlap_table, store)
    index, args.eval_index_sha256, sentences = eval_index(args.eval_manifest, store)
    if args.world == "learned":
        bundle = load_regressors(args.regressors, args.regressors_sha256, store)
        from signtranslator.data_engineering.world_regressor import Z_FREE_FEATURES
        if bundle.features not in Z_FREE_FEATURES:
            raise SystemExit(f"the bundle's features are {bundle.features!r}; UWB needs one of {sorted(Z_FREE_FEATURES)}")
    else:
        args.regressors_sha256 = None
    zips = args.zip or sorted((Path(k).name for k in store.list(args.prefix + "/")
                               if re.search(r"raw_keypoints_\d+\.zip$", k)), key=lambda n: int(re.findall(r"\d+", n)[-1]))
    if not zips:
        raise SystemExit("no keypoint zips under the prefix")
    units = []
    for name in zips:
        with open_zip(store, f"{args.prefix}/{name}") as archive:
            members = [i.filename for i in sorted(archive.infolist(), key=lambda i: i.header_offset)
                       if i.filename.endswith(".json")]
        units += units_of(members, name, args.chunk_size)
    mine = [u for k, u in enumerate(units) if k % args.shards == args.shard][:args.max_units or None]
    print(json.dumps(dict(zips=len(zips), units=len(units), shard_units=len(mine), fps_table_videos=len(fps_table),
                          overlap_videos=len(overlap["videos"]), eval_sentences=sentences)), flush=True)
    if args.dry_run:
        return 0
    texts = translations(store, args.prefix)
    results = []
    with Pool(max(1, min(args.workers, len(mine))), initializer=_init,
              initargs=(args, texts, fps_table, overlap, index)) as pool:
        for result in pool.imap_unordered(convert_unit, mine):
            results.append(result)
            print(json.dumps(result), flush=True)
    statuses = collections.Counter(r["status"] for r in results)
    summary = dict(shard=args.shard, shards=args.shards, units=len(mine), statuses=dict(statuses),
                   settings=settings(args), settings_fingerprint=fingerprint(settings(args)),
                   mismatches=[r for r in results if r["status"] == "settings-mismatch"],
                   run=dict(limit=args.limit or None, video=sorted(args.video) or None, max_units=args.max_units or None))
    store.put(f"{args.out_prefix}/shards/shard-{args.shard:03d}-of-{args.shards:03d}.json",
              json.dumps(summary, indent=1, sort_keys=True).encode())
    print(json.dumps(dict(statuses=dict(statuses))), flush=True)
    if statuses.get("settings-mismatch"):
        return 2
    return 0 if not statuses.get("incomplete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
