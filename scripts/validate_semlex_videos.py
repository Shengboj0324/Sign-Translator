"""Validate the Sem-Lex video path (``extract_semlex_videos.py``) on a smoke shard before the scale run.

Stage ``prepare``: from a smoke report of the extractor, stream the same split's archive from its start and keep the
videos of the report's clips that have a state (each checked against the report's source SHA-256) in ``--video-dir``,
then write the job list of ``legacy_holistic_arrays.py`` (full frames, legacy MediaPipe Holistic as Sem-Lex used it).
Run that job list in ``~/.venvs/semlex-legacy-mp``.

Stage ``score``: recogniser v2 at 24 fps on the in-vocabulary clips (Sem-Lex ``asllex`` labels through the ASL-LEX
code to v2's ASL Citizen gloss, as ``semlex_recognition_shift.py --mapping code``), with bootstrap 95% intervals over
clips and over signers:

- ``video``: the stored video states (the generator convention: upright chest, upright root, MediaPipe visibility);
- ``video_chest_only``: the same stored raw landmarks solved with the upright chest only (ASL Citizen's corpus
  convention, which v2 was trained on): the convention's own cost, on the same clips;
- ``adapter``: legacy Holistic on the same videos through the reviewed gate-v2 path (``convert_array`` with world
  regressors v2, learned visibility, the generator convention), so the video path is set against the pose-file path
  on identical clips;
- controls: ``blank`` (every channel removed, durations kept) and ``mismatched`` (each clip scored against another
  clip's gloss, a seeded derangement), for the video and the adapter states;
- paired differences: video - adapter, video - blank, video - mismatched, adapter - blank, adapter - mismatched,
  video - video_chest_only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

CITIZEN_MANIFEST = "acquisition/2026-10-17-v1/derived/asl-citizen-v1/manifest-v1/manifest.json"
CITIZEN_MANIFEST_SHA256 = "708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6"
ASLLEX_KEY = "acquisition/2026-10-17-v1/asl-lex-2.0/signdata.csv"
STATE_STATUSES = ("converted", "resumed", "already_present")


def prepare(args) -> dict:
    from convert_semlex_poses import make_store
    from extract_semlex_videos import BASE, members, open_archive
    store = make_store("r2")
    report = json.loads(store.get(args.report_key))
    wanted = {r["video_id"]: r["source_sha256"] for r in report["results"] if r["status"] in STATE_STATUSES}
    out = Path(args.video_dir)
    out.mkdir(parents=True, exist_ok=True)
    found = {}
    stream, _ = open_archive(store, report["archive_key"])
    with stream:
        for _, video_id, payload in members(stream, report["split"], dict(count=0, first=[])):
            if video_id in wanted and payload is not None and video_id not in found:
                if hashlib.sha256(payload).hexdigest() != wanted[video_id]:
                    raise SystemExit(f"{video_id}: archive bytes differ from the report's source hash")
                (out / f"{video_id}.webm").write_bytes(payload)
                found[video_id] = True
            if len(found) == len(wanted):
                break
    jobs = [dict(video=str(out / f"{v}.webm"), box=None, out=str(Path(args.holistic_dir) / f"{v}.npz"))
            for v in sorted(found)]
    Path(args.jobs).write_text(json.dumps(jobs, indent=1))
    return dict(clips=len(wanted), videos=len(found), jobs=args.jobs, base=BASE)


def code_mapping(store):
    """Sem-Lex ASL-LEX entry -> v2 gloss: through the entry's ASL-LEX code to the one ASL Citizen gloss with that code."""
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    asllex = {r["EntryID"]: r for r in csv.DictReader(io.StringIO(store.get(ASLLEX_KEY).decode("utf-8", "replace")))}
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(CITIZEN_MANIFEST) + b"\n")
        manifest = load_manifest(path, expected_sha256=CITIZEN_MANIFEST_SHA256)
    code_gloss = collections.defaultdict(set)
    for r in manifest["records"]:
        if r["gloss"].get("asl_lex_code"):
            code_gloss[r["gloss"]["asl_lex_code"]].add(r["gloss"]["label"])

    def gloss_of(label: str) -> str | None:
        entry = asllex.get(label)
        candidates = sorted(code_gloss.get(entry["Code"], ())) if entry else []
        return candidates[0] if len(candidates) == 1 else None
    return gloss_of


def derangement(labels, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        order = rng.permutation(len(labels))
        if not (labels[order] == labels).any():
            return labels[order]
    raise RuntimeError("no derangement of the labels found")


def paired(a: np.ndarray, b: np.ndarray, k: int = 5) -> dict:
    from semlex_recognition_shift import bootstrap
    diff = (a < k).astype(np.float64) - (b < k)
    return dict(difference=round(float(diff.mean()), 4), ci=bootstrap(diff))


def score_stage(args) -> dict:
    import torch
    from convert_semlex_poses import load_metadata, make_store, METADATA_KEY, METADATA_SHA256
    from evaluate_text_to_motion import blanked
    from extract_semlex_videos import solve
    from semlex_recognition_shift import ranks_of, score
    from signtranslator.data_engineering.semlex_pose import BlendshapeModel, Refusal, convert_array, load_face_resources
    from signtranslator.data_engineering.video_landmarks import load_raw_landmarks
    from signtranslator.data_engineering.world_regressor import WORLD_REGRESSORS, WorldRegressors
    from signtranslator.pose.multichannel import MultichannelMotion
    from signtranslator.pose.resample import resample_motion
    from validate_sign_recognizer import load_model
    torch.set_num_threads(1)
    store = make_store("r2")
    report = json.loads(store.get(args.report_key))
    metadata = load_metadata(store, METADATA_KEY, METADATA_SHA256)
    gloss_of = code_mapping(store)
    model, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    resources = load_face_resources(Path(args.models_dir) / "face_landmarker.task")
    blend = BlendshapeModel(resources)
    regressors = WorldRegressors.load(store.get(WORLD_REGRESSORS["key"]), expected_sha256=WORLD_REGRESSORS["sha256"])
    variants = collections.defaultdict(list)
    labels, signers, names, skipped = [], [], [], collections.Counter()
    for r in report["results"]:
        if r["status"] not in STATE_STATUSES:
            skipped["no state"] += 1
            continue
        rows = [row for row in metadata.get(r["video_id"], []) if row["split"] == report["split"]]
        if len({(row["label_type"], row["label"]) for row in rows}) != 1:
            skipped["ambiguous metadata label"] += 1
            continue
        row = rows[0]
        gloss = gloss_of(row["label"]) if row["label_type"] == "asllex" else None
        if gloss is None or gloss not in index:
            skipped["not in v2's vocabulary"] += 1
            continue
        holistic_path = Path(args.holistic_dir) / f"{r['video_id']}.npz"
        if not holistic_path.exists():
            skipped["no legacy Holistic array"] += 1
            continue
        with np.load(holistic_path, allow_pickle=False) as z:
            holistic = z["holistic"].copy()
        try:
            adapter, _ = convert_array(holistic, float(row["duration"]), resources, video_id=r["video_id"],
                                       source_sha256=r["source_sha256"], blendshape_model=blend,
                                       regressors=regressors, learned_visibility=True, flags="generator")
        except Refusal as error:
            skipped[f"adapter refused: {str(error)[:60]}"] += 1
            continue
        payload = store.get(r["state_key"])
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "s.npz"
            path.write_bytes(payload)
            video = MultichannelMotion.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())
        clip = load_raw_landmarks(store.get(r["landmarks_key"]), expected_sha256=r["landmarks_sha256"])
        from signtranslator.data_engineering.video_landmarks import canonical_from_landmarks
        chest_only, _ = canonical_from_landmarks(clip, sample_id=r["video_id"], source_id="x", source_sha256="0" * 64,
                                                 clock_id="x", inference_method="x", upright_chest=True)
        again, _ = solve(clip, r["video_id"], r["source_sha256"], video.channels["body"].inference_method)
        if not np.array_equal(again.channels["body"].values, video.channels["body"].values):
            raise SystemExit(f"{r['video_id']}: the stored state is not the solve of its stored landmarks")
        for name, state in (("video", video), ("video_chest_only", chest_only), ("adapter", adapter)):
            variants[name].append(resample_motion(state, 24.))
        labels.append(index[gloss])
        signers.append(row["signer_id"])
        names.append(r["video_id"])
    if not labels:
        raise SystemExit("no in-vocabulary clip to score")
    other = derangement(labels, args.seed).tolist()
    ranks = {name: ranks_of(model, states, labels) for name, states in variants.items()}
    ranks["video_blank"] = ranks_of(model, blanked(variants["video"]), labels)
    ranks["adapter_blank"] = ranks_of(model, blanked(variants["adapter"]), labels)
    ranks["video_mismatched"] = ranks_of(model, variants["video"], other)
    ranks["adapter_mismatched"] = ranks_of(model, variants["adapter"], other)
    scores = {name: score(r, signers) for name, r in ranks.items()}
    pairs = {"video_minus_adapter": ("video", "adapter"), "video_minus_blank": ("video", "video_blank"),
             "video_minus_mismatched": ("video", "video_mismatched"),
             "adapter_minus_blank": ("adapter", "adapter_blank"),
             "adapter_minus_mismatched": ("adapter", "adapter_mismatched"),
             "video_minus_video_chest_only": ("video", "video_chest_only")}
    differences = {name: dict(top5=paired(ranks[a], ranks[b], 5), top1=paired(ranks[a], ranks[b], 1))
                   for name, (a, b) in pairs.items()}
    per_signer = {s: dict(clips=int(np.sum(np.array(signers) == s)),
                          **{n: round(float((ranks[n][np.array(signers) == s] < 5).mean()), 4)
                             for n in ("video", "adapter")}) for s in sorted(set(signers))}
    return dict(report_key=args.report_key, run=args.recognizer_run, checkpoint=best["checkpoint_key"],
                regressors_sha256=regressors.sha256, mapping="code", clips=len(labels), signers=len(set(signers)),
                glosses=len(set(labels)), skipped=dict(skipped), scores=scores, differences=differences,
                per_signer_top5=per_signer, clip_names=names)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=("prepare", "score"))
    p.add_argument("--report-key", required=True, help="the extractor's smoke report in the store")
    p.add_argument("--video-dir")
    p.add_argument("--holistic-dir")
    p.add_argument("--jobs")
    p.add_argument("--models-dir", default=os.path.expanduser("~/.sign-translator-ops/models"))
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output")
    args = p.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    result = prepare(args) if args.stage == "prepare" else score_stage(args)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "clip_names"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
