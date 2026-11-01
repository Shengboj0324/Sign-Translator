"""How much of YouTube's training data has no usable wrist, and does it reach the generator? On CPU, read-only on stored states, nothing trains. A clip's wrist is usable on a side when two consecutive frames at 24 fps hold a valid wrist, the condition a speed number needs.

A fixed draw of the youtube-asl-v1 train records of ``round14b-mixed-v1``, the generator's mix, which the tokenizer's ``round3c-mixed-v1`` holds as well; a bootstrap by video, because validity is mostly a property of the video. The share of valid body columns is reported per class of clip: the prior's ``part_code_weights`` zeroes a part's codes where under half its unit columns are valid.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
YOUTUBE = "youtube-asl-v1"
MIX = (f"{O}/round14b-mixed-v1/manifest.json", "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d")
TOKENIZER_MIX = (f"{O}/round3c-mixed-v1/manifest.json", "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd")
ROUND13 = f"{O}/round13-mixed-v1/manifest.json"
PRIOR_RUN = "training-runs/tokens-rvq-d0b-prior-r14b-w14b/run.json"
TOKENIZER_RUN = "training-runs/tokens-rvq-d0b-r3d-d2/run.json"
EARLIER = f"{O}/arm-recovery/step-a-2026-10-28/state-validity-2000-seed7.json"
SEED = "youtube-wrist-validity"
SAMPLE = 3000
BOOT = 2000
SIDES = dict(left=4, right=5)
NOT_MEASURED = ("what the generator learns from these clips: part_code_weights zeroes the body codes where under half the body part's columns are valid, and valid body columns on these clips are 0.03%, so by code they teach no neutral body directly. Their neutral-valued body codes are still history for the hand codes, in about a fifth of the generator's samples. The tokenizer's channel reconstruction loss is masked by validity (rvq_loss, channel_loss_per_example), but its encoder and quantizer still see the neutral fill. Neither effect has been measured.")


def usable(valid: np.ndarray) -> bool:
    """Two consecutive valid frames: what a speed number needs."""
    return bool((valid[1:] & valid[:-1]).any())


def clip_row(state, record: dict) -> dict:
    """One clip's validity, never its motion."""
    from comp_a_lib import keypoint_view
    _, valid = keypoint_view(state)
    body = state.channels["body"]
    chest = body.valid[:, list(body.labels).index("Chest")]
    row = dict(video=record["sign"].split(":")[-1].rsplit("-", 1)[0], channel=(record.get("source") or {}).get("channel_id"), seconds=float((record.get("continuous") or {}).get("seconds") or 0.), chest=float(chest.mean()), body_columns=float(body.valid.mean()), right_hand=float(state.channels["right_hand"].valid.mean()),
               left_hand=float(state.channels["left_hand"].valid.mean()))
    for side, w in SIDES.items():
        row[f"{side}_usable"] = usable(valid[:, w])
        row[f"{side}_share"] = float(valid[:, w].mean())
    return row


def weight_share(records_by_corpus: dict, weights: dict, corpus: str) -> float:
    """A corpus's share of sampling: records times the corpus's weight, others weighing 1 (``train_motion_tokens.corpus_sample_weights``)."""
    total = sum(n * weights.get(c, 1.) for c, n in records_by_corpus.items())
    return records_by_corpus[corpus] * weights.get(corpus, 1.) / total


def counts_of(manifest: dict) -> dict:
    out = {}
    for r in manifest["records"]:
        if r.get("partition") == "train":
            out[r.get("corpus")] = out.get(r.get("corpus"), 0) + 1
    return out


def summarise(rows: list, *, boot: int = BOOT, seed: int = 0) -> dict:
    """Shares of clips and of seconds, with a bootstrap by video."""
    videos = sorted({r["video"] for r in rows})
    index = {v: [] for v in videos}
    for k, r in enumerate(rows):
        index[r["video"]].append(k)
    tests = dict(no_left=lambda r: not r["left_usable"], no_right=lambda r: not r["right_usable"], no_either_side=lambda r: not r["left_usable"] or not r["right_usable"], no_side_at_all=lambda r: not r["left_usable"] and not r["right_usable"],
                 under_half_frames_on_either_side=lambda r: min(r["left_share"], r["right_share"]) < .5, chest_never=lambda r: r["chest"] == 0.)
    flags = {k: np.array([bool(f(r)) for r in rows]) for k, f in tests.items()}
    seconds = np.array([r["seconds"] for r in rows])
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in tests}
    for _ in range(boot):
        pick = np.concatenate([index[videos[j]] for j in rng.integers(0, len(videos), len(videos))])
        for k, v in flags.items():
            draws[k].append(v[pick].mean())
    return dict(clips=len(rows), videos=len(videos), channels=len({r["channel"] for r in rows}), **{k: dict(share=round(float(v.mean()), 4), of_seconds=round(float(seconds[v].sum() / max(seconds.sum(), 1e-9)), 4), ci95_by_video=[round(float(np.percentile(draws[k], 2.5)), 4), round(float(np.percentile(draws[k], 97.5)), 4)]) for k, v in flags.items()},
                bimodal=dict(at_zero=round(float(np.mean([r["left_share"] == 0 and r["right_share"] == 0 for r in rows])), 4), at_least_90_both=round(float(np.mean([min(r["left_share"], r["right_share"]) >= .9 for r in rows])), 4)),
                body_columns_valid_on_clips_with_no_wrist=round(float(np.mean([r["body_columns"] for r in rows if not r["left_usable"] and not r["right_usable"]] or [0.])), 4),
                hands_on_clips_with_no_wrist=round(float(np.mean([max(r["left_hand"], r["right_hand"]) for r in rows if not r["left_usable"] and not r["right_usable"]] or [0.])), 4))


def args_of(run: dict) -> dict:
    return run["args"] if isinstance(run["args"], dict) else ast.literal_eval(run["args"])


def read(store, *, sample: int = SAMPLE) -> dict:
    from evaluate_text_to_motion import load_state
    from s2_lib import load_manifest_at
    mix, tok = load_manifest_at(store, *MIX), load_manifest_at(store, *TOKENIZER_MIX)
    youtube = sorted((r for r in mix["records"] if r.get("corpus") == YOUTUBE and r.get("partition") == "train" and (r.get("continuous") or {}).get("state_key")), key=lambda r: r["sign"])
    in_tokenizer = {r["continuous"]["state_key"] for r in tok["records"] if r.get("corpus") == YOUTUBE and r.get("partition") == "train"}
    picked = random.Random(SEED).sample(youtube, min(sample, len(youtube)))
    rows = []
    for k, r in enumerate(picked):
        rows.append(clip_row(load_state(store, r["continuous"]["state_key"], r["continuous"]["state_sha256"]), r))
        if (k + 1) % 250 == 0:
            print(f"{k + 1}/{len(picked)}", flush=True)
    prior_raw, tok_raw = store.get(PRIOR_RUN), store.get(TOKENIZER_RUN)
    prior_args, tok_args = args_of(json.loads(prior_raw)), args_of(json.loads(tok_raw))
    prior_weights = {k: float(v) for k, v in (w.split("=", 1) for w in prior_args["corpus_weight"])}
    tok_weights = {k: float(v) for k, v in (w.split("=", 1) for w in tok_args["corpus_weight"])}
    summary = summarise(rows)
    prior_share, tok_share = weight_share(counts_of(mix), prior_weights, YOUTUBE), weight_share(counts_of(tok), tok_weights, YOUTUBE)
    earlier = json.loads(store.get(EARLIER))
    round13 = None
    if store.exists(ROUND13):
        raw = json.loads(store.get(ROUND13))
        round13 = sum(1 for r in raw["records"] if r.get("corpus") == YOUTUBE and r.get("partition") == "train")
    of_mix = {k: dict(generator=round(prior_share * summary[k]["share"], 4), tokenizer=round(tok_share * summary[k]["share"], 4)) for k in ("no_left", "no_right", "no_either_side", "no_side_at_all", "under_half_frames_on_either_side")}
    return dict(kind="youtube-wrist-validity", report_only=True, cause="stored states: the body channel, chest included, is invalid and zero on these frames, before any reader; hands and face stay valid. Earlier evidence (plan, step A) ties it to framing: hips and often elbows out of frame, per video",
                sample=summary, youtube_records=len(youtube), youtube_records_also_in_tokenizer_mix=sum(r["continuous"]["state_key"] in in_tokenizer for r in youtube), generator=dict(run=PRIOR_RUN, run_sha256=hashlib.sha256(prior_raw).hexdigest(), mix=MIX[0], mix_sha256=MIX[1], corpus_weight=prior_args["corpus_weight"], part_valid_loss=True, youtube_share_of_sampling=round(prior_share, 4)),
                tokenizer=dict(run=TOKENIZER_RUN, run_sha256=hashlib.sha256(tok_raw).hexdigest(), mix=TOKENIZER_MIX[0], mix_sha256=TOKENIZER_MIX[1], corpus_weight=tok_args["corpus_weight"], youtube_share_of_sampling=round(tok_share, 4)), share_of_sampled_weight=of_mix,
                earlier_read=dict(key=EARLIER, clips=len(earlier), arm_at_zero=round(float(np.mean([r["arm"] == 0 for r in earlier])), 4), arm_under_half=round(float(np.mean([r["arm"] < .5 for r in earlier])), 4)), round13_youtube_train_records=round13,
                plan_decision="the plan (2026-10-27, round 13) dropped arm-untracked YouTube clips from generator training; round14b-mixed-v1 extends round10-mixed-v1 and holds all of them",
                not_measured=NOT_MEASURED)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", default="r2")
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.add_argument("--output-file", required=True)
    args = p.parse_args(argv)
    if Path(args.output_file).exists():
        raise SystemExit(f"{args.output_file} exists: refusing to overwrite")
    from train_text_to_motion import make_store
    out = read(make_store(args.store), sample=args.sample)
    data = json.dumps(out, indent=1, sort_keys=True).encode()
    with open(args.output_file, "xb") as stream:
        stream.write(data)
    print(args.output_file, hashlib.sha256(data).hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
