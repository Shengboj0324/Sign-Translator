"""Round 14 pre-check: does F12's motion tokenizer carry Sem-Lex signs? (declared 2026-10-28 13:59 UTC, before any
measurement; docs/NEXT_PHASE_PLAN.md, "Round 14 pre-check").

**Clips.** Sem-Lex **val** video states (``extract_semlex_videos.py``, ``states-video-v1/val/``), read through the
extractor's per-shard reports (each report's own SHA-256 is recorded; every state is checked against the SHA-256 its
report gives). A clip is eligible when its report has a state, its val metadata rows agree on one ``asllex`` label,
the id is not also listed in train (the manifest keeps those in train), and the label maps through its ASL-LEX code
to v2's ASL Citizen gloss (``validate_semlex_videos.code_mapping``, the mapping of the earlier video read). A
fixed-seed sample of ``--clips`` (300) is drawn spread over signers: signers in a seeded order, each signer's clips in
a seeded order, taken round-robin (``signer_spread_sample``), so every signer gives clips before any gives many.

**As the prior's dataset serves them.** Each clip becomes a manifest record (``continuous`` with the report's state
key, SHA-256 and available channels, the manifest rule of ``build_semlex_manifest.py``) and is read through
``TextMotionDataset`` with the template of F12's training (``layout_template(round10-mixed-v1,
how2sign-two-view-text-v1)``, the ``--conform-to-corpus`` of its run) at 24 fps: load and mask, ``conform``, resample.
The conform step is audited per channel (unchanged, absent-replaced, layout-renamed under a declared equivalence, or
refused) and the counts of changed and dropped clips are reported.

**Tokenizer.** The one in F12's step-60000 checkpoint (``tokenizer_state``, as ``exposure_diagnostic.py`` loads it),
checked tensor by tensor against the tokenizer checkpoint F12's run names (``tokenizer_key``, SHA-256 pinned by
``run.json``); the accepted decoder gains of that key and the generator's head smoothing (``make_post``).

**Forms,** each read by recogniser v2 at 24 fps:
- ``real``: the conformed state as served;
- ``recon_full``: its tokenizer codes decoded as a generation (every trained channel valid, gains, head smoothing);
- ``recon_masked``: ``recon_full`` with each channel's validity copied from ``real`` (instrument parity);
- ``blank``: ``real`` with every channel removed (only the duration left);
- ``real_unconformed`` (check only): the stored state resampled to 24 fps without the dataset path.

**Report.** Top-1 and top-5 per form with bootstrap 95% intervals over clips and over signers; paired per-clip
differences ``recon_masked - real`` and ``recon_full - real`` (top-5 and top-1) with 95% bootstrap intervals over clips
(and over signers, for information); per-signer values. **Verdict** (the declared bar): ``recon_masked - real`` top-5,
95% lower bound over clips above -0.15 is a pass.

**Secondary (not the verdict).** The same forms on the sampled clips that pass ``build_semlex_manifest.py``'s filters
(frame rate, hand tracking) cut to its sign window (``trim_window``), the form a Sem-Lex manifest would serve.
**Reference.** Per-channel reconstruction error of the tokenizer alone (no gains, no smoothing): ``native_errors``, the
geodesic angle in degrees (rotation channels) or mean absolute difference (face, blink) on the cells the real clip
observes, and the established unit-space metric of ``tokenizer_domain_check.channel_errors`` (which also counts the
neutral targets of unobserved joints on frames where the channel is valid), on the Sem-Lex sample and on
``--reference-clips`` validation clips each of ASLLVD,
RIT and How2Sign from round10-mixed-v1 through the same dataset path; for ASLLVD and RIT clips whose text is a v2 gloss,
the same recogniser forms and paired differences.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import replace
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

BASE = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
REPORTS_PREFIX = f"{BASE}/states-video-v1/reports/"
F12_RUN = "training-runs/tokens-rvq-d0b-prior-r12-f12"
F12_CHECKPOINT = f"{F12_RUN}/checkpoints/step-00060000.pt"
TEMPLATE_CORPUS = "how2sign-two-view-text-v1"
REFERENCE_CORPORA = ("asllvd-text-v1", "rit-text-r6", "how2sign-two-view-text-v1")
STATE_STATUSES = ("converted", "resumed", "already_present")
BAR = -0.15
OUTPUT_KEY = f"{BASE}/adapter-dev/2026-10-28-round14-precheck-tokenizer.json"


# ------------------------------------------------------------------------------------------------- pure helpers

def signer_spread_sample(items, n: int, seed: int) -> list:
    """``n`` of ``items`` ((key, signer) pairs) spread over signers: signers in a seeded order, each signer's keys in a
    seeded order, taken round-robin. Deterministic for a given set of items and seed (input order does not matter)."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for key, signer in sorted(set(items)):
        by[signer].append(key)
    signers = sorted(by)
    order = [signers[i] for i in rng.permutation(len(signers))]
    queues = {s: [by[s][i] for i in rng.permutation(len(by[s]))] for s in order}
    out = []
    while len(out) < n and any(queues.values()):
        for s in order:
            if queues[s] and len(out) < n:
                out.append((queues[s].pop(0), s))
    return out


def bootstrap_mean(values, groups=None, draws: int = 2000, seed: int = 0) -> list:
    """95% percentile interval of the mean of ``values`` over resampled clips, or over resampled groups (every clip of
    a drawn group, a cluster bootstrap) when ``groups`` is given (``semlex_recognition_shift.bootstrap``)."""
    from semlex_recognition_shift import bootstrap
    return bootstrap(values, groups, draws=draws, seed=seed)


def paired_difference(ranks_a, ranks_b, k: int, groups=None) -> dict:
    """Per-clip paired difference of top-``k`` hits, ``a - b``; the two rank arrays must be aligned clip by clip."""
    a, b = np.asarray(ranks_a), np.asarray(ranks_b)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired forms must hold the same, non-empty set of clips")
    diff = (a < k).astype(np.float64) - (b < k)
    out = dict(difference=round(float(diff.mean()), 4), ci_clips=bootstrap_mean(diff),
               wins=int((diff > 0).sum()), losses=int((diff < 0).sum()), clips=int(len(diff)))
    if groups is not None and len(set(groups)) >= 3:
        out["ci_signers"] = bootstrap_mean(diff, groups)
    return out


def verdict(lower_bound: float, bar: float = BAR) -> str:
    """The declared rule: the 95% lower bound of recon_masked - real (top-5, over clips) must lie above ``bar``."""
    return "pass" if lower_bound > bar else "fail"


def form_scores(ranks: np.ndarray, groups) -> dict:
    """Top-1 and top-5 with bootstrap 95% intervals over clips and over signers."""
    out = {}
    for k in (1, 5):
        hits = (np.asarray(ranks) < k).astype(np.float64)
        out[f"top{k}"] = round(float(hits.mean()), 4)
        out[f"top{k}_ci_clips"] = bootstrap_mean(hits)
        out[f"top{k}_ci_signers"] = bootstrap_mean(hits, groups) if len(set(groups)) >= 3 else None
    return out


def conform_kind(channel, reference) -> str:
    """How ``layout_adapter.conform`` treats one channel against the template's: ``same``, ``absent`` (no valid frame,
    replaced by the template's empty layout), ``equivalent`` (layout renamed under a declared equivalence) or
    ``refused`` (conform raises)."""
    from signtranslator.data.layout_adapter import EQUIVALENT
    c, ref = channel, reference
    if (c.labels, c.coordinate_frame, c.units, c.convention) == (ref.labels, ref.coordinate_frame, ref.units,
                                                                 ref.convention):
        return "same"
    if not c.valid.any():
        return "absent"
    if (c.labels == ref.labels and c.units == ref.units and c.values.shape[1:] == ref.values.shape[1:]
            and (c.coordinate_frame, c.convention) in EQUIVALENT.get((ref.coordinate_frame, ref.convention), set())):
        return "equivalent"
    return "refused"


def conform_audit(state, template) -> dict:
    """Per template channel its ``conform_kind`` (``missing`` when the state lacks it), and the state's channels the
    template does not carry (dropped by conform)."""
    kinds = {name: (conform_kind(state.channels[name], ref) if name in state.channels else "missing")
             for name, ref in template.channels.items()}
    return dict(kinds=kinds, extra=sorted(set(state.channels) - set(template.channels)))


def per_signer(ranks: dict, groups, forms) -> dict:
    groups = np.asarray(groups)
    out = {}
    for s in sorted(set(groups.tolist())):
        m = groups == s
        row = dict(clips=int(m.sum()))
        for f in forms:
            row[f"{f}_top5"] = round(float((ranks[f][m] < 5).mean()), 4)
        row["recon_masked_minus_real_top5"] = round(float((ranks["recon_masked"][m] < 5).mean()
                                                          - (ranks["real"][m] < 5).mean()), 4)
        out[s] = row
    return out


# ------------------------------------------------------------------------------------------------------ the job

class LocalCache:
    """Large checkpoints fetched once into ``directory`` (named by key); the SHA-256 of what was read is returned."""

    def __init__(self, directory: Path, store):
        self.directory, self.store = Path(directory), store
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, expected_sha256: str | None = None) -> tuple[bytes, str]:
        path = self.directory / key.replace("/", "__")
        data = path.read_bytes() if path.is_file() else None
        digest = hashlib.sha256(data).hexdigest() if data is not None else None
        if data is None or (expected_sha256 and digest != expected_sha256):
            data = self.store.get(key)
            digest = hashlib.sha256(data).hexdigest()
            path.write_bytes(data)
        if expected_sha256 and digest != expected_sha256:
            raise SystemExit(f"{key}: SHA-256 {digest} is not the pinned {expected_sha256}")
        return data, digest


def load_reports(store) -> tuple[dict, list]:
    """(video_id -> report result with a state, [report key and SHA-256]) over every val shard report."""
    keys = []
    paginator = store.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=store.bucket, Prefix=REPORTS_PREFIX):
        keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].rsplit("/", 1)[-1].startswith("val-shard-")]
    results, meta = {}, []
    for key in sorted(keys):
        payload = store.get(key)
        report = json.loads(payload)
        if report.get("split") != "val" or report.get("smoke"):
            continue
        meta.append(dict(key=key, sha256=hashlib.sha256(payload).hexdigest(), shard=report["shard"],
                         shards=report["shards"], status_counts=report["status_counts"]))
        for r in report["results"]:
            if r["status"] in STATE_STATUSES and r.get("state_sha256"):
                results[r["video_id"]] = r                      # sorted keys: a later report of a shard wins
    return results, meta


def tokenizer_of(args, store, cache):
    """(tokenizer, motion config, decoder gains, provenance) of F12's checkpoint."""
    import torch
    from signtranslator.generation.motion_rvq import MotionRVQ, RVQPriorConfig
    from evaluate_text_to_motion import accepted_decoder_gains
    run = json.loads(store.get(f"{F12_RUN}/run.json"))["args"]
    data, ck_sha = cache.get(args.checkpoint_key)
    ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    del data
    if ck.get("tokenizer_key") != run["tokenizer_key"]:
        raise SystemExit(f"F12's checkpoint names tokenizer {ck.get('tokenizer_key')}, its run {run['tokenizer_key']}")
    config = RVQPriorConfig.from_dict(ck["config"]).tokenizer
    tokenizer = MotionRVQ(config)
    tokenizer.load_state_dict(ck["tokenizer_state"])
    tokenizer.eval()
    own, tok_sha = cache.get(run["tokenizer_key"], run["tokenizer_sha256"])
    saved = torch.load(io.BytesIO(own), map_location="cpu", weights_only=False)
    weights = saved.get("ema") or saved["model"]
    state = tokenizer.state_dict()
    identical = set(weights) == set(state) and all(torch.equal(weights[k].cpu(), state[k]) for k in state)
    del saved, weights, ck
    gains = accepted_decoder_gains(store, run["tokenizer_key"])
    return tokenizer, config.motion, gains, dict(checkpoint_key=args.checkpoint_key, checkpoint_sha256=ck_sha,
                                                 tokenizer_key=run["tokenizer_key"], tokenizer_sha256=tok_sha,
                                                 tokenizer_identical_to_checkpoint=bool(identical),
                                                 decoder_gains=gains)


ROTATION_CHANNELS = ("body", "left_hand", "right_hand", "head_rotation")
SCALAR_CHANNELS = ("face", "blink")


def native_errors(recon, real, sums: dict, counts: dict) -> None:
    """Accumulate, per channel, the tokenizer's error on the real clip's valid (frame, element) cells: the geodesic
    angle in degrees for 6D rotation channels, the mean absolute difference for coefficient channels. Cells the real
    clip does not observe (untracked arms, legs, absent channels) are not counted."""
    import torch
    from signtranslator.pose.rotations import geodesic_distance, rotation_6d_to_matrix
    for name in ROTATION_CHANNELS + SCALAR_CHANNELS:
        if name not in real.channels or name not in recon.channels:
            continue
        valid = real.channels[name].valid & recon.channels[name].valid
        if not valid.any():
            continue
        a, b = real.channels[name].values[valid], recon.channels[name].values[valid]
        if name in ROTATION_CHANNELS:
            ra, rb = (rotation_6d_to_matrix(torch.as_tensor(np.asarray(x, dtype=np.float64))) for x in (a, b))
            err = np.degrees(geodesic_distance(ra, rb).numpy())
        else:
            err = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).reshape(len(a), -1).mean(-1)
        sums[name] = sums.get(name, 0.) + float(err.sum())
        counts[name] = counts.get(name, 0) + int(err.size)
        if name == "body" and valid.ndim == 2 and getattr(real.channels[name], "labels", None):
            joint = np.nonzero(valid)[1]                           # body per joint, to locate a body error
            for j, label in enumerate(real.channels[name].labels):
                if (joint == j).any():
                    key = f"body/{label}"
                    sums[key] = sums.get(key, 0.) + float(err[joint == j].sum())
                    counts[key] = counts.get(key, 0) + int((joint == j).sum())


def read_forms(tokenizer, motion_config, post, served, raw=None) -> tuple[dict, dict]:
    """(per form the list of states, aligned with ``served``; per channel the tokenizer's native error)."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    from evaluate_text_to_motion import at_rate, blanked
    from exposure_diagnostic import make_decode, masked_like
    decode = make_decode(tokenizer, motion_config, post)
    plain = make_decode(tokenizer, motion_config, lambda s: s)        # the tokenizer alone: no gains, no smoothing
    forms = collections.defaultdict(list)
    sums, counts = {}, {}
    with torch.no_grad():
        for n, (text, real) in enumerate(served):
            motion = collate_multichannel([real])
            unit = diffusion_targets(motion_config, motion)
            _, codes, _ = tokenizer.quantize(tokenizer.encode(unit))
            frames = int(motion.frame_valid.shape[1])
            if frames != len(real.timestamps):
                raise RuntimeError("collated frames differ from the state's")
            recon = decode(codes, frames, text)
            native_errors(plain(codes, frames, text), real, sums, counts)
            forms["real"].append(real)
            forms["recon_full"].append(recon)
            forms["recon_masked"].append(masked_like(recon, real))
            if raw is not None:
                forms["real_unconformed"].append(at_rate(raw[n], 24.))
    forms["blank"] = blanked(forms["real"])
    errors = {k: dict(mean=round(sums[k] / counts[k], 4), cells=counts[k],
                      unit="degrees" if k.split("/")[0] in ROTATION_CHANNELS else "absolute") for k in sorted(sums)}
    return forms, errors


def score_forms(recogniser, forms: dict, labels, groups) -> tuple[dict, dict]:
    from semlex_recognition_shift import ranks_of
    ranks = {name: ranks_of(recogniser, states, labels) for name, states in forms.items()}
    out = dict(scores={name: form_scores(r, groups) for name, r in ranks.items()},
               differences={f"{a}_minus_real": dict(top5=paired_difference(ranks[a], ranks["real"], 5, groups),
                                                    top1=paired_difference(ranks[a], ranks["real"], 1, groups))
                            for a in ("recon_masked", "recon_full", "blank") if a in ranks})
    return out, ranks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--clips", type=int, default=300)
    p.add_argument("--reference-clips", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-key", default=F12_CHECKPOINT)
    p.add_argument("--recognizer-run", default="sign-recognizer-v2")
    p.add_argument("--cache-dir", required=True, help="local directory for checkpoints and verified states")
    p.add_argument("--output", required=True, help="local JSON report")
    p.add_argument("--output-key", help=f"also write the report once to the store (declared: {OUTPUT_KEY})")
    args = p.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from signtranslator.data.text_motion_dataset import R2Store, TextMotionDataset
    from signtranslator.data_engineering.corpus_manifest import ContentCache, load_manifest, load_sample
    from convert_semlex_poses import METADATA_KEY, METADATA_SHA256, load_metadata
    from exposure_diagnostic import make_post
    from train_motion_tokens import layout_template
    from tokenizer_domain_check import channel_errors
    from validate_semlex_videos import code_mapping
    from validate_sign_recognizer import load_model
    store = R2Store(os.environ)
    if args.output_key and store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: the report is written once")
    cache = LocalCache(Path(args.cache_dir) / "checkpoints", store)
    content = ContentCache(Path(args.cache_dir) / "content", 16 * 2 ** 30)
    run = json.loads(store.get(f"{F12_RUN}/run.json"))["args"]

    # --- the Sem-Lex sample
    results, report_meta = load_reports(store)
    metadata = load_metadata(store, METADATA_KEY, METADATA_SHA256)
    gloss_of = code_mapping(store)
    recogniser, glosses, best = load_model(store, args.recognizer_run, "cpu")
    index = {g: i for i, g in enumerate(glosses)}
    skipped = collections.Counter()
    eligible = {}
    for video_id, r in sorted(results.items()):
        rows = metadata.get(video_id, [])
        val_rows = [row for row in rows if row["split"] == "val"]
        if any(row["split"] == "train" for row in rows):
            skipped["also listed in train (the manifest keeps it in train)"] += 1
            continue
        if len({(row["label_type"], row["label"]) for row in val_rows}) != 1:
            skipped["no single val metadata label"] += 1
            continue
        row = val_rows[0]
        gloss = gloss_of(row["label"]) if row["label_type"] == "asllex" else None
        if gloss is None or gloss not in index:
            skipped["not in v2's vocabulary"] += 1
            continue
        eligible[video_id] = dict(result=r, row=row, gloss=gloss)
    sample = signer_spread_sample([(v, e["row"]["signer_id"]) for v, e in eligible.items()], args.clips, args.seed)
    print(f"{len(results)} val states, {len(eligible)} eligible, {len(sample)} sampled over "
          f"{len({s for _, s in sample})} of {len({e['row']['signer_id'] for e in eligible.values()})} signers",
          flush=True)

    # --- the prior's dataset path
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(run["manifest_key"]) + b"\n")
        mix = load_manifest(path, expected_sha256=run["manifest_sha256"])
    if run["conform_to_corpus"] != TEMPLATE_CORPUS:
        raise SystemExit(f"F12 conformed to {run['conform_to_corpus']}")
    template = layout_template(mix, TEMPLATE_CORPUS, content, store.get)
    records = []
    for video_id, _ in sample:
        r = eligible[video_id]["result"]
        records.append(dict(sign=f"semlex-{video_id}", corpus="semlex-val-precheck", partition="val",
                            text=eligible[video_id]["gloss"].lower(),
                            continuous=dict(status="converted", state_key=r["state_key"], state_sha256=r["state_sha256"],
                                            available_channels=sorted(r["available_channels"]),
                                            seconds=round(r["frames"] / max(r["fps"], 1e-9), 4))))
    data = TextMotionDataset(dict(records=records), partition="val", cache=content, fetch=store.get,
                             max_seconds=run["max_seconds"], fps=24., template=template)
    if len(data.records) != len(records):
        print(f"dataset filter kept {len(data.records)} of {len(records)} records", flush=True)
    served, raw, kept, audit_rows, drops = [], [], [], [], collections.Counter()
    for k, record in enumerate(data.records):
        video_id = record["sign"][len("semlex-"):]
        state = load_sample(record, "continuous", cache=content, fetch=store.get)
        audit = conform_audit(state, template)
        try:
            item = data[k]
        except (ValueError, KeyError) as error:
            drops[f"{type(error).__name__}: {str(error)[:80]}"] += 1
            continue
        served.append(item)
        raw.append(state)
        kept.append(video_id)
        audit_rows.append(audit)
    kinds = collections.Counter((n, kind) for a in audit_rows for n, kind in a["kinds"].items())
    changed = sum(any(kind != "same" for kind in a["kinds"].values()) or bool(a["extra"]) for a in audit_rows)
    conform_report = dict(clips_in=len(data.records), clips_kept=len(kept), clips_dropped=sum(drops.values()),
                          drop_reasons=dict(drops), clips_changed=changed,
                          channel_kinds={f"{n}:{kind}": c for (n, kind), c in sorted(kinds.items())},
                          extra_channels_dropped=dict(collections.Counter(x for a in audit_rows for x in a["extra"])),
                          template=dict(corpus=TEMPLATE_CORPUS, sample_id=template.sample_id))
    print("conform:", json.dumps(conform_report), flush=True)

    tokenizer, motion_config, gains, provenance = tokenizer_of(args, store, cache)
    print("tokenizer:", json.dumps(provenance), flush=True)
    post = make_post(gains)
    labels = [index[eligible[v]["gloss"]] for v in kept]
    signers = [eligible[v]["row"]["signer_id"] for v in kept]
    forms, semlex_native = read_forms(tokenizer, motion_config, post, served, raw)
    primary, ranks = score_forms(recogniser, forms, labels, signers)
    lower = primary["differences"]["recon_masked_minus_real"]["top5"]["ci_clips"][0]
    print("primary:", json.dumps(primary), flush=True)
    clip_rows = [dict(video_id=v, signer=s, gloss=eligible[v]["gloss"], frames=len(served[n][1].timestamps),
                      **{f"{f}_rank": int(ranks[f][n]) for f in ranks}) for n, (v, s) in enumerate(zip(kept, signers))]

    # --- secondary: the manifest's filters and sign window
    from build_semlex_manifest import trim_window
    from signtranslator.data_engineering.corpus_manifest import window_state
    from signtranslator.data.layout_adapter import conform
    from evaluate_text_to_motion import at_rate
    trimmed, t_labels, t_signers, t_counts = [], [], [], collections.Counter()
    for n, v in enumerate(kept):
        state, row = raw[n], eligible[v]["row"]
        fps = len(state.timestamps) / (float(row["duration"]) / 1000.)
        hand = (state.channels["left_hand"].valid.all(1) | state.channels["right_hand"].valid.all(1)).mean()
        if fps < 20.:
            t_counts["drop:fps_below_min"] += 1
            continue
        if hand < .3:
            t_counts["drop:hand_tracked_below_min"] += 1
            continue
        window, reason = trim_window(state, .1, .25, .3, .5)
        t_counts[f"trim:{reason}"] += 1
        if window is None:
            continue
        trimmed.append((served[n][0], at_rate(conform(window_state(state, *window), template), 24.)))
        t_labels.append(labels[n])
        t_signers.append(signers[n])
    secondary = dict(counts=dict(t_counts), clips=len(trimmed))
    if trimmed:
        t_forms, t_native = read_forms(tokenizer, motion_config, post, trimmed)
        t_scores, _ = score_forms(recogniser, t_forms, t_labels, t_signers)
        secondary.update(t_scores, native_errors=t_native)
    print("secondary (trimmed):", json.dumps(secondary), flush=True)

    # --- reference: per-channel reconstruction error, and the same read on existing corpora. Two error measures:
    # ``native_errors`` (degrees / absolute, on the real clip's valid cells only) and ``unit_errors``, the
    # established tokenizer_domain_check metric (unit space, every cell of a frame where any element of the channel
    # is valid, so unobserved joints' neutral targets count too).
    def unit_errors(states):
        errors, scale = channel_errors(tokenizer, states)
        return dict(clips=len(states), unit_errors={k: round(v, 5) for k, v in sorted(errors.items())},
                    unit_scale={k: round(v, 5) for k, v in sorted(scale.items())})
    reference = dict(semlex_val=dict(native_errors=semlex_native, **unit_errors([s for _, s in served])))
    mix_data = TextMotionDataset(mix, partition="val", cache=content, fetch=store.get,
                                 max_seconds=run["max_seconds"], fps=24., template=template)
    rng = np.random.default_rng(args.seed)
    for corpus in REFERENCE_CORPORA:
        rows = [k for k, r in enumerate(mix_data.records) if r.get("corpus") == corpus]
        if corpus != TEMPLATE_CORPUS:      # clips the recogniser can grade (text a v2 gloss), as exposure_diagnostic
            rows = [k for k in rows if mix_data.records[k]["text"].upper().replace(" ", "") in index]
        chosen = sorted(rng.choice(rows, min(args.reference_clips, len(rows)), replace=False).tolist())
        items = [mix_data[k] for k in chosen]
        r_forms, r_native = read_forms(tokenizer, motion_config, post, items)
        entry = dict(eligible=len(rows), native_errors=r_native, **unit_errors([s for _, s in items]))
        if corpus != TEMPLATE_CORPUS:
            groups = [mix_data.records[k]["performer_sha256"][:12] for k in chosen]
            y = [index[mix_data.records[k]["text"].upper().replace(" ", "")] for k in chosen]
            scored, _ = score_forms(recogniser, r_forms, y, groups)
            entry.update(signers=len(set(groups)), **scored)
        reference[corpus] = entry
        print(f"reference {corpus}:", json.dumps(entry), flush=True)
    semlex = reference["semlex_val"]
    reference["semlex_native_error_ratio_to"] = {
        corpus: {k: round(v["mean"] / max(reference[corpus]["native_errors"][k]["mean"], 1e-9), 3)
                 for k, v in semlex["native_errors"].items() if k in reference[corpus]["native_errors"]}
        for corpus in REFERENCE_CORPORA}
    reference["semlex_unit_error_ratio_to"] = {
        corpus: {k: round(v / max(reference[corpus]["unit_errors"][k], 1e-9), 3)
                 for k, v in semlex["unit_errors"].items() if k in reference[corpus]["unit_errors"]}
        for corpus in REFERENCE_CORPORA}

    out = dict(
        declared="docs/NEXT_PHASE_PLAN.md, Round 14 pre-check (2026-10-28 13:59 UTC)",
        bar=dict(measure="recon_masked - real, top-5, 95% bootstrap lower bound over clips", above=BAR),
        verdict=verdict(lower), recon_masked_minus_real_top5_lower=lower,
        recognizer=dict(run=args.recognizer_run, checkpoint=best["checkpoint_key"], fps=24.),
        tokenizer=provenance, f12_manifest=dict(key=run["manifest_key"], sha256=run["manifest_sha256"]),
        reports=report_meta, mapping="ASL-LEX code -> ASL Citizen gloss (validate_semlex_videos.code_mapping)",
        sampling=dict(seed=args.seed, requested=args.clips, val_states=len(results), eligible=len(eligible),
                      eligible_signers=len({e["row"]["signer_id"] for e in eligible.values()}),
                      sampled=len(sample), skipped=dict(skipped), rule="signer_spread_sample (round-robin)"),
        conform=conform_report, clips=len(kept), signers=len(set(signers)), glosses=len(set(labels)),
        **primary, per_signer=per_signer(ranks, signers, ("real", "recon_full", "recon_masked", "blank")),
        secondary_trimmed=secondary, reference=reference, per_clip=clip_rows)
    payload = json.dumps(out, indent=1, sort_keys=True).encode()
    Path(args.output).write_bytes(payload)
    if args.output_key:
        store.put(args.output_key, payload)
    print(json.dumps(dict(verdict=out["verdict"], lower=lower, sha256=hashlib.sha256(payload).hexdigest())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
