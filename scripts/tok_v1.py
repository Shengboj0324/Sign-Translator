"""V1: does neutral fill limit generated signing, and how much would a tokenizer keep without it? Report only; nothing trains; this code declares nothing and holds no rule.

V0 read neutral fill as the main cost of the all-valid path on v1 (``identity`` against ``v0_nearest_fill``). The fill is what ``text_to_motion.diffusion_targets`` trains on, so a model may have learned
to emit it. D20 only, instrument v1, through ``tok_probes`` and ``tok_v0``'s reads, canary and digest, write-once under ``tok-probes-v1/``. ``--smoke-train N`` reads c02 train clips and says so.

**(a) The true cap.** ``rt_nearest``: real with nearest-valid fill, all valid, through d2's encoder, codes and decoder, no gains. ``no_quant_nearest``: the same with no codes. Compared with
``v0_nearest_fill`` (the same input, no network) and ``round_trip`` (neutral-filled, as d2 was trained, read as is).

**(b) Neutral hands, motion only, no reader** (``neutral``). A hand-frame's distance to the rest pose ``diffusion_targets`` fills (identity rotations) is the mean geodesic angle over its tracked joints. Frames
under a threshold count as neutral: every threshold in ``THRESHOLDS`` is reported, ``FLAG_DEGREES`` is the one (c) uses. Populations: real D20 clips on tracked frames only; real with neutral fill
(``identity``); nearest-fill; a9's and a3-gen top 10's cached word clips (best of 16; a3's new entries apart); composed a9@predicted and a3-gen top 10. ``word_alignment`` asks whether a word's neutral frames fall
where its training clips were untracked, on 20 time bins per word. ``corpus_neutral`` gives each corpus's neutral-filled hand frames in W14b's and d2's training mixes, on a sample.

**(c) A sensitivity, never a score.** ``a9p``: stored a9@predicted, composed again by ``round15_compose_eval --export-only``. ``a9p_replaced``: its frames flagged neutral replaced by that hand's nearest non-neutral frame.
This modifies generated output after the fact. It is not a generator's output, it cannot serve M1 or selection, and a rise says only that v1 reads these frames worse. ``composed-check`` compares ``a9p``'s hypotheses
with the stored a10 report.

``reads`` in the merged report gives each difference and share with a bootstrap interval, on likelihood minus blank first, then BLEU. ``scripts/tok_v1_rule.py`` reads it; the job never calls that.

The job is ``runpod/fleet/tok_v1.sh``.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "tok-v1"
JOB_FILE = "runpod/fleet/tok_v1.sh"
FPS = L.FPS
HANDS = ("left_hand", "right_hand")
PREFIX = f"{L.O}/tok-probes-v1"
CAP_FORMS = ("real", "blank", "identity", "round_trip", "v0_nearest_fill", "rt_nearest", "no_quant_nearest")
SENSITIVITY_FORMS = ("a9p", "a9p_replaced")
FORMS = CAP_FORMS + SENSITIVITY_FORMS
ANCHORS = ("real", "blank", "identity", "round_trip", "v0_nearest_fill")
COMPOSED_FORM = "core_crossfade@predicted"
A9_INDEX = f"{PREFIX}/composed/a9/index.json"
A3_INDEX = f"{PREFIX}/composed/a3-top10/index.json"
A9_CACHE, A9_CACHE_SHA = f"{L.O}/round20/phase-a/cache-w14b-nogains-r10/index.json", "507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b"
A3_CACHE, A3_CACHE_SHA = f"{L.O}/round20/comp-a/cache-a3-top30-nogains/index.json", "2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c"
A10_REPORT, A10_SHA = f"{L.O}/round20/phase-a/a10-predicted-nogains-r10.json", "1e912c9a3cac0f8c581e1542c53de41c8152196d371ee61e450e691836af537d"
TOKENS_FILE, TOKENS_SHA = "reports/comp_a3_tokens.json", "c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776"
STORED_FORMS = {"identity": (f"{L.O}/tok-probes/forms/form-identity.json", "24958febc663ab00f984f8f24bea791e3e5a0904a191311e49a213e4f58e0356"),
                "round_trip": (f"{L.O}/tok-probes/forms/form-round_trip.json", "18e6b17524db4df91b9770d197d7dd45d1a7314cd3685bd2008e5342243da4fb"),
                "v0_nearest_fill": (f"{L.O}/tok-probes-v0/forms/form-v0_nearest_fill.json", "e1bfd0ac62d72593b58370e48477a8db9bb7c85c6cc54b57790db64d7637939f")}
THRESHOLDS = (1., 2., 5., 10., 20.)      # degrees: a measure at each; none decides here
FLAG_DEGREES = 5.                        # what (c) replaces; the sensitivity to it is in the report
BINS = 20
WORD_CLIPS, CORPUS_CLIPS, SAMPLE_SEED = 8, 300, 20261103
BOOT = 2000
COMPARATORS = dict(identity=("real",), round_trip=("identity",), v0_nearest_fill=("identity", "real"), rt_nearest=("v0_nearest_fill", "round_trip"), no_quant_nearest=("v0_nearest_fill", "rt_nearest"),
                   a9p_replaced=("a9p",))
NOT_DECIDED = "report only: the declaration's reading is its own; no rule, bar or verdict is computed here"
DISCLOSURES = dict(
    sensitivity="a9p_replaced modifies generated output after the fact. It is a sensitivity of the instrument to these frames: never a score, never M1, never selection.",
    neutral_distance="Neutral means near the identity rotations of the finger joints. Wrist rotation and arm joints are in the body channel, not measured here. A real relaxed hand may sit near it, which is why real's tracked frames are the baseline.",
    alignment="A word's untracked share comes from at most 8 of its training clips, by time-normalised position, with no alignment of the sign inside the clip.",
    cap="rt_nearest sends d2 input it was not trained on (no neutral stretches), through a decoder trained on neutral-filled targets, so it reads what d2 keeps as it stands, not what a retrained d2 would.",
    a9p="Composed again, not stored: composed states were never saved. composed-check compares its hypotheses with the stored a10 read.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- neutral hand frames: pure measures ---------------------------------------------------------------------------------------

def hand_distance(channel) -> np.ndarray:
    """(T,) mean geodesic degrees to the identity rotation over the frame's valid joints; NaN on a frame with none."""
    from signtranslator.generation.motion_metrics import geodesic_degrees
    T, J = channel.values.shape[:2]
    identity = np.broadcast_to(np.array([1., 0., 0., 0., 1., 0.]), (T, J, 6))
    if not channel.valid.any():
        return np.full(T, np.nan)
    angle = np.where(channel.valid, geodesic_degrees(np.where(channel.valid[..., None], channel.values, identity), identity), 0.)
    count = channel.valid.sum(1)
    return np.where(count > 0, angle.sum(1) / np.maximum(count, 1), np.nan)


def neutral_shares(state, *, tracked_only_by=None) -> dict:
    """Per hand and threshold, this clip's share of frames counted neutral, among frames with a distance. ``tracked_only_by``: only the frames that state tracked. None where no frame counts."""
    out = {}
    for hand in HANDS:
        d = hand_distance(state.channels[hand])
        if tracked_only_by is not None:
            d = np.where(tracked_only_by.channels[hand].valid.any(1), d, np.nan)
        seen = np.isfinite(d)
        out[hand] = {str(t): (round(float((d[seen] < t).mean()), 5) if seen.any() else None) for t in THRESHOLDS}
    return out


def replaced_neutral(state, degrees: float = FLAG_DEGREES) -> tuple:
    """(``state`` with each hand's neutral frames holding the hand's nearest non-neutral frame, counts). A hand with no non-neutral frame stays and is counted."""
    from dataclasses import replace
    channels, flagged, frames, untouched = dict(state.channels), 0, 0, 0
    for hand in HANDS:
        c = state.channels[hand]
        d = hand_distance(c)
        neutral = np.isfinite(d) & (d < degrees)
        frames += len(d)
        if not neutral.any():
            continue
        if neutral.all():
            untouched += 1
            continue
        index = V.nearest_index(~neutral)
        channels[hand] = replace(c, values=c.values[index].astype(c.values.dtype), valid=c.valid[index], observed=c.observed[index], confidence=c.confidence[index])
        flagged += int(neutral.sum())
    return replace(state, channels=channels), dict(flagged_frames=flagged, hand_frames=frames, hands_all_neutral=untouched)


def time_bins(values: np.ndarray, bins: int = BINS) -> np.ndarray:
    """(bins,) means over time-normalised position; NaN in an empty bin."""
    T = len(values)
    index = np.minimum((np.arange(T) * bins / max(T, 1)).astype(int), bins - 1)
    out = np.full(bins, np.nan)
    for b in range(bins):
        hit = index == b
        seen = np.isfinite(values) & hit
        if seen.any():
            out[b] = float(values[seen].mean())
    return out


def untracked_profile(clips, hand: str) -> np.ndarray:
    """(bins,) over training clips: the share of a hand's joints untracked at each time-normalised position."""
    rows = [time_bins((~c.channels[hand].valid).mean(1).astype(np.float64)) for c in clips]
    if not rows:
        return np.full(BINS, np.nan)
    stacked = np.stack(rows)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(stacked).sum(0)
        return np.where(counts > 0, np.nansum(stacked, 0) / np.maximum(counts, 1), np.nan)


def word_alignment(rows: list[dict]) -> dict:
    """Per threshold: P(neutral | training mostly untracked) − P(neutral | training mostly tracked), over (entry, hand, bin), with a bootstrap over entries. ``rows``: one per entry,
    {entry, generated: {hand: (bins,) distance}, untracked: {hand: (bins,) share}}."""
    if not rows:
        return dict(entries=0)
    out = dict(entries=len(rows), bins=BINS, thresholds={})
    rng = np.random.default_rng(SAMPLE_SEED)
    draws = rng.integers(0, len(rows), (BOOT, len(rows)))
    for t in THRESHOLDS:
        counts = np.zeros((len(rows), 4))                    # neutral|untracked, untracked, neutral|tracked, tracked
        for i, row in enumerate(rows):
            for hand in HANDS:
                d, u = np.asarray(row["generated"][hand]), np.asarray(row["untracked"][hand])
                seen = np.isfinite(d) & np.isfinite(u)
                gone = seen & (u >= .5)
                kept = seen & (u < .5)
                counts[i] += [(d[gone] < t).sum(), gone.sum(), (d[kept] < t).sum(), kept.sum()]

        def lift(total):
            with np.errstate(divide="ignore", invalid="ignore"):
                return total[..., 0] / total[..., 1], total[..., 2] / total[..., 3]
        a, b = lift(counts.sum(0))
        s = counts[draws].sum(1)
        ra, rb = lift(s)
        out["thresholds"][str(t)] = dict(neutral_where_untracked=None if not np.isfinite(a) else round(float(a), 5), neutral_where_tracked=None if not np.isfinite(b) else round(float(b), 5),
                                         difference=None if not (np.isfinite(a) and np.isfinite(b)) else round(float(a - b), 5), ci95=P._quantiles(ra - rb),
                                         bins_untracked=int(counts[:, 1].sum()), bins_tracked=int(counts[:, 3].sum()))
    return out


def population(rows: list[dict]) -> dict:
    """Per hand and threshold over a population's clips: the mean share with a bootstrap, and over both hands."""
    out = dict(clips=len(rows))
    for hand in HANDS + ("both",):
        out[hand] = {}
        for t in THRESHOLDS:
            values = []
            for r in rows:
                parts = [r[h][str(t)] for h in (HANDS if hand == "both" else (hand,)) if r[h][str(t)] is not None]
                if parts:
                    values.append(float(np.mean(parts)))
            out[hand][str(t)] = M.bootstrap_mean(values, n=BOOT)
    return out


def paired_population(a: list[dict], b: list[dict]) -> dict:
    """Per threshold, a − b over clips both measure, both hands pooled: a composed form against real's tracked frames, clip by clip."""
    out = {}
    for t in THRESHOLDS:
        pairs = []
        for x, y in zip(a, b):
            px = [x[h][str(t)] for h in HANDS if x[h][str(t)] is not None]
            py = [y[h][str(t)] for h in HANDS if y[h][str(t)] is not None]
            if px and py:
                pairs.append((float(np.mean(px)), float(np.mean(py))))
        out[str(t)] = M.paired_mean_difference(*np.array(pairs).T, n=BOOT) if pairs else None
    return out


# ---- stored material ---------------------------------------------------------------------------------------------------------------

def load_composed(store, index_key: str, signs) -> list:
    """Each sign's composed state from an export index, in list order. A missing sign refuses."""
    from evaluate_text_to_motion import load_state
    index = json.loads(store.get(index_key))
    if index.get("kind") != "round15-composed-states":
        raise SystemExit(f"{index_key}: not a composed-state export")
    rows = {r["sign"]: r for r in index["states"] if r["form"] == COMPOSED_FORM}
    missing = [s for s in signs if s not in rows]
    if missing:
        raise SystemExit(f"{index_key}: {len(missing)} signs are not exported, for example {missing[0]}")
    return [load_state(store, rows[s]["state_key"], rows[s]["state_sha256"]) for s in signs]


def cache_entries(store, key: str, sha: str, only=None) -> list[dict]:
    """(entry, the best-of-16 state's key and hash) of a word cache, SHA-pinned; ``only`` keeps those entries."""
    raw = store.get(key)
    P.check_sha(raw, sha, key)
    rows = []
    for e in json.loads(raw)["entries"]:
        if only is None or e["entry"] in only:
            chosen = e["chosen"]["16"]
            rows.append(dict(entry=e["entry"], state_key=chosen["state_key"], state_sha256=chosen["state_sha256"]))
    return rows


def top_tokens(top: int) -> set:
    """The first ``top`` ranks of the committed list, SHA-pinned, as ``--vocabulary-extra-top`` takes them."""
    raw = (ROOT / TOKENS_FILE).read_bytes()
    P.check_sha(raw, TOKENS_SHA, TOKENS_FILE)
    return {t["token"] for t in json.loads(raw)["tokens"] if t["rank"] <= top}


def word_training_clips(manifest, cache, fetch, entries, *, clips: int = WORD_CLIPS, corpora=None) -> dict:
    """{entry: states} of each entry's training clips, a fixed draw of at most ``clips`` each, the windows training reads (``load_sample``). Never c00, never test."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    from signtranslator.generation.compositional import isolated_corpora, normalise
    corpora = set(corpora or isolated_corpora(manifest))
    wanted, pools = set(entries), {}
    for r in manifest["records"]:
        if r.get("partition") != "train" or r.get("corpus") not in corpora or not r.get("text") or r.get("signer_component") == "c00":
            continue
        entry = " ".join(normalise(r["text"]))
        if entry in wanted:
            pools.setdefault(entry, []).append(r)
    out = {}
    for entry, pool in pools.items():
        draw = random.Random(f"{SAMPLE_SEED}/{entry}").sample(sorted(pool, key=lambda r: r["sign"]), min(clips, len(pool)))
        out[entry] = [load_sample(r, "continuous", cache=cache, fetch=fetch) for r in draw]
    return out


def corpus_neutral(manifest, cache, fetch, *, clips: int = CORPUS_CLIPS, components=None) -> dict:
    """Per corpus in a training mix, a sample: the share of hand-frames with every joint untracked, and with any joint untracked (each neutral-filled by ``diffusion_targets``), as training reads the clips."""
    from signtranslator.data_engineering.corpus_manifest import load_sample
    corpora: dict = {}
    for r in manifest["records"]:
        if r.get("partition") == "train" and r.get("text") and (r.get("continuous") or {}).get("state_key") and r.get("signer_component") != "c00" \
                and (components is None or r.get("signer_component") in components):
            corpora.setdefault(r.get("corpus") or manifest.get("corpus_id"), []).append(r)
    out = {}
    for corpus, pool in sorted(corpora.items()):
        draw = random.Random(f"{SAMPLE_SEED}/{corpus}").sample(sorted(pool, key=lambda r: r["sign"]), min(clips, len(pool)))
        every, some = [], []
        for r in draw:
            state = load_sample(r, "continuous", cache=cache, fetch=fetch)
            whole = [float((~state.channels[h].valid).all(1).mean()) for h in HANDS]
            part = [float((~state.channels[h].valid).any(1).mean()) for h in HANDS]
            every.append(np.mean(whole))
            some.append(np.mean(part))
        out[corpus] = dict(records=len(pool), sampled=len(draw), every_joint_untracked=M.bootstrap_mean(every, n=BOOT), any_joint_untracked=M.bootstrap_mean(some, n=BOOT))
    total = sum(v["records"] for v in out.values())
    if total:
        out["weighted_by_records"] = dict(every_joint_untracked=round(sum(v["records"] * v["every_joint_untracked"]["mean"] for v in out.values()) / total, 5),
                                          any_joint_untracked=round(sum(v["records"] * v["any_joint_untracked"]["mean"] for v in out.values()) / total, 5))
    return out


# ---- forms -------------------------------------------------------------------------------------------------------------------------

def nearest_input(state, tokenizer):
    return V.v0_form("v0_nearest_fill", state, None, 0, tokenizer)


def cap_form(name: str, state, tokenizer):
    from evaluate_text_to_motion import blanked
    if name == "real":
        return state
    if name == "blank":
        return blanked([state])[0]
    if name in ("identity", "round_trip"):
        return P.decoded_form(name, state, tokenizer, None)
    filled = nearest_input(state, tokenizer)
    if name == "v0_nearest_fill":
        return filled
    if name == "rt_nearest":
        return P.decoded_form("round_trip", filled, tokenizer, None)
    if name == "no_quant_nearest":
        return P.decoded_form("no_quant", filled, tokenizer, None)
    raise SystemExit(f"unknown form {name}")


def build_form(name: str, refs, tokenizer, composed) -> tuple[list, list | None]:
    """(states, counts per clip, for the replaced form)."""
    if name in CAP_FORMS:
        return [cap_form(name, s, tokenizer) for s in refs], None
    if composed is None:
        raise SystemExit(f"{name} needs the composed export")
    if name == "a9p":
        return list(composed), None
    if name == "a9p_replaced":
        pairs = [replaced_neutral(s) for s in composed]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    raise SystemExit(f"unknown form {name}")


# ---- commands -------------------------------------------------------------------------------------------------------------------------

def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    torch.set_num_threads(args.threads)
    names = [f for f in args.forms.split(",") if f]
    unknown = [f for f in names if f not in FORMS]
    if unknown:
        raise SystemExit(f"unknown forms {unknown}")
    if (args.smoke_longest or args.smoke_never_tracked) and not args.smoke_train:
        raise SystemExit("--smoke-longest and --smoke-never-tracked belong to a smoke")
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    if args.smoke_longest:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in P.longest_train_records(store, args.smoke_longest) if r["sign"] not in have]
        described = dict(described, clips=len(records), longest=args.smoke_longest)
    if args.smoke_never_tracked:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in V.never_tracked_records(store, args.smoke_never_tracked) if r["sign"] not in have]
        described = dict(described, clips=len(records), never_tracked=args.smoke_never_tracked)
    P.check_read_list(records, declared)
    texts, signs = [r["text"] for r in records], [r["sign"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    composed = None
    if any(n in SENSITIVITY_FORMS for n in names):
        if declared:
            composed = load_composed(store, A9_INDEX, signs)
        else:                                                           # a smoke's stand-in on train clips, so nothing generated is read
            composed = [P.decoded_form("identity", s, tokenizer, None) for s in refs]
    reader = P.load_reader(store, args)
    candidates = M.retrieval_candidates(len(texts), 1, seed=0)
    digest, place = code_digest(), P.environment()
    for name in names:
        started = time.time()
        states, counts = build_form(name, refs, tokenizer, composed)
        built = time.time()
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place,
                      instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None), signs=signs, references=texts, counts=counts,
                      hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5) for s, r in zip(states, refs)],
                      motion=L.motion_statistics(states, refs), hands=P.hand_statistics(states, refs), tracked=P.tracked_statistics(states),
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2),
                                     gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def run_neutral(args) -> int:
    """(b), motion only, no reader. A smoke measures c02 train clips and reads no cache or composed form."""
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from evaluate_text_to_motion import load_state
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    if (out / "neutral.json").exists():
        raise SystemExit(f"{out / 'neutral.json'} exists (write-once)")
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    P.check_read_list(records, declared)
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    tracked = [neutral_shares(s, tracked_only_by=s) for s in refs]
    fill = [neutral_shares(P.decoded_form("identity", s, tokenizer, None)) for s in refs]
    nearest = [neutral_shares(nearest_input(s, tokenizer)) for s in refs]
    report = dict(kind=f"{PROBE}-neutral", report_only=True, declared_read=declared, list=described, code_digest=code_digest(), environment=P.environment(),
                  thresholds_degrees=list(THRESHOLDS), flag_degrees=FLAG_DEGREES, signs=[r["sign"] for r in records],
                  populations=dict(real_tracked_frames=population(tracked), real_with_neutral_fill=population(fill), real_with_nearest_fill=population(nearest)))
    with tempfile.TemporaryDirectory() as t:
        cache = ContentCache(Path(t) / "c", 2 * 2 ** 30)
        if declared:
            train = L.load_manifest_at(store, L.TRAIN_MIX, L.TRAIN_MIX_SHA)     # W14b's own training mix
            words = dict(a9_words=cache_entries(store, A9_CACHE, A9_CACHE_SHA), a3_new_words=cache_entries(store, A3_CACHE, A3_CACHE_SHA, top_tokens(10)))
            alignment = {}
            for name, rows in words.items():
                states = {r["entry"]: load_state(store, r["state_key"], r["state_sha256"]) for r in rows}
                report["populations"][name] = population([neutral_shares(s) for s in states.values()])
                training = word_training_clips(train, cache, store.get, states)
                alignment[name] = word_alignment([dict(entry=e, generated={h: time_bins(hand_distance(s.channels[h])) for h in HANDS}, untracked={h: untracked_profile(training[e], h) for h in HANDS})
                                                  for e, s in states.items() if training.get(e)])
                alignment[name]["entries_without_training_clips"] = sum(not training.get(e) for e in states)
            report["word_alignment"] = alignment
            signs = report["signs"]
            for name, key in (("a9_predicted", A9_INDEX), ("a3_top10_predicted", A3_INDEX)):
                composed = [neutral_shares(s) for s in load_composed(store, key, signs)]
                report["populations"][name] = population(composed)
                report[f"{name}_minus_real_tracked"] = paired_population(composed, tracked)
            run = json.loads(store.get(f"{L.TOKENIZER.split('/checkpoints/')[0]}/run.json"))
            d2 = L.load_manifest_at(store, run["manifest_key"], run["manifest_sha256"])
            report["corpus_neutral"] = dict(w14b=corpus_neutral(train, cache, store.get), d2=corpus_neutral(d2, cache, store.get), d2_manifest=dict(key=run["manifest_key"], sha256=run["manifest_sha256"]))
        else:
            report["corpus_neutral"] = dict(smoke_c02=corpus_neutral(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), cache, store.get, clips=args.smoke_train, components=("c02",)))
    report["peak_rss_bytes"] = P.peak_rss_bytes()
    P.write_report_once(out / "neutral.json", report)
    print(json.dumps(dict(neutral=str(out / "neutral.json"), peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))), flush=True)
    return 0


def run_path_check(args) -> int:
    """Anchors against their stored reads (each SHA-pinned), the canary's limits. Exit 3, voiding the read, on a miss. A smoke checks the pins only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    out = Path(args.out_dir)
    stored = {}
    for form, (key, sha) in STORED_FORMS.items():
        raw = store.get(key)
        P.check_sha(raw, sha, key)
        stored[form] = json.loads(raw)
    probe = json.loads((out / "form-identity.json").read_bytes())
    if not probe["declared_read"]:
        print(json.dumps(dict(path_check="not applicable: a smoke reads train clips; the pins are checked")), flush=True)
        return 0
    rows, misses = {}, []
    for form, kept in stored.items():
        read = json.loads((out / f"form-{form}.json").read_bytes())
        if read["signs"] != kept["signs"] or read["references"] != kept["references"]:
            raise SystemExit(f"the stored {form} read other clips or another order")
        identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], kept["hypotheses"])]))
        gap = float(np.mean(np.abs(np.asarray(read["nll"])[:, 0] - np.asarray(kept["nll"])[:, 0])))
        rows[form] = dict(identical=round(identical, 4), nll_mean_absolute_difference=round(gap, 6))
        misses += [f"{form} {why}" for why, bad in (("hypotheses", identical < P.CANARY["identical"]), ("NLL", gap > P.CANARY["nll"])) if bad]
    result = dict(reads=rows, limits=dict(identical=P.CANARY["identical"], nll=P.CANARY["nll"]), misses=misses, reproduced=not misses, code_digest=probe["code_digest"], environment=probe["environment"])
    if not (out / "path_check.json").exists():
        P.write_report_once(out / "path_check.json", result)
    print(json.dumps(result), flush=True)
    if not misses:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{probe['code_digest']}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=probe["code_digest"], path_check=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the anchors do not reproduce their stored reads: {misses}. The read is void: no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def run_composed_check(args) -> int:
    """a9p against the stored a10 report's hypotheses. It reports; only (c) depends on it. A smoke checks the pin only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    raw = store.get(A10_REPORT)
    P.check_sha(raw, A10_SHA, A10_REPORT)
    out = Path(args.out_dir)
    read = json.loads((out / "form-a9p.json").read_bytes())
    if not read["declared_read"]:
        print(json.dumps(dict(composed_check="not applicable: a smoke reads train clips; the pin is checked")), flush=True)
        return 0
    stored = json.loads(raw)
    if read["signs"] != stored["signs"]:
        raise SystemExit("the stored a10 report read other clips or another order")
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], stored["forms"][COMPOSED_FORM]["hypotheses"])]))
    result = dict(identical=round(identical, 4), limit=P.CANARY["identical"], reproduced=identical >= P.CANARY["identical"], stored=A10_REPORT, stored_sha256=A10_SHA, code_digest=read["code_digest"])
    P.write_report_once(out / "composed_check.json", result)
    print(json.dumps(result), flush=True)
    if not result["reproduced"]:
        print(f"ALERT a9p reproduces only {result['identical']} of a10's hypotheses: part (c) cannot be read", flush=True)
    return 0


def declared_reads(reads: dict) -> dict:
    """Differences and shares, as facts, on likelihood minus blank and on BLEU, jointly resampled over clips (n 2,000, seed 0). A block is None when a form is missing."""
    refs = reads["real"]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    forms = [f for f in FORMS if f in reads]
    dnll = {f: M.delta_nll(blank, np.asarray(reads[f]["nll"])[:, 0]) for f in forms}
    stats = {f: M.bleu_statistics(reads[f]["hypotheses"], refs) for f in forms}

    def block(metric: str) -> dict:
        if metric == "likelihood_minus_blank":
            point = {f: float(dnll[f].mean()) for f in forms}
            draws = {f: dnll[f][idx].mean(1) for f in forms}
        else:
            point = {f: float(M.bleu_from_statistics(stats[f].sum(0))) for f in forms}
            draws = {f: M.bleu_from_statistics(stats[f][idx].sum(1)) for f in forms}

        def difference(a, b):
            return dict(point=round(point[a] - point[b], 5), ci95=P._quantiles(draws[a] - draws[b])) if a in point and b in point else None

        def share(a, of):
            if a not in point or of not in point or point[of] == point["blank"]:
                return None
            with np.errstate(divide="ignore", invalid="ignore"):
                resampled = (draws[a] - draws["blank"]) / (draws[of] - draws["blank"])
            return dict(point=round((point[a] - point["blank"]) / (point[of] - point["blank"]), 4), ci95=P._quantiles(resampled))
        return dict(rt_nearest_minus_round_trip=difference("rt_nearest", "round_trip"), rt_nearest_minus_v0_nearest_fill=difference("rt_nearest", "v0_nearest_fill"),
                    no_quant_nearest_minus_rt_nearest=difference("no_quant_nearest", "rt_nearest"), no_quant_nearest_minus_v0_nearest_fill=difference("no_quant_nearest", "v0_nearest_fill"),
                    v0_nearest_fill_minus_identity=difference("v0_nearest_fill", "identity"), cap_share_of_nearest_fill=share("rt_nearest", "v0_nearest_fill"),
                    round_trip_share_of_real=share("round_trip", "real"), rt_nearest_share_of_real=share("rt_nearest", "real"), a9p_replaced_minus_a9p=difference("a9p_replaced", "a9p"))
    return dict(primary="likelihood_minus_blank", likelihood_minus_blank=block("likelihood_minus_blank"), bleu=block("bleu"), bootstrap=dict(n=BOOT, seed=0))


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    names = list(FORMS) if args.forms is None else [f for f in args.forms.split(",") if f]
    for anchor in ("real", "blank"):
        if anchor not in names:
            raise SystemExit(f"--forms must include {anchor}: every pair reads it")
    reads = L.load_form_reads(out_dir, names)
    first = next(iter(reads.values()))
    digest, place = P.check_same_run(reads, "the forms")
    canary, path_check = None, None
    if first["declared_read"]:
        path = out_dir / "canary.json"
        canary = json.loads(path.read_bytes()) if path.exists() else None
        if canary is None or canary.get("reproduced") is not True or canary.get("code_digest") != digest or canary.get("environment") != place:
            raise SystemExit(f"{path}: no passed canary of this code and environment; the read is void")
        path = out_dir / "path_check.json"
        path_check = json.loads(path.read_bytes()) if path.exists() else None
        if path_check is None or path_check.get("reproduced") is not True or path_check.get("code_digest") != digest:
            raise SystemExit(f"{path}: no passed path check of this code; the read is void")
    neutral_path = out_dir / "neutral.json"
    neutral = json.loads(neutral_path.read_bytes()) if neutral_path.exists() else None
    if neutral is not None and (neutral.get("code_digest") != digest or neutral.get("environment") != place or neutral.get("declared_read") != first["declared_read"]):
        raise SystemExit(f"{neutral_path}: another run's measure")
    composed_path = out_dir / "composed_check.json"
    panels = {name: P.form_panel(reads, name, tuple(c for c in COMPARATORS.get(name, ()) if c in reads)) for name in names}
    counts = reads["a9p_replaced"]["counts"] if "a9p_replaced" in reads else None
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=first["declared_read"], list=first["list"], instrument=first["instrument"],
                  tokenizer=first["tokenizer"], clips=len(first["signs"]), code_digest=digest, environment=place, disclosures=DISCLOSURES,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  smoke=None if first["declared_read"] else "SMOKE on c02 train clips: these numbers are plumbing, not a read", forms=panels, reads=declared_reads(reads), neutral=neutral,
                  sensitivity_counts=None if counts is None else dict(flagged_frames=sum(c["flagged_frames"] for c in counts), hand_frames=sum(c["hand_frames"] for c in counts),
                                                                      hands_all_neutral=sum(c["hands_all_neutral"] for c in counts), flag_degrees=FLAG_DEGREES),
                  hypotheses={name: reads[name]["hypotheses"] for name in names},
                  checks=dict(blank_identical_to_real=panels["blank"]["identical_to_real"], canary=canary, path_check=path_check,
                              composed_check=json.loads(composed_path.read_bytes()) if composed_path.exists() else None),
                  resources={name: dict(peak_rss_bytes=reads[name]["peak_rss_bytes"], gpu_peak_reserved_bytes=reads[name]["gpu_peak_reserved_bytes"], build_seconds=reads[name]["build_seconds"],
                                        read_seconds=reads[name]["read_seconds"]) for name in names})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, forms=len(panels))), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, or a declared read without its canary, path check or motion measure. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if report.get("declared_read"):
        for name in ("canary", "path_check"):
            if not (checks.get(name) or {}).get("reproduced"):
                problems.append(f"a declared read without its {name}")
        if not report.get("neutral"):
            problems.append("a declared read without its motion measure")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def run_export_list(args) -> int:
    """The list ``--export-only`` composes: D20's signs in order, or a smoke's first 5. Prints the file's hash."""
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    signs = [r["sign"] for r in records][:args.first or None]
    payload = json.dumps(dict(name=f"tok-v1-d20-first-{args.first}" if args.first else "tok-v1-d20", signs=signs), indent=1).encode()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(payload)
    print(hashlib.sha256(payload).hexdigest(), flush=True)
    return 0


def run_stored(args) -> int:
    from train_text_to_motion import make_store
    return 0 if make_store(args.store).exists(args.key) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help="read this many c02 train clips, not D20")
    read.add_argument("--smoke-longest", type=int, default=0)
    read.add_argument("--smoke-never-tracked", type=int, default=0)
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    neutral = sub.add_parser("neutral", help="(b), motion only")
    neutral.add_argument("--store", default="r2")
    neutral.add_argument("--out-dir", required=True)
    neutral.add_argument("--smoke-train", type=int, default=0)
    canary = sub.add_parser("canary")
    canary.add_argument("--store", default="r2")
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--void-prefix", required=True)
    path = sub.add_parser("path-check")
    path.add_argument("--store", default="r2")
    path.add_argument("--out-dir", required=True)
    path.add_argument("--void-prefix", required=True)
    composed = sub.add_parser("composed-check")
    composed.add_argument("--store", default="r2")
    composed.add_argument("--out-dir", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--forms")
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    export = sub.add_parser("export-list")
    export.add_argument("--store", default="r2")
    export.add_argument("--out", required=True)
    export.add_argument("--first", type=int, default=0)
    stored = sub.add_parser("stored")
    stored.add_argument("--store", default="r2")
    stored.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(preflight=run_preflight, read=run_read, neutral=run_neutral, canary=P.run_canary, merge=run_merge, check=run_check, upload=P.run_upload, stored=run_stored,
                **{"path-check": run_path_check, "composed-check": run_composed_check, "export-list": run_export_list})[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
