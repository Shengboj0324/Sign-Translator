"""Road to 3.8, stage S2's probes (docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design received", bars declared 2026-10-31
18:10 UTC; design: ``reports/s2_design.md``): the pieces every probe shares. Nothing here trains.

**Lists.** The probes read one validation list: D20, pinned by its SHA-256 and checked clip by clip. ``load_read_list``
refuses any other key without reading it (C20's list is never fetched), a list that is not a validation-pool list of
this manifest, and any clip of the 1,000 or the 300. No probe has an option for another list. ``--smoke-train N`` reads N
How2Sign c02 train clips instead; its outputs say ``declared_read`` false, and a verdict never mixes them with D20.

**Training records.** ``train_records`` returns partition ``train`` of the named signer components and refuses c00
(How2Sign signer 9) and any non-train partition by construction; ``check_read_records`` refuses test, held-out and c00
records anywhere in a read.

**Reads.** Every instrument probe (P2, P3, P4) reads a form per process (so up to 5 run side by side), writes
``form-<name>.json`` once, and a separate ``verdict`` step joins the forms, refusing mixed lists or instruments. The
per-form reads are those of ``s0_ceilings.py``: hypotheses at beam 4 and batch 32, and the teacher-forced reference NLL.

**Units.** ``pooled_rate`` is the unit of the stored AR figures: total nats over all clips divided by total seconds,
with seconds = steps / 24. On stage-1 codes (4 per step) that is the per-code mean × 96, so the stored per-code gains
give flan-T5 AR +0.0148 × 96 = +1.42 and W14b −0.00531 × 96 = −0.51 nats/s (``STORED_AR_GAIN_PER_CODE``).
"""
from __future__ import annotations

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

from signtranslator.eval import s0_metrics as M  # noqa: E402

O = "acquisition/2026-10-17-v1/youtube-asl-v1"
EVAL_MIX = f"{O}/round10-mixed-v1/manifest.json"
EVAL_MIX_SHA = "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a"
TRAIN_MIX = f"{O}/round14b-mixed-v1/manifest.json"
TRAIN_MIX_SHA = "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d"
D20_KEY = f"{O}/round20/d20.json"
D20_SHA = "d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe"
D20_CLIPS = 500
D20_SHORT = 86                              # D20's clips of at most SHORT_TOKENS tokens (the declared 86)
SHORT_TOKENS = 8
H2S = "how2sign-two-view-text-v1"
VAL_GROUPS = ("c01", "c03", "c05")
FPS = 24.
STAGE1_CODES_PER_SECOND = 96.               # 24 steps/s × 4 parts
STORED_AR_GAIN_PER_CODE = dict(flant5=.0148, w14b=-.00531)   # s1-text-kl-v1.json, "gain": 1.42 and −0.51 nats/s
TOKENIZER = "training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt"
TOKENIZER_SHA = "021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de"
TRANSLATOR_RUN = "back-translator-kp2d-uwb-ft-v1"
TRANSLATOR_MANIFEST_SHA = "004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd"
TRANSLATOR_SHA = "2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8"
SMOKE_SEED = 20261031


# ---- lists and guards ------------------------------------------------------------------------------------------------

def check_read_records(records) -> None:
    """Every record of a validation read is How2Sign val of the pilots' signer groups: never test, held-out or c00."""
    for r in records:
        if r.get("corpus") != H2S or r.get("partition") != "val" or r.get("signer_component") not in VAL_GROUPS:
            raise SystemExit(f"{r.get('sign')}: not a How2Sign val clip of {VAL_GROUPS}; S2's probes do not read it")


def check_train_records(records, components) -> None:
    for r in records:
        if r.get("partition") != "train" or r.get("signer_component") not in components or r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: not a train clip of {tuple(components)}")


def train_records(manifest, components, corpus: str = H2S) -> list[dict]:
    """How2Sign train records of ``components`` that have a text and a state, in manifest order. c00 is refused."""
    components = tuple(components)
    if "c00" in components:
        raise SystemExit("How2Sign signer 9 (c00) is never read")
    return [r for r in manifest["records"] if r.get("corpus") == corpus and r.get("partition") == "train"
            and r.get("signer_component") in components and r.get("text") and (r.get("continuous") or {}).get("state_key")]


def smoke_records(manifest, n: int) -> list[dict]:
    """``n`` How2Sign c02 train clips (fixed draw): the only clips a smoke reads."""
    pool = train_records(manifest, ("c02",))
    return random.Random(SMOKE_SEED).sample(pool, min(n, len(pool)))


def load_read_list(store, manifest, key: str = D20_KEY) -> list[dict]:
    """D20's manifest records, after every check in the module's docstring. Any other key is refused before it is read."""
    from pilot_content_check import validation_sentences
    from round20_lib import load_sentence_set
    if key != D20_KEY:
        raise SystemExit(f"{key}: S2's probes read D20 only ({D20_KEY}); C20, the 1,000 and fresh clips are refused")
    raw = store.get(key)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != D20_SHA:
        raise SystemExit(f"{key}: SHA-256 {digest}, expected {D20_SHA}")
    payload = json.loads(raw)
    if payload.get("set") != "D20":
        raise SystemExit(f"{key}: this list says it is {payload.get('set')!r}, not D20")
    records = load_sentence_set(payload, manifest, EVAL_MIX, EVAL_MIX_SHA)
    check_read_records(records)
    forbidden = {r["sign"] for n in (1000, 300) for r in validation_sentences(manifest, n)}
    hit = [r["sign"] for r in records if r["sign"] in forbidden]
    if hit:
        raise SystemExit(f"{len(hit)} clips of this list are in the 1,000 or the 300, for example {hit[0]}")
    return records


def short_clip_indices(records) -> list[int]:
    """The list's clips of at most ``SHORT_TOKENS`` tokens (``compositional.normalise``): P1's 86 on D20."""
    from signtranslator.generation.compositional import normalise
    return [i for i, r in enumerate(records) if len(normalise(r["text"])) <= SHORT_TOKENS]


def load_manifest_at(store, key: str, sha: str):
    from signtranslator.data_engineering.corpus_manifest import load_manifest
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / "m.json"
        path.write_bytes(store.get(key) + b"\n")
        return load_manifest(path, expected_sha256=sha)


def open_read_list(store, smoke_train: int) -> tuple[list[dict], bool, dict]:
    """(records, declared read?, list description) for ``--smoke-train``."""
    manifest = load_manifest_at(store, EVAL_MIX, EVAL_MIX_SHA)
    if smoke_train:
        records = smoke_records(manifest, smoke_train)
        check_train_records(records, ("c02",))
        return records, False, dict(list="smoke:c02-train", manifest_sha256=EVAL_MIX_SHA, clips=len(records))
    records = load_read_list(store, manifest)
    return records, True, dict(list=D20_KEY, sha256=D20_SHA, manifest_sha256=EVAL_MIX_SHA, clips=len(records))


def load_states(store, records, *, threads: int = 4) -> list:
    """Each record's state at 24 fps, SHA-checked."""
    from concurrent.futures import ThreadPoolExecutor
    from evaluate_text_to_motion import at_rate, load_state
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda r: at_rate(load_state(store, r["continuous"]["state_key"],
                                                          r["continuous"]["state_sha256"]), FPS), records))


def load_tokenizer(store):
    """d2, SHA-checked, in eval mode on the CPU."""
    import diagnose_tokenizer as dt
    digest = hashlib.sha256(store.get(TOKENIZER)).hexdigest()
    if digest != TOKENIZER_SHA:
        raise SystemExit(f"{TOKENIZER}: SHA-256 {digest}, expected {TOKENIZER_SHA}")
    tokenizer, _ = dt.load_tokenizer(store, TOKENIZER)
    return tokenizer.eval()


# ---- statistics ------------------------------------------------------------------------------------------------------

def pooled_rate(nats, seconds, *, boot: int = 2000, seed: int = 0) -> dict:
    """{rate, ci95, n}: total nats / total seconds over clips, the unit of the stored AR figures, bootstrapped over
    clips."""
    nats, seconds = np.asarray(nats, np.float64), np.asarray(seconds, np.float64)
    if nats.shape != seconds.shape or not len(nats):
        raise ValueError("nats and seconds must be paired and non-empty")
    if (seconds <= 0).any():
        raise ValueError("every clip needs a positive duration")
    idx = np.random.default_rng(seed).integers(0, len(nats), (boot, len(nats)))
    rates = nats[idx].sum(1) / seconds[idx].sum(1)
    return dict(rate=round(float(nats.sum() / seconds.sum()), 5),
                ci95=[round(float(np.percentile(rates, 2.5)), 5), round(float(np.percentile(rates, 97.5)), 5)],
                n=int(len(nats)))


def paired_rate_difference(a, b, seconds, *, boot: int = 2000, seed: int = 0) -> dict:
    """Pooled rate of a − b (nats per clip, paired), and the ratio of the two pooled rates, resampled jointly."""
    a, b, seconds = (np.asarray(x, np.float64) for x in (a, b, seconds))
    idx = np.random.default_rng(seed).integers(0, len(a), (boot, len(a)))
    sec = seconds[idx].sum(1)
    ra, rb = a[idx].sum(1) / sec, b[idx].sum(1) / sec
    ratios = ra[rb != 0] / rb[rb != 0]
    return dict(difference=round(float((a.sum() - b.sum()) / seconds.sum()), 5),
                ci95=[round(float(np.percentile(ra - rb, 2.5)), 5), round(float(np.percentile(ra - rb, 97.5)), 5)],
                ratio=round(float(a.sum() / b.sum()), 5) if b.sum() else None,
                ratio_ci95=[round(float(np.percentile(ratios, 2.5)), 5), round(float(np.percentile(ratios, 97.5)), 5)]
                if len(ratios) else None)


def bar(name: str, value, threshold, passed: bool, **extra) -> dict:
    return dict(name=name, value=value, threshold=threshold, passed=bool(passed), **extra)


# ---- write-once outputs ----------------------------------------------------------------------------------------------

def refuse_existing(store, output_key, output_file) -> None:
    if bool(output_key) == bool(output_file):
        raise SystemExit("give exactly one of --output-key and --output-file")
    if output_key and store is not None and store.exists(output_key):
        raise SystemExit(f"{output_key} exists (write-once)")
    if output_file and Path(output_file).exists():
        raise SystemExit(f"{output_file} exists (write-once)")


def write_once(store, output_key, output_file, report: dict) -> None:
    refuse_existing(store, output_key, output_file)
    payload = json.dumps(report, indent=1, sort_keys=True, default=float).encode()
    if output_key:
        store.put(output_key, payload)
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_bytes(payload)


def write_file_once(path: Path, report: dict) -> None:
    write_once(None, None, str(path), report)


# ---- instrument reads ------------------------------------------------------------------------------------------------

class Reader:
    """One translator's reads of a form, exactly as ``s0_ceilings.compute``: hypotheses (beam 4, batch 32) and the
    (n, k) candidate NLL matrix."""

    def __init__(self, model, kind: str, info: dict, device) -> None:
        self.model, self.kind, self.info, self.device = model, kind, info, device

    def read(self, states, texts, candidates, *, group: int = 2) -> dict:
        import s0_ceilings as s0
        from score_back_translation import translate
        hyps = translate(self.model, self.kind, states)
        items = s0.translator_items(self.model, self.kind, states)
        nll = s0.candidate_nll(self.model, items, texts, candidates, device=self.device, group=group)
        return dict(hypotheses=hyps, nll=nll)


class _OneFile:
    """A store holding one payload (``load_translator`` reads through a store's ``get``)."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get(self, key: str) -> bytes:
        return self.payload


READ_ONLY_KEYS = ("kind", "ema", "config", "config_sha256", "manifest_sha256", "step")


def instrument_payload(store, key: str, sha256: str | None, cache_dir: str | None) -> tuple[bytes, str]:
    """(checkpoint bytes, the original checkpoint's SHA-256), fetched once. v1's 3.4 GB file is mostly its optimizer, and each
    process that loads it peaks near 10 GB, so with ``cache_dir`` the first process checks the original's hash and keeps
    only the fields a read uses in ``<sha>.pt``, which later processes load directly. The file name is the original's
    hash, so a cached copy of another instrument is never read in its place."""
    import io
    import os
    import torch
    if cache_dir and sha256:
        path = Path(cache_dir) / f"{sha256}.pt"
        if path.exists():
            return path.read_bytes(), sha256
    payload = store.get(key)
    digest = hashlib.sha256(payload).hexdigest()
    if sha256 and digest != sha256:
        raise SystemExit(f"{key}: SHA-256 {digest}, expected {sha256}")
    if not (cache_dir and sha256):
        return payload, digest
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)     # our own checkpoint (R2)
    del payload
    if "ema" not in saved or saved["ema"] is None:
        raise SystemExit(f"{key}: no EMA weights to keep")
    slim = {k: saved[k] for k in READ_ONLY_KEYS if k in saved}
    del saved
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(slim, buffer)
    temporary = Path(cache_dir) / f"{sha256}.{os.getpid()}.tmp"
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(Path(cache_dir) / f"{sha256}.pt")          # atomic: another process never reads half a file
    return buffer.getvalue(), digest


def make_reader(store, *, run: str = TRANSLATOR_RUN, sha256: str | None = TRANSLATOR_SHA, device: str = "cpu",
                cache_dir: str | None = None) -> Reader:
    """The named instrument (default v1, SHA-pinned; another run through ``load_translator_run``, SHA-checked there)."""
    import torch
    from score_back_translation import load_translator, load_translator_run
    if run == TRANSLATOR_RUN:
        key = json.loads(store.get(f"training-runs/{run}/best.json"))["checkpoint_key"]
        payload, digest = instrument_payload(store, key, sha256, cache_dir)
        model, kind = load_translator(_OneFile(payload), key, TRANSLATOR_MANIFEST_SHA)
        del payload
    else:
        model, kind, key, digest = load_translator_run(store, run, TRANSLATOR_MANIFEST_SHA)
    return Reader(model.to(torch.device(device)).eval(), kind, dict(run=run, checkpoint_key=key, sha256=digest, kind=kind),
                  torch.device(device))


def observed_like(state, ref):
    """``state`` with its body and hands valid only where ``ref``'s are. A round trip fills unobserved frames with the
    neutral pose and marks them valid, so its unmasked wrist speed jumps at every tracking gap (2.7× real on 8 c02 clips,
    0.97× once masked); the statistics compare the frames real signing was tracked on. Forms of another length keep theirs."""
    from dataclasses import replace
    if len(state.timestamps) != len(ref.timestamps):
        return state
    channels = dict(state.channels)
    for name in ("body", "left_hand", "right_hand"):
        c, seen = channels[name], ref.channels[name].valid
        channels[name] = replace(c, valid=c.valid & seen, values=np.where((c.valid & seen)[..., None], c.values, 0.),
                                 observed=c.observed & seen, confidence=c.confidence * seen)
    return replace(state, channels=channels)


def motion_statistics(states, refs) -> dict:
    """Per clip: wrist speed and wrist spread in the back-translator's view (``sentence_exposure_diagnostic``), on the
    frames the real clip was tracked on (``observed_like``)."""
    from signtranslator.data.keypoints2d import from_state, normalise
    from sentence_exposure_diagnostic import keypoint_statistics
    speed, spread = [], []
    for s, ref in zip(states, refs):
        points, valid = from_state(observed_like(s, ref))
        k = keypoint_statistics(normalise(points, valid, y_down=False), valid, FPS)
        speed.append(k["wrist_speed"])
        spread.append(k["wrist_spread"])
    return dict(wrist_speed=speed, wrist_spread=spread)


def latent_distances(tokenizer, states, real_latents, mean, std, *, device: str = "cpu") -> np.ndarray:
    """Each form clip's DTW distance to its real clip in standardised d2 latents: ρ_faith's own distance
    (``s0_metrics.dtw_distances``). P2's curve and P1's samples are placed on this one axis."""
    import s0_ceilings as s0
    latents = s0.d2_latents(tokenizer, states, device)
    return M.dtw_distances([(z - mean) / std for z in latents], [(z - mean) / std for z in real_latents], device=device)


def add_read_arguments(parser, default_forms: str) -> None:
    parser.add_argument("--store", default="r2", help="r2, or dir:/path (tests)")
    parser.add_argument("--forms", default=default_forms, help="comma list; each process reads its own forms")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--translator-run", default=TRANSLATOR_RUN, help="v1 by default; v2 once accepted")
    parser.add_argument("--candidates", type=int, default=1,
                        help="own reference plus distractors per clip: 1 is ΔNLL only (Mac); 100 adds PMI ranks (pod)")
    parser.add_argument("--nll-group", type=int, default=2)
    parser.add_argument("--instrument-cache", help="directory for the instrument's slim checkpoint (read the first form "
                        "alone, so later processes do not each peak near 10 GB)")
    parser.add_argument("--smoke-train", type=int, default=0, help="read this many c02 train clips, not D20 (a smoke)")
    parser.add_argument("--device", default="cpu")


def read_forms(args, *, probe: str, build) -> int:
    """Read ``args.forms`` with ``build(name, context)`` -> states, one write-once ``form-<name>.json`` each."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    out_dir = Path(args.out_dir)
    names = [f for f in args.forms.split(",") if f]
    for name in names:
        if (out_dir / f"form-{name}.json").exists():
            raise SystemExit(f"{out_dir / f'form-{name}.json'} exists (write-once)")
    records, declared, described = open_read_list(store, args.smoke_train)
    texts = [r["text"] for r in records]
    refs = load_states(store, records)
    tokenizer = load_tokenizer(store)
    reader = make_reader(store, run=args.translator_run, sha256=TRANSLATOR_SHA if args.translator_run == TRANSLATOR_RUN else None,
                         device=args.device, cache_dir=args.instrument_cache)
    candidates = M.retrieval_candidates(len(texts), args.candidates, seed=0)
    context = dict(store=store, records=records, refs=refs, tokenizer=tokenizer, args=args)
    import s0_ceilings as s0
    real_latents = s0.d2_latents(tokenizer, refs, "cpu")
    mean, std = M.standardiser(real_latents)
    for name in names:
        started = time.time()
        states = build(name, context)
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"s2-{probe}-form", form=name, declared_read=declared, list=described, instrument=reader.info,
                      signs=[r["sign"] for r in records], groups=[r.get("signer_component") for r in records],
                      references=texts, candidates=candidates.tolist() if args.candidates > 1 else None,
                      hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      latent_distance=np.round(latent_distances(tokenizer, states, real_latents, mean, std), 5).tolist(),
                      seconds_over_reference=[round((len(s.timestamps)) / max(len(r.timestamps), 1), 5)
                                              for s, r in zip(states, refs)],
                      motion=motion_statistics(states, refs), read_seconds=round(time.time() - started, 1))
        write_file_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(seconds=report["read_seconds"])}), flush=True)
    return 0


def load_form_reads(out_dir, names) -> dict:
    """``{form: report}``; every form must exist and share one list, one instrument and one kind of read."""
    out, first = {}, None
    for name in names:
        path = Path(out_dir) / f"form-{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        r = json.loads(path.read_bytes())
        if r.get("form") != name:
            raise SystemExit(f"{path} holds form {r.get('form')!r}")
        signature = (r["signs"], r["instrument"]["sha256"], r["declared_read"], r["list"]["list"])
        if first is None:
            first = signature
        elif signature != first:
            raise SystemExit(f"{path}: other clips, another instrument, or a smoke mixed with a declared read")
        out[name] = r
    return out


def form_panel(reads: dict, name: str) -> dict:
    """BLEU with its interval, ΔNLL over blank, and paired differences against real and blank; PMI ranks when read."""
    r, refs = reads[name], reads[name]["references"]
    stats = M.bleu_statistics(r["hypotheses"], refs)
    bleu, ci = M.bleu_ci_fast(r["hypotheses"], refs)
    nll, blank = np.asarray(r["nll"]), np.asarray(reads["blank"]["nll"])
    dnll = M.delta_nll(blank[:, 0], nll[:, 0])
    out = dict(bleu=bleu, ci95=ci, length_ratio=M.length_ratio(stats), delta_nll=M.bootstrap_mean(dnll),
               latent_distance=M.bootstrap_mean(r["latent_distance"]),
               seconds_over_reference=round(float(np.mean(r["seconds_over_reference"])), 4),
               paired={a: dict(bleu=M.paired_bleu_fast(stats, M.bleu_statistics(reads[a]["hypotheses"], refs)),
                               delta_nll=M.paired_mean_difference(dnll, M.delta_nll(blank[:, 0], np.asarray(reads[a]["nll"])[:, 0])))
                       for a in ("real", "blank")})
    if nll.shape[1] > 1:
        ranks = M.mid_ranks(nll - blank)
        out["retrieval_pmi"] = M.retrieval_summary(ranks, nll.shape[1])
        out["mrr_pmi"] = M.bootstrap_mean(1. / ranks)
    speed = [x for x in r["motion"]["wrist_speed"] if x is not None]
    real_speed = [x for x in reads["real"]["motion"]["wrist_speed"] if x is not None]
    spread = [x for x in r["motion"]["wrist_spread"] if x is not None]
    real_spread = [x for x in reads["real"]["motion"]["wrist_spread"] if x is not None]
    out["wrist_speed_over_real"] = round(float(np.mean(speed) / np.mean(real_speed)), 4) if speed and real_speed else None
    out["wrist_spread_over_real"] = round(float(np.mean(spread) / np.mean(real_spread)), 4) if spread and real_spread else None
    return out


def verdict_shell(probe: str, reads: dict) -> dict:
    first = next(iter(reads.values()))
    return dict(kind=f"s2-{probe}-verdict", declared_read=first["declared_read"], list=first["list"],
                instrument=first["instrument"], declared="docs/NEXT_PHASE_PLAN.md, 'Road to 3.8: S2 design received'",
                note=None if first["declared_read"] else "SMOKE on train clips: no bar here is a declared result")
