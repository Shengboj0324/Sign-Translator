"""Path read: does instrument v1 read real D20 through the monocular lifting path materially lower than through two-view? Report only: nothing trains, no generated form and no composition is read, and this code gates and selects nothing.

The word clips composition uses come almost entirely from monocular-path training data (0.27% two-view among single-word clips: ``word_clip_path`` in ``reports/wrist_by_signer.json``, branch ``comp-a6``), while "real D20" has been read through two-view. If the path alone costs a lot on v1,
composition's "share of the gap" figures are measured against a target its clips cannot reach.

**How this will be read (stated before any read; the commit holding this text precedes the job).** Point difference d = real_monocular − real_two_view, BLEU, on the shared clips.

(a) monocular within 0.5 BLEU of two-view with overlapping CI: the path is not a large cap.
(b) lower by more than 0.5: the path caps composition, report the same-path gap share.
(c) higher: the instrument may prefer monocular, investigate before believing it.

Where those three leave a case open or overlap, this settles it now. "Higher" means d > 0.5, or d > 0 with a paired interval wholly above zero; a higher read inside noise is (a), and its sign is shown. Exactly 0.5 is within 0.5. Lower by at most 0.5 with non-overlapping intervals is none of the three, and the report says so. (b) has no interval condition, as stated; whether the intervals overlap is shown beside it.
Either way the case is a reading, not a rule: no step of the job branches on it, and it starts no training and no decision. BLEU is what the three cases name; likelihood minus blank is reported beside it and read the same way by eye.

**Forms** (D20, instrument v1, the selection and scoring of ``tok_v1`` and ``tok_v0``; no option names another list or instrument):

- ``real_two_view``: the stored D20 states, as V1's ``real``. The control: read on all 500, it must reproduce the pinned stored report (``tok_probes.canary_read``: BLEU within 0.05, at least 0.98 of hypotheses identical, mean per-clip NLL within 0.001). A miss stops the job before any monocular form and voids the read.
- ``blank``: V1's blank, ``blanked`` of the two-view clips; checked by the same control.
- ``real_monocular``: the same clips' monocular states (``derived/how2sign-video-estimate-v1``, extracted as YouTube's states are), at 24 fps, conformed by ``layout_adapter.conform`` to the template word clips are conformed to (``round15_compose_eval``: the first How2Sign train state of r10, ``TEMPLATE_SIGN``).
- ``real_monocular_unconformed``: conforming only renames the frame and convention of valid channels and empties channels with no valid frame; v1 reads body and hand rotations and their marks (``keypoints2d.from_state``). So each clip's v1 input is compared before any read. If it is byte-identical on every clip, a read can only repeat ``real_monocular``: the form is skipped, the comparison stands as what conforming costs (nothing), and ``monocular.json`` says why. If any clip differs, the form is read.
- ``round_trip_monocular``: V1's ``rt_nearest`` (``tok_v1.cap_form``: nearest-valid fill, every frame valid, d2's encoder, codes and decoder, no gains) on ``real_monocular``'s state, since word clips pass through the tokenizer. Beside it, the stored two-view ``rt_nearest`` (4.124 on all 500), restricted to the shared clips.

**Clips.** Monocular states are found by exact D20 record: the record's ``sign`` as a dictionary key among the rows of the 8 report shards pinned below, each by key and hash; the state key must then be exactly ``<prefix>/states/<sign>.npz``. Nothing is listed and no substring is matched. All 500 of D20 sit in those shards (checked 2026-11-01 from index objects; 0 in the official-val report), so the official val and test reports are never opened.
A record that is not a How2Sign val clip of c01, c03, c05 (or, in a smoke, a c02 train clip) is refused; so is c00, whatever else it says. C20 cannot be named: ``s2_lib.load_read_list`` refuses any other key unread. Clips without a usable monocular state are listed with their reasons. Every comparison, against ``real_two_view`` and the stored ``rt_nearest`` alike, runs on the clips every form read.

**Per form** (``merge``): BLEU with ``bleu_ci_fast``'s interval, likelihood minus blank, share of the real-minus-blank gap, valid-frame share, paired differences against ``real_two_view``, per-signer rows, clips lost with reasons. ``same_path``: each difference and share under one joint bootstrap (n 2,000, seed 0), with (b)'s figures:
the monocular gap, its ratio to the two-view gap, and the round trip's share of it. A stored figure S against two-view reads S / ratio against monocular, within the same blank. No generated report is opened here; that division is left to the reader.

**Smoke** (``--smoke-train 5``): five c02 train clips through every form, the stored form's identity, and D20's lookup from index objects alone. No D20 state is read and no instrument sees D20 in the smoke.

    python scripts/path_read.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/path_read.py lookups --out-dir W/forms
    python scripts/path_read.py read --forms real_two_view,blank --out-dir W/forms --instrument-cache W/i
    python scripts/path_read.py control --out-dir W/forms --void-prefix V
    python scripts/path_read.py read --forms real_monocular,real_monocular_unconformed,round_trip_monocular --resume --out-dir W/forms --instrument-cache W/i
    python scripts/path_read.py merge --out-dir W/forms --output-file W/report.json
    python scripts/path_read.py check --report W/report.json

The job is ``runpod/fleet/path_read.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v0 as V  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402

PROBE = "path-read"
JOB_FILE = "runpod/fleet/path_read.sh"
FPS = L.FPS
PREFIX = f"{L.O}/round20/path-read"
SMOKE_CLIPS = 5
BOOT = 2000
ANCHORS = ("real_two_view", "blank")
MONOCULAR_FORMS = ("real_monocular", "real_monocular_unconformed", "round_trip_monocular")
FORMS = ANCHORS + MONOCULAR_FORMS
UNCONFORMED = "real_monocular_unconformed"
MONOCULAR = "acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1"
REPORTS = f"{MONOCULAR}/reports/"
STATES = f"{MONOCULAR}/states/"
# the only monocular objects this code opens before a state: How2Sign's official train split, every clip of D20 and of the smoke. Never val or test.
REPORT_NAME = re.compile(r"train_rgb_front_clips-shard-\d{3}-of-008\.json")
MONOCULAR_REPORTS = {f"{REPORTS}train_rgb_front_clips-shard-{i:03d}-of-008.json": sha for i, sha in enumerate((
    "c4b5c02d8244ad61e61da231385318f07a73f738452f806d2ce3c2150d02fda7", "8caf0176fcf6318f4aa623658016676a89112bdc7d8c0c083a366e9885f135a8",
    "710e68880f745848a60efda1e2d3b217e9c64f64626ed03fbf62a75883b8a824", "2222b896371a3d1f0949671c241ba0d9d667dbda34ea80acd71bb13df14018de",
    "b4f8e34a4e98b4a4676df8ad693defe9a6e991fde20231f5308bcf28c20d784c", "6f2cd81d87be0386709ffe2ec73d566f932891a39bce01396a79e6c47baa697f",
    "a675fd716b2bc1fb3b398e25863c5851afaa25eca1f73025e256d4f3ee7cf407", "26361e9311dfce0ec5c6d2197c3f0c3e02a04e43da34c6f7e97d8a5fe8036804"))}
TEMPLATE_SIGN = "--7E2sU6zP4_10-5-rgb_front"            # r10's first How2Sign train record (c02): train_motion_tokens.layout_template's choice
STORED_RT = f"{L.O}/tok-probes-v1/forms/form-rt_nearest.json"
STORED_RT_BLEU, STORED_RT_CI = 4.124, [3.485, 4.764]    # docs/NEXT_PHASE_PLAN.md, V1 result
STORED_RT_DIGEST = "abcb46bd"                           # V1's code digest, the ledger's prefix
PRESTATED = dict(a="(a) monocular within 0.5 BLEU of two-view with overlapping CI: the path is not a large cap.",
                 b="(b) lower by more than 0.5: the path caps composition, report the same-path gap share.",
                 c="(c) higher: the instrument may prefer monocular, investigate before believing it.",
                 settled='"Higher" means d > 0.5, or d > 0 with a paired interval wholly above zero; a higher read inside noise is (a), and its sign is shown.')
LARGE = .5
NOT_DECIDED = "report only: the header's reading is stated before the read; no rule, bar or verdict here gates anything"
DISCLOSURES = dict(
    instrument="v1 was pretrained on UWB YouTube-ASL MediaPipe 2D keypoints, then fine-tuned on How2Sign FK keypoints built from the two-view manifest (full-v1) and selected on How2Sign val BLEU, the pool D20 comes from. A lower monocular read can be v1's fit to two-view input as well as lost content; this read cannot separate them. Either way it is what v1 can credit.",
    frames="A two-view clip is a slice of the same front clip's frames (triangulate_how2sign_sessions), so monocular clips run about 1.00-1.05x longer with extra frames at the ends. They are not trimmed: word clips carry no such trim. seconds_over_reference gives the ratio per form.",
    blank="Blank is V1's: the two-view clips' durations. Its length is therefore not exactly the monocular clip's, and no separate monocular-length blank is read.",
    marks="Validity differs by path: v1 reads per-point marks, so valid_frame_share belongs beside each read. round_trip_monocular is all-valid by construction, like V1's rt_nearest; real_monocular's share is the one that shows tracking.",
    scope="This reads How2Sign's signers through the monocular path. The word clips come from other corpora through that path (ASLLVD, RIT, WLASL, Sem-Lex, YouTube, ASLLRP), with other framing and signers, so this measures the path, not those corpora.",
    stored="The stored rt_nearest was read by V1's job; its environment is shown beside this read's, and its likelihood is taken against this read's blank.",
    panels="Inside a form's panel, wrist_speed, wrist_spread, hand_speed and hand_variation name 'over_real'; real there is real_two_view. Each form's motion is measured on its own path's tracked frames.")
PER_CLIP = ("signs", "groups", "references", "hypotheses", "nll", "seconds_over_reference", "valid_frame_share")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- what may be looked up -------------------------------------------------------------------------------------------------------

def check_lookup_records(records, declared: bool) -> None:
    """The only records a monocular lookup takes: D20's validation clips, or a smoke's c02 train clips. c00 and any other partition are refused, whatever else a record says."""
    for r in records:
        if r.get("signer_component") == "c00":
            raise SystemExit(f"{r.get('sign')}: How2Sign signer 9 (c00) is never read")
        if r.get("corpus") != L.H2S:
            raise SystemExit(f"{r.get('sign')}: not a How2Sign record")
    P.check_read_list(records, declared)


def check_report_key(key: str) -> None:
    """A report is opened only if it is pinned and named as an official train shard, tested before it is fetched."""
    if key not in MONOCULAR_REPORTS or not key.startswith(REPORTS) or not REPORT_NAME.fullmatch(key[len(REPORTS):]):
        raise SystemExit(f"{key}: not a pinned train-split report; refused unread")


def monocular_rows(store, wanted) -> dict:
    """{sign: its report row}, only for the exact signs in ``wanted``: other clips' rows are not kept. Every pinned shard is read and hash-checked; a sign with two different states refuses."""
    wanted, rows = set(wanted), {}
    for key, sha in sorted(MONOCULAR_REPORTS.items()):
        check_report_key(key)
        raw = store.get(key)
        P.check_sha(raw, sha, key)
        for row in json.loads(raw)["results"]:
            sign = row.get("sample_id")
            if sign not in wanted:
                continue
            here = dict(state_key=row.get("state_key"), state_sha256=row.get("state_sha256"), status=row.get("status"), shard=key[len(REPORTS):])
            if sign in rows and (rows[sign]["state_key"], rows[sign]["state_sha256"]) != (here["state_key"], here["state_sha256"]):
                raise SystemExit(f"{sign}: two different monocular states in the reports")
            rows.setdefault(sign, here)
    return rows


def state_key_of(sign: str) -> str:
    return f"{STATES}{sign}.npz"


def lost_clip(sign: str, reason: str, detail: str | None = None) -> dict:
    return dict(sign=sign, reason=reason, detail=detail)


def match_monocular(records, rows: dict, declared: bool) -> tuple[list[dict], list[dict]]:
    """(the records with a monocular state, in list order, as {index, sign, state_key, state_sha256}; the others, listed). A record matches its own row by its exact sign, and that row must name exactly this clip's state."""
    check_lookup_records(records, declared)
    found, lost = [], []
    for i, r in enumerate(records):
        sign, row = r["sign"], rows.get(r["sign"])
        if row is None:
            lost.append(lost_clip(sign, "no_monocular_row"))
        elif not row.get("state_key") or not row.get("state_sha256"):
            lost.append(lost_clip(sign, "no_state_in_row", row.get("status")))
        elif row["state_key"] != state_key_of(sign):
            lost.append(lost_clip(sign, "state_key_not_this_clip", row["state_key"]))
        else:
            found.append(dict(index=i, sign=sign, state_key=row["state_key"], state_sha256=row["state_sha256"]))
    return found, lost


def load_template(store):
    """The layout word clips are conformed to (``train_motion_tokens.layout_template`` on r10): the first How2Sign train record, checked by name, partition and signer."""
    from evaluate_text_to_motion import load_state
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    record = next((r for r in manifest["records"] if r.get("corpus") == L.H2S and r.get("partition") == "train"), None)
    if record is None or record["sign"] != TEMPLATE_SIGN or record.get("signer_component") == "c00":
        raise SystemExit(f"the template record is {record and record.get('sign')}, not {TEMPLATE_SIGN}")
    return load_state(store, record["continuous"]["state_key"], record["continuous"]["state_sha256"])


# ---- monocular clips -------------------------------------------------------------------------------------------------------------

@dataclass
class Clip:
    index: int
    sign: str
    raw: object
    conformed: object


def layout_of(state) -> dict:
    """The layout of the channels v1 reads. A clip's other channels may be unavailable under placeholder names in either path, which ``conform`` handles and v1 never sees."""
    return {name: (state.channels[name].labels, state.channels[name].coordinate_frame, state.channels[name].units, state.channels[name].convention) for name in V.MASKED}


def reader_input(state) -> np.ndarray:
    """What v1 receives from a state at its rate (``score_back_translation.translate``)."""
    from signtranslator.data.keypoints2d import features, from_state, normalise
    points, valid = from_state(state)
    return features(normalise(points, valid, y_down=False), valid)


def valid_frame_share(state) -> float:
    """The share of v1's points the state marks valid, over all frames."""
    from signtranslator.data.keypoints2d import from_state
    return round(float(from_state(state)[1].mean()), 5)


def load_clip(store, item: dict, template, ref) -> tuple[Clip | None, dict | None]:
    """(the clip, None) or (None, why it is lost). Conformed as ``round20_lib.utterance_clip`` conforms a word's material: to the template, then to 24 fps."""
    from evaluate_text_to_motion import at_rate, load_state
    from signtranslator.data.layout_adapter import conform
    sign = item["sign"]
    if item["state_key"] != state_key_of(sign):
        raise SystemExit(f"{sign}: refused {item['state_key']}: not this clip's monocular state")
    try:
        raw = load_state(store, item["state_key"], item["state_sha256"])
    except Exception as error:      # a missing object or a changed hash: this clip is lost, with its reason
        return None, lost_clip(sign, "state_unreadable", f"{type(error).__name__}: {error}")
    if len(raw.timestamps) < 2:
        return None, lost_clip(sign, "too_few_frames", str(len(raw.timestamps)))
    try:
        conformed = at_rate(conform(raw, template), FPS)
    except (ValueError, KeyError) as error:
        return None, lost_clip(sign, "conform_refused", f"{type(error).__name__}: {error}")
    if layout_of(conformed) != layout_of(ref):
        return None, lost_clip(sign, "layout_differs_from_two_view")
    return Clip(item["index"], sign, at_rate(raw, FPS), conformed), None


def open_monocular(store, records, refs, declared: bool) -> tuple[list[Clip], list[dict], object]:
    """Every record's monocular clip that can be read, the rest with their reasons, and the template."""
    found, lost = match_monocular(records, monocular_rows(store, [r["sign"] for r in records]), declared)
    template = load_template(store)
    with ThreadPoolExecutor(4) as pool:
        loaded = list(pool.map(lambda item: load_clip(store, item, template, refs[item["index"]]), found))
    clips = [c for c, _ in loaded if c is not None]
    lost += [why for _, why in loaded if why is not None]
    if not clips:
        raise SystemExit(f"no clip has a usable monocular state: {Counter(x['reason'] for x in lost)}")
    return clips, lost, template


def conform_tally(clips: list[Clip], template) -> dict:
    """Per channel, how ``conform`` treated it across clips (``tokenizer_domain_read.conform_kind``), and channels the template does not carry."""
    from tokenizer_domain_read import conform_audit
    kinds, extra = Counter(), Counter()
    for clip in clips:
        audit = conform_audit(clip.raw, template)
        kinds.update(f"{name}:{kind}" for name, kind in audit["kinds"].items())
        extra.update(audit["extra"])
    return dict(kinds=dict(sorted(kinds.items())), dropped=dict(sorted(extra.items())))


def unconformed_plan(clips: list[Clip]) -> dict:
    """Whether conforming changes anything v1 receives; the form is read only if it does on some clip."""
    differ = [c.sign for c in clips if not np.array_equal(reader_input(c.raw), reader_input(c.conformed))]
    if differ:
        reason = f"conforming changed v1's input on {len(differ)} of {len(clips)} clips, so the form is read"
    else:
        reason = (f"not read: v1's input is byte-identical with and without conforming on all {len(clips)} clips (conform renames the frame and convention of valid channels and empties channels with no valid frame; "
                  "v1 reads body and hand rotations and their marks). A read could only repeat real_monocular: conforming costs v1 nothing")
    return dict(read=bool(differ), clips_compared=len(clips), clips_differing=len(differ), differing=differ, reason=reason)


# ---- forms -----------------------------------------------------------------------------------------------------------------------

def build_form(name: str, refs, clips: list[Clip], tokenizer) -> tuple[list[int], list, list, list[dict]]:
    """(record indices, states, the states motion is masked by, clips this form lost). A form's motion is measured on its own path's tracked frames."""
    from evaluate_text_to_motion import blanked
    every = list(range(len(refs)))
    if name == "real_two_view":
        return every, list(refs), list(refs), []
    if name == "blank":
        return every, blanked(refs), list(refs), []
    if name == "real_monocular":
        return [c.index for c in clips], [c.conformed for c in clips], [c.conformed for c in clips], []
    if name == UNCONFORMED:
        return [c.index for c in clips], [c.raw for c in clips], [c.raw for c in clips], []
    if name == "round_trip_monocular":
        kept, states, masks, lost = [], [], [], []
        for c in clips:
            try:
                states.append(T.cap_form("rt_nearest", c.conformed, tokenizer))
            except (ValueError, RuntimeError, KeyError, IndexError) as error:
                lost.append(lost_clip(c.sign, "tokenizer_path_failed", f"{type(error).__name__}: {error}"))
                continue
            kept.append(c.index)
            masks.append(c.conformed)
        if not kept:
            raise SystemExit(f"no clip passed the tokenizer path: {lost[:3]}")
        return kept, states, masks, lost
    raise SystemExit(f"unknown form {name}")


def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def check_smoke(n: int) -> None:
    if n > SMOKE_CLIPS:
        raise SystemExit(f"a smoke reads at most {SMOKE_CLIPS} clips, not {n}")


def run_lookups(args) -> int:
    """The lookup alone, from index objects: no state, no reader. Written once, so a job shows its unmatched clips before it spends anything."""
    from train_text_to_motion import make_store
    P.check_pins()
    check_smoke(args.smoke_train)
    out = Path(args.out_dir)
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    check_lookup_records(records, declared)
    rows = monocular_rows(store, [r["sign"] for r in records])
    found, lost = match_monocular(records, rows, declared)
    report = dict(kind=f"{PROBE}-lookups", report_only=True, declared_read=declared, list=described, clips=len(records), matched=len(found), lost=lost, reports_opened=sorted(MONOCULAR_REPORTS), code_digest=code_digest(),
                  signers=dict(sorted(Counter(records[f["index"]].get("signer_component") for f in found).items())), shards=dict(sorted(Counter(rows[f["sign"]]["shard"] for f in found).items())), states=[f["state_key"] for f in found])
    P.write_report_once(out / "lookups.json", report)
    print(json.dumps(dict(lookups=str(out / "lookups.json"), declared_read=declared, clips=len(records), matched=len(found), lost=len(lost), signers=report["signers"], reports_opened=len(MONOCULAR_REPORTS))), flush=True)
    return 0


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    check_smoke(args.smoke_train)
    torch.set_num_threads(args.threads)
    names = [f for f in args.forms.split(",") if f]
    unknown = [f for f in names if f not in FORMS]
    if unknown:
        raise SystemExit(f"unknown forms {unknown}")
    out_dir = Path(args.out_dir)
    stored = [n for n in names if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    names = [n for n in names if n not in stored]
    plan_path = out_dir / "monocular.json"
    if UNCONFORMED in names and plan_path.exists() and not json.loads(plan_path.read_bytes())["unconformed"]["read"]:      # a resume: already settled, not read
        names.remove(UNCONFORMED)
    if not names:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    check_lookup_records(records, declared)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    clips, lost = [], []
    if any(n in MONOCULAR_FORMS for n in names):
        clips, lost, template = open_monocular(store, records, refs, declared)
        plan = unconformed_plan(clips)
        if not plan_path.exists():
            P.write_report_once(plan_path, dict(kind=f"{PROBE}-monocular", report_only=True, declared_read=declared, code_digest=code_digest(), clips_loaded=len(clips), lost=lost, conform=conform_tally(clips, template), unconformed=plan))
        print(json.dumps(dict(monocular_clips=len(clips), lost=len(lost), unconformed=plan["reason"])), flush=True)
        if UNCONFORMED in names and not plan["read"]:
            names.remove(UNCONFORMED)
    tokenizer = L.load_tokenizer(store) if "round_trip_monocular" in names else None
    reader = P.load_reader(store, args)
    digest, place = code_digest(), P.environment()
    for name in names:
        started = time.time()
        kept, states, masks, form_lost = build_form(name, refs, clips, tokenizer)
        built = time.time()
        here_texts = [texts[i] for i in kept]
        raw = reader.read(states, here_texts, M.retrieval_candidates(len(here_texts), 1, seed=0), group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place, instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None) if name == "round_trip_monocular" else None,
                      signs=[signs[i] for i in kept], groups=[groups[i] for i in kept], references=here_texts, lost=(lost if name in MONOCULAR_FORMS else []) + form_lost, hypotheses=raw["hypotheses"], nll=np.round(raw["nll"], 5).tolist(),
                      seconds_over_reference=[round(len(s.timestamps) / max(len(refs[i].timestamps), 1), 5) for s, i in zip(states, kept)], valid_frame_share=[valid_frame_share(s) for s in states],
                      motion=L.motion_statistics(states, masks), hands=P.hand_statistics(states, masks), tracked=P.tracked_statistics(states),
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=P.peak_rss_bytes(), gpu_peak_reserved_bytes=P.gpu_peak_reserved_bytes())
        P.write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(clips=len(kept), bleu=M.bleu_ci_fast(raw["hypotheses"], here_texts)[0], build_seconds=report["build_seconds"], read_seconds=report["read_seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2), gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


def run_control(args) -> int:
    """Exit 3, voiding the read, unless real_two_view and blank reproduce the pinned stored report, through ``tok_probes``' canary and its limits. A smoke reads train clips, so it checks the pin and the stored report's shape only, and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    stored = json.loads(raw)
    if any(len(stored["forms"][n]["per_clip"]["nll"]) != len(stored["signs"]) for n in ("real", "blank")):
        raise SystemExit(f"{P.CANARY_REPORT}: no per-clip NLL for real and blank")
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in ANCHORS}
    if not read["real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; the pin and the stored report's shape are checked")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "real_two_view and blank")
    result = dict(P.canary_read(dict(real=read["real_two_view"], blank=read["blank"]), stored), code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT real_two_view and blank do not reproduce the stored report: {result['misses']}. The read is void: no monocular form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


# ---- shared clips and the panels -------------------------------------------------------------------------------------------------

def restrict(read: dict, signs) -> dict:
    """``read`` on ``signs``, in that order, every per-clip field with them."""
    where = {s: i for i, s in enumerate(read["signs"])}
    pick = [where[s] for s in signs]
    out = dict(read)
    for key in PER_CLIP:
        if key in read:
            out[key] = [read[key][i] for i in pick]
    out["motion"] = {k: [v[i] for i in pick] for k, v in read["motion"].items()}
    out["hands"] = {h: {k: [v[i] for i in pick] for k, v in row.items()} for h, row in read["hands"].items()}
    out["tracked"] = {k: [v[i] for i in pick] for k, v in read["tracked"].items()}
    return out


def shared_signs(reads: dict) -> list[str]:
    """The clips every form read, in real_two_view's order. Refuses a form holding a clip outside the list, or another reference for one."""
    base = reads["real_two_view"]
    text = dict(zip(base["signs"], base["references"]))
    for name, read in reads.items():
        for sign, reference in zip(read["signs"], read["references"]):
            if text.get(sign) != reference:
                raise SystemExit(f"{name}: {sign} is not this list's clip with this reference")
    common = set.intersection(*(set(r["signs"]) for r in reads.values()))
    return [s for s in base["signs"] if s in common]


def lost_rows(reads: dict, shared) -> dict:
    """Every clip outside the shared set, with the reason each form lost it. A clip gone without a reason refuses."""
    keep, by_clip = set(shared), {}
    for name, read in reads.items():
        for row in read.get("lost") or []:
            by_clip.setdefault(row["sign"], {})[name] = dict(reason=row["reason"], detail=row.get("detail"))
    unexplained = [s for s in reads["real_two_view"]["signs"] if s not in keep and s not in by_clip]
    if unexplained:
        raise SystemExit(f"{len(unexplained)} clips fall outside the shared set with no reason, for example {unexplained[0]}")
    reasons = Counter(reason for forms in by_clip.values() for reason in {r["reason"] for r in forms.values()})
    return dict(count=len(by_clip), by_reason=dict(sorted(reasons.items())), clips=[dict(sign=s, forms=by_clip[s]) for s in sorted(by_clip)])


def aliased(reads: dict) -> dict:
    """``tok_probes.form_panel`` reads real under the name real."""
    return {("real" if k == "real_two_view" else k): v for k, v in reads.items()}


def panel(reads: dict, name: str) -> dict:
    out = P.form_panel(aliased(reads), "real" if name == "real_two_view" else name)
    out["paired"]["real_two_view"] = out["paired"].pop("real")
    out["identical_to_real_two_view"] = out.pop("identical_to_real")
    out["valid_frame_share"] = M.bootstrap_mean(reads[name]["valid_frame_share"])
    out["clips"] = len(reads[name]["signs"])
    return out


SIGNER_ROW = ("clips", "bleu", "share_of_gap", "likelihood_minus_blank", "valid_frame_share", "tracked_share", "seconds_over_reference", "paired")


def signer_rows(reads: dict) -> dict:
    """Per signer, the same panels on that signer's shared clips."""
    groups = dict(zip(reads["real_two_view"]["signs"], reads["real_two_view"]["groups"]))
    out = {}
    for signer in sorted(set(groups.values())):
        signs = [s for s in reads["real_two_view"]["signs"] if groups[s] == signer]
        here = {n: restrict(r, signs) for n, r in reads.items()}
        panels = {n: panel(here, n) for n in reads}
        out[signer] = {n: {k: p[k] for k in SIGNER_ROW} for n, p in panels.items()}
    return out


def same_path(reads: dict, stored: dict | None) -> dict:
    """Differences and shares under one joint bootstrap over the shared clips (n 2,000, seed 0), on BLEU and likelihood minus blank. A row whose form was not read is None."""
    forms = dict(reads, **({"stored_rt_nearest": stored} if stored else {}))
    refs = reads["real_two_view"]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = {f: M.delta_nll(blank, np.asarray(r["nll"])[:, 0]) for f, r in forms.items()}
    stats = {f: M.bleu_statistics(r["hypotheses"], refs) for f, r in forms.items()}

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
        return dict(real_two_view_gap=difference("real_two_view", "blank"), real_monocular_gap=difference("real_monocular", "blank"), monocular_gap_over_two_view_gap=share("real_monocular", "real_two_view"),
                    real_monocular_minus_real_two_view=difference("real_monocular", "real_two_view"), round_trip_monocular_minus_real_monocular=difference("round_trip_monocular", "real_monocular"),
                    round_trip_monocular_share_of_monocular_gap=share("round_trip_monocular", "real_monocular"), round_trip_monocular_share_of_two_view_gap=share("round_trip_monocular", "real_two_view"),
                    stored_rt_nearest_share_of_two_view_gap=share("stored_rt_nearest", "real_two_view"), round_trip_monocular_minus_stored_rt_nearest=difference("round_trip_monocular", "stored_rt_nearest"))
    return dict(bleu=block("bleu"), likelihood_minus_blank=block("likelihood_minus_blank"), bootstrap=dict(n=BOOT, seed=0), clips=len(refs))


def load_stored_rt_nearest(store) -> tuple[dict, str]:
    """(V1's stored rt_nearest form on all of D20, the file's hash). Its bytes were not read when this was written, so it is identified by what it holds: V1's kind, form, code, list, instrument and tokenizer, and its BLEU and interval recomputed to the plan's figures."""
    raw = store.get(STORED_RT)
    form = json.loads(raw)
    differs = [what for what, ok in (("kind", form.get("kind") == "tok-v1-form"), ("form", form.get("form") == "rt_nearest"), ("declared_read", form.get("declared_read") is True), ("report_only", form.get("report_only") is True),
                                     ("code_digest", str(form.get("code_digest")).startswith(STORED_RT_DIGEST)), ("list", (form.get("list") or {}).get("sha256") == L.D20_SHA), ("clips", len(form.get("signs") or ()) == L.D20_CLIPS),
                                     ("instrument", (form.get("instrument") or {}).get("sha256") == L.TRANSLATOR_SHA), ("tokenizer", (form.get("tokenizer") or {}).get("sha256") == L.TOKENIZER_SHA)) if not ok]
    if differs:
        raise SystemExit(f"{STORED_RT}: not V1's stored read: {differs}")
    bleu, ci = M.bleu_ci_fast(form["hypotheses"], form["references"])
    if bleu != STORED_RT_BLEU or ci != STORED_RT_CI:
        raise SystemExit(f"{STORED_RT}: reads {bleu} {ci}, not the plan's {STORED_RT_BLEU} {STORED_RT_CI}")
    return form, hashlib.sha256(raw).hexdigest()


def stored_block(form: dict, sha: str, reads: dict, place: dict, declared: bool) -> tuple[dict, dict | None]:
    """(the stored form's row, the form on the shared clips). A smoke reads other clips, so the form is checked and not compared."""
    bleu, ci = M.bleu_ci_fast(form["hypotheses"], form["references"])
    row = dict(key=STORED_RT, sha256=sha, code_digest=form["code_digest"], environment=form.get("environment"), same_environment=form.get("environment") == place, all_500=dict(bleu=bleu, ci95=ci, clips=len(form["signs"])))
    if not declared:
        return dict(row, shared=None, note="a smoke reads train clips: the stored form is checked, not compared"), None
    signs = reads["real_two_view"]["signs"]
    text = dict(zip(form["signs"], form["references"]))
    if any(text.get(s) != r for s, r in zip(signs, reads["real_two_view"]["references"])):
        raise SystemExit(f"{STORED_RT}: another reference for a shared clip, or a clip it did not read")
    kept = restrict(form, signs)
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = M.delta_nll(blank, np.asarray(kept["nll"])[:, 0])
    bleu, ci = M.bleu_ci_fast(kept["hypotheses"], kept["references"])
    trip = reads["round_trip_monocular"]
    row["shared"] = dict(clips=len(signs), bleu=bleu, ci95=ci, likelihood_minus_blank=M.bootstrap_mean(dnll),
                         round_trip_monocular_minus_stored=dict(bleu=M.paired_bleu_fast(M.bleu_statistics(trip["hypotheses"], trip["references"]), M.bleu_statistics(kept["hypotheses"], kept["references"])),
                                                                likelihood_minus_blank=M.paired_mean_difference(M.delta_nll(blank, np.asarray(trip["nll"])[:, 0]), dnll)))
    return row, kept


def overlaps(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def prestated_case(difference: float, paired_ci, two_view_ci, monocular_ci) -> str | None:
    """Which of the header's cases these numbers fall in: a, b, c, or None for none of them. A label on a reading stated before the read; nothing branches on it."""
    if difference < -LARGE:
        return "b"
    if difference > LARGE or (difference > 0 and paired_ci[0] > 0):
        return "c"
    return "a" if overlaps(two_view_ci, monocular_ci) else None        # within 0.5 by here


def reading_block(panels: dict, declared: bool) -> dict:
    two_view, monocular = panels["real_two_view"], panels["real_monocular"]
    paired = monocular["paired"]["real_two_view"]
    a, b = two_view["bleu"]["ci95"], monocular["bleu"]["ci95"]
    facts = dict(real_two_view=two_view["bleu"], real_monocular=monocular["bleu"], difference=paired["bleu"], likelihood_difference=paired["likelihood_minus_blank"], intervals_overlap=overlaps(a, b), monocular_higher=paired["bleu"]["difference"] > 0)
    return dict(prestated=PRESTATED, facts=facts, case=prestated_case(paired["bleu"]["difference"], paired["bleu"]["ci95"], a, b) if declared else None, case_note="a label on the reading stated in the script's header; nothing branches on it" if declared else "a smoke: five train clips carry no reading")


def load_reads(out_dir: Path) -> tuple[dict, dict]:
    """(every form read, what the monocular step recorded). All one list, instrument, kind of read, code and environment; the unconformed form is present exactly when the record says it was read."""
    plan_path = out_dir / "monocular.json"
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} is missing")
    plan = json.loads(plan_path.read_bytes())
    names = [n for n in FORMS if n != UNCONFORMED or plan["unconformed"]["read"]]
    reads, first = {}, None
    for name in names:
        path = out_dir / f"form-{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} is missing")
        read = json.loads(path.read_bytes())
        if read.get("form") != name:
            raise SystemExit(f"{path} holds form {read.get('form')!r}")
        signature = (read["instrument"]["sha256"], read["declared_read"], read["list"]["list"])
        if first is None:
            first = signature
        elif signature != first:
            raise SystemExit(f"{path}: another instrument, another list, or a smoke mixed with a declared read")
        reads[name] = read
    if not plan["unconformed"]["read"] and (out_dir / f"form-{UNCONFORMED}.json").exists():
        raise SystemExit(f"{UNCONFORMED} was read although the record says it was not")
    digest, _ = P.check_same_run(reads, "the forms")
    if plan.get("code_digest") != digest or plan.get("declared_read") != reads["real_two_view"]["declared_read"]:
        raise SystemExit(f"{plan_path}: another run's record")
    return reads, plan


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    out_dir = Path(args.out_dir)
    all_reads, plan = load_reads(out_dir)
    first = all_reads["real_two_view"]
    declared = first["declared_read"]
    digest, place = first["code_digest"], first["environment"]
    control = None
    if declared:
        path = out_dir / "control.json"
        control = json.loads(path.read_bytes()) if path.exists() else None
        if control is None or control.get("reproduced") is not True or control.get("code_digest") != digest or control.get("environment") != place:
            raise SystemExit(f"{path}: no passed control of this code and environment; the read is void")
    shared = shared_signs(all_reads)
    if not shared:
        raise SystemExit("the forms share no clip")
    reads = {n: restrict(r, shared) for n, r in all_reads.items()}
    stored_form, stored_sha = load_stored_rt_nearest(store)
    stored_row, stored_kept = stored_block(stored_form, stored_sha, reads, place, declared)
    panels = {n: panel(reads, n) for n in reads}
    lookups_path = out_dir / "lookups.json"
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=declared, smoke=None if declared else "SMOKE on c02 train clips: these numbers are plumbing, not a read", list=first["list"], instrument=first["instrument"], tokenizer=reads["round_trip_monocular"]["tokenizer"],
                  code_digest=digest, environment=place, disclosures=DISCLOSURES, reading=reading_block(panels, declared),
                  clips=dict(list=len(first["signs"]), shared=len(shared), outside_shared=lost_rows(all_reads, shared)), forms=panels, per_signer=signer_rows(reads), same_path=same_path(reads, stored_kept), stored_rt_nearest=stored_row,
                  monocular=dict(conform=plan["conform"], unconformed=plan["unconformed"], clips_loaded=plan["clips_loaded"]), lookups=json.loads(lookups_path.read_bytes()) if lookups_path.exists() else None,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  checks=dict(blank_identical_to_real_two_view=panels["blank"]["identical_to_real_two_view"], control=control),
                  resources={n: {k: r[k] for k in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for n, r in all_reads.items()})
    L.write_once(store, args.output_key, args.output_file, report)
    summary = {n: dict(bleu=p["bleu"]["bleu"], ci95=p["bleu"]["ci95"], gap_share=p["share_of_gap"]["bleu"], likelihood_minus_blank=p["likelihood_minus_blank"]["mean"], valid_frame_share=p["valid_frame_share"]["mean"]) for n, p in panels.items()}
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, shared=len(shared), of=len(first["signs"]), lost=report["clips"]["outside_shared"]["by_reason"], forms=summary, reading=report["reading"]["facts"]["difference"], case=report["reading"]["case"], stored_rt_nearest=stored_row["all_500"])), flush=True)
    return 0


def run_check(args) -> int:
    """Exit 3 on broken plumbing: not report_only, blank reading as real, a declared read without its passed control, or no shared clip. It judges no form."""
    report = json.loads(Path(args.report).read_text())
    checks, problems = report["checks"], []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    if checks["blank_identical_to_real_two_view"] >= 1:
        problems.append("blank reads exactly as real_two_view")
    if report.get("declared_read") and not (checks.get("control") or {}).get("reproduced"):
        problems.append("a declared read without its passed control")
    if not report["clips"]["shared"]:
        problems.append("no shared clip")
    print(json.dumps(dict(problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    lookups = sub.add_parser("lookups", help="index objects only: which clips have a monocular state")
    lookups.add_argument("--store", default="r2")
    lookups.add_argument("--out-dir", required=True)
    lookups.add_argument("--smoke-train", type=int, default=0, help=f"this many c02 train clips, at most {SMOKE_CLIPS}, not D20")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--resume", action="store_true", help="leave out forms already read in --out-dir")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"read this many c02 train clips, at most {SMOKE_CLIPS}, not D20")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="real_two_view and blank against the pinned stored report; exit 3, voiding the read, otherwise")
    control.add_argument("--store", default="r2")
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(preflight=run_preflight, lookups=run_lookups, read=run_read, control=run_control, merge=run_merge, check=run_check, upload=P.run_upload)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
