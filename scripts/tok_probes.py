"""Tokenizer probes B1, B2, B4, B5, B6 and the seam ablation C0 (``reports/post_s2_plan.md``, sections 4 and 5).

**Report only.** Nothing here trains a network, declares a rule or decides anything: the merged report says
``report_only`` and carries no pass, bar or verdict. The plan's reading of each probe is its own and stays there.

**What it reads.** ``read``: D20 only, pinned by SHA-256 through ``s2_lib.load_read_list``, which refuses any other key
unread (C20, the 1,000, fresh clips), any non-val clip and c00. No option names another list. Instrument v1 and its pins
as in ``s0_ceilings.py``; no option names another instrument. ``--smoke-train N`` reads N How2Sign c02 train clips instead,
and every output then says ``declared_read`` false. ``fit``: How2Sign c02 train clips only. ``words``: the gate halves of
RIT and ASLLVD in recogniser v3d's own data definition (partition ``gate:<corpus>``, never ``select:``, never test); its
``--smoke-train N`` reads N *train* words of each corpus, with no gate clip.

**The path every decoded form takes.** ``tokenizer_recognisability_check.round_trip``, no gains, on the CPU: the
reconstruction on frames the clip observes, the neutral pose on the others, **and every frame marked valid**. Real clips
mark untracked frames invalid, and v1 reads an observed flag per point. So a round trip differs from real in two ways,
values and validity, and only the first is the decoder's. Hence:

- ``identity``: the same path with no encoder and no decoder (real values, neutral fill, all valid). What the path itself
  costs. ``identity_masked``: its values under real's validity. A decoder that returns real reads as this, so its
  hypotheses should equal real's; ``checks`` reports the share and ``check`` alerts when it is not 1.
- ``round_trip``: d2 as every earlier read. ``round_trip_masked``: its values on real's tracked frames under real's
  validity, so it differs from real in values only.

**B1.** ``no_quant``, ``no_quant_masked``: d2's encoder into ``decode_latents`` with the continuous latents, no codes.

**B2** (``rt_only_<part>``: this part round trip, the rest real; ``real_only_<part>``: the reverse; both under real's
validity, so ``real_only`` of nothing would be real). Parts: ``body`` (the body channel, which holds both wrist rotations
and the chain that places the wrists), ``left``, ``right`` (the left_hand and right_hand channels: finger joints only),
``fingers`` (both of those), ``wrist`` (the body channel's LeftHand and RightHand rotations only), ``arms`` (the body
channel without them; this is what places the wrist point in signing space). Compare with ``round_trip_masked``.

**B4** (each through the round-trip path). ``q_<part><Q>``: that part's latents quantised with its first Q stages, the
others with all of d2's 8, Q in 1, 2, 4, part in body, left, right, hands (both), face. Face counts on v1 although v1 is
blind to the face channels, because one decoder reads all parts' latents. ``hands10``, ``hands12``: both hands' 8 stages
plus 2 or 4 post-hoc k-means stages fitted on their residuals. ``h8b4f2``, ``h12b6f4``: the plan's V3-H candidates.

**B5** (each through the round-trip path, every part, 8 stages). ``k<K>``, K in 256, 1024, 4096: post-hoc residual k-means
codebooks. ``k512`` is a control that the plan's list leaves out: it is d2's own size, so K's effect can be separated from
post hoc against learned. ``fsq10``: post-hoc residual FSQ with levels (8, 6, 5, 5), 1,200 codes, 10.23 bits per stage: per
stage and part, a PCA to 4 dimensions and a symmetric uniform grid bounded by the 99.5th percentile of each projection's absolute
value (the per-stage scale stands in for the invertible layer norm of 2508.15860). Its equal-bits comparator is ``k1024`` (10 bits). All these codebooks meet an
encoder and a decoder trained for d2's, so they read a lower bound on a retrained quantiser of that kind, and **FSQ more so:**
d2's latents have 128 dimensions and were never shaped to sit on a 4-dimensional grid, so 8 stages of PCA keep at most 32 of them.
A k-means book has no such limit. So fsq10 cannot close V3-F: below k1024 does not show that FSQ is worse.

**B6.** ``words``: v3d's top-1 and top-5 on each decoded form on RIT and ASLLVD words. **The plan says "each surviving form"
and declares no survival rule**, so by default every decoded form is read and nothing is filtered; a reader can filter later,
and ``--forms`` narrows the list, which the report records. Seam forms have no word read. v3d trained on d2 copies (recon_p
0.5, with gains), so it favours d2, and the gate's decoded read used gains; these forms have none.

**Seam ablation C0** (real frames only, no tokenizer). **How2Sign has no sign boundaries:** its manifest records hold no
gloss and no spans (0 of 24,496, checked 2026-11-01), as the plan says. So this uses section 5's declared fallback, and every
form report says so: cuts at wrist-activity minima at each sentence's word count.

**Composition density, the forms to read.** J = (r10's entries for this sentence, ``round15_compose_eval.stage1_selection``) − 1, the
same junctions per sentence as composition (mean entries 9.09, median 7 on D20). A sentence with one entry has no junction and stays
real; so do D20's 6 sentences with none (composition renders those blank, so this is milder there), and both are counted.

- ``seam_cmin<S>``: J windows centred on the J lowest activity minima (at least T / (2(J + 1)) frames apart; when too few minima exist,
  midpoints of the largest gaps, counted as ``filled``), S% of the clip's frames in all (S in 30, 45), each replaced by
  ``compositional.bridge`` from the frame before it to the frame after it. ``seam_crnd<S>``: the same lengths at random positions.
- ``seam_csq196_min``, ``seam_csq196_rnd``: segments squeezed 1.96× by ``compositional.time_scale`` with bridges filling the length, at
  the minima or at random cuts.
- ``seam_cval45``: ``seam_cmin45``'s values under real's validity. ``bridge`` marks a frame valid if either end is, so the other forms
  also heal tracking, which helps the read; this one cannot. A joint real tracks inside a window whose ends are both untracked keeps
  its real value, since no bridge exists for it.
- ``seam_cmax45``: windows on activity maxima, not minima. Minima can sit on holds, where a bridge replaces little; this brackets that.

**Word-count density, sensitivity only.** ``seam_min<S>``, ``seam_rnd<S>``, ``seam_fix<S>`` (S in 15, 30, 45; ``fix`` is even spacing, the
task's "fixed intervals", which section 5 does not name), ``seam_sq196_min``, ``seam_sq196_rnd``: J = max(1, ``normalise``'s tokens − 1).
That gives about 18.4 junctions and bridges of 1.2-3.4 frames, against composition's 8.09 and 2.6-7.8, and linear-bridge error grows
with span squared, so these are about 4× too mild to bound composition.

Every form's report gives its tracked share, so a heal is visible. Word echo (section 5) is not computed: it needs
r10's selection, and the hypotheses are stored for it.

**Per form** (``merge``): BLEU, likelihood minus blank (teacher-forced reference NLL, ``s0_ceilings.candidate_nll``), loop share
(``eval.loop_share``: a whitespace 3-gram occurring at least 3 times, overlaps counted), length ratio, wrist speed, tracked share,
and hand variation and speed (``motion_metrics.compare``'s own rule, per clip, on real's tracked frames), each with a
bootstrap interval over clips and paired against real and blank, the share of real's gap, and the named comparators (``comparators_of``). Blank has no tracked motion, so
its motion pairs are null, with the reason.

    python scripts/tok_probes.py fit --out-dir W/fit --device cuda
    python scripts/tok_probes.py read --out-dir W/forms --fit-dir W/fit --forms real,blank --instrument-cache W/i
    python scripts/tok_probes.py words --out-dir W/forms --fit-dir W/fit
    python scripts/tok_probes.py merge --out-dir W/forms --output-file W/report.json
    python scripts/tok_probes.py check --report W/report.json

Outputs are write-once and written atomically. Every report names its code (``eval.provenance.code_digest``: ``scripts/*.py``, ``signtranslator/**/*.py`` and
this job's shell file, which also catches a dirty tree; a pod has no ``.git``) and its environment (python, torch, transformers, tokenizers, GPU), and ``merge`` and
``restore`` refuse a mix; code that cannot be named voids the read.

**The canary.** Right after real and blank are read, against a SHA-pinned stored D20 report that read them through the same path (B0, ``s0_ceilings``):
BLEU within 0.05 of the plan's figures, at least 0.98 of hypotheses identical, and a mean absolute per-clip NLL difference of at most 0.001. On c02 train
clips on the CPU the batch group moves NLL by at most about 1e-6 and the candidate count by less, so 0.001 is not a batch effect; CPU against GPU is unmeasured.
A miss stops the job before any other form and voids the read: a void record is stored under this code's digest, this code never reads again, and a rerun
needs new code and a recorded cause (``restore --rerun-cause``).

``restore`` brings back one failed run's D20 forms and fit (the latest run, whole or not at all: same code, same environment), so a retry reads only what is
missing, and never real or blank from a run whose canary did not pass.

This script holds no decision rule. ``scripts/tok_probes_rule.py`` reads a report and applies a rule, separately. The job is ``runpod/fleet/tok_probes.sh``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import sys
import tempfile
import time
import warnings
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import s2_lib as L  # noqa: E402
from signtranslator.eval import provenance, s0_metrics as M  # noqa: E402
from signtranslator.eval.loop_share import loop_flags  # noqa: E402
from signtranslator.eval.provenance import environment  # noqa: E402

FPS = L.FPS
PROBE = "tok-probes"
JOB_FILE = "runpod/fleet/tok_probes.sh"
# every pin, stated here too: check_pins refuses a run whose shared constants have moved
PINS = dict(D20_KEY="acquisition/2026-10-17-v1/youtube-asl-v1/round20/d20.json",
            D20_SHA="d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe",
            EVAL_MIX="acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json",
            EVAL_MIX_SHA="ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a",
            TOKENIZER="training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt",
            TOKENIZER_SHA="021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de",
            TRANSLATOR_RUN="back-translator-kp2d-uwb-ft-v1",
            TRANSLATOR_MANIFEST_SHA="004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd",
            TRANSLATOR_SHA="2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8")
V3D_RUN = "sign-recognizer-v3d"
V3D_BEST_SHA = "3b50a8168e9365984a1f5eb42dd1525466e700bae4036c9bb485a23704cd7ba0"
V3D_CHECKPOINT_SHA = "0c6c10e232037e869f3983ed6df16df44b3e0a092cbfa34d40ca744cbf00a4e2"
V3D_MAX_SECONDS, V3D_MIN_HAND = 8., .2                     # v3d's run.json: max_seconds, min_hand_fraction
GLOSS_MAP = "acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json"
GLOSS_MAP_SHA = "db98b5915bf6fd51fb11b9401b43e66d55225fa54a9bd12dac253b3e26b15ed9"
WORD_CORPORA = (("acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/rit-sign-manifest-v2/manifest.json",
                 "40bb30f38addfb0c42d5684a1ab097a781b0c59290e24a44f06c90395f235e92"),
                ("acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/asllvd-sign-manifest-v1/manifest.json",
                 "86df1dfd00df235c7df2b0feaa021e618d9bfc591e13810153683475d9961842"))
CONFORM_TEMPLATE = f"{L.EVAL_MIX}@{L.EVAL_MIX_SHA}:{L.H2S}"  # v3d's run.json: conform_template

HANDS = ("left_hand", "right_hand")
WRIST_POINTS = (4, 5)                                       # keypoints2d.ARM: LeftWrist, RightWrist
HAND_ROOT_POINTS = (6, 27)                                  # keypoints2d.POINTS: each hand's wrist point, valid with its whole hand
SWAP_PARTS = ("body", "left", "right", "fingers", "wrist", "arms")
STAGE_PARTS = dict(body=("body",), left=("left_hand",), right=("right_hand",), hands=HANDS, face=("face",))
STAGE_GRID = (1, 2, 4)
FSQ_LEVELS = (8, 6, 5, 5)
KMEANS = (256, 512, 1024, 4096)
EXTENSION_STAGES = 4
SHARES = (15, 30, 45)
SQUEEZE = 1.96

PATH_FORMS = ("identity", "identity_masked", "round_trip", "round_trip_masked")
B1_FORMS = ("no_quant", "no_quant_masked")
B2_FORMS = tuple(f"{way}_{p}" for way in ("rt_only", "real_only") for p in SWAP_PARTS)
B4_FORMS = tuple(f"q_{p}{q}" for p in STAGE_PARTS for q in STAGE_GRID) + ("hands10", "hands12", "h8b4f2", "h12b6f4")
B5_FORMS = tuple(f"k{k}" for k in KMEANS) + ("fsq10",)
SEAM_SPECS = {f"seam_{r}{s}": dict(density="words", position=r, share=s) for r in ("min", "rnd", "fix") for s in SHARES}
SEAM_SPECS.update(seam_sq196_min=dict(density="words", position="min", squeeze=True), seam_sq196_rnd=dict(density="words", position="rnd", squeeze=True))
SEAM_WORD_FORMS = tuple(SEAM_SPECS)
SEAM_COMPOSITION_SPECS = {f"seam_c{r}{s}": dict(density="composition", position=r, share=s) for r in ("min", "rnd") for s in (30, 45)}
SEAM_COMPOSITION_SPECS.update(seam_csq196_min=dict(density="composition", position="min", squeeze=True),
                              seam_csq196_rnd=dict(density="composition", position="rnd", squeeze=True),
                              seam_cval45=dict(density="composition", position="min", share=45, values_only=True),
                              seam_cmax45=dict(density="composition", position="max", share=45))
SEAM_COMPOSITION_FORMS = tuple(SEAM_COMPOSITION_SPECS)
SEAM_SPECS.update(SEAM_COMPOSITION_SPECS)
SEAM_FORMS = SEAM_WORD_FORMS + SEAM_COMPOSITION_FORMS
DECODED_FORMS = PATH_FORMS + B1_FORMS + B2_FORMS + B4_FORMS + B5_FORMS
FORMS = ("real", "blank") + DECODED_FORMS + SEAM_FORMS
FORM_ID = {name: i for i, name in enumerate(FORMS)}
FIT_FORMS = ("hands10", "hands12", "h12b6f4") + B5_FORMS    # these read the fit
SEAM_SEED, FIT_SEED = 20261101, 20261102
FIT_CLIPS, FIT_STEPS, FIT_ITERATIONS = 4000, 500_000, 15    # 4,000 of c02's 17,945 train clips, mean 6.7 s: about 640k steps

D20_ENTRIES = dict(mean=9.09, median=7, empty=6)            # r10's stage-1 entries on D20 (reports/round20_dev.md)
IDENTITY_TOLERANCE = .01                                    # identical hypotheses: stored real reads of one list differ in 1 of 500
# stored D20 reads (phase-a: real 4.495 and 4.493, blank 1.6802; .002 is one hypothesis). B0 read real and blank on D20 through this path, at group 4
CANARY = dict(real=4.493, blank=1.68, bleu_tolerance=.05, identical=.98, nll=.001)
CANARY_REPORT = f"{L.O}/s2/b0-w14b-d20-v1.json"
CANARY_REPORT_SHA = "30ddbe7d1a7eb94b5dd40e28ae0aa8228d584efa8ed3d4632206e8d9b4e32665"
BOUNDARY_FIELDS = ("gloss", "spans", "word_spans", "alignment", "glosses")
BOUNDARY_RULE = dict(sign_boundaries="How2Sign records carry none (no gloss, no spans; the plan's section 5 says so too)",
                     used="section 5's declared fallback: wrist-activity minima at each sentence's word count",
                     composition_density="seam_c*: r10's stage-1 entries − 1, composition's junctions. Read these",
                     word_count="seam_min, rnd, fix, sq196: compositional.normalise tokens, stop words included. Sensitivity only: about "
                                "18.4 junctions with short bridges, about 4x too mild to bound composition")
READING = dict(path="identity against real: what the round-trip path itself costs. round_trip against round_trip_masked: the "
                    "validity share of the loss.",
               b1="no_quant near round_trip: the plan reads the decoder as the limit; near identity: the quantiser. Read "
                  "against identity, not only real, since a perfect decoder on this path reads identity.",
               b2="rt_only_<part> below real, and real_only_<part> above round_trip_masked: which part holds the loss.",
               b4="q_<part><Q> against round_trip: spare rate per part. hands10, hands12 against round_trip: extra hand rate.",
               b5="k<K> against k512: K. fsq10 against k1024: the quantiser's form at about equal bits, biased against FSQ (d2's "
                  "128-dimensional latents through a 4-dimensional grid). k512 against round_trip: post hoc against learned.",
               b6="v3d top-5 decoded against real, per corpus; v3d favours d2.",
               seam="seam_c* at composition density. Against real: what perfect transitions could buy, within limits. A keep does not "
                    "clear composition's seams, since the bridge endpoints come from one continuous motion, not stitched words. A fall "
                    "may overstate, since windows can sit on holds or moves (seam_cmax45 brackets it). seam_cval45 keeps real's validity; "
                    "the others heal tracking. The word-count forms are sensitivity only.")
DISCLOSURES = dict(
    fsq="Post-hoc FSQ is biased against FSQ (the review's simulated residual: 20.0 against 3.7 for the left hand, 59.5 against 7.6 for "
        "the body; not re-run here), so it cannot close V3-F.",
    v3d="v3d trained on d2 copies with gains, so it favours d2; these forms have no gains.",
    b6="B6 has only 119 RIT and 197 ASLLVD clips, which cannot resolve the gate's 0.02.",
    b2="Parts do not add up: rt_only_arms also moves the hand-local points (a hand's frame hangs from its forearm) and the shoulder "
       "normalisation (Pelvis and Chest set the shoulders), and hypotheses are not additive in their inputs.",
    composition_density="Six D20 sentences have no entry and stay real at composition density (composition renders them blank); "
                        "sentences with one entry have no junction.")

NOT_DECIDED = "report only: the plan's readings are its own; no rule, bar or verdict is computed here"


# ---- pins and guards -------------------------------------------------------------------------------------------------

def check_pins() -> None:
    """Refuses before anything is read when a shared pin differs from this script's."""
    moved = [k for k, v in PINS.items() if getattr(L, k) != v]
    if moved:
        raise SystemExit(f"pins moved in s2_lib: {moved}; this probe reads D20 {PINS['D20_SHA'][:8]} and instrument v1 only")


def check_sha(payload: bytes, sha: str, what: str) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha:
        raise SystemExit(f"{what}: SHA-256 {digest}, expected {sha}")


def check_read_list(records, declared: bool) -> None:
    if declared:
        L.check_read_records(records)
    else:
        L.check_train_records(records, ("c02",))


def records_with_boundaries(records) -> int:
    return sum(1 for r in records if any(r.get(k) for k in BOUNDARY_FIELDS))


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    """``eval.provenance.code_digest`` with this job's shell file, which holds pins: a pod has no ``.git``, and this also catches a dirty tree. Code that
    cannot be named voids the read."""
    try:
        return provenance.code_digest(extra_files=(JOB_FILE,))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


def check_same_run(reports: dict, what: str) -> tuple[str, dict]:
    """(code digest, environment) of reports that must be one run's; anything else refuses, and a report without a digest voids the read."""
    digests = {r.get("code_digest") for r in reports.values()}
    if None in digests:
        raise SystemExit(f"{what}: a report that does not name its code; the read is void")
    if len(digests) != 1:
        raise SystemExit(f"{what}: different code digests: {sorted(digests)}")
    environments = {json.dumps(r.get("environment"), sort_keys=True) for r in reports.values()}
    if len(environments) != 1:
        raise SystemExit(f"{what}: different environments: {sorted(environments)}")
    return digests.pop(), json.loads(environments.pop())


def gpu_peak_reserved_bytes() -> int | None:
    """``torch.cuda.max_memory_reserved`` of this process; None without a GPU."""
    import torch
    return int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None


def write_report_once(path: Path, report: dict) -> None:
    """Write-once and atomic: a partial upload or a kill never leaves half a file under this name."""
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(report, indent=1, sort_keys=True, default=float).encode())
    temporary.replace(path)


def peak_rss_bytes() -> int:
    """This process's peak resident memory (ru_maxrss: bytes on macOS, kilobytes on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


# ---- post-hoc quantisers (pure, on (N, D) float tensors) -------------------------------------------------------------

def _nearest(x, codes, *, chunk: int = 16384):
    """(index, squared distance) of each row of ``x`` to ``codes``, in chunks."""
    import torch
    index, distance = [], []
    norms = codes.pow(2).sum(-1)
    for start in range(0, len(x), chunk):
        row = x[start:start + chunk]
        d = row.pow(2).sum(-1, keepdim=True) - 2 * row @ codes.T + norms[None]
        best = d.min(-1)
        index.append(best.indices)
        distance.append(best.values.clamp_min(0.))
    return torch.cat(index), torch.cat(distance)


def kmeans(x, k: int, *, iterations: int, seed: int):
    """(k, D) centres by Lloyd's iterations from k distinct data rows; a code left empty is re-seeded with the rows farthest
    from their centres. Fewer rows than k (a smoke): the rows themselves, repeated."""
    import torch
    n = len(x)
    if n <= k:
        return x[torch.arange(k, device=x.device) % n].clone()
    generator = torch.Generator().manual_seed(seed)
    centres = x[torch.randperm(n, generator=generator)[:k].to(x.device)].clone()
    for _ in range(iterations):
        centres = _lloyd_step(x, centres)
    return centres


def _lloyd_step(x, centres):
    """One assignment and mean; each empty code moves to the row farthest from its centre."""
    import torch
    index, distance = _nearest(x, centres)
    counts = torch.bincount(index, minlength=len(centres))
    sums = torch.zeros_like(centres).index_add_(0, index, x)
    alive = counts > 0
    out = centres.clone()
    out[alive] = sums[alive] / counts[alive][:, None].to(x.dtype)
    dead = int((~alive).sum())
    if dead:
        out[~alive] = x[distance.topk(dead).indices]
    return out


def perplexity(index, k: int) -> float:
    import torch
    p = torch.bincount(index, minlength=k).double()
    p = p[p > 0] / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def fsq_fit(x, levels=FSQ_LEVELS) -> dict:
    """One FSQ stage on rows ``x``: mean, the top len(levels) principal directions and each one's bound (the 0.5-99.5% range of
    the projections' absolute values, floored)."""
    import torch
    mean = x.mean(0)
    centred = x - mean
    _, vectors = torch.linalg.eigh(centred.T @ centred / max(len(x) - 1, 1))
    basis = vectors[:, -len(levels):].flip(1)
    projected = centred @ basis
    bound = torch.quantile(projected.abs(), .995, dim=0).clamp_min(1e-6)
    return dict(mean=mean, basis=basis, bound=bound)


def fsq_stage(x, stage: dict, levels=FSQ_LEVELS):
    """(reconstruction of ``x``, per-row integer grid index) on this stage's grid: cell centres at
    -bound + (i + 0.5) * 2 * bound / L."""
    import torch
    sizes = torch.as_tensor(levels, device=x.device, dtype=torch.long)
    size = sizes.to(x.dtype)
    z = (x - stage["mean"]) @ stage["basis"]
    bound = stage["bound"]
    cell = ((z + bound) / (2 * bound) * size).floor().clamp(min=torch.zeros_like(size), max=size - 1)
    centre = -bound + (cell + .5) * 2 * bound / size
    weights = torch.cumprod(torch.cat((torch.ones(1, device=x.device, dtype=torch.long), sizes[:-1])), 0)
    return stage["mean"] + centre @ stage["basis"].T, (cell.long() * weights).sum(-1)


def apply_kmeans_stages(z, books):
    """Σ e_q of (…, D) latents ``z`` through residual stages ``books`` (each (K, D))."""
    import torch
    flat = z.reshape(-1, z.shape[-1])
    residual, total = flat, torch.zeros_like(flat)
    for book in books:
        index, _ = _nearest(residual, book)
        e = book[index]
        total = total + e
        residual = residual - e
    return total.reshape(z.shape)


def apply_fsq_stages(z, stages, levels=FSQ_LEVELS):
    import torch
    flat = z.reshape(-1, z.shape[-1])
    residual, total = flat, torch.zeros_like(flat)
    for stage in stages:
        e, _ = fsq_stage(residual, stage, levels)
        total = total + e
        residual = residual - e
    return total.reshape(z.shape)


# ---- the fit (c02 train only) ----------------------------------------------------------------------------------------

def fit_records(store, clips: int) -> list[dict]:
    """A fixed draw of How2Sign c02 train clips. ``train_records`` refuses c00 and returns train only; checked again here."""
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    pool = L.train_records(manifest, ("c02",))
    records = random.Random(FIT_SEED).sample(pool, min(clips, len(pool)))
    L.check_train_records(records, ("c02",))
    return records


def collect_latents(store, tokenizer, records, device: str, *, max_steps: int, chunk: int = 64) -> dict:
    """{part: (N, D) float32} d2 encoder outputs on the valid steps of ``records``, subsampled to ``max_steps`` rows per part."""
    import torch
    from signtranslator.data.multichannel import collate_multichannel
    from signtranslator.generation.text_to_motion import diffusion_targets
    parts = [p for p, _ in tokenizer.parts]
    rows = {p: [] for p in parts}
    with torch.no_grad():
        for start in range(0, len(records), chunk):
            for state in L.load_states(store, records[start:start + chunk]):
                batch = collate_multichannel([state])
                unit = diffusion_targets(tokenizer.config.motion, batch).to(device)
                latents = tokenizer.encode(unit)
                valid = tokenizer.token_valid(batch.frame_valid.to(device))
                for p in parts:
                    rows[p].append(latents[p][valid].float().cpu().numpy())
    out = {p: np.concatenate(v) for p, v in rows.items()}
    keep = None
    for p in parts:
        if len(out[p]) > max_steps:
            if keep is None:
                keep = np.sort(np.random.default_rng(FIT_SEED).choice(len(out[p]), max_steps, replace=False))
            out[p] = out[p][keep]
    return out


def write_npz_once(path: Path, arrays: dict) -> str:
    """Write-once, atomic; the file's SHA-256."""
    if path.exists():
        raise SystemExit(f"{path} exists (write-once)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.replace(path)
    return digest


def fit_stage_seed(name: str, part: str, stage: int) -> int:
    return zlib.crc32(f"{FIT_SEED}/{name}/{part}/{stage}".encode())


def fit_codebooks(tokenizer, latents: dict, device: str, *, iterations: int) -> tuple[dict, dict]:
    """({array key: array}, {fit: {part: per-stage statistics}}): every post-hoc quantiser, fitted on ``latents``."""
    import torch
    stages, arrays, stats = tokenizer.config.stages, {}, {}

    def stage_stats(before, after, index, codes) -> dict:
        return dict(mse_before=round(float(before.pow(2).sum(-1).mean()), 5), mse_after=round(float(after.pow(2).sum(-1).mean()), 5),
                    perplexity=round(perplexity(index, codes), 2),
                    used=round(float((torch.bincount(index, minlength=codes) > 0).float().mean()), 4))

    for k in KMEANS:
        name = f"k{k}"
        stats[name] = {}
        for part, z in latents.items():
            residual = torch.as_tensor(z, device=device)
            rows = []
            for q in range(stages):
                book = kmeans(residual, k, iterations=iterations, seed=fit_stage_seed(name, part, q))
                index, _ = _nearest(residual, book)
                after = residual - book[index]
                rows.append(stage_stats(residual, after, index, k))
                arrays[f"{name}/{part}/{q}"] = book.cpu().numpy()
                residual = after
            stats[name][part] = rows
    codes = int(np.prod(FSQ_LEVELS))
    stats["fsq10"] = {}
    for part, z in latents.items():
        residual = torch.as_tensor(z, device=device)
        rows = []
        for q in range(stages):
            stage = fsq_fit(residual)
            e, index = fsq_stage(residual, stage)
            after = residual - e
            rows.append(stage_stats(residual, after, index, codes))
            for key, value in stage.items():
                arrays[f"fsq10/{part}/{q}/{key}"] = value.cpu().numpy()
            residual = after
        stats["fsq10"][part] = rows
    stats["hands_extension"] = {}
    model = tokenizer.to(device)
    with torch.no_grad():
        full = torch.as_tensor(np.stack([latents[p] for p, _ in model.parts]), device=device)
        parts = [p for p, _ in model.parts]
        quantised = {p: [] for p in parts}
        for start in range(0, full.shape[1], 65536):
            done = model.quantize({p: full[i, start:start + 65536][None] for i, p in enumerate(parts)})[0]
            for p in parts:
                quantised[p].append(done[p][0])
        for part in HANDS:
            residual = full[parts.index(part)] - torch.cat(quantised[part])
            rows = []
            for q in range(EXTENSION_STAGES):
                book = kmeans(residual, model.config.codebook, iterations=iterations, seed=fit_stage_seed("ext", part, q))
                index, _ = _nearest(residual, book)
                after = residual - book[index]
                rows.append(stage_stats(residual, after, index, model.config.codebook))
                arrays[f"ext/{part}/{q}"] = book.cpu().numpy()
                residual = after
            stats["hands_extension"][part] = rows
    return arrays, stats


def run_fit(args) -> int:
    import torch
    from train_text_to_motion import make_store
    check_pins()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    if (out / "fit.json").exists():
        raise SystemExit(f"{out / 'fit.json'} exists (write-once)")
    store = make_store(args.store)
    started = time.time()
    declared = not args.smoke_train
    records = fit_records(store, args.smoke_train or args.fit_clips)
    signs = [r["sign"] for r in records]
    tokenizer = L.load_tokenizer(store).to(args.device)
    meta_path, latent_path = out / "latents.json", out / "latents.npz"
    if meta_path.exists() and latent_path.exists():                     # the stored latents (a crash in the fits keeps them)
        meta = json.loads(meta_path.read_text())
        if meta["signs"] != signs or meta["tokenizer_sha256"] != L.TOKENIZER_SHA:
            raise SystemExit(f"{meta_path}: other clips or another tokenizer")
        check_sha(latent_path.read_bytes(), meta["sha256"], str(latent_path))
        with np.load(latent_path) as stored:
            latents = {p: stored[p] for p in stored.files}
    else:
        latents = collect_latents(store, tokenizer, records, args.device, max_steps=args.fit_steps)
        sha = write_npz_once(latent_path, latents)
        L.write_file_once(meta_path, dict(signs=signs, tokenizer_sha256=L.TOKENIZER_SHA, sha256=sha))
    arrays, stats = fit_codebooks(tokenizer, latents, args.device, iterations=args.iterations)
    sha = write_npz_once(out / "codebooks.npz", arrays)
    report = dict(kind=f"{PROBE}-fit", report_only=True, declared_fit=declared, clips=len(records), signs=signs,
                  components=sorted({r["signer_component"] for r in records}), partitions=sorted({r["partition"] for r in records}),
                  corpora=sorted({r["corpus"] for r in records}), manifest_sha256=L.EVAL_MIX_SHA, tokenizer_sha256=L.TOKENIZER_SHA,
                  steps={p: int(len(z)) for p, z in latents.items()}, iterations=args.iterations, codebooks_sha256=sha,
                  fsq=dict(levels=list(FSQ_LEVELS), codes=int(np.prod(FSQ_LEVELS)), bits=round(math.log2(np.prod(FSQ_LEVELS)), 2)),
                  statistics=stats, peak_rss_bytes=peak_rss_bytes(), gpu_peak_reserved_bytes=gpu_peak_reserved_bytes(), code_digest=code_digest(),
                  environment=environment(),
                  seconds=round(time.time() - started, 1),
                  note="a post-hoc fit: the encoder and decoder were trained for d2's codebooks")
    write_report_once(out / "fit.json", report)
    print(json.dumps(dict(fit=str(out / "fit.json"), clips=len(records), steps=report["steps"], seconds=report["seconds"],
                          peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))), flush=True)
    return 0


def load_fit(fit_dir, *, declared: bool, read_signs=()) -> dict:
    """The stored fit's arrays, after checking they are what they say: c02 train clips of How2Sign, d2, the file's hash, a
    declared fit for a declared read, and no clip of the read list."""
    if not fit_dir:
        raise SystemExit("this form reads the fit: give --fit-dir")
    path = Path(fit_dir)
    report = json.loads((path / "fit.json").read_text())
    if report["components"] != ["c02"] or report["partitions"] != ["train"] or report["corpora"] != [L.H2S]:
        raise SystemExit(f"{path}: a fit reads How2Sign c02 train clips only, not {report['components']} {report['partitions']}")
    if report["tokenizer_sha256"] != L.TOKENIZER_SHA:
        raise SystemExit(f"{path}: fitted on another tokenizer")
    if declared and not report["declared_fit"]:
        raise SystemExit(f"{path}: a smoke fit cannot feed a declared read")
    if report.get("code_digest") != code_digest() or report.get("environment") != environment():
        raise SystemExit(f"{path}: fitted by other code or in another environment")
    if declared and set(report["signs"]) & set(read_signs):
        raise SystemExit(f"{path}: the fit holds clips of the read list")
    raw = (path / "codebooks.npz").read_bytes()
    check_sha(raw, report["codebooks_sha256"], str(path / "codebooks.npz"))
    del raw
    with np.load(path / "codebooks.npz") as stored:
        arrays = {k: stored[k] for k in stored.files}
    return dict(arrays=arrays, report=dict(codebooks_sha256=report["codebooks_sha256"], clips=report["clips"],
                                           declared_fit=report["declared_fit"], components=report["components"]))


def fit_books(fit: dict, name: str, part: str, stages: int) -> list:
    import torch
    return [torch.as_tensor(fit["arrays"][f"{name}/{part}/{q}"]) for q in range(stages)]


def fit_fsq(fit: dict, part: str, stages: int) -> list[dict]:
    import torch
    return [{key: torch.as_tensor(fit["arrays"][f"fsq10/{part}/{q}/{key}"]) for key in ("mean", "basis", "bound")}
            for q in range(stages)]


# ---- decoded forms (one state at a time) ----------------------------------------------------------------------------

class _Latents:
    """``round_trip``'s plan hook: the decoder reads ``compute(latents)``, so the masks and fills are every round trip's."""

    def __init__(self, compute) -> None:
        self.compute = compute

    def eval(self):
        return self

    def reconstruct(self, latents, valid):
        return self.compute(latents)


class _NoNetwork:
    """A stand-in tokenizer for ``round_trip``: the 'decoder' returns the unit vector it was given."""

    def __init__(self, tokenizer) -> None:
        self.config = tokenizer.config

    def encode(self, unit):
        return dict(unit=unit)

    def quantize(self, latents, stages=None):
        return latents, None, None

    def decode_latents(self, quantised, frames):
        return quantised["unit"][:, :frames]


def stage_map(name: str, tokenizer) -> dict:
    """{part: stages} of a B4 form; every part not named keeps all of d2's stages."""
    stages = tokenizer.config.stages
    out = {p: stages for p, _ in tokenizer.parts}
    if name == "h8b4f2":
        out.update(left_hand=8, right_hand=8, body=4, face=2)
    elif name == "h12b6f4":
        out.update(left_hand=12, right_hand=12, body=6, face=4)
    elif name in ("hands10", "hands12"):
        out.update(left_hand=int(name[5:]), right_hand=int(name[5:]))
    elif name.startswith("q_"):
        body = name[2:]
        part = body.rstrip("0123456789")
        if part not in STAGE_PARTS:
            raise SystemExit(f"unknown form {name}")
        for p in STAGE_PARTS[part]:
            out[p] = int(body[len(part):])
    else:
        raise SystemExit(f"unknown form {name}")
    return out


def by_stages(tokenizer, latents: dict, stages: dict, fit: dict | None) -> dict:
    """Each part quantised with its stages: d2's first Q, and beyond d2's own the fitted extension on its residual."""
    own, out, full = tokenizer.config.stages, {}, {}
    for part, q in stages.items():
        used = min(q, own)
        if used not in full:
            full[used] = tokenizer.quantize(latents, used)[0]
        z = full[used][part]
        if q > own:
            if fit is None:
                raise SystemExit("an extended form reads the fit")
            z = z + apply_kmeans_stages(latents[part] - z, fit_books(fit, "ext", part, q - own))
        out[part] = z
    return out


def latent_hook(name: str, tokenizer, fit: dict | None):
    """The form's latents -> decoder input; None is d2's own quantiser."""
    if name in ("round_trip", "round_trip_masked"):
        return None
    if name in ("no_quant", "no_quant_masked"):
        return _Latents(lambda latents: latents)
    if name.startswith("k") and name[1:].isdigit():
        if int(name[1:]) not in KMEANS:
            raise SystemExit(f"unknown form {name}")
        stages = tokenizer.config.stages
        return _Latents(lambda latents: {p: apply_kmeans_stages(z, fit_books(fit, name, p, stages)) for p, z in latents.items()})
    if name == "fsq10":
        stages = tokenizer.config.stages
        return _Latents(lambda latents: {p: apply_fsq_stages(z, fit_fsq(fit, p, stages)) for p, z in latents.items()})
    stages = stage_map(name, tokenizer)
    return _Latents(lambda latents: by_stages(tokenizer, latents, stages, fit))


def joints_of(channel, keep) -> np.ndarray:
    return np.array([keep(label) for label in channel.labels])


def swap_selection(part: str, real) -> dict:
    """{channel: per-joint bool} of a swap part, in this state's layout."""
    wrist = ("LeftHand", "RightHand")
    all_of = lambda name: np.ones(len(real.channels[name].labels), bool)  # noqa: E731
    if part == "body":
        return dict(body=all_of("body"))
    if part == "left":
        return dict(left_hand=all_of("left_hand"))
    if part == "right":
        return dict(right_hand=all_of("right_hand"))
    if part == "fingers":
        return {h: all_of(h) for h in HANDS}
    if part in ("wrist", "arms"):
        labels = real.channels["body"].labels
        if not set(wrist) <= set(labels):
            raise SystemExit(f"the body channel has no {wrist} joints: the layout does not allow {part!r}")
        mask = joints_of(real.channels["body"], lambda label: label in wrist)
        return dict(body=mask if part == "wrist" else ~mask)
    raise SystemExit(f"unknown part {part}")


def overlay(real, trip, selection: dict):
    """``real`` with the selected joints' values taken from ``trip`` on the frames real tracked them. Validity, every other
    channel and the clock stay real's, so a ``trip`` equal to real gives real."""
    channels = dict(real.channels)
    for name, joints in selection.items():
        c, t = real.channels[name], trip.channels[name]
        if c.values.shape != t.values.shape:
            raise SystemExit(f"{name}: the round trip's shape {t.values.shape} differs from the clip's {c.values.shape}")
        take = c.valid & joints[None]
        channels[name] = replace(c, values=np.where(take[..., None], t.values, c.values).astype(c.values.dtype))
    return replace(real, channels=channels)


def everything(tokenizer, real) -> dict:
    return {name: np.ones(len(real.channels[name].labels), bool) for name in tokenizer.config.motion.trained_channels
            if name in real.channels}


def decoded_form(name: str, state, tokenizer, fit: dict | None):
    """One clip's form, through ``round_trip`` (no gains)."""
    from tokenizer_recognisability_check import round_trip
    motion = tokenizer.config.motion
    if name in ("identity", "identity_masked"):
        trip = round_trip(_NoNetwork(tokenizer), motion, state, FPS)
        return trip if name == "identity" else overlay(state, trip, everything(tokenizer, state))
    if name in B2_FORMS:
        way, part = name.split("_")[:-1], name.split("_")[-1]
        trip = round_trip(tokenizer, motion, state, FPS)
        selected = swap_selection(part, state)
        if way == ["rt", "only"]:
            return overlay(state, trip, selected)
        rest = everything(tokenizer, state)
        for channel, joints in selected.items():
            rest[channel] = rest[channel] & ~joints
        return overlay(state, trip, rest)
    trip = round_trip(tokenizer, motion, state, FPS, plan=latent_hook(name, tokenizer, fit))
    return overlay(state, trip, everything(tokenizer, state)) if name.endswith("_masked") else trip


# ---- the seam ablation (real frames only) ---------------------------------------------------------------------------

def wrist_steps(state) -> np.ndarray:
    """(T, 2) per-frame wrist displacement in the back-translator's view, NaN where a wrist has no valid pair; first row NaN."""
    from signtranslator.data.keypoints2d import from_state, normalise
    points, valid = from_state(state)
    p = normalise(points, valid, y_down=False)
    out = np.full((len(p), 2), np.nan)
    for k, w in enumerate(WRIST_POINTS):
        ok = valid[1:, w] & valid[:-1, w]
        out[1:, k] = np.where(ok, np.linalg.norm(p[1:, w] - p[:-1, w], axis=-1), np.nan)
    return out


def wrist_activity(state) -> np.ndarray | None:
    """(T,) mean tracked wrist speed, smoothed over 5 frames, gaps interpolated; None when no wrist pair is tracked."""
    steps = wrist_steps(state)
    seen = np.isfinite(steps)
    if not seen.any():
        return None
    have = seen.any(1)
    value = np.where(seen, steps, 0.).sum(1) / np.maximum(seen.sum(1), 1)
    frames = np.arange(len(value))
    value = np.interp(frames, frames[have], value[have])
    pad = np.pad(value, 2, mode="edge")
    return np.convolve(pad, np.ones(5) / 5, mode="valid")


def seam_wanted(spec: dict, text: str, entries: int | None) -> int:
    """Junctions a form wants: composition's (entries − 1, so 0 for one entry or none), or the word-count sensitivity's."""
    from signtranslator.generation.compositional import normalise
    if spec["density"] == "composition":
        if entries is None:
            raise SystemExit("a composition-density form needs the sentence's entries")
        return max(0, entries - 1)
    return max(1, len(normalise(text)) - 1)


def seam_count(wanted: int, frames: int) -> tuple[int, bool]:
    """(junctions, capped): at most what fits (each window a frame, a real frame on every side)."""
    fits = max((frames - 1) // 2, 0)
    return min(wanted, fits), wanted > fits


def minima_cuts(a: np.ndarray, count: int) -> tuple[list[int], int]:
    """(sorted cut frames, how many were filled by gap midpoints): the lowest local minima of activity ``a``, at least
    T / (2(count + 1)) frames apart."""
    T = len(a)
    spacing = max(2, T // (2 * (count + 1)))
    candidates = [t for t in range(1, T - 1) if a[t] <= a[t - 1] and a[t] <= a[t + 1]]
    chosen: list[int] = []
    for t in sorted(candidates, key=lambda f: (a[f], f)):
        if len(chosen) == count:
            break
        if all(abs(t - c) >= spacing for c in chosen):
            chosen.append(t)
    return _fill(chosen, count, T)


def _fill(chosen: list[int], count: int, frames: int) -> tuple[list[int], int]:
    filled = 0
    cuts = sorted(chosen)
    while len(cuts) < count:
        ends = [0] + cuts + [frames - 1]
        gap, at = max((b - a, a) for a, b in zip(ends, ends[1:]))
        if gap < 2:
            break
        cuts = sorted(cuts + [at + gap // 2])
        filled += 1
    return cuts, filled


def even_cuts(count: int, frames: int) -> list[int]:
    return _fill([], count, frames)[0]


def windows_at_cuts(cuts, total: int, frames: int) -> list[tuple[int, int]]:
    """Inclusive windows centred on ``cuts``, ``total`` frames in all (as near equal as each cut's room allows), disjoint, with
    at least one real frame between and at the ends."""
    if not cuts or total <= 0:
        return []
    lows, highs = [1], []
    for a, b in zip(cuts, cuts[1:]):
        middle = (a + b) // 2
        highs.append(max(a, min(middle, b - 2)))
        lows.append(highs[-1] + 2)
    highs.append(frames - 2)
    room = [max(h - l + 1, 0) for l, h in zip(lows, highs)]
    sizes = [0] * len(cuts)
    left = min(total, sum(room))
    while left:
        open_cells = [i for i, r in enumerate(room) if sizes[i] < r]
        if not open_cells:
            break
        share, extra = divmod(left, len(open_cells))
        if not share:
            for i in open_cells[:extra]:
                sizes[i] += 1
            left = 0
            break
        for i in open_cells:
            step = min(share, room[i] - sizes[i])
            sizes[i] += step
            left -= step
    out = []
    for c, l, h, n in zip(cuts, lows, highs, sizes):
        if n:
            start = int(np.clip(c - n // 2, l, h - n + 1))
            out.append((start, start + n - 1))
    return out


def windows_from_lengths(lengths, gaps) -> list[tuple[int, int]]:
    out, at = [], 0
    for size, gap in zip(lengths, gaps):
        at += gap
        out.append((at, at + size - 1))
        at += size
    return out


def random_windows(lengths, frames: int, rng) -> list[tuple[int, int]]:
    """The same window lengths at uniformly random positions, at least one real frame around each."""
    free = frames - sum(lengths) - len(lengths) - 1
    if free < 0:
        raise ValueError("the windows do not fit")
    lengths = list(rng.permutation(np.asarray(lengths, dtype=np.int64)))
    gaps = 1 + rng.multinomial(free, np.full(len(lengths) + 1, 1. / (len(lengths) + 1)))
    return windows_from_lengths([int(x) for x in lengths], [int(g) for g in gaps[:-1]])


def fixed_windows(lengths, frames: int) -> list[tuple[int, int]]:
    """The same window lengths at even spacing."""
    free = frames - sum(lengths)
    n = len(lengths) + 1
    if free < n:
        raise ValueError("the windows do not fit")
    base, extra = divmod(free, n)
    return windows_from_lengths(list(lengths), [base + (1 if i < extra else 0) for i in range(n - 1)])


def apply_windows(state, windows):
    """``state`` with each window's frames replaced by the linear bridge between the frames around it."""
    from signtranslator.generation.compositional import bridge, concatenate, take_frames
    pieces, at = [], 0
    for a, b in windows:
        before = take_frames(state, np.arange(at, a))
        pieces += [before, bridge(before, take_frames(state, np.array([b + 1])), b - a + 1)]
        at = b + 1
    pieces.append(take_frames(state, np.arange(at, len(state.timestamps))))
    return concatenate(pieces, 0, sample_id=state.sample_id)


def squeeze_at(state, cuts, ratio: float = SQUEEZE):
    """The segments between ``cuts`` squeezed ``ratio``× in time, with linear bridges filling the clip's length; ((form), the
    bridged frames as windows)."""
    from signtranslator.generation.compositional import bridge, concatenate, take_frames, time_scale
    T = len(state.timestamps)
    ends = [0] + list(cuts) + [T]
    segments = [take_frames(state, np.arange(a, b)) for a, b in zip(ends, ends[1:])]
    squeezed = [time_scale(s, max(1, int(round(len(s.timestamps) / ratio)))) for s in segments]
    spare = T - sum(len(s.timestamps) for s in squeezed)
    gaps = len(squeezed) - 1
    base, extra = divmod(spare, gaps) if gaps else (0, 0)
    pieces, windows, at = [], [], 0
    for i, s in enumerate(squeezed):
        pieces.append(s)
        at += len(s.timestamps)
        if i < gaps:
            n = base + (1 if i < extra else 0)
            if n:
                pieces.append(bridge(s, squeezed[i + 1], n))
                windows.append((at, at + n - 1))
                at += n
    form = concatenate(pieces, 0, sample_id=state.sample_id)
    if len(form.timestamps) != T:
        raise AssertionError("the squeezed form must keep the clip's length")
    return form, windows


def seam_rng(clip: int, form: str) -> np.random.Generator:
    return np.random.default_rng([SEAM_SEED, clip, FORM_ID[form]])


def values_only(real, form):
    """``form``'s values under ``real``'s validity: no bridge heals tracking. Where the form has no value for a joint real tracks
    (both ends of its window untracked), real's own value stays."""
    channels = dict(real.channels)
    for name, c in real.channels.items():
        f = form.channels[name]
        take = c.valid & f.valid
        channels[name] = replace(c, values=np.where(take[..., None], f.values, c.values).astype(c.values.dtype))
    return replace(real, channels=channels)


def seam_form(name: str, state, text: str, clip: int, entries: int | None = None):
    """(form, info) of one clip. The cuts come from the real clip; every control keeps the same window lengths, so a control
    never moves with its own form's outcome. Seeds depend on the clip and the form only."""
    spec = SEAM_SPECS.get(name)
    if spec is None:
        raise SystemExit(f"unknown form {name}")
    T = len(state.timestamps)
    count, capped = seam_count(seam_wanted(spec, text, entries), T)
    info = dict(frames=T, density=spec["density"], entries=entries, junctions=count, capped=capped, filled=0, untracked=False, windows=0,
                bridged_frames=0, path_share=None)
    if count < 1 or T < 5:                                  # no junction, or nothing fits: the clip stays real, and says so
        info.update(skipped=True)
        return state, info
    activity = wrist_activity(state)
    info["untracked"] = activity is None
    if activity is None:
        cuts, filled = even_cuts(count, T), count
    else:
        cuts, filled = minima_cuts(-activity if spec["position"] == "max" else activity, count)
    info["filled"] = filled
    if spec.get("squeeze"):
        if spec["position"] == "rnd":
            cuts = sorted(int(c) for c in seam_rng(clip, name).choice(np.arange(1, T), size=len(cuts), replace=False))
        form, windows = squeeze_at(state, cuts)
    else:
        placed = windows_at_cuts(cuts, int(round(spec["share"] / 100 * T)), T)
        lengths = [b - a + 1 for a, b in placed]
        if spec["position"] in ("min", "max"):
            windows = placed
        elif spec["position"] == "rnd":
            windows = random_windows(lengths, T, seam_rng(clip, name))
        else:
            windows = fixed_windows(lengths, T)
        form = apply_windows(state, windows)
    if spec.get("values_only"):
        form = values_only(state, form)
    mask = np.zeros(T, bool)
    for a, b in windows:
        mask[a:b + 1] = True
    path = np.nansum(wrist_steps(form), axis=1)
    info.update(windows=len(windows), bridged_frames=int(mask.sum()),
                path_share=round(float(path[mask].sum() / path.sum()), 4) if path.sum() > 0 else None)
    return form, info


def composition_entries(store, records, declared: bool) -> list[int]:
    """Per clip, the r10 entries composition signs: ``round15_compose_eval.stage1_selection`` on the evaluation mix's vocabulary.
    On D20 they must give the recorded figures, or the vocabulary is not what composition read."""
    from round15_compose_eval import stage1_selection
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    _, words = stage1_selection([r["text"] for r in records], training_vocabulary(manifest, isolated_corpora(manifest)))
    counts = [len(w) for w in words]
    if declared:
        check_d20_entries(counts)
    return counts


def check_d20_entries(counts) -> None:
    found = dict(mean=round(float(np.mean(counts)), 2), median=float(np.median(counts)), empty=sum(c == 0 for c in counts))
    if found != {k: float(v) if k == "median" else v for k, v in D20_ENTRIES.items()}:
        raise SystemExit(f"r10's entries on this list are {found}, not the recorded {D20_ENTRIES}")


def build_form(name: str, context: dict) -> tuple[list, list | None]:
    """(the form's states, per-clip information or None)."""
    from evaluate_text_to_motion import blanked
    refs = context["refs"]
    if name == "real":
        return list(refs), None
    if name == "blank":
        return blanked(refs), None
    if name in SEAM_FORMS:
        entries = context.get("entries") or [None] * len(refs)
        pairs = [seam_form(name, s, r["text"], i, entries[i]) for i, (s, r) in enumerate(zip(refs, context["records"]))]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if name not in FORM_ID:
        raise SystemExit(f"unknown form {name}")
    fit = context.get("fit") if name in FIT_FORMS else None
    if name in FIT_FORMS and fit is None:
        raise SystemExit(f"{name} reads the fit: give --fit-dir")
    return [decoded_form(name, s, context["tokenizer"], fit) for s in refs], None


# ---- per-clip reads --------------------------------------------------------------------------------------------------

def tracked_statistics(states) -> dict:
    """Per clip, the share of frames v1 sees as tracked: wrist points, and hand points (``keypoints2d.from_state``), the form's own
    validity, unmasked, so a bridge's healing shows."""
    from signtranslator.data.keypoints2d import from_state
    wrists, hands = [], []
    for s in states:
        _, valid = from_state(s)
        wrists.append(round(float(valid[:, list(WRIST_POINTS)].mean()), 5))
        hands.append(round(float(valid[:, list(HAND_ROOT_POINTS)].mean()), 5))
    return dict(wrists=wrists, hands=hands)


def hand_statistics(states, refs) -> dict:
    """Per hand and clip, on real's tracked frames: ``motion_metrics.compare``'s speed and variation (a median over joints and
    frames, and of per-element temporal standard deviations). None where the hand has no valid frame."""
    from signtranslator.generation.motion_metrics import angular_speeds
    out = {h: dict(speed=[], variation=[]) for h in HANDS}
    for s, ref in zip(states, refs):
        masked = L.observed_like(s, ref)
        for h in HANDS:
            c = masked.channels[h]
            if not c.valid.any():
                out[h]["speed"].append(None)
                out[h]["variation"].append(None)
                continue
            speeds = angular_speeds(masked, h)
            speeds = speeds[np.isfinite(speeds)]
            out[h]["speed"].append(round(float(np.median(speeds)), 5) if speeds.size else None)
            values = np.where(c.valid[..., None], c.values, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)      # joints with too few valid frames: NaN, as compare
                out[h]["variation"].append(round(float(np.nanmedian(np.nanstd(values, axis=0))), 6))
    return out


def load_reader(store, args):
    """Instrument v1. With a cached slim copy, memory-mapped with its weights assigned, so loading does not hold three copies of
    them (4.02 GB on the Mac; the limit is 4). Otherwise ``s2_lib.make_reader``, which checks the original's hash and writes the slim
    copy. The file's name is the original's hash, so another instrument's copy is never read in its place."""
    import torch
    path = Path(args.instrument_cache) / f"{L.TRANSLATOR_SHA}.pt" if args.instrument_cache else None
    if path is None or not path.exists():
        return L.make_reader(store, run=L.TRANSLATOR_RUN, sha256=L.TRANSLATOR_SHA, device=args.device, cache_dir=args.instrument_cache)
    from signtranslator.generation.keypoints_to_text_t5 import KeypointsToT5Config, build
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("kind") != "keypoints_to_text_t5":
        raise SystemExit(f"{path}: not instrument v1's kind of checkpoint")
    model = build(KeypointsToT5Config.from_dict(saved["config"]), pretrained=False)
    model.load_state_dict(saved["ema"], assign=True)
    key = json.loads(store.get(f"training-runs/{L.TRANSLATOR_RUN}/best.json"))["checkpoint_key"]
    device = torch.device(args.device)
    return L.Reader(model.to(device).eval(), "keypoints", dict(run=L.TRANSLATOR_RUN, checkpoint_key=key, sha256=L.TRANSLATOR_SHA, kind="keypoints"), device)


def longest_train_records(store, k: int) -> list[dict]:
    """The ``k`` longest How2Sign c02 train clips: a smoke's memory and speed must meet them, not only a typical draw."""
    pool = L.train_records(L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA), ("c02",))
    records = sorted(pool, key=lambda r: (-float(r["continuous"].get("seconds") or 0.), r["sign"]))[:k]
    L.check_train_records(records, ("c02",))
    return records


def run_read(args) -> int:
    import torch
    from train_text_to_motion import make_store
    check_pins()
    torch.set_num_threads(args.threads)
    names = [f for f in args.forms.split(",") if f]
    unknown = [f for f in names if f not in FORM_ID]
    if unknown:
        raise SystemExit(f"unknown forms {unknown}")
    if args.smoke_longest and not args.smoke_train:
        raise SystemExit("--smoke-longest belongs to a smoke")
    out_dir = Path(args.out_dir)
    for name in names:
        if (out_dir / f"form-{name}.json").exists():
            raise SystemExit(f"{out_dir / f'form-{name}.json'} exists (write-once)")
    store = make_store(args.store)
    records, declared, described = L.open_read_list(store, args.smoke_train)
    if args.smoke_longest:
        have = {r["sign"] for r in records}
        records = list(records) + [r for r in longest_train_records(store, args.smoke_longest) if r["sign"] not in have]
        described = dict(described, clips=len(records), longest=args.smoke_longest)
    check_read_list(records, declared)
    texts, signs = [r["text"] for r in records], [r["sign"] for r in records]
    refs = L.load_states(store, records)
    needs_tokenizer = any(n in DECODED_FORMS for n in names)
    tokenizer = L.load_tokenizer(store) if needs_tokenizer else None
    fit = load_fit(args.fit_dir, declared=declared, read_signs=signs) if any(n in FIT_FORMS for n in names) else None
    entries = composition_entries(store, records, declared) if any(n in SEAM_COMPOSITION_FORMS for n in names) else None
    reader = load_reader(store, args)
    candidates = M.retrieval_candidates(len(texts), args.candidates, seed=0)
    context = dict(records=records, refs=refs, tokenizer=tokenizer, fit=fit, entries=entries)
    digest, place = code_digest(), environment()
    for name in names:
        started = time.time()
        states, info = build_form(name, context)
        built = time.time()
        raw = reader.read(states, texts, candidates, group=args.nll_group)
        report = dict(kind=f"{PROBE}-form", report_only=True, form=name, declared_read=declared, list=described, code_digest=digest, environment=place,
                      instrument=reader.info, tokenizer=dict(key=L.TOKENIZER, sha256=L.TOKENIZER_SHA, gains=None) if tokenizer else None,
                      fit=fit["report"] if fit and name in FIT_FORMS else None, signs=signs,
                      groups=[r.get("signer_component") for r in records], references=texts,
                      candidates=candidates.tolist() if args.candidates > 1 else None, hypotheses=raw["hypotheses"],
                      nll=np.round(raw["nll"], 5).tolist(), seconds_over_reference=[round(len(s.timestamps) / max(len(r.timestamps), 1), 5)
                                                                                     for s, r in zip(states, refs)],
                      motion=L.motion_statistics(states, refs), hands=hand_statistics(states, refs), tracked=tracked_statistics(states),
                      seam=info, boundaries=dict(BOUNDARY_RULE, records_with_boundaries=records_with_boundaries(records)) if info else None,
                      build_seconds=round(built - started, 2), read_seconds=round(time.time() - built, 2), peak_rss_bytes=peak_rss_bytes(),
                      gpu_peak_reserved_bytes=gpu_peak_reserved_bytes())
        write_report_once(out_dir / f"form-{name}.json", report)
        print(json.dumps({name: dict(build_seconds=report["build_seconds"], read_seconds=report["read_seconds"],
                                     peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2), gpu_reserved_bytes=report["gpu_peak_reserved_bytes"])}), flush=True)
        del states, raw
    return 0


# ---- B6: words -------------------------------------------------------------------------------------------------------

def word_records(store, labels) -> list[dict]:
    """RIT's and ASLLVD's records in v3d's own data definition (``rc.assemble``: test partitions dropped, c00 refused, each
    val set split into select and gate), each manifest and the gloss map SHA-pinned."""
    from signtranslator.generation import recognizer_corpora as rc
    payload = store.get(GLOSS_MAP)
    check_sha(payload, GLOSS_MAP_SHA, GLOSS_MAP)
    label_set = set(labels)
    signbank = rc.signbank_targets(json.loads(payload), label_set)
    corpora = []
    for key, sha in WORD_CORPORA:
        manifest = L.load_manifest_at(store, key, sha)
        corpora.append((manifest, rc.CorpusSpec(key, sha, "signbank", 1.), rc.label_mapper("signbank", label_set, signbank=signbank)))
    return rc.assemble(corpora, label_set)[0]


def word_partitions(records, smoke: bool) -> dict:
    """{corpus: [records]}: the gate half of each corpus, or for a smoke its train records. Nothing else, whatever the input."""
    from signtranslator.generation import recognizer_corpora as rc
    out: dict = {}
    for r in records:
        if r.get("source_partition") in rc.FORBIDDEN_PARTITIONS or r.get("partition") in rc.FORBIDDEN_PARTITIONS:
            raise SystemExit(f"{r.get('sign')}: a test record reached the word read")
        rc.check_loadable(r)
        if r["partition"] == ("train" if smoke else f"gate:{r['corpus']}"):
            out.setdefault(r["corpus"], []).append(r)
    return {c: sorted(rs, key=lambda r: r["sign"]) for c, rs in out.items()}


def load_v3d(store, device: str):
    """(v3d in eval mode, its labels), the best.json and the checkpoint it names SHA-pinned."""
    import validate_sign_recognizer as vsr
    best = store.get(f"training-runs/{V3D_RUN}/best.json")
    check_sha(best, V3D_BEST_SHA, f"{V3D_RUN}/best.json")
    payload = store.get(json.loads(best)["checkpoint_key"])
    check_sha(payload, V3D_CHECKPOINT_SHA, json.loads(best)["checkpoint_key"])
    del payload
    model, labels, _ = vsr.load_model(store, V3D_RUN, device)
    return model, labels


def word_states(store, records, labels, limit: int | None, cache, *, longest: bool = False) -> tuple:
    """(states, label indices, signs) as v3d's gates load them: conformed, 24 fps, at most 8 s, the hand filter. ``longest`` (a smoke):
    the longest of them, not the first."""
    from concurrent.futures import ThreadPoolExecutor
    from signtranslator.generation.sign_recognizer import GlossMotionDataset
    from train_sign_recognizer import conform_template
    index = {g: i for i, g in enumerate(labels)}
    template, _ = conform_template(store, CONFORM_TEMPLATE, cache)
    name = records[0]["partition"]
    dataset = GlossMotionDataset(dict(records=records), partition=name, labels=index, cache=cache, fetch=store.get,
                                 min_hand_fraction=V3D_MIN_HAND, fps=FPS, template=template, hand_fraction_fallback=True)
    ids = [i for i, r in enumerate(dataset.records) if r["continuous"].get("seconds", 0.) <= V3D_MAX_SECONDS]
    if longest:
        ids = sorted(ids, key=lambda i: (-float(dataset.records[i]["continuous"].get("seconds", 0.)), i))
    ids = ids[:limit] if limit else ids
    with ThreadPoolExecutor(4) as pool:
        states = [item[1] for item in pool.map(dataset.__getitem__, ids)]
    return states, [index[dataset.records[i]["gloss"]["label"]] for i in ids], [dataset.records[i]["sign"] for i in ids]


def word_form(name: str, state, tokenizer, fit):
    if name == "real":
        return state
    if name in FIT_FORMS and fit is None:
        raise SystemExit(f"{name} reads the fit: give --fit-dir")
    return decoded_form(name, state, tokenizer, fit if name in FIT_FORMS else None)


def run_words(args) -> int:
    import torch
    from evaluate_text_to_motion import blanked
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from train_text_to_motion import make_store
    import validate_sign_recognizer as vsr
    check_pins()
    torch.set_num_threads(args.threads)
    store = make_store(args.store)
    smoke = bool(args.smoke_train)
    names = ("real", "blank") + DECODED_FORMS if args.forms is None else tuple(f for f in args.forms.split(",") if f)
    unknown = [f for f in names if f not in FORM_ID or f in SEAM_FORMS]
    if unknown:
        raise SystemExit(f"not word forms: {unknown}")
    out_dir = Path(args.out_dir)
    if args.resume:                                                     # a retry reads only what is not stored
        names = tuple(n for n in names if not (out_dir / f"word-{n}.json").exists())
    if args.list:
        labels = json.loads(store.get(f"training-runs/{V3D_RUN}/run.json"))["labels"]
        sets = word_partitions(word_records(store, labels), smoke)
        print(json.dumps({c: len(rs) for c, rs in sets.items()}), flush=True)
        return 0
    for name in names:
        if (out_dir / f"word-{name}.json").exists():
            raise SystemExit(f"{out_dir / f'word-{name}.json'} exists (write-once)")
    model, labels = load_v3d(store, args.device)
    sets = word_partitions(word_records(store, labels), smoke)
    if not sets:
        raise SystemExit("no word records in the read")
    cache = ContentCache(Path(tempfile.gettempdir()) / "sign-translator-cache", 2 * 2 ** 30)
    limit = args.smoke_train or args.limit or None
    loaded = {c: word_states(store, rs, labels, limit, cache, longest=smoke) for c, rs in sets.items()}
    tokenizer = L.load_tokenizer(store) if any(n in DECODED_FORMS for n in names) else None
    fit = load_fit(args.fit_dir, declared=not smoke) if any(n in FIT_FORMS for n in names) else None
    rule = "every decoded form: the plan declares no survival rule" if args.forms is None else "the --forms list, chosen by the operator"
    for name in names:
        started = time.time()
        ranks = {}
        for corpus, (states, y, signs) in loaded.items():
            forms = blanked(states) if name == "blank" else [word_form(name, s, tokenizer, fit) for s in states]
            ranks[corpus] = dict(signs=signs, ranks=vsr.ranks_from_logits(vsr.logits_of(model, forms, args.device), y).tolist())
        report = dict(kind=f"{PROBE}-word", report_only=True, form=name, declared_read=not smoke, forms_rule=rule,
                      instrument=dict(run=V3D_RUN, best_sha256=V3D_BEST_SHA, checkpoint_sha256=V3D_CHECKPOINT_SHA),
                      partition="train (a smoke)" if smoke else "gate", corpora=ranks, seconds=round(time.time() - started, 2),
                      fit=fit["report"] if fit and name in FIT_FORMS else None, peak_rss_bytes=peak_rss_bytes(), gpu_peak_reserved_bytes=gpu_peak_reserved_bytes(),
                      code_digest=code_digest(), environment=environment())
        write_report_once(out_dir / f"word-{name}.json", report)
        print(json.dumps({f"word-{name}": dict(seconds=report["seconds"], peak_rss_gb=round(report["peak_rss_bytes"] / 2 ** 30, 2))}), flush=True)
    return 0


# ---- merge: panels, report only ---------------------------------------------------------------------------------------

BOOT = 2000


def _quantiles(values) -> list | None:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return [round(float(np.percentile(finite, 2.5)), 4), round(float(np.percentile(finite, 97.5)), 4)]


def _share(values, blank, real):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(real != blank, (values - blank) / np.where(real != blank, real - blank, 1.), np.nan)


def length_panel(stats, others: dict, idx) -> dict:
    def ratio(s):
        total = s[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return total[:, -2] / total[:, -1]
    own = ratio(stats)
    return dict(ratio=M.length_ratio(stats), ci95=_quantiles(own),
                paired={a: dict(difference=round(float((M.length_ratio(stats) or 0.) - (M.length_ratio(s) or 0.)), 4),
                                ci95=_quantiles(own - ratio(s))) for a, s in others.items()})


def paired_present(x, y, *, n=1000, seed=0) -> dict | None:
    """Paired mean difference and ratio over the clips where both are present; None when none are."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pairs:
        return None
    a, b = np.array(pairs, np.float64).T
    difference, ratio = M.paired_mean_difference(a, b, n=n, seed=seed), M.ratio_of_means(a, b, n=n, seed=seed)
    return dict(clips=len(pairs), difference=difference["difference"], difference_ci95=difference["ci95"], ratio=ratio["ratio"],
                ratio_ci95=ratio["ci95"])


def median_ratio(x, y, idx) -> dict | None:
    """``compare``'s ratio of medians over clips, resampled jointly; None without values on both sides."""
    a, b = (np.array([np.nan if v is None else v for v in s], np.float64) for s in (x, y))
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return None
    if np.nanmedian(b) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                 # a resample with no value: NaN, dropped by _quantiles
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.nanmedian(a[idx], axis=1) / np.nanmedian(b[idx], axis=1)
    return dict(ratio=round(float(np.nanmedian(a) / np.nanmedian(b)), 4), ci95=_quantiles(ratios))


def form_panel(reads: dict, name: str, comparators=()) -> dict:
    r, refs = reads[name], reads[name]["references"]
    n = len(refs)
    idx = np.random.default_rng(0).integers(0, n, (BOOT, n))
    stats = {k: M.bleu_statistics(reads[k]["hypotheses"], refs) for k in {name, "real", "blank", *comparators}}
    bleu_of = {k: M.bleu_from_statistics(s[idx].sum(1)) for k, s in stats.items()}
    blank_nll = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = {k: M.delta_nll(blank_nll, np.asarray(reads[k]["nll"])[:, 0]) for k in stats}
    loops = {k: np.array(loop_flags(reads[k]["hypotheses"]), np.float64) for k in stats}
    bleu, ci = M.bleu_ci_fast(r["hypotheses"], refs)
    point = {k: float(M.bleu_from_statistics(s.sum(0))) for k, s in stats.items()}
    real_dnll = float(dnll["real"].mean())
    resampled = dnll[name][idx].mean(1), dnll["real"][idx].mean(1)
    others = {k: stats[k] for k in ("real", "blank")}
    motion, real_motion = r["motion"], reads["real"]["motion"]
    speed = paired_present(motion["wrist_speed"], real_motion["wrist_speed"])
    out = dict(
        bleu=dict(bleu=bleu, ci95=ci),
        share_of_gap=dict(bleu=None if point["real"] == point["blank"] else round((point[name] - point["blank"]) / (point["real"] - point["blank"]), 4),
                          bleu_ci95=_quantiles(_share(bleu_of[name], bleu_of["blank"], bleu_of["real"])),
                          likelihood=None if real_dnll == 0 else round(float(dnll[name].mean()) / real_dnll, 4),
                          likelihood_ci95=_quantiles(_share(resampled[0], 0., resampled[1]))),
        likelihood_minus_blank=M.bootstrap_mean(dnll[name]),
        loop_share=M.bootstrap_mean(loops[name]),
        length_ratio=length_panel(stats[name], others, idx),
        wrist_speed=dict(over_real=speed, over_blank=None,
                         blank_note="blank has no tracked wrist, so no paired read"),
        wrist_spread=dict(over_real=paired_present(motion["wrist_spread"], real_motion["wrist_spread"])),
        hand_variation={h: dict(over_real=median_ratio(r["hands"][h]["variation"], reads["real"]["hands"][h]["variation"], idx), over_blank=None)
                        for h in HANDS},
        hand_speed={h: dict(over_real=median_ratio(r["hands"][h]["speed"], reads["real"]["hands"][h]["speed"], idx), over_blank=None)
                    for h in HANDS},
        tracked_share={k: dict(M.bootstrap_mean(r["tracked"][k]), minus_real=M.paired_mean_difference(r["tracked"][k], reads["real"]["tracked"][k]))
                       for k in ("wrists", "hands")},
        seconds_over_reference=round(float(np.mean(r["seconds_over_reference"])), 4),
        identical_to_real=round(float(np.mean([a == b for a, b in zip(r["hypotheses"], reads["real"]["hypotheses"])])), 4),
        paired={a: dict(bleu=M.paired_bleu_fast(stats[name], stats[a]),
                        likelihood_minus_blank=M.paired_mean_difference(dnll[name], dnll[a]),
                        loop_share=M.paired_mean_difference(loops[name], loops[a]))
                for a in ("real", "blank")})
    out["comparators"] = {c: dict(bleu=M.paired_bleu_fast(stats[name], stats[c]), likelihood_minus_blank=M.paired_mean_difference(dnll[name], dnll[c]),
                                  loop_share=M.paired_mean_difference(loops[name], loops[c])) for c in comparators}
    nll = np.asarray(r["nll"])
    if nll.shape[1] > 1:
        ranks = M.mid_ranks(nll - np.asarray(reads["blank"]["nll"]))
        out["retrieval_pmi"] = M.retrieval_summary(ranks, nll.shape[1])
    if r.get("seam"):
        rows = r["seam"]
        frames, bridged = sum(c["frames"] for c in rows), sum(c["bridged_frames"] for c in rows)
        path = [c["path_share"] for c in rows if c.get("path_share") is not None]
        windows = sum(c["windows"] for c in rows)
        out["seam"] = dict(bridged_frame_share=round(bridged / max(frames, 1), 4), path_share=round(float(np.mean(path)), 4) if path else None,
                           density=rows[0]["density"], junctions_per_clip=round(float(np.mean([c["junctions"] for c in rows])), 3),
                           frames_per_bridge=round(bridged / windows, 3) if windows else None,
                           clips_without_entries=sum(c["entries"] == 0 for c in rows), filled_cuts=sum(c["filled"] for c in rows), capped_clips=sum(bool(c["capped"]) for c in rows),
                           untracked_clips=sum(bool(c["untracked"]) for c in rows), skipped_clips=sum(bool(c.get("skipped")) for c in rows))
    return out


def comparators_of(name: str) -> tuple[str, ...]:
    """The paired reads beside real and blank; those that a merge did not read are dropped."""
    if name == "identity":
        return ("round_trip",)
    if name == "identity_masked":
        return ("round_trip_masked",)
    if name == "no_quant":
        return ("round_trip", "identity")
    if name == "no_quant_masked":
        return ("round_trip_masked", "identity_masked")
    if name == "round_trip_masked" or name in B4_FORMS or name == "k512":
        return ("round_trip",)
    if name == "fsq10":
        return ("k1024", "round_trip")
    if name in B5_FORMS:
        return ("k512", "round_trip")
    if name in B2_FORMS:
        return ("round_trip_masked",)
    if name in SEAM_SPECS and name not in ("seam_rnd15", "seam_rnd30", "seam_rnd45", "seam_crnd30", "seam_crnd45", "seam_sq196_rnd", "seam_csq196_rnd"):
        spec = SEAM_SPECS[name]
        prefix = "seam_c" if spec["density"] == "composition" else "seam_"
        if spec.get("squeeze"):
            return (f"{prefix}sq196_rnd",)
        if name == "seam_cval45":
            return ("seam_cmin45", "seam_crnd45")
        return (f"{prefix}rnd{spec['share']}",)
    return ()


def word_panel(words: dict, name: str, corpus: str) -> dict:
    ranks = np.asarray(words[name]["corpora"][corpus]["ranks"])
    real = np.asarray(words["real"]["corpora"][corpus]["ranks"])
    blank = np.asarray(words["blank"]["corpora"][corpus]["ranks"])
    out = dict(clips=int(len(ranks)))
    for k in (1, 5):
        hit = (ranks < k).astype(np.float64)
        out[f"top{k}"] = M.bootstrap_mean(hit)
        out[f"top{k}_minus_real"] = M.paired_mean_difference(hit, (real < k).astype(np.float64))
        out[f"top{k}_minus_blank"] = M.paired_mean_difference(hit, (blank < k).astype(np.float64))
    return out


def load_words(out_dir: Path, declared: bool, digest: str | None = None, place: dict | None = None, fit_block: dict | None = None) -> dict:
    """The word reports, after checking they are one read of the forms' code, environment and fit."""
    out = {}
    for path in sorted(out_dir.glob("word-*.json")):
        r = json.loads(path.read_bytes())
        if r["declared_read"] != declared:
            raise SystemExit(f"{path}: a smoke mixed with a declared read")
        out[r["form"]] = r
    if not out:
        return {}
    first = next(iter(out.values()))
    for name, r in out.items():
        if {c: v["signs"] for c, v in r["corpora"].items()} != {c: v["signs"] for c, v in first["corpora"].items()}:
            raise SystemExit(f"word-{name}: other clips")
        if r.get("fit") and fit_block is not None and r["fit"] != fit_block:
            raise SystemExit(f"word-{name}: read another fit than the forms")
    for anchor in ("real", "blank"):
        if anchor not in out:
            raise SystemExit(f"word-{anchor}.json is missing")
    found = check_same_run(out, "the word reports")
    if digest is not None and found != (digest, place):
        raise SystemExit("the word reports were read by other code or in another environment than the forms")
    return out


def b1_block(reads: dict, suffix: str) -> dict | None:
    """B1's differences and position on the path, as facts: D = identity − round_trip (what B1 can move at all), and
    p = (no_quant − round_trip) / D, 0 at the round trip and 1 at the identity path; on BLEU and on likelihood minus blank, jointly resampled."""
    names = {k: f"{k}{suffix}" for k in ("no_quant", "identity", "round_trip")}
    if any(n not in reads for n in names.values()) or "blank" not in reads:
        return None
    refs = reads[names["no_quant"]]["references"]
    idx = np.random.default_rng(0).integers(0, len(refs), (BOOT, len(refs)))
    stats = {k: M.bleu_statistics(reads[n]["hypotheses"], refs) for k, n in names.items()}
    blank = np.asarray(reads["blank"]["nll"])[:, 0]
    dnll = {k: M.delta_nll(blank, np.asarray(reads[n]["nll"])[:, 0]) for k, n in names.items()}

    def position(values, resampled) -> dict:
        d, gain = values["identity"] - values["round_trip"], values["no_quant"] - values["round_trip"]
        with np.errstate(divide="ignore", invalid="ignore"):
            draws = (resampled["no_quant"] - resampled["round_trip"]) / (resampled["identity"] - resampled["round_trip"])
        return dict(d=round(float(d), 4), p=round(float(gain / d), 4) if d else None, p_ci95=_quantiles(draws))
    bleu = {k: float(M.bleu_from_statistics(v.sum(0))) for k, v in stats.items()}
    bleu_draws = {k: M.bleu_from_statistics(v[idx].sum(1)) for k, v in stats.items()}
    mean = {k: float(v.mean()) for k, v in dnll.items()}
    mean_draws = {k: v[idx].mean(1) for k, v in dnll.items()}
    pairs = (("no_quant", "round_trip"), ("identity", "round_trip"), ("identity", "no_quant"))
    return dict(forms=names,
                bleu=dict(position(bleu, bleu_draws), **{f"{a}_minus_{b}": M.paired_bleu_fast(stats[a], stats[b]) for a, b in pairs}),
                likelihood_minus_blank=dict(position(mean, mean_draws), **{f"{a}_minus_{b}": M.paired_mean_difference(dnll[a], dnll[b]) for a, b in pairs}))


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    check_pins()
    store = make_store(args.store) if args.output_key else None
    L.refuse_existing(store, args.output_key, args.output_file)
    names = list(FORMS) if args.forms is None else [f for f in args.forms.split(",") if f]
    for anchor in ("real", "blank"):
        if anchor not in names:
            raise SystemExit(f"--forms must include {anchor}: every pair reads it")
    out_dir = Path(args.out_dir)
    reads = L.load_form_reads(out_dir, names)
    first = next(iter(reads.values()))
    fits = {json.dumps(r["fit"], sort_keys=True) for r in reads.values() if r.get("fit")}
    if len(fits) > 1:
        raise SystemExit("forms read different fits")
    digest, place = check_same_run(reads, "the forms")
    canary = None
    if first["declared_read"]:
        path = out_dir / "canary.json"
        canary = json.loads(path.read_bytes()) if path.exists() else None
        if canary is None or canary.get("reproduced") is not True or canary.get("code_digest") != digest or canary.get("environment") != place:
            raise SystemExit(f"{path}: no passed canary of this code and environment; the read is void")
    panels = {name: form_panel(reads, name, tuple(c for c in comparators_of(name) if c in reads)) for name in names}
    words = load_words(out_dir, first["declared_read"], digest, place, json.loads(next(iter(fits))) if fits else None)
    word_block = None
    if words:
        corpora = sorted(next(iter(words.values()))["corpora"])
        word_block = dict(forms_rule=next(iter(words.values()))["forms_rule"], instrument=next(iter(words.values()))["instrument"],
                          caveat="v3d trained on d2 copies with gains, so it favours d2; these forms have no gains",
                          forms={name: {c: word_panel(words, name, c) for c in corpora} for name in words})
    seam = next((r["boundaries"] for r in reads.values() if r.get("boundaries")), None)
    report = dict(kind=f"{PROBE}-report", report_only=True, note=NOT_DECIDED, declared_read=first["declared_read"], list=first["list"],
                  instrument=first["instrument"], tokenizer=next((r["tokenizer"] for r in reads.values() if r.get("tokenizer")), None),
                  fit=json.loads(next(iter(fits))) if fits else None, clips=len(first["signs"]),
                  smoke=None if first["declared_read"] else "SMOKE on c02 train clips: these numbers are plumbing, not a read",
                  reading=READING, disclosures=DISCLOSURES, boundaries=seam, sensitivity_only=[n for n in names if n in SEAM_WORD_FORMS], code_digest=digest, environment=place,
                  rerun=json.loads((out_dir / "rerun.json").read_bytes()) if (out_dir / "rerun.json").exists() else None,
                  b1=dict(unmasked=b1_block(reads, ""), masked=b1_block(reads, "_masked")), forms=panels, words=word_block,
                  hypotheses={name: reads[name]["hypotheses"] for name in names},
                  checks=dict(identity_masked_identical_to_real=panels["identity_masked"]["identical_to_real"] if "identity_masked" in panels else None,
                              blank_identical_to_real=panels["blank"]["identical_to_real"],
                              canary=canary,
                              rule=f"a decoder that returns real must read real, within {IDENTITY_TOLERANCE} of its hypotheses; blank must not; "
                                   "real and blank must reproduce the pinned stored report"),
                  resources={name: dict(peak_rss_bytes=reads[name]["peak_rss_bytes"], gpu_peak_reserved_bytes=reads[name]["gpu_peak_reserved_bytes"], build_seconds=reads[name]["build_seconds"],
                                        read_seconds=reads[name]["read_seconds"]) for name in names})
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, forms=len(panels))), flush=True)
    return 0


def canary_read(reads: dict, stored: dict) -> dict:
    """Real and blank against the pinned stored report: BLEU, identical hypotheses, per-clip NLL. Clips in another order refuse."""
    rows, misses = {}, []
    for name in ("real", "blank"):
        r, kept = reads[name], stored["forms"][name]
        if r["signs"] != stored["signs"] or r["references"] != stored["references"]:
            raise SystemExit("the stored report read other clips or another order")
        bleu = M.bleu_ci_fast(r["hypotheses"], r["references"])[0]
        identical = float(np.mean([a == b for a, b in zip(r["hypotheses"], kept["hypotheses"])]))
        gap = float(np.mean(np.abs(np.asarray(r["nll"])[:, 0] - np.asarray(kept["per_clip"]["nll"]))))
        rows[name] = dict(bleu=bleu, stored_bleu=CANARY[name], bleu_gap=round(abs(bleu - CANARY[name]), 4), identical=round(identical, 4),
                          nll_mean_absolute_difference=round(gap, 6))
        misses += [f"{name} {why}" for why, bad in (("BLEU", rows[name]["bleu_gap"] > CANARY["bleu_tolerance"]), ("hypotheses", identical < CANARY["identical"]),
                                                    ("NLL", gap > CANARY["nll"])) if bad]
    return dict(reads=rows, limits=CANARY, stored_report=CANARY_REPORT, stored_sha256=CANARY_REPORT_SHA, misses=misses, reproduced=not misses)


def run_canary(args) -> int:
    """Exit 3, voiding the read, unless real and blank reproduce the pinned stored report. A smoke reads train clips, so it checks the pin and the stored
    report's shape only, and says so."""
    from train_text_to_motion import make_store
    check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    raw = store.get(CANARY_REPORT)
    check_sha(raw, CANARY_REPORT_SHA, CANARY_REPORT)
    stored = json.loads(raw)
    if any(len(stored["forms"][n]["per_clip"]["nll"]) != len(stored["signs"]) for n in ("real", "blank")):
        raise SystemExit(f"{CANARY_REPORT}: no per-clip NLL for real and blank")
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in ("real", "blank")}
    if not read["real"]["declared_read"]:
        print(json.dumps(dict(canary="not applicable: a smoke reads train clips; the pin and the stored report's shape are checked")), flush=True)
        return 0
    digest, place = check_same_run(read, "real and blank")
    result = dict(canary_read(read, stored), code_digest=digest, environment=place)
    path = out / "canary.json"
    if not path.exists():
        write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=digest, environment=place, canary=result), indent=1, sort_keys=True).encode())
    print(f"ALERT real and blank do not reproduce the stored report: {result['misses']}. The read is void: no retry; a rerun needs new code and a recorded cause",
          flush=True)
    return 3


def run_check(args) -> int:
    """Exit 3 when the path fails its own planted case on real clips, or the canary is missing or failed (the job stops on a smoke). It judges plumbing, no
    form."""
    report = json.loads(Path(args.report).read_text())
    checks = report["checks"]
    problems = []
    if report.get("report_only") is not True:
        problems.append("the report does not say report_only")
    same = checks["identity_masked_identical_to_real"]
    if same is not None and same < 1 - IDENTITY_TOLERANCE:
        problems.append(f"identity_masked reads real on a share of {same}, below {1 - IDENTITY_TOLERANCE}")
    if checks["blank_identical_to_real"] >= 1:
        problems.append("blank reads exactly as real")
    if checks.get("canary") and not checks["canary"]["reproduced"]:
        problems.append(f"real and blank do not reproduce the stored report: {checks['canary']['misses']}")
    if report.get("declared_read") and not checks.get("canary"):
        problems.append("a declared read without its canary")
    print(json.dumps(dict(checks={k: v for k, v in checks.items() if k != "rule"}, problems=problems)), flush=True)
    for line in problems:
        print(f"ALERT {line}", flush=True)
    return 3 if problems else 0


UPLOAD_EXCLUDED = ("latents.npz", "latents.json")          # local, and rebuilt from the clips


def run_upload(args) -> int:
    """Every report and codebook under ``--out-dir`` to ``--prefix``. Default: write-once, refused whole if any key exists. ``--partial``
    (a failed job's work): keys already stored are left alone, the rest uploaded."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    paths = sorted(p for p in Path(args.out_dir).rglob("*") if p.is_file() and p.suffix in (".json", ".npz") and p.name not in UPLOAD_EXCLUDED
                   and not p.name.endswith(".tmp.npz"))
    if not paths:
        if args.partial:
            print(json.dumps(dict(uploaded=0, prefix=args.prefix)), flush=True)
            return 0
        raise SystemExit(f"{args.out_dir}: nothing to upload")
    keys = [(p, f"{args.prefix.rstrip('/')}/{p.relative_to(args.out_dir)}") for p in paths]
    taken = [k for _, k in keys if store.exists(k)]
    if taken and not args.partial:
        raise SystemExit(f"{taken[0]} exists (write-once); nothing uploaded")
    done = set(taken)
    for p, k in keys:
        if k not in done:
            store.put(k, p.read_bytes())
    print(json.dumps(dict(uploaded=len(keys) - len(done), kept=len(done), prefix=args.prefix)), flush=True)
    return 0


RESTORABLE = re.compile(r"^([^/]+)/d20/(fit/fit\.json|fit/codebooks\.npz|forms/canary\.json|forms/(?:form|word)-[^/]+\.json)$")


def _json_or_none(raw: bytes) -> dict | None:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def check_void(store, args, digest: str, out: Path) -> None:
    """This code's voided read never reads again; after any void, a rerun needs new code and a recorded cause."""
    prefix = args.void_prefix.rstrip("/") + "/"
    voided = store.list(prefix)
    if f"{prefix}{digest}.json" in voided:
        raise SystemExit(f"{prefix}{digest}.json: this code's read is void; a rerun needs new code and a recorded cause")
    if voided and not args.rerun_cause:
        raise SystemExit(f"{len(voided)} voided reads under {prefix}: give --rerun-cause, what was wrong and what the new code changes")
    if args.rerun_cause and not (out / "forms" / "rerun.json").exists():
        write_report_once(out / "forms" / "rerun.json", dict(cause=args.rerun_cause, voided=voided, code_digest=digest))


def run_restore(args) -> int:
    """A retry's first step: one failed run's stored D20 forms, words and fit, the latest run only, whole or not at all. Refused when the finished read is
    already stored or this code's read was voided, so no hours are spent first. A run of other code or another environment is left alone and the read starts
    fresh, since nothing mixes then. Local forms that do not match stop here, not hours later at the merge."""
    from train_text_to_motion import make_store
    check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    out, digest, place = Path(args.out_dir), code_digest(), environment()
    check_void(store, args, digest, out)
    local = {p.name: json.loads(p.read_bytes()) for p in (out / "forms").glob("[fw]*-*.json")} if (out / "forms").exists() else {}
    if local and check_same_run(local, "the local forms") != (digest, place):
        raise SystemExit(f"{out / 'forms'}: local forms of other code or another environment")
    prefix = args.prefix.rstrip("/") + "/"
    stored: dict = {}
    for key in store.list(prefix):
        match = RESTORABLE.match(key[len(prefix):])
        if match:
            stored.setdefault(match[1], {})[match[2]] = key
    if not stored:
        print(json.dumps(dict(restored=0, run=None, code_digest=digest)), flush=True)
        return 0
    run = max(stored)
    files = stored[run]
    payloads = {rel: store.get(key) for rel, key in files.items() if rel != "fit/codebooks.npz"}
    reports = {rel: r for rel, r in ((rel, _json_or_none(raw)) for rel, raw in payloads.items()) if r is not None}
    foreign = [rel for rel, r in reports.items() if r.get("code_digest") != digest or r.get("environment") != place
               or r.get("declared_read", r.get("declared_fit")) is not True]
    canary = reports.get("forms/canary.json")
    if foreign or (canary is not None and canary.get("reproduced") is not True):
        print(json.dumps(dict(restored=0, run=run, refused=foreign[:5] or "its canary failed", code_digest=digest)), flush=True)
        print(f"ALERT run {run} is not reused: other code, another environment or a failed canary. The read starts fresh", flush=True)
        return 0
    restored = 0
    for rel, r in reports.items():
        anchor = rel in ("forms/form-real.json", "forms/form-blank.json")
        if rel.startswith("fit/") or rel == "forms/canary.json" or (out / rel).exists() or (anchor and canary is None):
            continue
        write_report_once(out / rel, r)
        restored += 1
    fit_report = reports.get("fit/fit.json")
    if fit_report is not None and "fit/codebooks.npz" in files and not (out / "fit/fit.json").exists():
        books = store.get(files["fit/codebooks.npz"])
        if hashlib.sha256(books).hexdigest() == fit_report["codebooks_sha256"]:
            (out / "fit").mkdir(parents=True, exist_ok=True)
            (out / "fit/codebooks.npz").write_bytes(books)
            write_report_once(out / "fit/fit.json", fit_report)
            restored += 2
    print(json.dumps(dict(restored=restored, run=run, code_digest=digest)), flush=True)
    return 0


def form_groups(processes: int, skip=(), stored_in: str | None = None) -> list[str]:
    """Forms in up to ``processes`` comma lists, deterministic: fit forms spread, so no one process holds every codebook pass. With
    ``stored_in``, forms already read there are left out."""
    done = {p.name[len("form-"):-len(".json")] for p in Path(stored_in).glob("form-*.json")} if stored_in else set()
    names = [f for f in FORMS if f not in set(skip) and f not in done]
    groups: list[list[str]] = [[] for _ in range(max(1, processes))]
    for i, name in enumerate(names):
        groups[i % len(groups)].append(name)
    return [",".join(g) for g in groups if g]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="post-hoc codebooks from c02 train latents (no network trains)")
    fit.add_argument("--store", default="r2")
    fit.add_argument("--out-dir", required=True)
    fit.add_argument("--fit-clips", type=int, default=FIT_CLIPS)
    fit.add_argument("--fit-steps", type=int, default=FIT_STEPS)
    fit.add_argument("--iterations", type=int, default=FIT_ITERATIONS)
    fit.add_argument("--smoke-train", type=int, default=0, help="a smoke fit on this many c02 train clips; it cannot feed a declared read")
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--threads", type=int, default=1)
    read = sub.add_parser("read", help="one write-once form-<name>.json per form")
    read.add_argument("--store", default="r2")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--fit-dir")
    read.add_argument("--forms", default=",".join(FORMS))
    read.add_argument("--candidates", type=int, default=1, help="1: likelihood only; 100 adds PMI ranks")
    read.add_argument("--nll-group", type=int, default=2)
    read.add_argument("--instrument-cache", help="the slim instrument's directory (read the first form alone)")
    read.add_argument("--smoke-train", type=int, default=0, help="read this many c02 train clips, not D20")
    read.add_argument("--smoke-longest", type=int, default=0, help="a smoke's list also holds this many of c02 train's longest clips")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    words = sub.add_parser("words", help="B6: v3d on decoded RIT and ASLLVD words")
    words.add_argument("--store", default="r2")
    words.add_argument("--out-dir", required=True)
    words.add_argument("--fit-dir")
    words.add_argument("--forms", help="default: real, blank and every decoded form (the plan declares no survival rule)")
    words.add_argument("--limit", type=int, default=0, help="at most this many clips per corpus")
    words.add_argument("--smoke-train", type=int, default=0, help="this many train words per corpus, no gate clip")
    words.add_argument("--list", action="store_true", help="print each read's clip count from the manifests; loads no clip")
    words.add_argument("--resume", action="store_true", help="leave out forms already read in --out-dir")
    words.add_argument("--device", default="cpu")
    words.add_argument("--threads", type=int, default=1)
    merge = sub.add_parser("merge", help="the report: report_only, decides nothing")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--out-dir", required=True)
    merge.add_argument("--forms", help="default: every form (a missing form refuses)")
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check", help="the path's planted case on real clips; exit 3 on a problem")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload", help="write-once upload of a work directory")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true", help="a failed job's work: leave stored keys alone, upload the rest")
    restore = sub.add_parser("restore", help="a failed job's stored D20 work, same commit only")
    restore.add_argument("--store", default="r2")
    restore.add_argument("--prefix", required=True)
    restore.add_argument("--out-dir", required=True)
    restore.add_argument("--final-key", required=True, help="the finished report's key: refused if it exists")
    restore.add_argument("--void-prefix", required=True)
    restore.add_argument("--rerun-cause", help="after a voided read: what was wrong and what the new code changes")
    canary = sub.add_parser("canary", help="real and blank against the pinned stored report; exit 3, voiding the read, otherwise")
    canary.add_argument("--store", default="r2")
    canary.add_argument("--out-dir", required=True)
    canary.add_argument("--void-prefix", required=True)
    groups = sub.add_parser("groups", help="print one comma list of forms per process")
    groups.add_argument("--processes", type=int, required=True)
    groups.add_argument("--skip", default="")
    groups.add_argument("--stored-in", help="leave out forms already read in this directory")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "groups":
        print("\n".join(form_groups(args.processes, [s for s in args.skip.split(",") if s], args.stored_in)), flush=True)
        return 0
    return dict(fit=run_fit, read=run_read, words=run_words, merge=run_merge, check=run_check, upload=run_upload, restore=run_restore,
                canary=run_canary)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
