"""Parent controls: why does composition read above real signing on v1's parent, and what does the mirror make of the same forms? Report only: nothing trains, and this code gates and selects nothing. Composition of generated word clips (A5's eight word draws, top10 added words) read BLEU 3.30 against real's 2.13, margin over still 0.452 against 0.237, on ``back-translator-kp2d-uwb-v2`` step 76,000, sha ``f1ff9446...`` (``scripts/parent_comp.py``, ``round20/parent-comp/report.json``, sha ``a41781a8...``). That is flagged as not credible until explained. These forms separate two causes: the generator, or the word-by-word format.

**How this will be read (stated before any read; the commit holding this text precedes the job).** No label and no bar: this job sets none, no label gates anything and no target moves.
- **Oracle at or above composition: the format reads high, not the generator's skill.**
- **Oracle far below composition: generated clips suit this reader better than real ones, which supports the shared-source explanation.** Generated clips and the parent's training data share a landmark source (YouTube-ASL, MediaPipe).
- **Shuffling that barely lowers BLEU or margin: the reader isn't using word order.**
**None of this shows human comprehension,** and none of it is a claim about M1 or release. "Far below" is deliberately given no number here: the report gives the paired differences and their intervals, and whoever reads them says what they make of them.
- **The mirror is report only and never moves M1. Generated motion reading high on it is flagged on the same terms as on the parent,** since it shares the parent's lineage (the parent's weights, trained on the MediaPipe source of word clips), **and How2Sign monocular trained it,** so it was fitted to the path word clips come through, and its training holds 234 of D20's clips. The same three readings apply on it, each on its own reading.

**The second instrument (``--instrument``: ``parent``, ``mirror``).** ``mirror`` is ``back-translator-kp2d-uwb-ft-mono-v1`` step 9,000, sha ``1a996cfc...``, the parent fine-tuned on How2Sign monocular states (``scripts/symmetric_ft.py``, label ``unresolved``), loaded by ``parent_lib.load_mono`` from its trained record and held to that hash. Each instrument has its own forms, control, void and report. Through the mirror: the composition, its still and its blank on each draw it reads (it has no stored forms), real and nearest on both paths with their stills, blank at the reference's length, and every control form. **Its shares take its own path's anchors,** real and nearest monocular over the monocular still, since that is its peak; the two-view anchors stand beside them. By clips in and outside its training (v1's list less the refused clips, the same on D20), report only: the plan's table shows that split matters most on this instrument.
**Its control:** real monocular 4.715, nearest monocular 4.783, nearest two-view 3.702, blank 1.159 (the mirror report, sha ``223dd372...``), each within 0.05, and the same forms' hypotheses and NLL against its stored forms (``parent_lib.same_read``). A miss voids the mirror's read, status 3, no retry; a stored object off its hash or digest stops with status 1, nothing voided.
**What the mirror reads of generated motion, and why: the composed state as it stands, the very bytes the parent read.** No conforming, no marks, no trim.
- **Not conformed to the monocular layout.** The reader's input is ``from_state``, ``normalise`` and ``features``: forward kinematics of body and hand rotations and their validity. ``conform`` only renames a channel's frame and convention, so names cannot reach it (``path_read`` found it byte-identical on all 500 real clips; here ``tests/test_parent_controls.py`` holds it for a composed state). Word clips came through the monocular path already, and were conformed to two-view's names only.
- **No monocular marks.** A composed state is valid on every read point, as d2's decode assembles it, so it is the marks-free kind: **its like on the mirror is ``nearest_monocular`` (4.783), with ``real_monocular`` (4.715) beside it.** Generated motion has no tracking gaps of its own. Copying a real clip's would hand the reader that clip's gap pattern, and the path read showed marks can carry a read by themselves, so the form would stop being generated motion alone.
- **Not trimmed or stretched to a monocular clock.** The predicted length is the product being measured, and a clip's own length would leak its reference. Monocular clips run longer than their two-view slices, and the predicted seconds are fitted on two-view durations, so these forms are shorter than what the mirror trained on: the report gives the ratio (``length_over_monocular``), and a margin over the form's own still at its own length cancels flags and duration as on the parent.
- **The same bytes on both instruments** make the two reads a comparison of readers rather than of forms.

**Forms, all on D20: the same 500 clips, the same predicted lengths and timing as the composition's ``core_crossfade@predicted`` of the same draw, on each instrument.**
- ``oracle``: the composition's word list, each word replaced by a REAL word clip from the train records of ``asllvd-text-v1``, ``rit-text-r6``, ``wlasl-text-r6`` in ``round10-mixed-v1/manifest.json``, through d2's round trip (``build_recon_states.decode_state``, step 5000, EMA, no gains: A5's recipe, codes decoded as a generation so every trained channel reads valid, as a generated clip does), then the same cut and stitching. **The real clip is a chance pick, not best of 16:** ``pick_index``, the hash of (draw, entry), one clip per entry in a draw, as a word cache holds one sign per entry. No instrument chooses it, while generated clips were each the recogniser's best of 16. That difference favours the generated clips and is part of what this reads. If a picked clip cannot be served, the next in the pool's fixed order, counted.
- ``shuffled``: the composition's generated clips and planned frames, words in a seeded random order per clip, seeds 0, 1, 2 (``compositional_baseline.shuffle_order``: never the identity from two words up; seed 0 is the order of the repo's own ``/shuffled`` control).
- ``oracle_shuffled``: ``oracle`` under the same permutation, restricted to the words the oracle holds.
- ``covered``, **not in the brief's forms:** the generated clips of exactly the oracle's words, under the oracle's plan. **Why it is read:** the oracle always lacks the added words (below), so oracle against composition mixes "real clips" with "fewer words". Oracle against ``covered`` is the same words with the same frames planned, and ``covered`` against composition is the word-list effect alone. It sets nothing.

**What composition really does (read from the code: ``round15_compose_eval.run``, ``round20_lib.cached_material``, ``round15_stage2.render_form``, ``compositional.render``).** Each word's clip is cut to a core: for a generated clip the sign's median position in the entry's ASLLVD train clips (``sign_core``), else over all of them, margin 0. The sentence takes ``R = round(predicted seconds x 24)`` frames, predicted seconds from the token count (pin -0.083, 0.3691). ``fitted_plan`` shares R among the words by their isolated sign seconds and among the joins by the gap ratio, with no lead and no trail, so **R is the clip's length whatever the words are.** Each core is time-resampled to its planned frames, and each join is ``bridge``, a linear interpolation on each channel's manifold from one core's last frame to the next core's first, never observed.
**An uncovered word is dropped:** a token with no vocabulary entry is simply absent, nothing stands in for it, and the words that remain share R. A sentence left with no word is its blank clip at R. **The oracle takes that same fallback:** a word with no real clip in the three corpora is dropped from its list, and the rest share R.
**Two exact consequences, both reported.** Composition's seven added entries (``you, it, your, this, so, i, or``, from aliases too) have no real clip here: composition serves them only from ASLLRP continuous segments, which the three corpora do not hold, and it never takes a clip of the same spelling from elsewhere (ASLLVD's letter ``i``); the oracle keeps that rule. Sem-Lex-only entries have none either. So the oracle's coverage is below composition's on most clips, and coverage bands are given both by the form's own coverage and by composition's.
**One difference in the cut,** ``round15_stage2.oracle_material``'s: an ASLLVD real clip with its own annotated sign is cut there; anything else takes ``sign_core`` as a generated clip does. Stitching and plan are the same.
**"Same frames" in a shuffle:** each word keeps its planned core frames, and the gap lengths keep their places (``moved_plan``). Every word's rendered core is frame for frame the unshuffled form's, and the total is R. The joins are bridged anew, since their neighbours changed. ``render_form``'s own ``order`` would re-allocate frames by weight, so it is not used for this.

**Fidelity, before any instrument.** Every draw's composition is rebuilt from its stored word clips and windows by this code's plan. Each non-empty clip must hash to the stored ``core_crossfade@predicted`` state of the pinned export, or, since another pod's CPU can round a float differently, hold the same clock and marks with every value within 1e-5 (both counted), or the draw stops with status 1. So the shuffles start from what was read. No read happens without that index.
**Controls.** The four anchors against the parts job's stored forms, pinned by hash (``parent_comp.run_control``); and the first draw's composition read again against its pinned stored form (at least 0.98 of hypotheses identical, BLEU within 0.05, NLL within 0.001). A miss voids the read: status 3, no retry. On the parent, composition's comp, still and blank are then the pinned forms of ``round20/parent-comp`` (digest ``dc3d89b0``), not read again, which also pairs these forms with the very forms of the 3.30.

**Margin** is ``parent_comp``'s: mean NLL of 4 mismatched references minus NLL of its own (``retrieval_candidates``, seed 0), over the form's own still (its nearest-filled middle frame held at its own length, on its own clock). Each form, each seed. **A clip with no word is its blank and counts, with margin exactly zero.** Blank for gap shares: the draw's blank at the composition's length; every form here is that long, which ``check_still_lengths`` and ``same_lengths`` enforce.

**Reported per form, over the draws** (t interval on the draws alone, bootstrap over clips n 2,000 seed 0, draws fixed; seeds averaged within a draw for the shuffled groups, each seed also alone): BLEU; margin over its own still; share of real's margin and of nearest's; BLEU and likelihood gap shares; by word coverage bands (under .25, .25-.5, over .5) by its own coverage and by composition's; per signer, report only; loop share (``eval.loop_share``); content-word overlap with the reference (share of the reference's words outside ``compositional.STOP_WORDS`` found in the hypothesis, the clip's mean; a stated definition, **not the same as the plan's sampled 16.3% and 11.1%, whose definition was not recorded**, so read it only among this report's rows). **Paired differences** with the same draws and clips: every form against composition, oracle against covered, oracle_shuffled against oracle.
**Oracle coverage:** words kept and dropped, by reason; clips fully covered, clips left empty; corpus of the picks; picks that could not be served.

**Disclosures.** The generator trained on these real clips, so they are not independent of it. Real clips are MediaPipe monocular estimates of isolated signers, as generated clips' training data is, so an oracle reading high does not separate source from format as cleanly as a two-view clip would. A different signer at every word. d2's round trip smooths real motion (it keeps about 0.84 of finger speed: ``hand_motion_compare``). A real clip's untracked frames go in as the dataset serves them, with no fill, as the prior's training codes were made.

**Cost, estimated, not measured.** A form costs about 60 s (read 34 s, build about 26 s: ``resources`` of the stored parent report, RTX 6000 Ada), and each read command about 1.5 min to start. Parent, 8 draws x 16 forms: 140 min. Mirror, 3 draws x 19 forms, 7 anchors, the monocular clips: 74 min. Reads run ``READ_PROCS`` at a time, 2 by default: the stored peak of 19.8 GB reserved allows two on 48 GB, and 1.7 times the speed is a guess, so 214 min becomes about 126. Compose about 21 min per draw, 4 at a time: about 50 min. Pins, model fit and both smokes 30 min; two anchor sets' checks 8 min; merges 15 min. **About 3.8 h and about $3.25** at the $0.85/h implied by parent-comp's $1.70 for about 2 h, range about $2.70 to $4.30: the compose time and the parallel speed are the guesses. Without parallel reads, about 5.3 h and $4.50.
**So the mirror reads three draws (1, 2, 3), not eight.** Two draws would leave its t interval almost empty; each further draw costs about $0.20. The parent keeps all eight, which slightly passes the brief's "about $3". ``DRAWS="1 2 3 4 5 6"`` with the mirror's three is about $2.85. **The clock rule, fixed here and blind to every score:** before starting each draw past the minimum (parent four, mirror two), the job starts no more of that instrument's draws if elapsed time plus one more draw's measured wall time would pass its limit (parent ``PARENT_BUDGET_SECONDS`` 10,800 s; mirror ``BUDGET_SECONDS`` 14,400 s, 4 h, about $3.40), and merges what it holds. Each draw's forms are read whole, and a report's ``draws`` says which stand. The measured cost must be reported beside this.

**Refused.** A record outside the three corpora or outside ``train``, and c00 in any case, stops before it is fetched (``check_word_record``). D20 only (``s2_lib``; C20 cannot be named). No v1 read, and no instrument but the two pinned: neither is ever used here for a loss, a rerank or a best-of-N, and no instrument sees a real clip before the pick. A stored form or export on other bytes stops. A smoke (``--smoke-train 5``): 5 D20 sentences composed, read against c02 train references, so the parent sees no D20 text or clip: numbers are plumbing.

    python scripts/parent_controls.py inputs --model-file W/model.json
    python scripts/parent_controls.py compose --draw K --model-file W/model.json --prefix P --work W --device cuda
    python scripts/parent_controls.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/parent_controls.py read --forms control --instrument I --out-dir W/forms --instrument-cache W/i
    python scripts/parent_controls.py control --instrument I --out-dir W/forms --void-prefix V
    python scripts/parent_controls.py read --forms comp-check --draws K --composed P --out-dir W/forms
    python scripts/parent_controls.py comp-check --draw K --out-dir W/forms --void-prefix V
    python scripts/parent_controls.py read --draws 1,2 --instrument I --composed P --resume --out-dir W/forms --instrument-cache W/i
    python scripts/parent_controls.py upload --out-dir W/forms --prefix D/forms
    python scripts/parent_controls.py merge --draws 1,2 --instrument I --composed P --prefix D/forms --output-file W/report.json
    python scripts/parent_controls.py check --report W/report.json

The job is ``runpod/fleet/parent_controls.sh``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import hand_motion_compare as H  # noqa: E402
import parent_comp as C  # noqa: E402
import parent_lib as Q  # noqa: E402
import path_parts as PP  # noqa: E402
import path_read as R  # noqa: E402
import s2_lib as L  # noqa: E402
import symmetric_ft as SYM  # noqa: E402
import tok_probes as P  # noqa: E402
import tok_v1 as T  # noqa: E402
from signtranslator.eval import loop_share, provenance, s0_metrics as M  # noqa: E402

PROBE = "parent-controls"
JOB_FILE = "runpod/fleet/parent_controls.sh"
PREFIX = f"{L.O}/round20/parent-controls"
STORED_COMPOSED = f"{C.PREFIX}/composed"
STORED_FORMS = f"{C.PREFIX}/pods/a/forms"
STORED_REPORT, STORED_REPORT_SHA = f"{C.PREFIX}/report.json", "a41781a8dfe72f7daac0f3bb56c152ff9a4c089b2b1f77321b1905e896f47993"
STORED_DIGEST = "dc3d89b0be92c70ec5bf6cee0d95b41c0821d6140f0d8916ccf2f29cedf6c7b5"
# reports/parent_comp_inventory.json: the stored export's index and the comp, still and blank forms of each draw, by their hashes
STORED = {
    1: dict(index="82ea1571809fc12cb25d7d99ad99a72c01efe15659cc263c5734bd9028fff0b3", comp="07b980790a1f00ec2d404fe466a580eea31fdc198bb6f875157ccdb82f397b4e", still="6129db1b0a5f6648a1f38953544554334056510a5084fcb1fed1636684cec00f", blank="8d4f0230183b9bf5b701541d06c774fc282ce54f172484984528372599cd2223"),
    2: dict(index="db1031c4828c7bb45d9643c049ed07c05d43ef32633740f49e2cd922e272e25b", comp="54ac67e8761ed3c99b9d275cbce88e3db13e6875986bb928bad9d871d6337d31", still="731e87793795f271a3eab4a5bfec9b97daf7f014c358bca3205ae7880e76d895", blank="9e5ca7c8bcaf9a0424493a113ae227126ec6ddc4d294f3c80b0af1f53ecdb5e7"),
    3: dict(index="71fcdbecfbf31397e811b48f05314072a8c7fe58b5e6bcec203e8d1bcd4fc0a0", comp="c7ba21f691f0d84557933f5c347a248ecb0b44044196810514656b8072b4e472", still="b5071b1f52f1a803fa1d691239c26b499518a32bb48dff153bdcb30662aa0e5b", blank="277dec588ed5310319d197976230ad8922882f9e997f8c4806746cb582c782db"),
    4: dict(index="cf3eb9346765366496abaa321022f121e63175d5b050d920a15c41d27c6e09a6", comp="1e3fc3296b10b335e6b22e9b67f5fdbcb52f9adb8ac2bc654bc6582a4b777312", still="94b786634a45cbe42bd8f48e3e1e4906acb8f793820a48f6159dae6f8fd89835", blank="6080034ed72cda4a5a8f6d62f8d4970589960b0574ecc4813c69efa3e6c90c9e"),
    5: dict(index="f8ae5570092abed0153dec88125782d29ba78eeb17e895e11ba818f8d3a421b2", comp="aebf2dab4f775e7ccb560f8d11b290d640dc957c3cb4bca2362f9694058ab948", still="fd860950d1a48d58b22739bf2e626b133461f8d25ea8e7dc22686f7eb1295414", blank="b0bce8e069c6edc1ed59f62638c100ae0f9a02f1b3383ff44c51eb0bf8229c0c"),
    6: dict(index="a885e039a799405b26acb9619a251372cb091b5f6d87cd2748874008d78441da", comp="be7f37c73e95ec1f4bdb6e84f463e52d71d02c99285eec78e97213d9d9bf94e0", still="0c8683a1087d78bd2d4e583646bc2bcd19d7018d5cbda3fc67917d507c5dc963", blank="4b6602525e5cd8c93d58dfbcca457553900913e44d25c25cf0b95bca85dc1d8c"),
    7: dict(index="e38333fdb597f82515ccda5eb893b895eff5ededfcd3fd63abfacca7d81fc133", comp="e1ae0b1b5702b11eecf256f51c8b59f88e7688e0e81d365ebc1fe8323137868b", still="69b6d76b94f0c504034fe940de5c735f7889f3351679698e4ac9da327bc1d4c7", blank="b87422fd288395021e5329a4ecdb39d989b87d6b10430ec652af633cd53fab7b"),
    8: dict(index="caa96c174d953ce80983e197ba62845ebe14f549be63cd83ba56dde0499d9d01", comp="d37a5296e1863b396665c4da28979bbfbe81bfad77ab54e98f5f90e9b58d446f", still="a08205e60ba068f1c0ef25b83e661b5571cb98571a228d707e0cdb719ad7ea19", blank="6a16525eb0b71d614d12dde5d25216be1f9d9b7c0053eb5eda0abc7b917b0de1")}
DRAWS = C.DRAWS
SEEDS = (0, 1, 2)
INSTRUMENTS = dict(parent=dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA),
                   mirror=dict(run=Q.MONO_RUN, checkpoint_key=f"{Q.MONO_PREFIX}/checkpoints/step-00009000.pt", sha256="1a996cfc9c1917181fef849c6a9b8dde0406c34e42fd48a75f1df63fb189ca59"))
MIRROR_ANCHORS = ("real_two_view", "real_monocular", "nearest_two_view", "nearest_monocular", "blank", "still_two_view", "still_monocular")
MIRROR_VALUES = dict(real_monocular=4.715, nearest_monocular=4.783, nearest_two_view=3.702, blank=1.159)      # the mirror report, docs/NEXT_PHASE_PLAN.md: the anchors reproduce these or its read is void
MIRROR_TOLERANCE = P.CANARY["bleu_tolerance"]
MIRROR_REPORT, MIRROR_REPORT_SHA_PREFIX = f"{Q.MONO_PREFIX}/read/report.json", "223dd372"
MIRROR_FORMS = f"{Q.MONO_PREFIX}/read/forms"
MIRROR_DIGEST_PREFIX = "de05576e"
PICK_SEED = "parent-controls|oracle"
TIMING = "sign-fitted"
CORE_MARGIN = 0.
TOP = 10
IDENTITY_TOLERANCE = 1e-5
KINDS = ("oracle", "covered", *(f"shuffled_s{s}" for s in SEEDS), *(f"oracle_shuffled_s{s}" for s in SEEDS))
GROUPS = dict(shuffled=tuple(f"shuffled_s{s}" for s in SEEDS), oracle_shuffled=tuple(f"oracle_shuffled_s{s}" for s in SEEDS))
FORM = re.compile(r"d([1-8])_(oracle_shuffled_s[0-2]|shuffled_s[0-2]|oracle|covered)(_still)?")
BASE_FORM = re.compile(r"d([1-8])_(comp_reread|comp|still|blank)")
SMOKE_CLIPS = C.SMOKE_CLIPS
COST_ESTIMATE = dict(measured=False, hours=3.8, dollars=3.25, range_dollars=[2.7, 4.3], dollars_per_hour=.85, forms_per_draw=dict(parent=16, mirror=19), draws=dict(parent=8, mirror=3), seconds_per_form=60, read_procs=2, parallel_speed_guess=1.7, compose_minutes=50, budget_seconds=dict(parent=10800, mirror=14400), basis="stored parent-comp report's resources (read 34 s, build 26 s per form, RTX 6000 Ada) and its about $1.70 for about 2 h; the compose time is the guess; the measured cost is to be reported beside this")
DISCLOSURES = dict(
    pick="The real clip is a chance pick, one per entry in a draw, by a hash of draw and entry: no instrument chose it. Each generated clip was its recogniser's best of 16.",
    independence="The generator trained on these real clips, and they share a monocular MediaPipe path with its training data and with the parent's: an oracle reading high does not separate source from format as a two-view clip would.",
    added_words="The seven added entries have no real clip in the three corpora here, and a clip of the same spelling from elsewhere is never taken, so the oracle drops them. Oracle against composition therefore mixes real clips with fewer words: oracle against covered is the same words.",
    cut="An ASLLVD real clip with its own annotated sign is cut there; every other clip, generated clips included, takes its entry's median position. Plan and stitching are the same.",
    shuffle="Each word keeps its planned core frames and the gap lengths keep their places, so every word's rendered core is frame for frame the unshuffled form's. The joins are bridged anew.",
    round_trip="d2's round trip smooths real motion; a real clip's untracked frames go in as the dataset serves them, with no fill.",
    content_overlap="Content-word overlap here is the share of the reference's words outside the stop list found in the hypothesis, per clip. It is not the plan's sampled figures, whose definition was not recorded: read it only among this report's rows.",
    mirror="The mirror is report only and never moves M1. It shares the parent's lineage, and How2Sign monocular trained it, so generated motion reading high on it is flagged on the same terms as on the parent. Its training holds 234 of D20's clips, on signers c01 and c05, and c03 selected it.",
    mirror_form="The mirror reads the composed state as it stands, the bytes the parent read: not conformed, since names cannot reach its input; no marks, so its like is nearest_monocular; at the predicted length, shorter than the monocular clips it trained on.",
    humans="None of this shows human comprehension, and none of it is a claim about M1 or release.")
READING = ("Oracle at or above composition: the format reads high, not the generator's skill. Oracle far below composition: generated clips suit this reader better than real ones, which supports the shared-source explanation. Shuffling that barely lowers BLEU or margin: the reader isn't using word order. The mirror is report only and never moves M1; generated motion reading high on it is flagged on the same terms as on the parent, since it shares the parent's lineage and How2Sign monocular trained it. No label is computed and nothing is gated. None of it shows human comprehension.")


@functools.lru_cache(maxsize=1)
def code_digest() -> str:
    try:
        return provenance.code_digest(extra_files=(JOB_FILE, C.JOB_FILE))
    except OSError as error:
        raise SystemExit(f"no code digest ({error}): a read that cannot name its code is void")


# ---- the instruments --------------------------------------------------------------------------------------------------------------------

def check_instrument(name: str) -> dict:
    if name not in INSTRUMENTS:
        raise SystemExit(f"--instrument {name}: one of {sorted(INSTRUMENTS)}")
    return INSTRUMENTS[name]


def load_instrument(name: str, store, args):
    """The named instrument, on its pinned checkpoint. The mirror's trained record names the checkpoint and ``load_mono`` holds it to that record's hash, so a record that moved on refuses here."""
    pin = check_instrument(name)
    reader = PP.load_unfitted(store, args) if name == "parent" else Q.load_mono(store, args)
    if reader.info["sha256"] != pin["sha256"] or reader.info["checkpoint_key"] != pin["checkpoint_key"]:
        raise SystemExit(f"{name}: loaded {reader.info['checkpoint_key']} sha {reader.info['sha256']}, not the pinned {pin['checkpoint_key']} sha {pin['sha256']}")
    return reader


def anchors_of(instrument: str) -> tuple:
    return C.ANCHORS if instrument == "parent" else MIRROR_ANCHORS


def own_path(forms: dict, instrument: str) -> dict:
    """The anchors a form's shares take, under ``parent_comp``'s names: the instrument's own path. Its arithmetic is not copied."""
    if instrument == "parent":
        return {a: forms[a] for a in C.ANCHORS}
    return dict(real_two_view=forms["real_monocular"], blank=forms["blank"], nearest_two_view=forms["nearest_monocular"], still_two_view=forms["still_monocular"])


def anchor_report(forms: dict, instrument: str, idx: np.ndarray) -> dict:
    """Each anchor's margin over its own path's still, and its BLEU."""
    two_view = C.anchor_rows({a: forms[a] for a in C.ANCHORS}, idx)
    if instrument == "parent":
        return two_view
    own = C.anchor_rows(own_path(forms, instrument), idx)
    return dict(two_view, real_monocular=own["real_two_view"], nearest_monocular=own["nearest_two_view"], still_monocular=own["still_two_view"])


def monocular_clips(store, records, refs, declared: bool) -> list:
    """Every record's monocular clip, in list order: the mirror's anchors read the same clips as every other form, so a clip lost refuses."""
    clips, lost, _ = R.open_monocular(store, records, refs, declared)
    if lost or [c.index for c in clips] != list(range(len(records))):
        raise SystemExit(f"{len(lost)} clips have no monocular state, or the clips are out of order: the mirror's anchors would read other clips")
    return clips


def load_mirror_form(store, name: str) -> dict:
    """A stored form of the mirror's own read, of its digest and checkpoint. The report's hash holds the values; the forms are checked by what they say they are."""
    if name not in MIRROR_ANCHORS:
        raise SystemExit(f"{name}: no stored form of the mirror's read under this name")
    key = f"{MIRROR_FORMS}/form-mono__{name}.json"
    read = json.loads(store.get(key))
    if read.get("form") != f"mono__{name}" or read.get("instrument", {}).get("sha256") != INSTRUMENTS["mirror"]["sha256"] or read.get("declared_read") is not True or not str(read.get("code_digest")).startswith(MIRROR_DIGEST_PREFIX):
        raise SystemExit(f"{key}: not the stored mirror read's form")
    return read


def check_mirror_report(store) -> None:
    raw = store.get(MIRROR_REPORT)
    digest = hashlib.sha256(raw).hexdigest()
    if not digest.startswith(MIRROR_REPORT_SHA_PREFIX):
        raise SystemExit(f"{MIRROR_REPORT}: SHA-256 {digest}, expected {MIRROR_REPORT_SHA_PREFIX}...")


def mirror_misses(read: dict, name: str, stored: dict) -> tuple[dict, list[str]]:
    """One anchor against the report's value and its stored form."""
    bleu = M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
    try:
        row = Q.same_read(read, stored)
    except SystemExit as error:
        row = dict(misses=[str(error)])
    misses = [f"{name} {m}" for m in row["misses"]]
    if abs(bleu - MIRROR_VALUES[name]) > MIRROR_TOLERANCE:
        misses.append(f"{name} reads BLEU {bleu}, not the report's {MIRROR_VALUES[name]}")
    return dict(row, read_bleu=bleu, report_bleu=MIRROR_VALUES[name]), misses


def run_mirror_control(args) -> int:
    """Exit 3, voiding the mirror's read, unless its anchors reproduce the report's values and stored forms. A smoke reads train clips, so it compares nothing and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in MIRROR_ANCHORS}
    if not read["real_monocular"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the mirror's control forms")
    check_mirror_report(store)      # a pin failure stops here with status 1: no control runs, nothing is voided
    stored = {name: load_mirror_form(store, name) for name in MIRROR_VALUES}
    rows, misses = {}, []
    for name in MIRROR_VALUES:
        rows[name], missed = mirror_misses(read[name], name, stored[name])
        misses += missed
    result = dict(stored=rows, misses=misses, reproduced=not misses, code_digest=digest, environment=place)
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-mirror-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the mirror's anchors do not reproduce its report: {misses}. The mirror's read is void: no form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def run_control(args) -> int:
    check_instrument(args.instrument)
    return C.run_control(args) if args.instrument == "parent" else run_mirror_control(args)


# ---- what may be read --------------------------------------------------------------------------------------------------------------------

def check_word_record(record: dict) -> dict:
    """The record, or SystemExit before anything is fetched: exactly one of the three corpora, exactly ``train``, never c00. A missing field is a refusal."""
    if record.get("signer_component") == "c00":
        raise SystemExit("refused: How2Sign signer 9 (c00) is never read")
    if record.get("partition") != "train":
        raise SystemExit(f"refused: a word record whose partition is not exactly train ({record.get('sign')})")
    if record.get("corpus") not in H.WORD_CORPORA:
        raise SystemExit(f"refused: a word record outside {H.WORD_CORPORA} ({record.get('sign')})")
    return record


def real_pools(manifest: dict, added) -> dict:
    """{entry: its train records of the three corpora}, in ``word_pools``'s fixed order. An added entry has none, whatever is spelled the same: composition's own rule."""
    from round15_oracle import word_pools
    pools = word_pools(manifest, H.WORD_CORPORA)
    for entry in added:
        pools.pop(entry, None)
    for records in pools.values():
        for record in records:
            check_word_record(record)
    return pools


def pick_index(draw: int, entry: str, size: int) -> int:
    """The pick: fixed by (draw, entry), a chance pick, no instrument."""
    digest = hashlib.sha256(f"{PICK_SEED}|draw{draw}|{entry}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


class RealClips:
    """One draw's real clip per entry, as a word cache holds one generated sign per entry. ``load(record, entry)`` serves and round-trips it; a clip that cannot be served gives way to the next in the pool's order, counted."""

    def __init__(self, pools: dict, draw: int, load) -> None:
        self.pools, self.draw, self.load, self.memo, self.failed = pools, draw, load, {}, 0

    def of(self, entry: str):
        """(record, state), or None: no real clip, so composition's fallback, dropped."""
        if entry not in self.memo:
            self.memo[entry] = self._resolve(entry)
        return self.memo[entry]

    def _resolve(self, entry: str):
        pool = self.pools.get(entry) or []
        start = pick_index(self.draw, entry, len(pool)) if pool else 0
        for offset in range(len(pool)):
            record = check_word_record(pool[(start + offset) % len(pool)])
            try:
                return record, self.load(record, entry)
            except (ValueError, RuntimeError):      # a state whose window holds nothing: counted, not hidden
                self.failed += 1
        return None


def round_tripper(store, device: str, work: Path):
    """``load(record, entry)``: the record as the prior's dataset serves it, then d2's codes decoded as a generation: A5's recipe, no gains."""
    from signtranslator.data_engineering.corpus_manifest import ContentCache
    from build_recon_states import decode_state, load_tokenizer, serve
    from exposure_diagnostic import make_post
    raw = store.get(L.TOKENIZER)
    P.check_sha(raw, L.TOKENIZER_SHA, L.TOKENIZER)
    tokenizer, _ = load_tokenizer(raw, device)
    post = make_post(None)
    template = R.load_template(store)
    cache = ContentCache(work / "cache-words", 2_000_000_000)

    def load(record: dict, entry: str):
        check_word_record(record)
        return decode_state(serve(record, template, cache=cache, fetch=store.get), tokenizer, tokenizer.config.motion, post, text=entry, device=device)
    return load


# ---- the plan's inputs -------------------------------------------------------------------------------------------------------------------

def load_model(path) -> dict:
    """The stage-2 model the composition used, by its pinned hash."""
    import round15_stage2_fit as fit
    model = json.loads(Path(path).read_bytes())
    fit.check_model(model)
    if model["sha256"] != C.MODEL_SHA:
        raise SystemExit(f"the stage-2 model is {model['sha256']}, not the pinned {C.MODEL_SHA}")
    return model


def load_context(store, model_file) -> SimpleNamespace:
    """What ``round15_compose_eval.run`` holds for this arm: the plan's model, vocabulary and core fractions, the sentence's predicted seconds from the pin, and the three corpora's pools. The mix parses whole and is cut at once to train records of the vocabulary's corpora; no other record stays, is used or is printed."""
    from comp_a_lib import arm_of, extra_vocabulary, load_token_list
    from round15_compose_eval import parse_duration_pin
    import round15_stage2_fit as fit
    from signtranslator.generation.compositional import isolated_corpora, training_vocabulary
    model = load_model(model_file)
    manifest = L.load_manifest_at(store, L.EVAL_MIX, L.EVAL_MIX_SHA)
    corpora = isolated_corpora(manifest)
    if model.get("mix_key") != L.EVAL_MIX or list(model.get("corpora") or ()) != list(corpora):
        raise SystemExit("the stage-2 model was fitted on another vocabulary")
    kept = dict(records=[r for r in manifest["records"] if r.get("corpus") in corpora and r.get("partition") == "train" and r.get("signer_component") != "c00"])
    del manifest
    tokens = load_token_list(ROOT / T.TOKENS_FILE, T.TOKENS_SHA)
    arm = arm_of(tokens, TOP)
    vocab = dict(training_vocabulary(kept, corpora), **extra_vocabulary(arm))
    plan_model = dict(model, timing=dict(model["timing"], annotated={k: v for k, v in model["timing"]["annotated"].items() if k not in arm["entries"]}))
    fractions = fit.sign_fractions(kept, vocab)
    widening = tokens.get("min_segment_seconds")
    fractions = dict(fractions, entries={k: v for k, v in fractions["entries"].items() if k not in arm["entries"]}, min_seconds={e: widening for e in arm["entries"]} if widening else {})
    intercept, slope, _ = parse_duration_pin(C.DURATION)
    return SimpleNamespace(model=model, plan_model=plan_model, vocab=vocab, fractions=fractions, duration=dict(intercept=intercept, slope=slope), pools=real_pools(kept, arm["entries"]), added=frozenset(arm["entries"]))


# ---- orders and plans --------------------------------------------------------------------------------------------------------------------

def word_order(n: int, clip: int, seed: int) -> list[int]:
    """Seed ``seed``'s order of a clip's words: the repo's own, never the identity from two words up."""
    from compositional_baseline import shuffle_order
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed}: the seeds are {SEEDS}")
    return shuffle_order(n, clip, seed)


def restricted(order, kept) -> list[int]:
    """``order`` over a clip's words, as an order over the kept ones: the same permutation, with the dropped words taken out."""
    where = {position: k for k, position in enumerate(kept)}
    return [where[position] for position in order if position in where]


def moved_plan(plan: dict, order) -> dict:
    """The plan with each word keeping its planned core frames, and the gap lengths in their places: the total is unchanged, and every word's rendered core is the same frames."""
    if sorted(order) != list(range(len(plan["cores"]))):
        raise ValueError("not an order of this plan's words")
    return dict(plan, cores=[plan["cores"][k] for k in order])


def render_order(pieces, plan: dict, order, sample_id: str):
    from signtranslator.generation.compositional import parts_from_pieces, render
    return render(parts_from_pieces(pieces, order), moved_plan(plan, order), sample_id=sample_id, bridge_mode="linear")


# ---- composing one clip ------------------------------------------------------------------------------------------------------------------

def _never(record):
    raise AssertionError(f"{record.get('sign')}: every real clip is served before the material is built")


def compose_clip(row: dict, ctx, clip_states: dict, real: RealClips) -> dict:
    """{kind: dict(state or None for a blank, words, order, plan, ...)} of one stored composition row, and ``comp`` rebuilt. ``state`` None: no word, so the blank clip at R, built at read time against that read's reference."""
    from signtranslator.generation.composition_v2 import predicted_seconds
    from signtranslator.generation.compositional import parts_from_pieces
    import round15_stage2 as r2
    i, frames = row["clip"], row["frames"]
    empty = dict(state=None, words=[], order=[], dropped=[])
    if row["empty"]:
        return {kind: dict(empty) for kind in ("comp", *KINDS)}
    seconds = predicted_seconds(ctx.duration, row["text"])
    entries = [w["word"] for w in row["words"]]
    job = lambda words: dict(words=words, weights=r2.plan_words(words, ctx.plan_model, ctx.vocab)["weights"])      # noqa: E731
    pieces = [dict(state=clip_states[w["clip_sha256"]], lo=w["lo"], hi=w["hi"]) for w in row["words"]]
    comp, plan = r2.render_form(dict(pieces=pieces, parts=parts_from_pieces(pieces)), job(entries), seconds, ctx.model, timing=TIMING, sample_id=f"round15-eval-{C.PREDICTED}:{i}")
    out = dict(comp=dict(state=comp, words=row["words"], order=list(range(len(entries))), dropped=[], plan=plan))
    orders = {s: word_order(len(entries), i, s) for s in SEEDS}
    for s, order in orders.items():
        out[f"shuffled_s{s}"] = dict(state=render_order(pieces, plan, order, f"parent-controls-shuffled-s{s}:{i}"), words=[row["words"][k] for k in order], order=order, dropped=[], plan=moved_plan(plan, order))
    picked = [real.of(e) for e in entries]
    kept = [p for p, x in enumerate(picked) if x is not None]
    dropped = [dict(word=entries[p], reason=drop_reason(entries[p], ctx, real)) for p, x in enumerate(picked) if x is None]
    if not kept:
        out.update({kind: dict(empty, dropped=dropped) for kind in KINDS if kind.startswith(("oracle", "covered"))})
    else:
        oracle_job = job([entries[p] for p in kept])
        material = r2.oracle_material(oracle_job, i, None, loaded={picked[p][0]["sign"]: picked[p][1] for p in kept}, load=_never, rule=None, timing=TIMING, fractions=ctx.fractions, margin=CORE_MARGIN, tighten=False, records=[picked[p][0] for p in kept])
        oracle, oracle_plan = r2.render_form(material, oracle_job, seconds, ctx.model, timing=TIMING, sample_id=f"parent-controls-oracle:{i}")
        rows = [{k: s.get(k) for k in ("word", "sign", "corpus", "lo", "hi", "reason")} for s in material["segments"]]
        out["oracle"] = dict(state=oracle, words=rows, order=list(range(len(kept))), dropped=dropped, plan=oracle_plan)
        kept_pieces = [pieces[p] for p in kept]
        covered, covered_plan = r2.render_form(dict(pieces=kept_pieces, parts=parts_from_pieces(kept_pieces)), oracle_job, seconds, ctx.model, timing=TIMING, sample_id=f"parent-controls-covered:{i}")
        out["covered"] = dict(state=covered, words=[row["words"][p] for p in kept], order=list(range(len(kept))), dropped=dropped, plan=covered_plan)
        for s, order in orders.items():
            mine = restricted(order, kept)
            out[f"oracle_shuffled_s{s}"] = dict(state=render_order(material["pieces"], oracle_plan, mine, f"parent-controls-oracle-shuffled-s{s}:{i}"), words=[rows[k] for k in mine], order=mine, dropped=dropped, plan=moved_plan(oracle_plan, mine))
    for kind, made in out.items():
        if made["state"] is not None and len(made["state"].timestamps) != frames:
            raise SystemExit(f"clip {i} {kind}: {len(made['state'].timestamps)} frames, not the composition's {frames}")
    return out


def drop_reason(entry: str, ctx, real: RealClips) -> str:
    if entry in ctx.added:
        return "added_entry"
    return "not_served" if real.pools.get(entry) else "no_clip_in_the_three_corpora"


def same_composition(state, row: dict, store) -> tuple[bool, bool]:
    """(the same bytes as the stored export's, every value within ``IDENTITY_TOLERANCE`` with the same clock and marks). Another pod's CPU can round a float differently, which changes a hash and no frame."""
    from build_recon_states import state_bytes
    from evaluate_text_to_motion import load_state
    if state_bytes(state)[1] == row["state_sha256"]:
        return True, True
    stored = load_state(store, row["state_key"], row["state_sha256"])
    if len(state.timestamps) != len(stored.timestamps) or not np.array_equal(state.timestamps, stored.timestamps) or set(state.channels) != set(stored.channels):
        return False, False
    return False, all(np.array_equal(c.valid, stored.channels[name].valid) and np.abs(np.asarray(c.values, np.float64) - np.asarray(stored.channels[name].values, np.float64)).max(initial=0.) <= IDENTITY_TOLERANCE for name, c in state.channels.items())


def coverage_of(rows: list[dict], failed: int) -> dict:
    """Oracle coverage over the clips' composed words."""
    words = [(len(r["composition_words"]), len(r["words"])) for r in rows]
    selected, kept = sum(a for a, _ in words), sum(b for _, b in words)
    reasons, corpora = {}, {}
    for r in rows:
        for d in r["dropped"]:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
        for w in r["words"]:
            corpora[w["corpus"]] = corpora.get(w["corpus"], 0) + 1
    filled = [(a, b) for a, b in words if a]
    return dict(words_selected=selected, words_kept=kept, share_kept=round(kept / selected, 4) if selected else None, dropped=dict(sorted(reasons.items())), corpora=dict(sorted(corpora.items())), clips=len(rows), clips_with_words=len(filled), clips_fully_covered=sum(a == b for a, b in filled), clips_left_empty=sum(b == 0 for _, b in filled), mean_clip_share=round(float(np.mean([b / a for a, b in filled])), 4) if filled else None, picks_not_served=failed)


def order_stats(rows: dict) -> dict:
    """How far the permutations moved anything: clips that could move at all, and clips whose word sequence came out unchanged (a repeated word; for the oracle, too few kept words)."""
    out = {}
    for kind in KINDS:
        if "shuffled" not in kind:
            continue
        here = rows[kind]
        moved = [r for r in here if len(r["words"]) >= 2]
        same = [r for r in moved if [w["word"] for w in r["words"]] == sorted_words(r)]
        out[kind] = dict(clips_with_two_words=len(moved), unchanged_word_sequence=len(same), identity_order=sum(r["order"] == sorted(r["order"]) for r in moved))
    return out


def sorted_words(row: dict) -> list:
    """The clip's words in sentence order, given its row's own order."""
    back = [None] * len(row["order"])
    for at, k in enumerate(row["order"]):
        back[k] = row["words"][at]["word"]
    return back


# ---- the compose step --------------------------------------------------------------------------------------------------------------------

def index_key(prefix: str, k: int) -> str:
    return f"{prefix.rstrip('/')}/draw{k}/index.json"


def load_stored_export(store, k: int, records, clips: int) -> list[dict]:
    """The pinned export's predicted rows, in D20's order, its first ``clips`` for a smoke. Its settings and hash are checked; another export refuses."""
    raw = store.get(C.export_index_key(STORED_COMPOSED, k))
    P.check_sha(raw, STORED[k]["index"], C.export_index_key(STORED_COMPOSED, k))
    return C.load_export(store, STORED_COMPOSED, k, records, True)[C.PREDICTED][:clips]


def check_index(index: dict, k: int, clips: int) -> None:
    differs = [what for what, ok in (("kind", index.get("kind") == f"{PROBE}-composed"), ("draw", index.get("draw") == k), ("clips", index.get("clips") == clips), ("seeds", index.get("seeds") == list(SEEDS)), ("code", index.get("code_digest") == code_digest())) if not ok]
    if differs:
        raise SystemExit(f"draw {k}: the stored index differs in {differs}: it is written once")


def put_states(store, prefix: str, k: int, sign: str, made: dict, pool, pending: list) -> dict:
    """Each state of a clip under its own key, written once; {kind: (key, sha)}. A failed write comes back through ``pending``."""
    from build_recon_states import state_bytes
    tag = hashlib.sha256(sign.encode()).hexdigest()[:20]
    out = {}
    for kind, item in made.items():
        if item["state"] is None or kind == "comp":
            continue
        data, sha = state_bytes(item["state"])
        key = f"{prefix.rstrip('/')}/draw{k}/states/{tag}-{kind}.npz"
        out[kind] = (key, sha)
        pending.append(pool.submit(put_once, store, key, data, sha))
    return out


def put_once(store, key: str, data: bytes, sha: str) -> None:
    if store.exists(key):
        if hashlib.sha256(store.get(key)).hexdigest() != sha:
            raise SystemExit(f"{key} exists with other content: written once")
        return
    store.put(key, data)


def clean_word(word: dict) -> dict:
    return {k: word.get(k) for k in ("word", "sign", "corpus", "lo", "hi", "reason", "clip_sha256") if word.get(k) is not None}


def run_compose(args) -> int:
    """One draw's forms, write-once, no instrument loaded: a stored index of this code is left, another refuses. A rebuilt composition off its stored bytes stops the draw, with no index."""
    from evaluate_text_to_motion import load_state
    from train_text_to_motion import make_store
    P.check_pins()
    C.check_draw(args.draw)
    if args.clips not in (L.D20_CLIPS, *range(1, SMOKE_CLIPS + 1)):
        raise SystemExit(f"--clips {args.clips}: all of D20 ({L.D20_CLIPS}) or a smoke of at most {SMOKE_CLIPS}")
    store = make_store(args.store)
    key = index_key(args.prefix, args.draw)
    if store.exists(key):
        check_index(json.loads(store.get(key)), args.draw, args.clips)
        print(json.dumps(dict(composed=key, stored=True)), flush=True)
        return 0
    started = time.time()
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    rows = load_stored_export(store, args.draw, records, args.clips)
    ctx = load_context(store, args.model_file)
    wanted = {w["clip_sha256"]: w["clip_key"] for r in rows for w in r["words"]}
    with ThreadPoolExecutor(16) as pool:
        clip_states = dict(zip(wanted, pool.map(lambda sha: load_state(store, wanted[sha], sha), wanted)))
    real = RealClips(ctx.pools, args.draw, round_tripper(store, args.device, Path(args.work) / f"draw{args.draw}"))
    table = {kind: [] for kind in ("comp", *KINDS)}
    identical, close = 0, 0
    pending = []
    with ThreadPoolExecutor(8) as pool:
        for n, row in enumerate(rows):
            made = compose_clip(row, ctx, clip_states, real)
            if not row["empty"]:
                same, near = same_composition(made["comp"]["state"], row, store)
                if not near:
                    raise SystemExit(f"draw {args.draw}: clip {row['clip']}'s rebuilt composition is not its stored state: nothing is read")
                identical, close = identical + same, close + near
            where = put_states(store, args.prefix, args.draw, row["sign"], made, pool, pending)
            for kind, item in made.items():
                key_sha = (row["state_key"], row["state_sha256"]) if kind == "comp" and not row["empty"] else where.get(kind, (None, None))
                table[kind].append(dict(clip=row["clip"], sign=row["sign"], text=row["text"], form=kind, empty=item["state"] is None, state_key=key_sha[0], state_sha256=key_sha[1], frames=row["frames"], reference_frames=row["reference_frames"], reference_sha256=row["reference_sha256"], order=item["order"], words=[clean_word(w) for w in item["words"]], dropped=item["dropped"], composition_words=[w["word"] for w in row["words"]]))
            if (n + 1) % 100 == 0:
                print(f"draw {args.draw}: {n + 1}/{len(rows)} clips, {time.time() - started:.0f} s", flush=True)
    for done in pending:
        done.result()
    index = dict(kind=f"{PROBE}-composed", report_only=True, draw=args.draw, clips=args.clips, seeds=list(SEEDS), code_digest=code_digest(), environment=P.environment(), pick=PICK_SEED, stored_export=dict(key=C.export_index_key(STORED_COMPOSED, args.draw), sha256=STORED[args.draw]["index"]), identity=dict(non_empty=close, byte_identical=identical, within_tolerance=close, clips=len(rows)), coverage=coverage_of(table["oracle"], real.failed), orders=order_stats(table), seconds=round(time.time() - started, 1), rows=table)
    store.put(key, json.dumps(index, indent=1, default=float).encode())
    print(json.dumps(dict(composed=key, stored=False, seconds=index["seconds"], identity=index["identity"])), flush=True)
    return 0


# ---- reading -----------------------------------------------------------------------------------------------------------------------------

def read_states(store, rows, refs) -> list:
    """The form's states. A row with no word is its blank clip at the form's length, against this read's reference."""
    from evaluate_text_to_motion import load_state

    def one(pair):
        row, ref = pair
        return C.blank_at_length(ref, row["frames"]) if row["empty"] else load_state(store, row["state_key"], row["state_sha256"])
    with ThreadPoolExecutor(16) as pool:
        return list(pool.map(one, zip(rows, refs)))


def load_composed(store, prefix: str, k: int, records, declared: bool) -> dict:
    """{kind: rows, in list order}, of this code. On D20 each row is the record's. A smoke's rows are the first clips, read against other references."""
    key = index_key(prefix, k)
    if not store.exists(key):
        raise SystemExit(f"{key}: draw {k} is not composed")
    index = json.loads(store.get(key))
    check_index(index, k, len(records) if not declared else L.D20_CLIPS)
    out = {}
    for kind, rows in index["rows"].items():
        if len(rows) != len(records):
            raise SystemExit(f"draw {k} {kind}: {len(rows)} clips, not {len(records)}")
        if declared:
            by_sign = {r["sign"]: r for r in rows}
            if any(r["sign"] not in by_sign for r in records):
                raise SystemExit(f"draw {k} {kind}: a clip is missing")
            for record in records:
                row = by_sign[record["sign"]]
                if row["text"] != record["text"] or row["reference_sha256"] != record["continuous"]["state_sha256"]:
                    raise SystemExit(f"{record['sign']}: draw {k}'s row is another clip's")
            out[kind] = [by_sign[r["sign"]] for r in records]
        else:
            out[kind] = sorted(rows, key=lambda r: r["clip"])
    return out


def has_stored_composition(declared: bool, instrument: str) -> bool:
    """The parent's read of the composition stands, by its pins. A smoke has no stored forms, and the mirror never read it."""
    return declared and instrument == "parent"


def forms_of_draw(k: int, declared: bool, instrument: str = "parent") -> list[str]:
    C.check_draw(k)
    names = [n for kind in KINDS for n in (f"d{k}_{kind}", f"d{k}_{kind}_still")]
    return names if has_stored_composition(declared, instrument) else names + [f"d{k}_comp", f"d{k}_still", f"d{k}_blank"]


def run_read(args) -> int:
    """``--forms control``: the instrument's anchors. ``--forms comp-check --draws K``: the parent's stored composition of draw K, read again. ``--draws``: each draw's forms and their stills, and comp, still and blank where they have no stored forms: the mirror's, and a smoke's."""
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    torch.set_num_threads(args.threads)
    check_instrument(args.instrument)
    if args.forms not in (None, "control", "comp-check"):
        raise SystemExit(f"--forms {args.forms}: only control and comp-check")
    if args.forms == "comp-check" and args.instrument != "parent":
        raise SystemExit("--forms comp-check is the parent's: the mirror has no stored composition")
    if not args.forms and not args.draws:
        raise SystemExit("give --forms or --draws")
    draws = [int(x) for x in args.draws.split(",") if x] if args.draws else []
    for k in draws:
        C.check_draw(k)
    if args.forms == "control" and draws:
        raise SystemExit("--forms control reads the anchors: no --draws")
    if args.forms == "comp-check" and len(draws) != 1:
        raise SystemExit("--forms comp-check reads one draw")
    out_dir = Path(args.out_dir)
    store = make_store(args.store)
    records, declared, described = C.open_list(store, args)
    if args.forms == "control":
        wanted = list(anchors_of(args.instrument))
    elif args.forms == "comp-check":
        wanted = [f"d{draws[0]}_comp_reread"]
    else:
        wanted = [n for k in draws for n in forms_of_draw(k, declared, args.instrument)]
    stored = [n for n in wanted if (out_dir / f"form-{n}.json").exists()]
    if stored and not args.resume:
        raise SystemExit(f"{out_dir / f'form-{stored[0]}.json'} exists (write-once)")
    wanted = [n for n in wanted if n not in stored]
    if not wanted:
        print(json.dumps(dict(read=0, stored=len(stored))), flush=True)
        return 0
    digest, place = code_digest(), P.environment()
    if args.forms != "control" and declared:
        R.passed_control(out_dir, digest, place)
    if not args.forms and has_stored_composition(declared, args.instrument):
        passed_comp_check(out_dir, digest, place)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    refs = L.load_states(store, records)
    tokenizer = L.load_tokenizer(store)
    candidates = M.retrieval_candidates(len(texts), C.CANDIDATES, seed=0)
    reader = load_instrument(args.instrument, store, args)
    shared = dict(reader=reader, texts=texts, signs=signs, groups=groups, candidates=candidates, refs=refs, args=args, declared=declared, described=described, digest=digest, place=place, out_dir=out_dir, probe=PROBE)
    anchors = [n for n in wanted if n in anchors_of(args.instrument)]
    clips = monocular_clips(store, records, refs, declared) if args.instrument == "mirror" and anchors else []
    for name in anchors:
        started = time.time()
        if args.instrument == "parent":
            C.read_one(name, C.build_anchor(name, refs, tokenizer), list(refs), None, started=started, **shared)
        else:
            _, states, masks = SYM.build_form(name, refs, clips, tokenizer)
            C.read_one(name, states, masks, None, started=started, **shared)
    for k in draws:
        composed = load_composed(store, args.composed, k, records, declared)
        here = [n for n in wanted if n.startswith(f"d{k}_")]
        for kind in ("comp", *KINDS):
            names = [n for n in here if n in (f"d{k}_{kind}", f"d{k}_{kind}_still", *((f"d{k}_still", f"d{k}_blank", f"d{k}_comp_reread") if kind == "comp" else ()))]
            if not names:
                continue
            started = time.time()
            rows = composed[kind]
            states = read_states(store, rows, refs)
            for name in names:
                how = "blank" if name.endswith("_blank") else "still" if name.endswith("still") else "as_composed"
                C.read_one(name, C.kind_states(how, states, rows, refs, tokenizer), list(refs), rows, started=started, **shared)
                started = time.time()
            del states
    return 0


def passed_comp_check(out_dir: Path, digest: str, place: dict) -> dict:
    path = out_dir / "comp-check.json"
    record = json.loads(path.read_bytes()) if path.exists() else None
    if record is None or record.get("reproduced") is not True or record.get("code_digest") != digest or record.get("environment") != place:
        raise SystemExit(f"{path}: the stored composition has not been reproduced by this code and environment; the read is void")
    return record


def stored_form_key(k: int, kind: str) -> str:
    return f"{STORED_FORMS}/form-d{k}_{kind}.json"


def load_stored_form(store, k: int, kind: str) -> dict:
    """A form of the composition's own read, by its pinned hash, of its own code and instrument."""
    C.check_draw(k)
    if kind not in ("comp", "still", "blank"):
        raise SystemExit(f"{kind}: no stored form is pinned under this name")
    key = stored_form_key(k, kind)
    raw = store.get(key)
    P.check_sha(raw, STORED[k][kind], key)
    read = json.loads(raw)
    if read.get("form") != f"d{k}_{kind}" or read.get("instrument", {}).get("sha256") != PP.UNFITTED_SHA or read.get("declared_read") is not True or read.get("code_digest") != STORED_DIGEST:
        raise SystemExit(f"{key}: not the stored parent read's form")
    return read


def run_comp_check(args) -> int:
    """Exit 3, voiding the read, unless the composition read again here reproduces its stored form: this environment reads as that one did."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    read = json.loads((out / f"form-d{args.draw}_comp_reread.json").read_bytes())
    stored = load_stored_form(store, args.draw, "comp")      # a pin failure stops here with status 1: nothing is voided
    try:
        row = Q.same_read(read, stored)
    except SystemExit as error:
        row = dict(misses=[str(error)])
    result = dict(draw=args.draw, stored=row, reproduced=not row["misses"], code_digest=read["code_digest"], environment=read["environment"])
    path = out / "comp-check.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{read['code_digest']}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-void", code_digest=read["code_digest"], environment=read["environment"], comp_check=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the stored composition is not reproduced: {row['misses']}. The read is void: no form follows, no retry; a rerun needs new code and a recorded cause", flush=True)
    return 3


def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def run_inputs(args) -> int:
    """The job's first step: every pinned stored object, the stored report and the stage-2 model, fitted once to a file for the compose steps. Prints hashes, no score."""
    from train_text_to_motion import make_store
    import round15_stage2_fit as fit
    P.check_pins()
    store = make_store(args.store)
    P.check_sha(store.get(STORED_REPORT), STORED_REPORT_SHA, STORED_REPORT)
    for k in DRAWS:
        P.check_sha(store.get(C.export_index_key(STORED_COMPOSED, k)), STORED[k]["index"], C.export_index_key(STORED_COMPOSED, k))
        for kind in ("comp", "still", "blank"):
            load_stored_form(store, k, kind)
    for name in C.ANCHORS:
        Q.stored_form(store, Q.STORED_PARENT, name)
    check_mirror_report(store)
    for name in MIRROR_VALUES:
        load_mirror_form(store, name)
    path = Path(args.model_file)
    if not path.exists():
        model, _ = fit.fit_model(fit.agc._store(args.store), L.EVAL_MIX, L.EVAL_MIX_SHA)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model))
    print(json.dumps(dict(inputs=dict(draws=list(DRAWS), report=STORED_REPORT_SHA[:8], model=load_model(path)["sha256"][:8]))), flush=True)
    return 0


# ---- the report --------------------------------------------------------------------------------------------------------------------------

def content_words(text: str) -> set:
    from signtranslator.generation.compositional import STOP_WORDS, normalise
    return {t for t in normalise(text) if t not in STOP_WORDS}


def content_overlap(hypothesis: str, reference: str) -> float | None:
    """The share of the reference's content words found in the hypothesis. None for a reference with none."""
    wanted = content_words(reference)
    return len(wanted & content_words(hypothesis)) / len(wanted) if wanted else None


def hypothesis_values(read: dict) -> dict:
    """Per clip: the loop flag, the content-word overlap (NaN for a reference with none), the hypothesis's words."""
    overlap = [content_overlap(h, r) for h, r in zip(read["hypotheses"], read["references"])]
    return dict(loop_share=np.asarray(loop_share.loop_flags(read["hypotheses"]), np.float64), content_word_overlap=np.array([np.nan if x is None else x for x in overlap], np.float64), words=np.array([len(h.split()) for h in read["hypotheses"]], np.float64))


def nan_means(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Each row of ``idx``: the mean over the clips that have a value. A row with none reads NaN, which ``_quantiles`` leaves out."""
    picked = values[idx]
    kept = ~np.isnan(picked)
    counts = kept.sum(1)
    return np.divide(np.where(kept, picked, 0.).sum(1), counts, out=np.full(len(picked), np.nan), where=counts > 0)


def hypothesis_row(members: dict, idx: np.ndarray) -> dict:
    """{draw: [forms]} -> loop share, content-word overlap and hypothesis words, seeds averaged within a draw, over the draws."""
    draws = sorted(members)
    values = {k: [hypothesis_values(read) for read in members[k]] for k in draws}
    every = np.arange(idx.shape[1])[None]
    out = {}
    for name in ("loop_share", "content_word_overlap", "words"):
        out[name] = C.over_draws([float(np.mean([nan_means(v[name], every)[0] for v in values[k]])) for k in draws], np.stack([np.mean([nan_means(v[name], idx) for v in values[k]], 0) for k in draws]))
    return out


def as_parent_view(k: int, read: dict, still: dict, blank: dict) -> dict:
    """A form under ``parent_comp``'s names, so its arithmetic is not copied. The true-length slots hold the same form: no row here reads them."""
    return {f"d{k}_comp": read, f"d{k}_still": still, f"d{k}_comp_true": read, f"d{k}_still_true": still, f"d{k}_blank": blank}


def same_lengths(k: int, name: str, read: dict, composition: dict) -> None:
    if read["frames"] != composition["frames"]:
        raise SystemExit(f"draw {k}: {name} is not as long as the composition on every clip")


def view_arrays(kind_reads: dict, base: dict, anchors: dict, idx: np.ndarray, name: str) -> tuple[dict, dict, dict]:
    """(arrays, rows, views) of one form over the draws: ``{k: (read, still)}``, its own still at its own length, the draw's stored blank."""
    arrays, rows, views = {}, {}, {}
    for k, (read, still) in sorted(kind_reads.items()):
        same_lengths(k, name, read, base[k]["comp"])
        view = as_parent_view(k, read, still, base[k]["blank"])
        C.check_still_lengths(k, view)
        built = C.draw_rows(k, view, anchors, idx)
        rows[k] = built.pop("row")
        arrays[k] = built
        views.update(view)
    return arrays, rows, views


BLOCK_ROWS = ("margin", "share_of_real", "share_of_nearest", "bleu", "bleu_gap_share", "likelihood_gap_share")


def _bleu(stats) -> np.ndarray:
    return M.bleu_from_statistics(stats)


METRICS = dict(margin=(lambda x: float(x["margin"].mean()), lambda x, idx: x["margin"][idx].mean(1)),
               share_of_real=(lambda x: float(x["margin"].mean() / x["real"].mean()) if x["real"].mean() else float("nan"), lambda x, idx: C.ratio_draws(x["margin"][idx].mean(1), x["real"][idx].mean(1))),
               bleu=(lambda x: float(_bleu(x["stats"].sum(0))), lambda x, idx: _bleu(x["stats"][idx].sum(1))),
               bleu_gap_share=(lambda x: _gap_point(x), lambda x, idx: C.ratio_draws(_bleu(x["stats"][idx].sum(1)) - _bleu(x["blank_stats"][idx].sum(1)), _bleu(x["real_stats"][idx].sum(1)) - _bleu(x["blank_stats"][idx].sum(1)))),
               likelihood_gap_share=(lambda x: float(x["likelihood_top"].mean() / x["likelihood_bottom"].mean()) if x["likelihood_bottom"].mean() else float("nan"), lambda x, idx: C.ratio_draws(x["likelihood_top"][idx].mean(1), x["likelihood_bottom"][idx].mean(1))))


def _gap_point(x: dict) -> float:
    form, real, blank = (float(_bleu(x[f].sum(0))) for f in ("stats", "real_stats", "blank_stats"))
    return (form - blank) / (real - blank) if real != blank else float("nan")


def group_rows(members: list[dict], idx: np.ndarray) -> dict:
    """Several seeds' arrays ({k: arrays} each): each metric averaged over the seeds within a draw, then over the draws."""
    draws = sorted(members[0])
    out = {}
    for name, (point, boot) in METRICS.items():
        out[name] = C.over_draws([float(np.mean([point(m[k]) for m in members])) for k in draws], np.stack([np.mean([boot(m[k], idx) for m in members], 0) for k in draws]))
    return out


def paired_rows(mine: list[dict], other: list[dict], idx: np.ndarray, bands: dict | None = None) -> dict:
    """Mine minus other on the same draws and clips: every metric, over the draws, seeds averaged within a draw. ``bands``, the composition's arrays: margin and BLEU differences on the same clips, by composition's coverage."""
    draws = sorted(mine[0])
    out = {}
    for name, (point, boot) in METRICS.items():
        here = [float(np.mean([point(m[k]) for m in mine])) - float(np.mean([point(m[k]) for m in other])) for k in draws]
        out[name] = C.over_draws(here, np.stack([np.mean([boot(m[k], idx) for m in mine], 0) - np.mean([boot(m[k], idx) for m in other], 0) for k in draws]))
    if bands is not None:
        out["by_composition_coverage"] = {f"{lo}-{hi}": band_difference(mine, other, draws, bands, lo, hi) for lo, hi in C.COVERAGE_BINS}
    return out


def band_mask(x: dict, lo: float, hi: float) -> np.ndarray:
    return x["non_empty"] & (x["coverage"] > lo) & (x["coverage"] <= hi)


def band_difference(mine: list[dict], other: list[dict], draws, bands: dict, lo: float, hi: float) -> dict:
    margin, bleu, clips = [], [], []
    for k in draws:
        mask = band_mask(bands[k], lo, hi)
        clips.append(int(mask.sum()))
        if not mask.any():
            continue
        margin.append(float(np.mean([m[k]["margin"][mask].mean() for m in mine]) - np.mean([m[k]["margin"][mask].mean() for m in other])))
        bleu.append(float(np.mean([_bleu(m[k]["stats"][mask].sum(0)) for m in mine]) - np.mean([_bleu(m[k]["stats"][mask].sum(0)) for m in other])))
    return dict(clips_per_draw=round(float(np.mean(clips)), 1), margin=round(float(np.mean(margin)), 5) if margin else None, bleu=round(float(np.mean(bleu)), 4) if bleu else None)


def coverage_rows(members: list[dict], bands: dict | None = None) -> dict:
    """Report only: margin, its share of real and BLEU, by coverage: each seed's own (``bands`` None) or a fixed set of arrays' (composition's), the draws' means. A band with no clip has none."""
    draws = sorted(members[0])

    def row(lo: float, hi: float) -> dict:
        margin, real, bleu, clips = [], [], [], []
        for k in draws:
            for m in members:
                mask = band_mask((bands or m)[k], lo, hi)
                clips.append(int(mask.sum()))
                if mask.any():
                    margin.append(float(m[k]["margin"][mask].mean()))
                    real.append(float(m[k]["real"][mask].mean()))
                    bleu.append(float(_bleu(m[k]["stats"][mask].sum(0))))
        if not margin:
            return dict(clips_per_draw=0.)
        return dict(clips_per_draw=round(float(np.mean(clips)), 1), margin=round(float(np.mean(margin)), 5), real_margin=round(float(np.mean(real)), 5), share_of_real=round(float(np.mean(margin) / np.mean(real)), 4) if np.mean(real) else None, bleu=round(float(np.mean(bleu)), 4))
    return {f"{lo}-{hi}": row(lo, hi) for lo, hi in C.COVERAGE_BINS}


def signer_bleu(arrays: dict, groups) -> dict:
    """Report only: each signer's BLEU, the mean of the draws."""
    labels = np.asarray(groups)
    return {str(g): round(float(np.mean([_bleu(arrays[k]["stats"][labels == g].sum(0)) for k in sorted(arrays)])), 4) for g in sorted(set(groups))}


def training_rows(arrays: dict, masks: dict) -> dict:
    """Report only: margin, its share of real and BLEU, on clips in and outside v1's training list, which is the mirror's on D20: here training status and signer are confounded."""
    out = C.subset_rows(arrays, masks)
    for name, mask in masks.items():
        if out[name]:
            out[name]["bleu"] = round(float(np.mean([_bleu(arrays[k]["stats"][mask].sum(0)) for k in sorted(arrays)])), 4)
    return out


def form_block(name: str, arrays: dict, rows: dict, views: dict, anchors: dict, first: dict, reads: dict, idx: np.ndarray, composition: dict, masks: dict) -> dict:
    """One form's rows."""
    with warnings.catch_warnings():      # parent_comp's length block, which no row here reads, regresses on a ratio that may not vary
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        block = C.composition_block(arrays, idx)
        pod = dict(view=dict(forms={**anchors, **views}))
        signers = C.signer_rows(pod, {k: "view" for k in arrays}, first["signs"], first["groups"])
    for signer, bleu in signer_bleu(arrays, first["groups"]).items():
        signers[signer]["bleu"] = bleu
    return dict(form=name, draws=sorted(arrays), per_draw={str(k): rows[k] for k in sorted(rows)}, **{k: block[k] for k in BLOCK_ROWS}, by_own_coverage=coverage_rows([arrays]), by_composition_coverage=coverage_rows([arrays], composition), by_v1_training=training_rows(arrays, masks), per_signer=signers, per_signer_note="report only: few clips make these rows noisy", hypotheses=hypothesis_row({k: [reads[k]] for k in reads}, idx), empty_clips={str(k): rows[k]["empty"] for k in sorted(rows)})


def load_forms(store, prefix: str) -> tuple[dict, dict | None, dict | None]:
    """({name: form}, the anchors' control, the composition check). Every key is named exactly; other files are left."""
    base = f"{prefix.rstrip('/')}/"
    forms, control, comp_check = {}, None, None
    for key in store.list(base):
        name = key[len(base):]
        if name == "control.json":
            control = json.loads(store.get(key))
        elif name == "comp-check.json":
            comp_check = json.loads(store.get(key))
        elif name.startswith("form-") and name.endswith(".json"):
            form = name[5:-5]
            if form not in MIRROR_ANCHORS and FORM.fullmatch(form) is None and BASE_FORM.fullmatch(form) is None:
                raise SystemExit(f"unknown form {form}")
            read = json.loads(store.get(key))
            if read.get("form") != form:
                raise SystemExit(f"{key} holds form {read.get('form')!r}")
            forms[form] = read
    return forms, control, comp_check


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    pin = check_instrument(args.instrument)
    store = make_store(args.store)
    L.refuse_existing(store, args.output_key, args.output_file)
    forms, control, comp_check = load_forms(store, args.prefix)
    if not forms:
        raise SystemExit(f"{args.prefix}: no form")
    digests = {r.get("code_digest") for r in forms.values()}
    if digests != {code_digest()}:
        raise SystemExit(f"the forms are of code {sorted(str(d) for d in digests)}, not this code {code_digest()}: a stale read")
    if len({(r["declared_read"], r["list"]["list"], tuple(r["signs"]), tuple(r["references"])) for r in forms.values()}) != 1:
        raise SystemExit("the forms read other clips, another list, or a smoke mixed with a declared read")
    if {r["instrument"]["sha256"] for r in forms.values()} != {pin["sha256"]}:
        raise SystemExit(f"a form was not read by the {args.instrument}")
    if any(a not in forms for a in anchors_of(args.instrument)):
        raise SystemExit("the anchors are missing")
    _, place = P.check_same_run(forms, "the forms")
    first = forms["real_two_view"]
    declared = first["declared_read"]
    stored_composition = has_stored_composition(declared, args.instrument)
    if declared and (not control or control.get("reproduced") is not True or control.get("code_digest") != code_digest() or control.get("environment") != place):
        raise SystemExit("no passed control of this code and environment; the read is void")
    if stored_composition and (not comp_check or comp_check.get("reproduced") is not True or comp_check.get("code_digest") != code_digest() or comp_check.get("environment") != place):
        raise SystemExit("no passed composition check of this code and environment; the read is void")
    draws = [int(x) for x in args.draws.split(",") if x]
    if not draws:
        raise SystemExit("--draws names no draw")
    for k in draws:
        C.check_draw(k)
        missing = [n for n in forms_of_draw(k, declared, args.instrument) if n not in forms]
        if missing:
            raise SystemExit(f"draw {k}: {missing[0]} is missing, of {len(missing)} forms")
    base = {k: {kind: load_stored_form(store, k, kind) if stored_composition else forms[f"d{k}_{kind}"] for kind in ("comp", "still", "blank")} for k in draws}
    for k in draws:
        for kind, read in base[k].items():
            if read["signs"] != first["signs"] or read["references"] != first["references"]:
                raise SystemExit(f"draw {k}: the stored {kind} holds other clips or another order")
    n = len(first["signs"])
    idx = np.random.default_rng(0).integers(0, n, (C.BOOT, n))
    anchors = own_path(forms, args.instrument)
    masks = Q.subset_masks(first["signs"], Q.v1_training_signs(store))
    composed = {k: json.loads(store.get(index_key(args.composed, k))) for k in draws}
    for k, index in composed.items():
        check_index(index, k, L.D20_CLIPS if declared else n)
    with np.errstate(divide="ignore", invalid="ignore"):      # a resample's ratio can hold a zero denominator: ``_quantiles`` leaves what is not finite out
        # composition first: its coverage is the bands' fixed set
        comp_reads = {k: (base[k]["comp"], base[k]["still"]) for k in draws}
        comp_arrays, comp_rows, comp_views = view_arrays(comp_reads, base, anchors, idx, "the composition")
        arrays, blocks = {}, {}
        for kind in KINDS:
            reads = {k: (forms[f"d{k}_{kind}"], forms[f"d{k}_{kind}_still"]) for k in draws}
            here, rows, views = view_arrays(reads, base, anchors, idx, kind)
            arrays[kind] = here
            blocks[kind] = form_block(kind, here, rows, views, anchors, first, {k: reads[k][0] for k in draws}, idx, comp_arrays, masks)
        composition = form_block("composition", comp_arrays, comp_rows, comp_views, anchors, first, {k: base[k]["comp"] for k in draws}, idx, comp_arrays, masks)
        groups = {name: dict(seeds=list(SEEDS), **group_rows([arrays[m] for m in members], idx), by_own_coverage=coverage_rows([arrays[m] for m in members]), by_composition_coverage=coverage_rows([arrays[m] for m in members], comp_arrays), hypotheses=hypothesis_row({k: [forms[f"d{k}_{m}"] for m in members] for k in draws}, idx)) for name, members in GROUPS.items()}
        shuffled, oracle_shuffled = [arrays[m] for m in GROUPS["shuffled"]], [arrays[m] for m in GROUPS["oracle_shuffled"]]
        paired = dict(oracle_minus_composition=paired_rows([arrays["oracle"]], [comp_arrays], idx, comp_arrays), covered_minus_composition=paired_rows([arrays["covered"]], [comp_arrays], idx, comp_arrays), oracle_minus_covered=paired_rows([arrays["oracle"]], [arrays["covered"]], idx, comp_arrays), shuffled_minus_composition=paired_rows(shuffled, [comp_arrays], idx, comp_arrays), oracle_shuffled_minus_composition=paired_rows(oracle_shuffled, [comp_arrays], idx, comp_arrays), oracle_shuffled_minus_oracle=paired_rows(oracle_shuffled, [arrays["oracle"]], idx, comp_arrays))
    real_hypotheses = hypothesis_row({0: [anchors["real_two_view"]]}, idx)      # the instrument's own path
    length_over_monocular = None if args.instrument == "parent" else round(float(np.mean([a / b for a, b in zip(base[draws[0]]["comp"]["frames"], forms["real_monocular"]["frames"])])), 4)
    coverage = {str(k): composed[k]["coverage"] for k in draws}
    alerts = [f"draw {k}: its oracle holds no word" for k in draws if not composed[k]["coverage"]["words_kept"]]
    report = dict(kind=f"{PROBE}-report", report_only=True, note=C.NOT_DECIDED, reading=READING, declared_read=declared, smoke=None if declared else "SMOKE: composed D20 sentences against c02 train references; these numbers are plumbing, not a read", list=first["list"], code_digest=code_digest(), environment=place, instrument=first["instrument"], instrument_name=args.instrument, own_path_anchors=list(anchors), length_over_monocular=length_over_monocular, draws=draws, draws_note=None if tuple(draws) == DRAWS else f"fewer than eight: {draws}", seeds=list(SEEDS), disclosures=DISCLOSURES, cost_estimate=COST_ESTIMATE, alerts=alerts, stored=dict(report_sha256=STORED_REPORT_SHA, digest=STORED_DIGEST) if stored_composition else None,
                  anchors=anchor_report(forms, args.instrument, idx), real_hypotheses=real_hypotheses, composition=composition, forms=blocks, groups=groups, paired=paired, oracle_coverage=coverage, orders={str(k): composed[k]["orders"] for k in draws}, identity={str(k): composed[k]["identity"] for k in draws},
                  checks=dict(controls=dict(anchors=control, **(dict(comp_check=comp_check) if stored_composition else {})), blank_identical_to_real=round(float(np.mean([a == b for a, b in zip(anchors["blank"]["hypotheses"], anchors["real_two_view"]["hypotheses"])])), 4)),
                  resources=dict(forms={k: {x: r[x] for x in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for k, r in forms.items()}, compose_seconds={str(k): composed[k]["seconds"] for k in draws}))
    L.write_once(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, report_only=True, declared_read=declared, instrument=args.instrument, draws=draws, anchors=report["anchors"], composition={k: composition[k] for k in BLOCK_ROWS}, forms={f: {k: blocks[f][k] for k in ("margin", "bleu")} for f in ("oracle", "covered")}, groups={f: {k: groups[f][k] for k in ("margin", "bleu")} for f in groups}, paired={f: {k: paired[f][k] for k in ("margin", "bleu")} for f in paired}, oracle_coverage=coverage, alerts=alerts)), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs", help="the job's first step: every pin, the model fitted once; prints no score")
    inputs.add_argument("--store", default="r2")
    inputs.add_argument("--model-file", required=True)
    compose = sub.add_parser("compose", help="one draw's forms; no instrument is loaded")
    compose.add_argument("--store", default="r2")
    compose.add_argument("--draw", type=int, required=True)
    compose.add_argument("--clips", type=int, default=L.D20_CLIPS)
    compose.add_argument("--work", required=True)
    compose.add_argument("--prefix", required=True)
    compose.add_argument("--model-file", required=True)
    compose.add_argument("--device", default="cpu")
    pre = sub.add_parser("preflight", help="the read is not finished or void")
    pre.add_argument("--store", default="r2")
    pre.add_argument("--out-dir", required=True)
    pre.add_argument("--final-key", required=True)
    pre.add_argument("--void-prefix", required=True)
    pre.add_argument("--rerun-cause")
    read = sub.add_parser("read")
    read.add_argument("--store", default="r2")
    read.add_argument("--instrument", default="parent", help=f"one of {sorted(INSTRUMENTS)}, each pinned")
    read.add_argument("--out-dir", required=True)
    read.add_argument("--forms", help="control: the anchors; comp-check: one draw's stored composition read again")
    read.add_argument("--draws", help="comma list: each draw's forms")
    read.add_argument("--composed", help="the prefix of this job's composed forms")
    read.add_argument("--resume", action="store_true")
    read.add_argument("--nll-group", type=int, default=4)
    read.add_argument("--instrument-cache")
    read.add_argument("--smoke-train", type=int, default=0, help=f"c02 train references, at most {SMOKE_CLIPS}, not D20")
    read.add_argument("--device", default="cpu")
    read.add_argument("--threads", type=int, default=1)
    control = sub.add_parser("control", help="the anchors against their stored reads; exit 3, voiding the read, otherwise")
    control.add_argument("--store", default="r2")
    control.add_argument("--instrument", default="parent")
    control.add_argument("--out-dir", required=True)
    control.add_argument("--void-prefix", required=True)
    check_comp = sub.add_parser("comp-check", help="a draw's composition, read again, against its stored form; exit 3, voiding the read, otherwise")
    check_comp.add_argument("--store", default="r2")
    check_comp.add_argument("--draw", type=int, required=True)
    check_comp.add_argument("--out-dir", required=True)
    check_comp.add_argument("--void-prefix", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--store", default="r2")
    merge.add_argument("--instrument", default="parent")
    merge.add_argument("--draws", default=",".join(str(k) for k in DRAWS))
    merge.add_argument("--composed", required=True)
    merge.add_argument("--prefix", required=True)
    merge.add_argument("--output-key")
    merge.add_argument("--output-file")
    check = sub.add_parser("check")
    check.add_argument("--report", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--store", default="r2")
    upload.add_argument("--out-dir", required=True)
    upload.add_argument("--prefix", required=True)
    upload.add_argument("--partial", action="store_true")
    put = sub.add_parser("put-report")
    put.add_argument("--store", default="r2")
    put.add_argument("--file", required=True)
    put.add_argument("--key", required=True)
    log = sub.add_parser("upload-log")
    log.add_argument("--store", default="r2")
    log.add_argument("--file", required=True)
    log.add_argument("--key", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return dict(inputs=run_inputs, compose=run_compose, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=C.run_check, upload=PP.run_upload, **{"comp-check": run_comp_check, "upload-log": R.run_upload_log, "put-report": C.run_put_report})[args.command](args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
