"""Parent controls: why does composition read above real signing on v1's parent, and what does the mirror make of the same forms? Report only: nothing trains, and this code gates and selects nothing. Composition of generated word clips (A5's eight word draws, top10 added words) read BLEU 3.30 against real's 2.13, margin over still 0.452 against 0.237, on ``back-translator-kp2d-uwb-v2`` step 76,000, sha ``f1ff9446...`` (``scripts/parent_comp.py``, ``round20/parent-comp/report.json``, sha ``a41781a8...``). That is flagged as not credible until explained. These forms are controls on it: real clips in the same format, the same clips in another order. They narrow the explanations, and they do not by themselves tell generator, format and source apart, for the reasons stated below.

**How this will be read (stated before any read; the commit holding this text precedes the job).** No label and no bar: this job sets none, no label gates anything and no target moves.

**The rule for every contrast, on one instrument at a time.** A contrast is decided on its **paired t interval over draws (``t_ci95``), on two metrics: margin over still, and BLEU.** It is **above** or **below** only if the t interval excludes zero on BOTH metrics in the same direction. An interval holding zero, or metrics that disagree, is **unresolved**. The clip bootstrap interval is reported beside and decides nothing. **Three mirror draws make unresolved likely there.** Parent minus mirror crosses instruments, stands beside the reading and is not part of it. The report states each contrast and the cell (``reading_contrasts``); a cell gates nothing.

**Two contrasts, and the rows that answer them.**
- **Generated against real clips is ``oracle`` minus ``covered``:** the same words under the same plan. It is never oracle against composition, which also differs in words.
- **The format is ``oracle`` against real signing** (``paired.form_minus_real``): real word clips, stitched word by word, against real sentences. On the parent that is real two-view. **On the mirror the like is ``nearest_monocular``** (``paired.form_minus_nearest``), since composed motion has no marks, **with ``real_monocular`` beside it.**

**The four cells, and unresolved.**
- **Oracle above the format's like, and above covered:** the format reads high, and real clips read higher in it than generated ones.
- **Oracle above the format's like, but below covered, the middle case:** the format reads high AND generated clips read higher still. Both hold, and neither explains the 3.30 alone.
- **Oracle below the format's like, and above covered:** real clips in this format do not reach real signing, and generated clips fall below them. Composition's lead would then sit in **the words the oracle lacks and the frames they free:** read ``covered_minus_composition``.
- **Oracle below the format's like, and below covered:** the format alone does not lift real clips, and generated clips read higher than real ones in it.
- **Either contrast unresolved: the outcome is unresolved,** and the report says which. No outcome sits between these.
**The words the oracle lacks and the frames they free** are the seven frequent added entries, Sem-Lex-only words and words not served, with R reshared among those left. So covered differs from composition in words and in pace at once.

**What an oracle read cannot do.** Neither a low nor a high oracle separates format from source by itself, since **real word clips are MediaPipe monocular too:** the landmark source of the parent's training data and of the generator's. A high oracle may be the format or that shared source. **Every explanation for a low oracle stays live,** and these rows do not choose among them:
- selection: each generated clip was a recogniser's best of 16, and each real clip is a chance pick;
- d2 decoding: a generated clip is codes the prior learned to emit, and a real clip is encoded and decoded, which smooths it;
- a different signer at every word, where the generator has no signer of its own;
- the cut: the sign's position is fitted on ASLLVD's annotations and applied to WLASL and RIT clips, where it can miss the sign;
- fewer words: the oracle lacks seven frequent ones, and each core is stretched to fill R;
- a shared source that favours generated clips over the real clips it came from.

**What a shuffle drop means.** Each shuffle against its unshuffled partner, on clips whose word sequence actually moved (``paired.moved``): ``shuffled`` against composition, ``covered_shuffled`` against ``covered``, ``oracle_shuffled`` against ``oracle``. No number is set for large or small.
- **A large drop:** the reader may use word order, so English order may be part of what it reads. It may also come from the rebuilt joins or the different still, which these rows do not fully separate. It does not make the order signing, or something a person would follow.
- **A small drop:** the reader may not be using word order. But BLEU can see order-invariant signals, such as length, which the token count fixes, and pose; and a form with near-zero margin has nothing to lose. It says nothing about whether order matters to people.
**Seeds are not independent replicates:** a draw's seeds share its clips, words and plan, so the draws carry the variance and seeds are averaged within a draw. **A two-word clip cannot vary:** it has one other order, the same under every seed. The order depends on draw, seed and clip, so it varies across draws (``word_order``). A shuffled still is another frame, so ``still_raw`` and the margin over the unshuffled form's still stand beside the margin (``paired.still_effect``).

**None of this shows human comprehension,** and none of it is a claim about M1 or release.
- **The mirror is report only and never moves M1. Generated motion reading high on it is flagged on the same terms as on the parent,** since it shares the parent's lineage (the parent's weights, trained on the MediaPipe source of word clips), **and How2Sign monocular trained it,** so it was fitted to the path word clips come through, and its training holds 234 of D20's clips. The same readings apply on it, each on its own anchors.
- **A parent void or a parent check alert stops the job before the mirror reads:** its anchors would then have no parent to stand beside.

**The second instrument (``--instrument``: ``parent``, ``mirror``).** ``mirror`` is ``back-translator-kp2d-uwb-ft-mono-v1`` step 9,000, sha ``1a996cfc...``, the parent fine-tuned on How2Sign monocular states (``scripts/symmetric_ft.py``, label ``unresolved``), loaded by ``parent_lib.load_mono`` from its trained record and held to that hash. Each instrument has its own forms, control, void and report. Through the mirror: the composition, its still and its blank on each draw it reads (it has no stored forms), real and nearest on both paths with their stills, blank at the reference's length, and every control form. **Its shares take its own path's anchors,** real and nearest monocular over the monocular still, since that is its peak; the two-view anchors stand beside them. A share of real two-view stands beside each share (``share_of_two_view_real``). Report only, for each form on either instrument: BLEU of the form, real, nearest and blank, and the BLEU gap share, on clips in v1's training list, outside it, and outside training and selection, the clean 178 (``symmetric_ft.clip_subsets``; the mirror's list is v1's less the refused clips, the same on D20): the plan's table shows that split matters most here. The mirror's report also holds the parent's rows on the mirror's own draws, and parent minus mirror draw by draw (``parent_on_mirror_draws``).
**Its control:** all seven anchors against the stored report's values (real monocular 4.715, nearest monocular 4.783, nearest two-view 3.702, blank 1.159 among them, within 0.05), and their hypotheses and NLL against its stored forms (``parent_lib.same_read``), report and forms held to full hashes. A miss voids the mirror's read, status 3; a stored object off its hash stops with status 1, nothing voided.
**The GPU.** The mirror's stored forms were read on an NVIDIA L40S, so a different type may void its read: **the job refuses any GPU but ``EXPECT_GPU``, the NVIDIA L40S, and the pod must be requested as that type.** The parent's stored composition was read on an RTX 6000 Ada, and whether an L40S reproduces it is what ``comp-check`` tests.
**What the mirror reads of generated motion, and why: the composed state as it stands, the very bytes the parent read.** No conforming, no marks, no trim.
- **Not conformed to the monocular layout.** The reader's input is ``from_state``, ``normalise`` and ``features``: forward kinematics of body and hand rotations and their validity. ``conform`` only renames a channel's frame and convention, so names cannot reach it (``path_read`` found it byte-identical on all 500 real clips; here ``tests/test_parent_controls.py`` holds it for a composed state). Word clips came through the monocular path already, and were conformed to two-view's names only.
- **No monocular marks.** A composed state is valid on every read point, as d2's decode assembles it, so it is the marks-free kind: **its like on the mirror is ``nearest_monocular`` (4.783), with ``real_monocular`` (4.715) beside it.** Generated motion has no tracking gaps of its own. Copying a real clip's would hand the reader that clip's gap pattern, and the path read showed marks can carry a read by themselves, so the form would stop being generated motion alone.
- **Not trimmed or stretched to a monocular clock.** The predicted length is the product being measured, and a clip's own length would leak its reference. Monocular clips run longer than their two-view slices, and the predicted seconds are fitted on two-view durations, so these forms are shorter than what the mirror trained on: the report gives the ratio (``length_over_monocular``), and a margin over the form's own still at its own length cancels flags and duration as on the parent.
- **The same bytes on both instruments** make the two reads a comparison of readers rather than of forms.

**The third instrument (``--instrument v1``, ``INSTRUMENTS=v1`` in the job): how much of the official 2.40 is word order?** ``v1`` is ``L.TRANSLATOR_RUN``, step 9,000, sha ``2e3ded3f...``, the official instrument. **Reads only: it is never used here for a loss, a rerank or a best-of-N,** and no choice in this job depends on it. **Declared before any read of it:**
- **The same rule:** a contrast is above or below only if the t interval over draws excludes zero on both margin over still and BLEU in the same direction; otherwise unresolved; the clip interval decides nothing. All eight draws, with no clock rule on them: this is the ruler that matters.
- **Anchors:** real two-view (4.493) and nearest two-view (4.621), with real monocular beside them. Oracle minus covered, and oracle against real two-view, read as above.
- **A shuffle drop on moved clips near the parent's and the mirror's, about 1 BLEU,** means much of the official 2.40 is word order. **A small drop** means it mostly is not. Neither is the better outcome, and the limits on what a drop means, stated above, hold here. ``official_gap`` gives composition's and its shuffled form's shares of the official gap, (form minus blank) over (real minus blank), and the share of composition's gap that shuffling removes.
- **v1 trained on 234 of D20's clips and selected on 265 more,** so ``by_clip_subset`` rows matter. **No target moves: M1 stays read on the held-out set, and nothing here touches that set. None of it shows human comprehension.**
**Why composition, its still and blank are read again.** The stored A5 reports hold v1's hypotheses and each clip's own log-probability, but no still and no NLL of mismatched references, so they are not this recipe's margin. They are the check: draw one's stored composition must reproduce before anything else is read through v1, every draw's hypotheses and BLEU are set beside its stored report and ``a5-summary.json``, and a miss raises an alert. Their own-reference log-probability is reported beside and is no miss, since those reports may have grouped their batches otherwise.
**Its controls, before any spend on draws:** real and blank against the pinned canary (real 4.493, blank 1.680, ``s2/b0-w14b-d20-v1.json``), real, blank and nearest on both paths against the parts job's pinned forms, and draw one's stored composition (``comp-check``). A miss voids v1's read, status 3. **The forms read the stored composed states of this job's first read, no new compose:** each draw's index is held to its full hash and of digest ``0936c98a``, and each state to its hash in that index. **The GPU:** the parts job's v1 forms state an NVIDIA L40S, as the mirror's did, and the job refuses another type. The A5 reports state none, though their hypotheses reproduced on an RTX 6000 Ada (fidelity 32 of 32 per draw).
**Cost, estimated from the measured pace:** the parent's draws took about 12 min each with 2 reads running, for 16 forms, so about 14 min for v1's 19, 8 draws 115 min; pins, the checkpoint's first fetch, the smoke, anchors and checks about 30 min; merge 10 min. **About 2.6 h and about $2.90** at the $1.11/h of the last job's $3.40 for 3 h 4 min, range about $2.40 to $3.60. That v1 reads at the parent's pace is assumed.

**Forms, all on D20: the same 500 clips, the same predicted lengths and timing as the composition's ``core_crossfade@predicted`` of the same draw, on each instrument.**
- ``oracle``: the composition's word list, each word replaced by a REAL word clip from the train records of ``asllvd-text-v1``, ``rit-text-r6``, ``wlasl-text-r6`` in ``round10-mixed-v1/manifest.json``, through d2's round trip (``build_recon_states.decode_state``, step 5000, EMA, no gains: A5's recipe, codes decoded as a generation so every trained channel reads valid, as a generated clip does), then the same cut and stitching. **The real clip is a chance pick, not best of 16:** ``pick_index``, the hash of (draw, entry), one clip per entry in a draw, as a word cache holds one sign per entry. No instrument chooses it, while generated clips were each the recogniser's best of 16. That difference favours the generated clips and is part of what this reads. If a picked clip cannot be served, the next in the pool's fixed order, counted.
- ``covered``: the generated clips of exactly the oracle's words, under the oracle's plan. The oracle always lacks the added words (below), so oracle against composition mixes "real clips" with "fewer words". Oracle against ``covered`` is the same words with the same frames planned, and ``covered`` against composition is the word-list effect alone.
- ``shuffled``: the composition's generated clips and planned frames, words in a seeded random order per clip, seeds 0 and 1 (``compositional_baseline.shuffle_order`` keyed by draw and seed: never the identity from two words up).
- ``oracle_shuffled`` and ``covered_shuffled``: ``oracle`` and ``covered`` under the same permutation, restricted to the words they hold.

**What composition really does (read from the code: ``round15_compose_eval.run``, ``round20_lib.cached_material``, ``round15_stage2.render_form``, ``compositional.render``).** Each word's clip is cut to a core: for a generated clip the sign's median position in the entry's ASLLVD train clips (``sign_core``), else over all of them, margin 0. The sentence takes ``R = round(predicted seconds x 24)`` frames, predicted seconds from the token count (pin -0.083, 0.3691). ``fitted_plan`` shares R among the words by their isolated sign seconds and among the joins by the gap ratio, with no lead and no trail, so **R is the clip's length whatever the words are.** Each core is time-resampled to its planned frames, and each join is ``bridge``, a linear interpolation on each channel's manifold from one core's last frame to the next core's first, never observed.
**An uncovered word is dropped:** a token with no vocabulary entry is simply absent, nothing stands in for it, and the words that remain share R. A sentence left with no word is its blank clip at R. **The oracle takes that same fallback:** a word with no real clip in the three corpora is dropped from its list, and the rest share R.
**Two exact consequences, both reported.** Composition's seven added entries (``you, it, your, this, so, i, or``, from aliases too) have no real clip here: composition serves them only from ASLLRP continuous segments, which the three corpora do not hold, and it never takes a clip of the same spelling from elsewhere (ASLLVD's letter ``i``); the oracle keeps that rule. Sem-Lex-only entries have none either. So the oracle's coverage is below composition's on most clips, and coverage bands are given both by the form's own coverage and by composition's.
**One difference in the cut,** ``round15_stage2.oracle_material``'s: an ASLLVD real clip with its own annotated sign is cut there; anything else takes ``sign_core`` as a generated clip does. Stitching and plan are the same.
**"Same frames" in a shuffle:** each word keeps its planned core frames, and the gap lengths keep their places (``moved_plan``). Every word's rendered core is frame for frame the unshuffled form's, and the total is R. The joins are bridged anew, since their neighbours changed. ``render_form``'s own ``order`` would re-allocate frames by weight, so it is not used for this.

**Fidelity, before any instrument.** Every draw's composition is rebuilt from its stored word clips and windows by this code's plan. Each non-empty clip must hash to the stored ``core_crossfade@predicted`` state of the pinned export, or, since another pod's CPU can round a float differently, hold the same clock and marks with every value within 1e-5 (both counted), or the draw stops with status 1. So the shuffles start from what was read. No read happens without that index.
**Controls, before the compose is paid for.** The four anchors against the parts job's stored forms, pinned by hash (``parent_comp.run_control``); and the first draw's stored composition read again against its pinned stored form (at least 0.98 of hypotheses identical, BLEU within 0.05, NLL within 0.001). A miss voids the read: status 3, no retry. On the parent, composition's comp, still and blank are then the pinned forms of ``round20/parent-comp`` (digest ``dc3d89b0``), not read again, which also pairs these forms with the very forms of the 3.30.

**Resuming.** Rerunning the job resumes it. A stage whose final key holds the bytes of its local report is skipped, and its check runs again, so a parent check alert stops the job again and reads nothing; forms that stand are not read again; ``merge`` and ``put-report`` accept identical bytes and refuse others; the smoke is not repeated for the same code; a stage's draws, stop reason, seconds and budget stand from the first time its reads finished; elapsed seconds and the clock are cumulative across a resume; a changed ``DRAWS`` or ``MIRROR_DRAWS`` refuses.
**On a new pod,** a finished stage's report is restored from its final key, if it is of this code, this environment and these planned draws (``stage-state``). What the old pod held locally and had not uploaded is lost: an unfinished stage reads its forms again, up to about $1.20 for the parent and $0.60 for the mirror, unless a failure had uploaded them, and those partial forms are not restored.
**A rerun after a void** needs new code and ``RERUN_CAUSE``, which may not be blank and is copied into the report: its work and outputs go under ``runs/<code digest>/``, which is what lets new code write at all, and the voids it must answer stay in the common prefix.

**Margin** is ``parent_comp``'s: mean NLL of 4 mismatched references minus NLL of its own (``retrieval_candidates``, seed 0), over the form's own still (its nearest-filled middle frame held at its own length, on its own clock). Each form, each seed. **A clip with no word is its blank and counts, with margin exactly zero.** Blank for gap shares: the draw's blank at the composition's length; every form here is that long, which ``check_still_lengths`` and ``same_lengths`` enforce.

**Reported per form, over the draws** (t interval on the draws alone, bootstrap over clips n 2,000 seed 0, draws fixed; seeds averaged within a draw for the shuffle groups, each seed also alone; intervals on coverage and on bands, over the clips in them): BLEU; margin over its own still; share of real's margin and of nearest's; BLEU and likelihood gap shares; by word coverage bands (under .25, .25-.5, over .5) by its own coverage and by composition's; per signer, report only; loop share (``eval.loop_share``); content-word overlap with the reference (share of the reference's words outside ``compositional.STOP_WORDS`` found in the hypothesis, the clip's mean; a stated definition, **not the same as the plan's sampled 16.3% and 11.1%, whose definition was not recorded**, so read it only among this report's rows). **Paired differences** with the same draws and clips: every form against composition, each form against real and against nearest, oracle against covered, and each shuffle against its partner, on all clips and on moved clips.
**Oracle coverage:** words kept and dropped, by reason; clips fully covered, clips left empty; corpus of the picks, with the oracle's rows by each clip's main corpus; picks that could not be served. A pick gives way only on a window holding no frames: anything else, an out-of-memory or a hash failure, stops the draw. More than 2% of a draw's entries with a pool finally not served raises an ALERT and stops it; in a smoke it is reported and the smoke goes on.

**Disclosures.** The generator trained on these real clips, so they are not independent of it. Real clips are MediaPipe monocular estimates of isolated signers, as generated clips' training data is, so an oracle reading high does not separate source from format as cleanly as a two-view clip would. A different signer at every word. d2's round trip smooths real motion (it keeps about 0.84 of finger speed: ``hand_motion_compare``). A real clip's untracked frames go in as the dataset serves them, with no fill, as the prior's training codes were made. **Before any spend,** ``inputs`` opens the cache's prior and holds its stored tokenizer to this d2's weights and key, so generated and real clips share one decoder; compares the mirror's stored forms' clips with D20; and runs the validity check on the stored compositions. **What is parsed:** shared helpers (``load_manifest``, ``fit_model``, ``open_read_list``) parse the whole mix, and ``v1_training_signs`` holds some ids of the sentence set no job here names at merge, as earlier reads did: nothing of either reaches an output.

**Cost, estimated, not measured.** A form costs about 60 s (read 34 s, build about 26 s: ``resources`` of the stored parent report, on an RTX 6000 Ada; the L40S is assumed alike), and each read command about 1.5 min to start. Parent, 8 draws x 16 forms: 140 min. Mirror, 3 draws x 19 forms, 7 anchors, the monocular clips: 74 min. Reads run ``READ_PROCS`` at a time, 2 by default: the stored peak of 19.8 GB reserved allows two on 48 GB, and 1.7 times the speed is a guess, so 214 min becomes about 126. Compose about 21 min per draw, 4 at a time: about 50 min. Pins, the decoder check, model fit and both smokes 35 min; two anchor sets' checks 8 min; merges 20 min. **About 4 h and about $3.40** at the $0.85/h implied by parent-comp's $1.70 for about 2 h, range about $2.80 to $4.50: the compose time, the parallel speed and the L40S's rate are the guesses. Without parallel reads, about 5.5 h and $4.70.
**So the mirror reads three draws (1, 2, 3), not eight.** Two draws would leave its t interval almost empty; each further draw costs about $0.20. Those three draws are few for a t interval, as the reading says. The parent keeps all eight, which passes the brief's "about $3". ``DRAWS="1 2 3 4 5 6"`` with the mirror's three is about $3.
**The clock rule, fixed here and blind to every score:** once an instrument has started its minimum of draws (parent four, mirror two), the job starts no further draw of it if the time since the job began, plus the wall time since its reads began divided by the draws finished, would pass its limit (parent ``PARENT_BUDGET_SECONDS`` 10,800 s; mirror ``BUDGET_SECONDS`` 14,400 s, 4 h, about $3.40). So the mirror's third draw can be left out. ``MIRROR_DRAWS`` must be a prefix of ``DRAWS``. Each draw's forms are read whole. A report's ``stage`` gives its planned draws, the draws read, the stop reason, the elapsed seconds and the budget. Every step runs under a stated limit: ``READ_TIMEOUT`` 3,600 s for an anchor set or a draw's forms, ``COMPOSE_TIMEOUT`` 5,400 s, ``MERGE_TIMEOUT`` 3,600 s for a merge or the pins, ``STEP_TIMEOUT`` 1,800 s for the rest. The measured cost must be reported beside this.

**Refused.** A record outside the three corpora or outside ``train``, and c00 in any case, stops before it is fetched (``check_word_record``). D20 only (``s2_lib``; C20 cannot be named). No v1 read, and no instrument but the two pinned: neither is ever used here for a loss, a rerank or a best-of-N, and no instrument sees a real clip before the pick. A stored form or export on other bytes stops. A smoke (``--smoke-train 5``): 5 D20 sentences composed, read against c02 train references, so the parent sees no D20 text or clip: numbers are plumbing.

    python scripts/parent_controls.py inputs --model-file W/model.json --expect-gpu NVIDIA_L40S
    python scripts/parent_controls.py digest
    python scripts/parent_controls.py stage-state --key K --file W/report.json --record W/stage.txt --planned-draws 1,2
    python scripts/parent_controls.py compose --draw K --model-file W/model.json --prefix P --work W --device cuda
    python scripts/parent_controls.py preflight --out-dir W --final-key K --void-prefix V
    python scripts/parent_controls.py read --forms control --instrument I --out-dir W/forms --instrument-cache W/i
    python scripts/parent_controls.py control --instrument I --out-dir W/forms --void-prefix V
    python scripts/parent_controls.py read --forms comp-check --draws K --composed P --out-dir W/forms
    python scripts/parent_controls.py comp-check --draw K --out-dir W/forms --void-prefix V
    python scripts/parent_controls.py read --draws 1,2 --instrument I --composed P --resume --out-dir W/forms --instrument-cache W/i
    python scripts/parent_controls.py upload --out-dir W/forms --prefix D/forms
    python scripts/parent_controls.py merge --draws 1,2 --instrument I --composed P --prefix D/forms --output-file W/report.json --parent-forms D/forms --planned-draws 1,2 --stop-reason all_planned --elapsed-seconds 100 --budget-seconds 10800
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
SEEDS = (0, 1)
INSTRUMENTS = dict(parent=dict(run=PP.UNFITTED_RUN, checkpoint_key=PP.UNFITTED_KEY, sha256=PP.UNFITTED_SHA),
                   mirror=dict(run=Q.MONO_RUN, checkpoint_key=f"{Q.MONO_PREFIX}/checkpoints/step-00009000.pt", sha256="1a996cfc9c1917181fef849c6a9b8dde0406c34e42fd48a75f1df63fb189ca59"),
                   v1=dict(run=L.TRANSLATOR_RUN, checkpoint_key=f"training-runs/{L.TRANSLATOR_RUN}/checkpoints/step-00009000.pt", sha256=L.TRANSLATOR_SHA))
MIRROR_ANCHORS = ("real_two_view", "real_monocular", "nearest_two_view", "nearest_monocular", "blank", "still_two_view", "still_monocular")
MIRROR_VALUES = dict(real_monocular=4.715, nearest_monocular=4.783, nearest_two_view=3.702, blank=1.159)      # the mirror report, docs/NEXT_PHASE_PLAN.md: the anchors reproduce these or its read is void
MIRROR_TOLERANCE = P.CANARY["bleu_tolerance"]
MIRROR_REPORT, MIRROR_REPORT_SHA = f"{Q.MONO_PREFIX}/read/report.json", "223dd372b5f1ba2a4f0980b67203bbbd0a8884bd03202970902d8ceaf26c11b2"
MIRROR_FORMS = f"{Q.MONO_PREFIX}/read/forms"
MIRROR_DIGEST = "de05576ec6b714f4e16095732a17e2576df0444cc0f3d9d8072f2ee82d4f268e"
MIRROR_STORED = dict(real_two_view="cfeff3dae866fcffe8a5433549817b56c5c0963832d54571d372485501f822b9", real_monocular="f53a324c6a469c096f97db53951498dbe7ecbca518149d7e5db1d70d753537d4", nearest_two_view="2eaa1773278279349d36978a66b4840237f9fc54b256e41e4012692e38aea0b8", nearest_monocular="b78e470e2f277a2fb23875f3decfbf674e1296b036dd151617ff8ed0fa94df3b", blank="159163666dd144a5451d8153dedf0052b6ea43135144ea78ab68ca5b0e9b533d", still_two_view="f48b11703c4c0846448732f5d98f8f9b62ab0af32c0cd479f021b29376121063", still_monocular="9b567d5428ee5d9c7de3fdbc9ba44deccca99cef783911329accd282545e1a99")      # fetched and hashed 2026-11-02; those forms state NVIDIA L40S
REQUIRED_GPU = "NVIDIA L40S"
# fetched and hashed 2026-11-02: the stored composed forms v1 reads, of this job's first read
COMPOSED_DIGEST = "0936c98aec02269bc8bca63c29d75b2628adff3fdd8ef71986449341032821a0"
COMPOSED_INDEX = {1: "02726a3d077b5e0e9c1ce239c33e12def2ff2c059daeffc5bf3e00db84b3ad82", 2: "8beadf2abf52313a6ec93c01a7a2f928bee0163c8abeb99524a2ecb39ba77a12", 3: "c38c28726cd0bf69a187948acc81e6c085b5d623933df0c33afd846fbe0fc918", 4: "c0f33728dff47eaeb82b3e73e902f2aed217562ba0582cd93162beea83d7a968", 5: "c1bc2e88e737983fdabd98f8cba1dc5dca36c4cb109ce3c5d70c616002a9163b", 6: "d3a045484ab022ee617e476b2dad0cd33d059fb80b5cfadacb7e5c0139dbc8c7", 7: "d034682ed66b49c3a6a62e03cc4e517206475a9028ec94da272f0433929d408e", 8: "49c43821b988c2f0179985fafd9a101ee0c35b0579e8eaf449b0fcfb4720e0fb"}
V1_A5_BLEU = (2.406, 2.469, 2.22, 2.449, 2.08, 2.35, 2.556, 2.687)      # a5-summary.json, sha 4f3d1f38
V1_STORED_ANCHORS = ("real_two_view", "blank", "nearest_two_view", "nearest_monocular")
V1_COST_ESTIMATE = dict(measured=False, hours=2.6, dollars=2.9, range_dollars=[2.4, 3.6], dollars_per_hour=1.11, forms_per_draw=19, draws=8, minutes_per_draw=14, basis="the measured pace of the parent's draws (about 12 min for 16 forms with 2 reads running) and the last job's $3.40 for 3 h 4 min; that v1 reads at that pace is assumed; the measured cost is to be reported beside this")
EMPTY_WINDOW = "contains no frames"      # corpus_manifest.window_state's refusal: the one failure a pick gives way on
MAX_NOT_SERVED = .02
PICK_SEED = "parent-controls|oracle"
TIMING = "sign-fitted"
CORE_MARGIN = 0.
TOP = 10
IDENTITY_TOLERANCE = 1e-5
GROUPS = {name: tuple(f"{name}_s{s}" for s in SEEDS) for name in ("shuffled", "oracle_shuffled", "covered_shuffled")}
KINDS = ("oracle", "covered", *(kind for kinds in GROUPS.values() for kind in kinds))
PARTNERS = dict(shuffled="composition", oracle_shuffled="oracle", covered_shuffled="covered")
FORM = re.compile(r"d([1-8])_(oracle_shuffled_s[01]|covered_shuffled_s[01]|shuffled_s[01]|oracle|covered)(_still)?")
BASE_FORM = re.compile(r"d([1-8])_(comp_reread|comp|still|blank)")
SMOKE_CLIPS = C.SMOKE_CLIPS
RERUN_ROUTE = "A rerun needs new code and RERUN_CAUSE, which writes under runs/<code digest>/."
COST_ESTIMATE = dict(measured=False, hours=4., dollars=3.4, range_dollars=[2.8, 4.5], dollars_per_hour=.85, forms_per_draw=dict(parent=16, mirror=19), draws=dict(parent=8, mirror=3), seconds_per_form=60, read_procs=2, parallel_speed_guess=1.7, compose_minutes=50, budget_seconds=dict(parent=10800, mirror=14400), basis="stored parent-comp report's resources (read 34 s, build 26 s per form, RTX 6000 Ada) and its about $1.70 for about 2 h; the compose time is the guess; the measured cost is to be reported beside this")
DISCLOSURES = dict(
    pick="The real clip is a chance pick, one per entry in a draw, by a hash of draw and entry: no instrument chose it. Each generated clip was its recogniser's best of 16.",
    independence="The generator trained on these real clips, and they share a monocular MediaPipe path with its training data and with the parent's: an oracle reading high does not separate source from format as a two-view clip would.",
    words_lacked="The oracle lacks the seven added entries, which have no real clip in the three corpora and no clip of the same spelling taken from elsewhere, Sem-Lex-only words and words not served, and R is reshared among those left. Oracle against composition therefore mixes real clips with the words the oracle lacks and the frames they free: oracle against covered is the same words.",
    cut="An ASLLVD real clip with its own annotated sign is cut there; every other clip, generated clips included, takes its entry's median position. Plan and stitching are the same.",
    shuffle="Each word keeps its planned core frames and the gap lengths keep their places, so every word's rendered core is frame for frame the unshuffled form's. The joins are bridged anew.",
    round_trip="d2's round trip smooths real motion; a real clip's untracked frames go in as the dataset serves them, with no fill.",
    content_overlap="Content-word overlap here is the share of the reference's words outside the stop list found in the hypothesis, per clip. It is not the plan's sampled figures, whose definition was not recorded: read it only among this report's rows.",
    mirror="The mirror is report only and never moves M1. It shares the parent's lineage, and How2Sign monocular trained it, so generated motion reading high on it is flagged on the same terms as on the parent. Its training holds 234 of D20's clips, on signers c01 and c05, and c03 selected it.",
    mirror_form="The mirror reads the composed state as it stands, the bytes the parent read: not conformed, since names cannot reach its input; no marks, so its like is nearest_monocular; at the predicted length, shorter than the monocular clips it trained on.",
    oracle_limits="Neither a low nor a high oracle separates format from source by itself: real word clips are MediaPipe monocular too. Live for a low oracle: selection by best of 16, d2 decoding, a different signer per word, a cut fitted on ASLLVD and applied to WLASL and RIT, fewer words with each core stretched, a source favouring generated clips.",
    rule="A contrast is above or below only if its paired t interval over draws excludes zero on both margin over still and BLEU in the same direction; otherwise it is unresolved. The clip bootstrap interval decides nothing. Three mirror draws make unresolved likely there.",
    shuffle_drop="A large drop may also come from the rebuilt joins or the different still, which these rows do not fully separate. On a small drop, BLEU can see order-invariant signals such as length and pose, and a form with near-zero margin has nothing to lose.",
    seeds="Seeds are not independent replicates: a draw's seeds share its clips, words and plan. A two-word clip has one other order, the same under every seed. Moved rows take only clips whose word sequence changed.",
    parsing="Shared helpers parse the whole mix, and v1_training_signs holds ids of the sentence set no job here names at merge, as earlier reads did. Nothing of either reaches an output.",
    gpu="The mirror's stored forms state an NVIDIA L40S, and the job refuses another type. The parent's stored composition states an RTX 6000 Ada, and comp-check tests whether the L40S reproduces it.",
    stops="A parent void or check alert stops the job before the mirror reads.",
    v1="v1 is the official instrument, read only here, never for a loss, a rerank or a best-of-N. It trained on 234 of D20's clips and selected on 265 more, so by_clip_subset rows matter. Its forms are the stored composed states of this job's first read, each held to its hash. Stored A5 reports hold no still and no mismatched NLL, so composition, still and blank are read again and checked against them.",
    humans="None of this shows human comprehension, and none of it is a claim about M1 or release.")
READING = ("A contrast is above or below only if its paired t interval over draws excludes zero on both margin over still and BLEU in the same direction; otherwise it is unresolved. Generated against real clips is oracle minus covered, never oracle against composition. The format is oracle against real, and on the mirror against nearest_monocular with real_monocular beside it. Oracle above the like and above covered: the format reads high, and real clips read higher than generated ones. Oracle above the like but below covered: the format reads high AND generated clips read higher. Oracle below the like and above covered: composition's lead would sit in the words the oracle lacks and the frames they free. Oracle below the like and below covered: the format alone does not lift real clips, and generated clips read higher. Either contrast unresolved: unresolved. Neither a low nor a high oracle separates format from source by itself, since real word clips are MediaPipe monocular too. The mirror is report only and never moves M1; generated motion reading high on it is flagged on the same terms as on the parent, since it shares the parent's lineage and How2Sign monocular trained it. No label gates anything. None of it shows human comprehension.")
READING_V1 = ("The same rule: above or below only if the t interval over draws excludes zero on both margin over still and BLEU in the same direction; otherwise unresolved; the clip interval decides nothing. Anchors: real two-view and nearest two-view, with real monocular beside them. A shuffle drop on moved clips near the parent's and the mirror's, about 1 BLEU, means much of the official 2.40 is word order; a small drop means it mostly is not. v1 is read only, never for a loss, a rerank or a best-of-N. v1 trained on 234 of D20's clips and selected on 265 more, so by_clip_subset rows matter. No target moves, M1 stays read on the held-out set and nothing here touches that set. None of it shows human comprehension.")
CELLS = {("above", "above"): "format reads high, and real clips read higher than generated ones", ("above", "below"): "the middle case: format reads high AND generated clips read higher", ("below", "above"): "real clips do not reach real signing and generated clips fall below them: composition's lead would sit in the words the oracle lacks and the frames they free", ("below", "below"): "format alone does not lift real clips, and generated clips read higher"}



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
    reader = PP.load_unfitted(store, args) if name == "parent" else Q.load_mono(store, args) if name == "mirror" else P.load_reader(store, args)
    if reader.info["sha256"] != pin["sha256"] or reader.info["checkpoint_key"] != pin["checkpoint_key"]:
        raise SystemExit(f"{name}: loaded {reader.info['checkpoint_key']} sha {reader.info['sha256']}, not the pinned {pin['checkpoint_key']} sha {pin['sha256']}")
    return reader


def anchors_of(instrument: str) -> tuple:
    return C.ANCHORS if instrument == "parent" else MIRROR_ANCHORS


def own_path(forms: dict, instrument: str) -> dict:
    """The anchors a form's shares take, under ``parent_comp``'s names: the instrument's own path, two-view but for the mirror. Its arithmetic is not copied."""
    if instrument != "mirror":
        return {a: forms[a] for a in C.ANCHORS}
    return dict(real_two_view=forms["real_monocular"], blank=forms["blank"], nearest_two_view=forms["nearest_monocular"], still_two_view=forms["still_monocular"])


def anchor_report(forms: dict, instrument: str, idx: np.ndarray) -> dict:
    """Each anchor's margin over its own path's still, and its BLEU."""
    two_view = C.anchor_rows({a: forms[a] for a in C.ANCHORS}, idx)
    if instrument == "parent":
        return two_view
    own = C.anchor_rows(own_path(forms, "mirror"), idx)
    return dict(two_view, real_monocular=own["real_two_view"], nearest_monocular=own["nearest_two_view"], still_monocular=own["still_two_view"])


def monocular_clips(store, records, refs, declared: bool) -> tuple[list, list, list]:
    """(clips, records, references), one for one in list order. On D20 every record has its clip, since the mirror's anchors read the same clips as every other form, and a clip lost refuses. A smoke reads the clips it has, as the mirror's own smoke did."""
    clips, lost, _ = R.open_monocular(store, records, refs, declared)
    kept = [c.index for c in clips]
    if kept != sorted(kept) or len(set(kept)) != len(kept):
        raise SystemExit("the monocular clips are out of order: the mirror's anchors would read other clips")
    if declared and (lost or kept != list(range(len(records)))):
        raise SystemExit(f"{len(lost)} clips have no monocular state: the mirror's anchors would read other clips")
    if not declared and len(kept) < 2:
        raise SystemExit(f"a smoke needs two clips with a monocular state, not {len(kept)}")
    if lost:
        print(json.dumps(dict(smoke_lost=len(lost), reasons=sorted({x.get("reason") for x in lost}))), flush=True)
    return clips, [records[i] for i in kept], [refs[i] for i in kept]


def load_mirror_form(store, name: str) -> dict:
    """A stored form of the mirror's own read, by its pinned hash, of its digest and checkpoint."""
    if name not in MIRROR_STORED:
        raise SystemExit(f"{name}: no stored form of the mirror's read under this name")
    key = f"{MIRROR_FORMS}/form-mono__{name}.json"
    raw = store.get(key)
    P.check_sha(raw, MIRROR_STORED[name], key)
    read = json.loads(raw)
    if read.get("form") != f"mono__{name}" or read.get("instrument", {}).get("sha256") != INSTRUMENTS["mirror"]["sha256"] or read.get("declared_read") is not True or read.get("code_digest") != MIRROR_DIGEST:
        raise SystemExit(f"{key}: not the stored mirror read's form")
    return read


def load_mirror_report(store) -> dict:
    """The mirror's report by its hash, and its parsed values against the four this read declares."""
    raw = store.get(MIRROR_REPORT)
    P.check_sha(raw, MIRROR_REPORT_SHA, MIRROR_REPORT)
    report = json.loads(raw)
    values = report["forms"]
    differs = [name for name, value in MIRROR_VALUES.items() if values[name]["bleu"] != value]
    if differs or any(name not in values for name in MIRROR_STORED) or report.get("code_digest") != MIRROR_DIGEST:
        raise SystemExit(f"{MIRROR_REPORT}: it does not hold the declared values or all seven forms: {differs}")
    return report


def mirror_misses(read: dict, name: str, stored: dict, value: float) -> tuple[dict, list[str]]:
    """One anchor against the report's value and its stored form."""
    bleu = M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
    try:
        row = Q.same_read(read, stored)
    except SystemExit as error:
        row = dict(misses=[str(error)])
    misses = [f"{name} {m}" for m in row["misses"]]
    if abs(bleu - value) > MIRROR_TOLERANCE:
        misses.append(f"{name} reads BLEU {bleu}, not the report's {value}")
    return dict(row, read_bleu=bleu, report_bleu=value), misses


def run_mirror_control(args) -> int:
    """Exit 3, voiding the mirror's read, unless all seven anchors reproduce the report's values and stored forms. A smoke reads train clips, so it compares nothing and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in MIRROR_ANCHORS}
    if not read["real_monocular"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "the mirror's control forms")
    report = load_mirror_report(store)      # a pin failure stops here with status 1: no control runs, nothing is voided
    stored = {name: load_mirror_form(store, name) for name in MIRROR_ANCHORS}
    rows, misses = {}, []
    for name in MIRROR_ANCHORS:
        rows[name], missed = mirror_misses(read[name], name, stored[name], report["forms"][name]["bleu"])
        misses += missed
    result = dict(stored=rows, misses=misses, reproduced=not misses, code_digest=digest, environment=place, stored_environment=stored["real_monocular"].get("environment"))
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    key = f"{args.void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-mirror-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())
    print(f"ALERT the mirror's anchors do not reproduce its report: {misses}. The mirror's read is void: no form follows, and this code never reads again. {RERUN_ROUTE}", flush=True)
    return 3


def check_gpu(expected: str | None) -> None:
    """The stored forms' GPU type, or the job stops before it spends anything. Underscores stand for spaces in the job's argument."""
    found = P.environment().get("gpu")
    if expected and found != expected.replace("_", " "):
        raise SystemExit(f"this pod's GPU is {found}, not {expected.replace('_', ' ')}: a different type may void the read")


def check_decoder(store, checkpoint_key: str) -> dict:
    """The cache's prior stores the tokenizer it was trained on: it must name this d2 and hold its weights, so generated and real clips share one decoder."""
    import io
    import torch
    raw = store.get(L.TOKENIZER)
    P.check_sha(raw, L.TOKENIZER_SHA, L.TOKENIZER)
    d2 = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    prior = torch.load(io.BytesIO(store.get(checkpoint_key)), map_location="cpu", weights_only=False)
    want, have = d2.get("ema") or d2["model"], prior.get("tokenizer_state")
    named = prior.get("tokenizer_key")
    if have is None or set(want) != set(have) or any(not torch.equal(want[k], have[k]) for k in want) or named not in (None, L.TOKENIZER):
        raise SystemExit(f"{checkpoint_key}: its stored tokenizer is not this d2's ({named}): word clips were decoded by another")
    return dict(checkpoint_key=checkpoint_key, tokenizer_key=named, tensors=len(want), same_weights=True)


def void_read(store, void_prefix: str, digest: str, place: dict, result: dict, name: str) -> None:
    key = f"{void_prefix.rstrip('/')}/{digest}.json"
    if not store.exists(key):
        store.put(key, json.dumps(dict(kind=f"{PROBE}-{name}-void", code_digest=digest, environment=place, control=result), indent=1, sort_keys=True).encode())


def run_v1_control(args) -> int:
    """Exit 3, voiding v1's read, unless its real and blank reproduce the pinned canary and its anchors the parts job's pinned forms. A smoke reads train clips, so it compares nothing and says so."""
    from train_text_to_motion import make_store
    P.check_pins()
    out = Path(args.out_dir)
    store = make_store(args.store)
    read = {n: json.loads((out / f"form-{n}.json").read_bytes()) for n in MIRROR_ANCHORS}
    if not read["real_two_view"]["declared_read"]:
        print(json.dumps(dict(control="not applicable: a smoke reads train clips; nothing compared")), flush=True)
        return 0
    digest, place = P.check_same_run(read, "v1's control forms")
    raw = store.get(P.CANARY_REPORT)
    P.check_sha(raw, P.CANARY_REPORT_SHA, P.CANARY_REPORT)      # a pin failure stops here or below with status 1: no control runs, nothing is voided
    stored = {name: Q.stored_form(store, Q.STORED_V1, name) for name in V1_STORED_ANCHORS}
    try:
        canary = P.canary_read(dict(real=read["real_two_view"], blank=read["blank"]), json.loads(raw))
    except SystemExit as error:
        canary = dict(reads=None, misses=[str(error)], reproduced=False)
    rows, misses = {}, [f"canary {m}" for m in canary["misses"]]
    for name in V1_STORED_ANCHORS:
        try:
            rows[name] = Q.same_read(read[name], stored[name])
        except SystemExit as error:
            rows[name] = dict(misses=[str(error)])
        misses += [f"{name} {m}" for m in rows[name]["misses"]]
    result = dict(canary=canary, stored=rows, misses=misses, reproduced=not misses, code_digest=digest, environment=place, stored_environment=stored["real_two_view"].get("environment"))
    path = out / "control.json"
    if not path.exists():
        P.write_report_once(path, result)
    print(json.dumps(result), flush=True)
    if result["reproduced"]:
        return 0
    void_read(store, args.void_prefix, digest, place, result, "v1")
    print(f"ALERT v1's anchors do not reproduce its stored reads: {misses}. v1's read is void: no form follows, and this code never reads again. {RERUN_ROUTE}", flush=True)
    return 3


def run_control(args) -> int:
    check_instrument(args.instrument)
    return dict(parent=C.run_control, mirror=run_mirror_control, v1=run_v1_control)[args.instrument](args)


def a5_row(read: dict, top10: dict, stored_bleu: float) -> tuple[dict, list[str]]:
    """A read of the stored composition through v1 against its stored A5 report: the clips, the hypotheses and BLEU decide; the own-reference log-probability stands beside."""
    if read["signs"] != top10["signs"] or read["references"] != top10["references"]:
        return dict(misses=["other clips or another order"]), ["other clips or another order"]
    form = top10["forms"][C.PREDICTED]
    identical = float(np.mean([a == b for a, b in zip(read["hypotheses"], form["hypotheses"])]))
    bleu = M.bleu_ci_fast(read["hypotheses"], read["references"])[0]
    own = np.array([np.nan if x is None else -float(x) for x in top10["likelihood"]["per_form"][C.PREDICTED]], np.float64)
    gap = np.abs(np.asarray(read["nll"], np.float64)[:, 0] - own)
    misses = [why for why, bad in (("BLEU", abs(bleu - stored_bleu) > P.CANARY["bleu_tolerance"]), ("hypotheses", identical < P.CANARY["identical"])) if bad]
    return dict(bleu=bleu, stored_bleu=stored_bleu, identical=round(identical, 4), own_nll_mean_absolute_difference=round(float(gap[np.isfinite(gap)].mean()), 6) if np.isfinite(gap).any() else None, misses=misses), misses


def needs_comp_check(declared: bool, instrument: str) -> bool:
    return declared and instrument != "mirror"


def load_index(store, prefix: str, k: int, instrument: str, clips: int | None = None) -> dict:
    """A draw's composed index. v1 reads the stored one of this job's first read, by its full hash; the others read their own run's."""
    key = index_key(prefix, k)
    raw = store.get(key)
    index = json.loads(raw)
    if instrument != "v1":
        check_index(index, k, index.get("clips") if clips is None else clips)
        return index
    P.check_sha(raw, COMPOSED_INDEX[k], key)
    if index.get("kind") != f"{PROBE}-composed" or index.get("draw") != k or index.get("clips") != L.D20_CLIPS or index.get("seeds") != list(SEEDS) or index.get("code_digest") != COMPOSED_DIGEST:
        raise SystemExit(f"{key}: not the stored composed forms of this job's first read")
    return index


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
    """One draw's real clip per entry, as a word cache holds one generated sign per entry. ``load(record, entry)`` serves and round-trips it; a clip whose window holds no frames gives way to the next in the pool's order, counted. Anything else raises."""

    def __init__(self, pools: dict, draw: int, load) -> None:
        self.pools, self.draw, self.load, self.memo, self.failed = pools, draw, load, {}, 0

    def entries_with_a_pool(self) -> int:
        return sum(1 for entry in self.memo if self.pools.get(entry))

    def entries_not_served(self) -> int:
        """Entries with a pool left with no clip: the entries, not the failed attempts."""
        return sum(1 for entry, got in self.memo.items() if got is None and self.pools.get(entry))

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
            except ValueError as error:      # only a window holding no frames gives way, and it is counted: an out-of-memory or a hash failure stops the draw
                if EMPTY_WINDOW not in str(error):
                    raise
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

def word_order(n: int, clip: int, seed: int, draw: int) -> list[int]:
    """Seed ``seed``'s order of a clip's words in draw ``draw``: never the identity from two words up, so a two-word clip has one order under every seed."""
    from compositional_baseline import shuffle_order
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed}: the seeds are {SEEDS}")
    C.check_draw(draw)
    return shuffle_order(n, clip, 100 * draw + seed)


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
    orders = {s: word_order(len(entries), i, s, real.draw) for s in SEEDS}
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
        covered_words = [row["words"][p] for p in kept]
        out["covered"] = dict(state=covered, words=covered_words, order=list(range(len(kept))), dropped=dropped, plan=covered_plan)
        for s, order in orders.items():
            mine = restricted(order, kept)
            out[f"covered_shuffled_s{s}"] = dict(state=render_order(kept_pieces, covered_plan, mine, f"parent-controls-covered-shuffled-s{s}:{i}"), words=[covered_words[k] for k in mine], order=mine, dropped=dropped, plan=moved_plan(covered_plan, mine))
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


def coverage_of(rows: list[dict], failed: int, with_pool: int = 0, not_served: int = 0) -> dict:
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
    interval = None
    if filled:
        pairs = np.asarray(filled, np.float64)
        idx = np.random.default_rng(0).integers(0, len(pairs), (C.BOOT, len(pairs)))
        interval = dict(share_kept=P._quantiles(pairs[idx, 1].sum(1) / pairs[idx, 0].sum(1)), mean_clip_share=P._quantiles((pairs[idx, 1] / pairs[idx, 0]).mean(1)))
    return dict(ci95=interval, words_selected=selected, words_kept=kept, share_kept=round(kept / selected, 4) if selected else None, dropped=dict(sorted(reasons.items())), corpora=dict(sorted(corpora.items())), clips=len(rows), clips_with_words=len(filled), clips_fully_covered=sum(a == b for a, b in filled), clips_left_empty=sum(b == 0 for _, b in filled), mean_clip_share=round(float(np.mean([b / a for a, b in filled])), 4) if filled else None, picks_not_served=failed, entries_with_a_pool=with_pool, entries_not_served=not_served)


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
    with_pool, not_served = real.entries_with_a_pool(), real.entries_not_served()
    if not_served > MAX_NOT_SERVED * max(with_pool, 1):
        line = f"draw {args.draw}: {not_served} of {with_pool} entries with a pool not served, above {MAX_NOT_SERVED:.0%}"
        if args.clips == L.D20_CLIPS:
            print(f"ALERT {line}: no index is written", flush=True)
            raise SystemExit(line)
        print(f"{line}: reported, since a smoke's few entries say little", flush=True)
    index = dict(kind=f"{PROBE}-composed", report_only=True, draw=args.draw, clips=args.clips, seeds=list(SEEDS), code_digest=code_digest(), environment=P.environment(), pick=PICK_SEED, stored_export=dict(key=C.export_index_key(STORED_COMPOSED, args.draw), sha256=STORED[args.draw]["index"]), identity=dict(non_empty=close, byte_identical=identical, within_tolerance=close, clips=len(rows)), coverage=coverage_of(table["oracle"], real.failed, with_pool, not_served), orders=order_stats(table), seconds=round(time.time() - started, 1), rows=table)
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


def load_composed(store, prefix: str, k: int, records, declared: bool, instrument: str = "parent") -> dict:
    """{kind: rows, in list order}, of this code. On D20 each row is the record's. A smoke's rows are the first clips, read against other references."""
    key = index_key(prefix, k)
    if not store.exists(key):
        raise SystemExit(f"{key}: draw {k} is not composed")
    index = load_index(store, prefix, k, instrument, L.D20_CLIPS if declared else None)
    out = {}
    for kind, rows in index["rows"].items():
        if len(rows) != len(records) and (declared or len(rows) < len(records)):
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
            out[kind] = sorted(rows, key=lambda r: r["clip"])[:len(records)]      # a smoke that lost a monocular clip reads the first it can
    return out


def has_stored_composition(declared: bool, instrument: str) -> bool:
    """The parent's read of the composition stands, by its pins. A smoke has no stored forms, and the mirror never read it."""
    return declared and instrument == "parent"


def forms_of_draw(k: int, declared: bool, instrument: str = "parent") -> list[str]:
    C.check_draw(k)
    names = [n for kind in KINDS for n in (f"d{k}_{kind}", f"d{k}_{kind}_still")]
    return names if has_stored_composition(declared, instrument) else names + [f"d{k}_comp", f"d{k}_still", f"d{k}_blank"]


def run_read(args) -> int:
    """``--forms control``: the instrument's anchors. ``--forms comp-check --draws K``: the parent's stored composition of draw K, read again from its stored export, before anything is composed. ``--draws``: each draw's forms and their stills, and comp, still and blank where they have no stored forms: the mirror's, and a smoke's."""
    import torch
    from train_text_to_motion import make_store
    P.check_pins()
    torch.set_num_threads(args.threads)
    check_instrument(args.instrument)
    if args.forms not in (None, "control", "comp-check"):
        raise SystemExit(f"--forms {args.forms}: only control and comp-check")
    if args.forms == "comp-check" and args.instrument == "mirror":
        raise SystemExit("--forms comp-check reads a stored composition, the parent's or v1's: the mirror has none")
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
    if not args.forms and needs_comp_check(declared, args.instrument):
        passed_comp_check(out_dir, digest, place)
    refs = L.load_states(store, records)
    anchors = [n for n in wanted if n in anchors_of(args.instrument)]
    clips = []
    if args.instrument != "parent" and (anchors or not declared):      # a smoke's every read holds the same clips
        clips, records, refs = monocular_clips(store, records, refs, declared)
    texts, signs, groups = [r["text"] for r in records], [r["sign"] for r in records], [r["signer_component"] for r in records]
    tokenizer = L.load_tokenizer(store)
    candidates = M.retrieval_candidates(len(texts), C.CANDIDATES, seed=0)
    reader = load_instrument(args.instrument, store, args)
    shared = dict(reader=reader, texts=texts, signs=signs, groups=groups, candidates=candidates, refs=refs, args=args, declared=declared, described=described, digest=digest, place=place, out_dir=out_dir, probe=PROBE)
    for name in anchors:
        started = time.time()
        if args.instrument == "parent":
            C.read_one(name, C.build_anchor(name, refs, tokenizer), list(refs), None, started=started, **shared)
        else:
            _, states, masks = SYM.build_form(name, refs, clips, tokenizer)
            C.read_one(name, states, masks, None, started=started, **shared)
    if args.forms == "comp-check":
        name = wanted[0]
        rows = load_stored_export(store, draws[0], records, len(records))
        started = time.time()
        C.read_one(name, read_states(store, rows, refs), list(refs), rows, started=started, **shared)
        return 0
    for k in draws:
        composed = load_composed(store, args.composed, k, records, declared, args.instrument)
        here = [n for n in wanted if n.startswith(f"d{k}_")]
        for kind in ("comp", *KINDS):
            names = [n for n in here if n in (f"d{k}_{kind}", f"d{k}_{kind}_still", *((f"d{k}_still", f"d{k}_blank") if kind == "comp" else ()))]
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
    if args.instrument == "v1":
        row, _ = a5_row(read, C.load_top10(store, args.draw), V1_A5_BLEU[args.draw - 1])      # a pin failure stops here with status 1: nothing is voided
    else:
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
    print(f"ALERT the stored composition is not reproduced: {row['misses']}. The read is void: no form follows, and this code never reads again. {RERUN_ROUTE}", flush=True)
    return 3


def check_cause(cause: str | None) -> None:
    if cause is not None and not cause.strip():
        raise SystemExit("--rerun-cause is blank: a rerun records what was wrong and what the new code changes")


def run_preflight(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    check_cause(args.rerun_cause)
    store = make_store(args.store)
    if store.exists(args.final_key):
        raise SystemExit(f"{args.final_key} exists: this read is finished (write-once)")
    P.check_void(store, args, code_digest(), Path(args.out_dir))
    print(json.dumps(dict(preflight="clear", code_digest=code_digest())), flush=True)
    return 0


def run_inputs_v1(store) -> int:
    """v1's pins before any spend: the composed indexes, the A5 reports and summary, the canary, the parts job's forms, all of D20's clips. No model is fitted and nothing is composed. Prints hashes, no score."""
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    signs, texts = [r["sign"] for r in records], [r["text"] for r in records]
    raw = store.get(C.A5_SUMMARY)
    P.check_sha(raw, C.A5_SUMMARY_SHA, C.A5_SUMMARY)
    if tuple(json.loads(raw)["forms"]["predicted"]["top10_bleu"]) != V1_A5_BLEU:
        raise SystemExit(f"{C.A5_SUMMARY}: it does not hold the declared per-draw BLEU")
    for k in DRAWS:
        load_index(store, f"{PREFIX}/composed", k, "v1")
        top10 = C.load_top10(store, k)
        if top10["signs"] != signs or top10["references"] != texts:
            raise SystemExit(f"draw {k}: its stored A5 report holds other clips than D20, or another order")
    P.check_sha(store.get(P.CANARY_REPORT), P.CANARY_REPORT_SHA, P.CANARY_REPORT)
    for name in V1_STORED_ANCHORS:
        stored = Q.stored_form(store, Q.STORED_V1, name)
        if stored["signs"] != signs or stored["references"] != texts or stored["instrument"]["sha256"] != L.TRANSLATOR_SHA:
            raise SystemExit(f"v1's stored {name} holds other clips than D20, another order or another instrument")
    print(json.dumps(dict(inputs=dict(instrument="v1", draws=list(DRAWS), summary=C.A5_SUMMARY_SHA[:8], canary=P.CANARY_REPORT_SHA[:8], indexes={k: COMPOSED_INDEX[k][:8] for k in DRAWS}))), flush=True)
    return 0


def run_inputs(args) -> int:
    """The job's first step: every pinned stored object, the stored report and the stage-2 model, fitted once to a file for the compose steps. Prints hashes, no score."""
    from train_text_to_motion import make_store
    import round15_stage2_fit as fit
    P.check_pins()
    check_gpu(args.expect_gpu)
    store = make_store(args.store)
    if args.instrument == "v1":
        return run_inputs_v1(store)
    if not args.model_file:
        raise SystemExit("--model-file: the parent and the mirror's job fits the model once")
    P.check_sha(store.get(STORED_REPORT), STORED_REPORT_SHA, STORED_REPORT)
    alerts = []
    for k in DRAWS:
        P.check_sha(store.get(C.export_index_key(STORED_COMPOSED, k)), STORED[k]["index"], C.export_index_key(STORED_COMPOSED, k))
        for kind in ("comp", "still", "blank"):
            read = load_stored_form(store, k, kind)
            if kind == "comp":
                alerts += validity_alerts("the stored composition", {k: read})
    if alerts:
        print(f"ALERT {alerts}", flush=True)
        raise SystemExit("a stored composition is not valid on every point: the header's reason for reading it as it stands fails")
    for name in C.ANCHORS:
        Q.stored_form(store, Q.STORED_PARENT, name)
    load_mirror_report(store)
    records, declared, _ = L.open_read_list(store, 0)
    P.check_read_list(records, declared)
    signs, texts = [r["sign"] for r in records], [r["text"] for r in records]
    for name in MIRROR_ANCHORS:
        stored = load_mirror_form(store, name)
        if stored["signs"] != signs or stored["references"] != texts:
            raise SystemExit(f"the mirror's stored {name} holds other clips than D20, or another order")
    keys = {(json.loads(store.get(C.export_index_key(STORED_COMPOSED, k))).get("word_cache") or {}).get("checkpoint_key") for k in DRAWS}
    if len(keys) != 1 or None in keys:
        raise SystemExit(f"the stored exports name priors {sorted(str(k) for k in keys)}, not one")
    decoder = check_decoder(store, keys.pop())
    path = Path(args.model_file)
    if not path.exists():
        model, _ = fit.fit_model(fit.agc._store(args.store), L.EVAL_MIX, L.EVAL_MIX_SHA)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model))
    print(json.dumps(dict(inputs=dict(draws=list(DRAWS), report=STORED_REPORT_SHA[:8], mirror_report=MIRROR_REPORT_SHA[:8], model=load_model(path)["sha256"][:8], decoder=decoder))), flush=True)
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
               share_of_nearest=(lambda x: float(x["margin"].mean() / x["nearest"].mean()) if x["nearest"].mean() else float("nan"), lambda x, idx: C.ratio_draws(x["margin"][idx].mean(1), x["nearest"][idx].mean(1))),
               bleu=(lambda x: float(_bleu(x["stats"].sum(0))), lambda x, idx: _bleu(x["stats"][idx].sum(1))),
               bleu_gap_share=(lambda x: _gap_point(x), lambda x, idx: C.ratio_draws(_bleu(x["stats"][idx].sum(1)) - _bleu(x["blank_stats"][idx].sum(1)), _bleu(x["real_stats"][idx].sum(1)) - _bleu(x["blank_stats"][idx].sum(1)))),
               likelihood_gap_share=(lambda x: float(x["likelihood_top"].mean() / x["likelihood_bottom"].mean()) if x["likelihood_bottom"].mean() else float("nan"), lambda x, idx: C.ratio_draws(x["likelihood_top"][idx].mean(1), x["likelihood_bottom"][idx].mean(1))),
               still_raw=(lambda x: float(x["still_raw"].mean()), lambda x, idx: x["still_raw"][idx].mean(1)))


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


def paired_rows(mine: list[dict], other: list[dict], idx: np.ndarray, bands: dict | None = None, metrics=tuple(METRICS)) -> dict:
    """Mine minus other on the same draws and clips: each metric, over the draws, seeds averaged within a draw. ``bands``, the composition's arrays: margin and BLEU differences on the same clips, by composition's coverage."""
    draws = sorted(mine[0])
    out = {}
    for name in metrics:
        point, boot = METRICS[name]
        here = [float(np.mean([point(m[k]) for m in mine])) - float(np.mean([point(m[k]) for m in other])) for k in draws]
        out[name] = C.over_draws(here, np.stack([np.mean([boot(m[k], idx) for m in mine], 0) - np.mean([boot(m[k], idx) for m in other], 0) for k in draws]))
    if bands is not None:
        out["by_composition_coverage"] = {f"{lo}-{hi}": band_difference(mine, other, draws, bands, lo, hi) for lo, hi in C.COVERAGE_BINS}
    return out


def as_real(arrays: dict) -> dict:
    """The arrays of real signing on this instrument's own path, so a form minus real is the same arithmetic."""
    return {k: dict(x, margin=x["real"], stats=x["real_stats"]) for k, x in arrays.items()}


def as_nearest(arrays: dict, stats) -> dict:
    """The arrays of nearest on this instrument's own path, as ``as_real``."""
    return {k: dict(x, margin=x["nearest"], stats=stats) for k, x in arrays.items()}


def contrast(row: dict) -> str:
    """The declared rule: above or below only if the t interval over draws excludes zero on both margin and BLEU in the same direction; anything else is unresolved. The clip interval is not read."""
    found = set()
    for metric in ("margin", "bleu"):
        interval = row[metric].get("t_ci95") if isinstance(row.get(metric), dict) else None
        if not interval or not all(np.isfinite(x) for x in interval):
            return "unresolved"
        found.add("above" if interval[0] > 0 else "below" if interval[1] < 0 else "unresolved")
    return found.pop() if len(found) == 1 else "unresolved"


def reading_contrasts(paired: dict, instrument: str) -> dict:
    """Each declared contrast by the rule, and the cell. It describes the rows above and gates nothing."""
    like = paired["form_minus_nearest"]["oracle"] if instrument == "mirror" else paired["form_minus_real"]["oracle"]
    format_, generated = contrast(like), contrast(paired["oracle_minus_covered"])
    return dict(rule="above or below only if the t interval over draws excludes zero on both margin and BLEU in the same direction; the clip interval decides nothing", gates_nothing=True, format_like="nearest_monocular" if instrument == "mirror" else "real_two_view", format=format_, format_beside=dict(parent=None, mirror=dict(like="real_monocular", contrast=contrast(paired["form_minus_real"]["oracle"])), v1=dict(like="nearest_two_view", contrast=contrast(paired["form_minus_nearest"]["oracle"])))[instrument], generated_against_real_clips=generated, cell=CELLS.get((format_, generated), "unresolved"), unresolved=[name for name, value in (("format", format_), ("generated_against_real_clips", generated)) if value == "unresolved"], words_lacked=contrast(paired["covered_minus_composition"]), shuffle_on_moved_clips={name: contrast(row) if "margin" in row else "unresolved" for name, row in paired["moved"].items()})


def moved_mask(rows: list[dict]) -> np.ndarray:
    """Over the clips in list order: those whose word sequence this form's order changed."""
    return np.array([len(r["words"]) >= 2 and [w["word"] for w in r["words"]] != sorted_words(r) for r in rows], bool)


def moved_rows(mine: list[dict], masks: list[dict], other: list[dict], idx: np.ndarray) -> dict:
    """Mine minus its unshuffled partner on moved clips alone, each seed's own, seeds averaged within a draw: margin and BLEU, with the bootstrap over clips counting a resample's moved clips only."""
    draws = sorted(mine[0])
    margin, bleu, margin_boot, bleu_boot, counts = [], [], [], [], []
    for k in draws:
        a, b, c, d = [], [], [], []
        for m, mask, o in zip(mine, masks, other * len(mine)):
            x, y, moved = m[k], o[k], mask[k]
            counts.append(int(moved.sum()))
            if not moved.any():
                continue
            a.append(float((x["margin"] - y["margin"])[moved].mean()))
            b.append(float(_bleu(x["stats"][moved].sum(0)) - _bleu(y["stats"][moved].sum(0))))
            picked = moved[idx]
            c.append(np.divide(((x["margin"] - y["margin"])[idx] * picked).sum(1), picked.sum(1), out=np.full(len(idx), np.nan), where=picked.sum(1) > 0))
            d.append(_bleu((x["stats"][idx] * picked[..., None]).sum(1)) - _bleu((y["stats"][idx] * picked[..., None]).sum(1)))
        if a:
            margin.append(float(np.mean(a)))
            bleu.append(float(np.mean(b)))
            margin_boot.append(np.mean(c, 0))
            bleu_boot.append(np.mean(d, 0))
    if not margin:
        return dict(moved_clips_per_form=0.)
    return dict(moved_clips_per_form=round(float(np.mean(counts)), 1), margin=C.over_draws(margin, np.stack(margin_boot)), bleu=C.over_draws(bleu, np.stack(bleu_boot)))


def still_effect(mine: list[dict], other: list[dict], idx: np.ndarray) -> dict:
    """A shuffled still is another frame: its raw margin minus the partner's, and the form's margin over the partner's still instead of its own."""
    draws = sorted(mine[0])
    over = lambda x, y: x["margin"] + x["still_raw"] - y["still_raw"]      # noqa: E731
    per_draw = [float(np.mean([over(m[k], o[k]).mean() for m, o in zip(mine, other * len(mine))])) for k in draws]
    boots = np.stack([np.mean([over(m[k], o[k])[idx].mean(1) for m, o in zip(mine, other * len(mine))], 0) for k in draws])
    return dict(still_raw_difference=paired_rows(mine, other, idx, metrics=("still_raw",))["still_raw"], margin_over_partners_still=C.over_draws(per_draw, boots))


def band_mask(x: dict, lo: float, hi: float) -> np.ndarray:
    return x["non_empty"] & (x["coverage"] > lo) & (x["coverage"] <= hi)


def band_interval(values: list) -> list | None:
    """The interval of the bootstrap means over a band's clips (draws x resamples), the draws averaged."""
    return P._quantiles(np.mean(values, 0)) if values else None


def _band_idx(n: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, n, (C.BOOT, n))


def band_difference(mine: list[dict], other: list[dict], draws, bands: dict, lo: float, hi: float) -> dict:
    margin, bleu, clips, margin_boot, bleu_boot = [], [], [], [], []
    for k in draws:
        mask = band_mask(bands[k], lo, hi)
        clips.append(int(mask.sum()))
        if not mask.any():
            continue
        idx = _band_idx(int(mask.sum()))
        margin.append(float(np.mean([m[k]["margin"][mask].mean() for m in mine]) - np.mean([m[k]["margin"][mask].mean() for m in other])))
        bleu.append(float(np.mean([_bleu(m[k]["stats"][mask].sum(0)) for m in mine]) - np.mean([_bleu(m[k]["stats"][mask].sum(0)) for m in other])))
        margin_boot.append(np.mean([m[k]["margin"][mask][idx].mean(1) for m in mine], 0) - np.mean([m[k]["margin"][mask][idx].mean(1) for m in other], 0))
        bleu_boot.append(np.mean([_bleu(m[k]["stats"][mask][idx].sum(1)) for m in mine], 0) - np.mean([_bleu(m[k]["stats"][mask][idx].sum(1)) for m in other], 0))
    return dict(clips_per_draw=round(float(np.mean(clips)), 1), margin=round(float(np.mean(margin)), 5) if margin else None, margin_ci95=band_interval(margin_boot), bleu=round(float(np.mean(bleu)), 4) if bleu else None, bleu_ci95=band_interval(bleu_boot))


def coverage_rows(members: list[dict], bands: dict | None = None) -> dict:
    """Report only: margin, its share of real and BLEU, by coverage: each seed's own (``bands`` None) or a fixed set of arrays' (composition's), the draws' means, with intervals over the band's clips. A band with no clip has none."""
    draws = sorted(members[0])

    def row(lo: float, hi: float) -> dict:
        margin, real, bleu, clips, margin_boot, bleu_boot = [], [], [], [], [], []
        for k in draws:
            for m in members:
                mask = band_mask((bands or m)[k], lo, hi)
                clips.append(int(mask.sum()))
                if mask.any():
                    idx = _band_idx(int(mask.sum()))
                    margin.append(float(m[k]["margin"][mask].mean()))
                    real.append(float(m[k]["real"][mask].mean()))
                    bleu.append(float(_bleu(m[k]["stats"][mask].sum(0))))
                    margin_boot.append(m[k]["margin"][mask][idx].mean(1))
                    bleu_boot.append(_bleu(m[k]["stats"][mask][idx].sum(1)))
        if not margin:
            return dict(clips_per_draw=0.)
        return dict(clips_per_draw=round(float(np.mean(clips)), 1), margin=round(float(np.mean(margin)), 5), margin_ci95=band_interval(margin_boot), real_margin=round(float(np.mean(real)), 5), share_of_real=round(float(np.mean(margin) / np.mean(real)), 4) if np.mean(real) else None, bleu=round(float(np.mean(bleu)), 4), bleu_ci95=band_interval(bleu_boot))
    return {f"{lo}-{hi}": row(lo, hi) for lo, hi in C.COVERAGE_BINS}


def signer_bleu(arrays: dict, groups) -> dict:
    """Report only: each signer's BLEU, the mean of the draws."""
    labels = np.asarray(groups)
    return {str(g): round(float(np.mean([_bleu(arrays[k]["stats"][labels == g].sum(0)) for k in sorted(arrays)])), 4) for g in sorted(set(groups))}


def subset_rows(arrays: dict, masks: dict, nearest) -> dict:
    """Report only, on clips in and outside v1's training list and outside training and selection: margin, its share of real, and BLEU of the form, real, nearest and blank with the BLEU gap share, the draws' means. Here training status and signer are confounded."""
    out = C.subset_rows(arrays, masks)
    draws = sorted(arrays)
    for name, mask in masks.items():
        if not out[name]:
            continue
        form, real, blank = (float(np.mean([_bleu(arrays[k][field][mask].sum(0)) for k in draws])) for field in ("stats", "real_stats", "blank_stats"))
        out[name].update(bleu=round(form, 4), real_bleu=round(real, 4), nearest_bleu=round(float(_bleu(nearest[mask].sum(0))), 4), blank_bleu=round(blank, 4), bleu_gap_share=round((form - blank) / (real - blank), 4) if real != blank else None)
    return out


def form_block(name: str, arrays: dict, rows: dict, views: dict, anchors: dict, first: dict, reads: dict, idx: np.ndarray, composition: dict, masks: dict, two_view_real: np.ndarray) -> dict:
    """One form's rows."""
    with warnings.catch_warnings():      # parent_comp's length block, which no row here reads, regresses on a ratio that may not vary
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        block = C.composition_block(arrays, idx)
        pod = dict(view=dict(forms={**anchors, **views}))
        signers = C.signer_rows(pod, {k: "view" for k in arrays}, first["signs"], first["groups"])
    for signer, bleu in signer_bleu(arrays, first["groups"]).items():
        signers[signer]["bleu"] = bleu
    return dict(form=name, draws=sorted(arrays), per_draw={str(k): rows[k] for k in sorted(rows)}, **{k: block[k] for k in BLOCK_ROWS}, by_own_coverage=coverage_rows([arrays]), by_composition_coverage=coverage_rows([arrays], composition), by_clip_subset=subset_rows(arrays, masks, M.bleu_statistics(anchors["nearest_two_view"]["hypotheses"], anchors["nearest_two_view"]["references"])), still_raw=group_rows([arrays], idx)["still_raw"], share_of_two_view_real=two_view_share(arrays, two_view_real, idx), per_signer=signers, per_signer_note="report only: few clips make these rows noisy", hypotheses=hypothesis_row({k: [reads[k]] for k in reads}, idx), empty_clips={str(k): rows[k]["empty"] for k in sorted(rows)})


def two_view_share(arrays: dict, two_view_real: np.ndarray, idx: np.ndarray) -> dict:
    """The form's margin over real two-view's, over its own still: on the parent the same as ``share_of_real``, on the mirror the other path's."""
    draws = sorted(arrays)
    return C.over_draws([float(arrays[k]["margin"].mean() / two_view_real.mean()) if two_view_real.mean() else float("nan") for k in draws], np.stack([C.ratio_draws(arrays[k]["margin"][idx].mean(1), two_view_real[idx].mean(1)) for k in draws]))


def corpus_rows(oracle: dict, covered: dict, composed: dict) -> dict:
    """Report only: the oracle's rows by each clip's main pick corpus (most words; a tie, the first by name), with covered on the same clips."""
    from collections import Counter
    labels = {}
    for k in sorted(oracle):
        names = []
        for row in sorted(composed[k]["rows"]["oracle"], key=lambda r: r["clip"]):
            count = Counter(w["corpus"] for w in row["words"])
            names.append(min(count, key=lambda c: (-count[c], c)) if count else None)
        labels[k] = np.array(names[:len(oracle[k]["margin"])], object)
    out = {}
    for corpus in sorted({x for v in labels.values() for x in v if x is not None}):
        masks = {k: labels[k] == corpus for k in labels}
        used = [k for k in masks if masks[k].any()]
        mean = lambda arrays, field: float(np.mean([arrays[k][field][masks[k]].mean() for k in used]))      # noqa: E731
        bleu = lambda arrays: float(np.mean([_bleu(arrays[k]["stats"][masks[k]].sum(0)) for k in used]))      # noqa: E731
        out[corpus] = dict(clips_per_draw=round(float(np.mean([masks[k].sum() for k in labels])), 1), oracle_margin=round(mean(oracle, "margin"), 5), covered_margin=round(mean(covered, "margin"), 5), oracle_bleu=round(bleu(oracle), 4), covered_bleu=round(bleu(covered), 4))
    return out


def validity_alerts(name: str, reads: dict) -> list[str]:
    """A composed clip with signing is valid on every point the reader reads. Under that, the header's reason for reading it as it stands fails."""
    out = []
    for k, read in reads.items():
        low = [v for v, empty in zip(read["valid_frame_share"], read["empty"]) if not empty and v < 1 - 1e-6]
        if low:
            out.append(f"draw {k} {name}: {len(low)} clips with signing have a valid_frame_share under 1, the lowest {min(low)}")
    return out


def write_report(store, key: str | None, file: str | None, report: dict) -> bool:
    """Write-once, and identical bytes stand: True when it wrote. A merge that runs again on the same forms changes nothing."""
    if bool(key) == bool(file):
        raise SystemExit("give exactly one of --output-key and --output-file")
    payload = json.dumps(report, indent=1, sort_keys=True, default=float).encode()
    existing = (store.get(key) if store.exists(key) else None) if key else (Path(file).read_bytes() if Path(file).exists() else None)
    if existing == payload:
        return False
    if existing is not None:
        raise SystemExit(f"{key or file} exists and holds other bytes (write-once)")
    L.write_once(store, key, file, report)
    return True


def run_put_report(args) -> int:
    """The merged report to its final key, write-once; identical bytes stand."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    payload = Path(args.file).read_bytes()
    if store.exists(args.key):
        if store.get(args.key) != payload:
            raise SystemExit(f"{args.key} exists and holds other bytes (write-once)")
        print(json.dumps(dict(report=args.key, stored=True)), flush=True)
        return 0
    store.put(args.key, payload)
    print(json.dumps(dict(report=args.key, stored=False)), flush=True)
    return 0


EXIT_NOT_THERE = 5


def run_stage_state(args) -> int:
    """Exit 0 when a stage's final key holds the bytes of its local report: the stage is done. On a new pod, with no local report, the report is restored from the key if it is of this code, this environment and these planned draws, with the stage's record, so the stage stands. 5 when the key is absent: run it. Anything else refuses: the key belongs to another run."""
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if not store.exists(args.key):
        print(json.dumps(dict(stage=args.key, state="to run")), flush=True)
        return EXIT_NOT_THERE
    local, raw = Path(args.file), store.get(args.key)
    if local.exists():
        if local.read_bytes() != raw:
            raise SystemExit(f"{args.key} exists and {args.file} does not hold its bytes: this read is finished elsewhere (write-once)")
        print(json.dumps(dict(stage=args.key, state="done")), flush=True)
        return 0
    report = json.loads(raw)
    stage = report.get("stage") or {}
    planned = [int(x) for x in args.planned_draws.split(",") if x] if args.planned_draws else None
    differs = [what for what, ok in (("kind", report.get("kind") == f"{PROBE}-report"), ("code", report.get("code_digest") == code_digest()), ("environment", report.get("environment") == P.environment()), ("planned draws", planned is not None and stage.get("planned_draws") == planned)) if not ok]
    if differs:
        raise SystemExit(f"{args.key} exists and is not restorable here: it differs in {differs}. Other code needs RERUN_CAUSE, which writes under runs/<code digest>/")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
    if args.record and not Path(args.record).exists():
        Path(args.record).write_text(f"{','.join(str(k) for k in stage['draws'])} {stage['stop_reason']} {stage['elapsed_seconds']} {stage['budget_seconds']}\n")
    print(json.dumps(dict(stage=args.key, state="restored", draws=stage.get("draws"))), flush=True)
    return 0


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


def open_read(store, prefix: str, instrument: str, draws: list[int], composed_prefix: str, *, want_declared=None) -> SimpleNamespace:
    """One instrument's forms, checked as one run of this code, with each form's arrays on its own path's anchors."""
    pin = check_instrument(instrument)
    forms, control, comp_check = load_forms(store, prefix)
    if not forms:
        raise SystemExit(f"{prefix}: no form")
    digests = {r.get("code_digest") for r in forms.values()}
    if digests != {code_digest()}:
        raise SystemExit(f"the forms are of code {sorted(str(d) for d in digests)}, not this code {code_digest()}: a stale read")
    if len({(r["declared_read"], r["list"]["list"], tuple(r["signs"]), tuple(r["references"])) for r in forms.values()}) != 1:
        raise SystemExit("the forms read other clips, another list, or a smoke mixed with a declared read")
    if {r["instrument"]["sha256"] for r in forms.values()} != {pin["sha256"]}:
        raise SystemExit(f"a form was not read by the {instrument}")
    if any(a not in forms for a in anchors_of(instrument)):
        raise SystemExit("the anchors are missing")
    _, place = P.check_same_run(forms, "the forms")
    first = forms["real_two_view"]
    declared = first["declared_read"]
    if want_declared is not None and declared != want_declared:
        raise SystemExit(f"the {instrument} forms under {prefix} are of another kind of run")
    stored_composition = has_stored_composition(declared, instrument)
    if declared and (not control or control.get("reproduced") is not True or control.get("code_digest") != code_digest() or control.get("environment") != place):
        raise SystemExit("no passed control of this code and environment; the read is void")
    if needs_comp_check(declared, instrument) and (not comp_check or comp_check.get("reproduced") is not True or comp_check.get("code_digest") != code_digest() or comp_check.get("environment") != place):
        raise SystemExit("no passed composition check of this code and environment; the read is void")
    for k in draws:
        C.check_draw(k)
        missing = [n for n in forms_of_draw(k, declared, instrument) if n not in forms]
        if missing:
            raise SystemExit(f"draw {k}: {missing[0]} is missing, of {len(missing)} forms")
    base = {k: {kind: load_stored_form(store, k, kind) if stored_composition else forms[f"d{k}_{kind}"] for kind in ("comp", "still", "blank")} for k in draws}
    for k in draws:
        for kind, read in base[k].items():
            if read["signs"] != first["signs"] or read["references"] != first["references"]:
                raise SystemExit(f"draw {k}: the stored {kind} holds other clips or another order")
    n = len(first["signs"])
    idx = np.random.default_rng(0).integers(0, n, (C.BOOT, n))
    anchors = own_path(forms, instrument)
    composed = {k: load_index(store, composed_prefix, k, instrument, L.D20_CLIPS if declared else None) for k in draws}
    comp_arrays, comp_rows, comp_views = view_arrays({k: (base[k]["comp"], base[k]["still"]) for k in draws}, base, anchors, idx, "the composition")
    built = {}
    for kind in KINDS:
        reads = {k: (forms[f"d{k}_{kind}"], forms[f"d{k}_{kind}_still"]) for k in draws}
        here, rows, views = view_arrays(reads, base, anchors, idx, kind)
        built[kind] = SimpleNamespace(arrays=here, rows=rows, views=views, reads={k: reads[k][0] for k in draws})
    two_view_real = Q.margins(forms["real_two_view"]) - Q.margins(forms["still_two_view"])
    return SimpleNamespace(forms=forms, control=control, comp_check=comp_check, place=place, first=first, declared=declared, stored_composition=stored_composition, base=base, idx=idx, anchors=anchors, composed=composed, comp_arrays=comp_arrays, comp_rows=comp_rows, comp_views=comp_views, built=built, two_view_real=two_view_real)


def members_of(read: SimpleNamespace, name: str) -> list[dict]:
    """A named form's arrays, or a group's, one per seed."""
    if name == "composition":
        return [read.comp_arrays]
    return [read.built[kind].arrays for kind in GROUPS.get(name, (name,))]


def paired_block(read: SimpleNamespace, composed_rows: dict) -> dict:
    """Every paired row: on all clips, against real, and on moved clips with each shuffled still's effect."""
    idx, comp = read.idx, read.comp_arrays
    names = ("oracle", "covered", *GROUPS)
    out = {f"{name}_minus_composition": paired_rows(members_of(read, name), [comp], idx, comp) for name in names}
    out["oracle_minus_covered"] = paired_rows(members_of(read, "oracle"), members_of(read, "covered"), idx, comp)
    for name, partner in PARTNERS.items():
        if partner != "composition":
            out[f"{name}_minus_{partner}"] = paired_rows(members_of(read, name), members_of(read, partner), idx, comp)
    out["form_minus_real"] = {name: paired_rows(members_of(read, name), [as_real(comp)], idx, metrics=("margin", "bleu")) for name in ("composition", *names)}
    nearest = M.bleu_statistics(read.anchors["nearest_two_view"]["hypotheses"], read.anchors["nearest_two_view"]["references"])
    out["form_minus_nearest"] = {name: paired_rows(members_of(read, name), [as_nearest(comp, nearest)], idx, metrics=("margin", "bleu")) for name in ("composition", *names)}
    out["moved"], out["still_effect"] = {}, {}
    for name, partner in PARTNERS.items():
        masks = [{k: composed_rows[k][kind] for k in composed_rows} for kind in GROUPS[name]]
        out["moved"][f"{name}_minus_{partner}"] = moved_rows(members_of(read, name), masks, members_of(read, partner), idx)
        out["still_effect"][f"{name}_against_{partner}"] = still_effect(members_of(read, name), members_of(read, partner), idx)
    return out


def parent_on_mirror_draws(store, args, read: SimpleNamespace, draws: list[int]) -> dict | None:
    """The parent's rows on the mirror's own draws and clips, and parent minus mirror draw by draw. Each share is over its instrument's own path's real."""
    if not args.parent_forms:
        return None
    parent = open_read(store, args.parent_forms, "parent", draws, args.composed, want_declared=read.declared)
    if parent.first["signs"] != read.first["signs"]:
        if not read.declared:      # a smoke's mirror lost a monocular clip the parent read
            return None
        raise SystemExit("the parent's forms read other clips than the mirror's")
    out = {}
    for name in ("composition", "oracle", "covered", *GROUPS):
        mine, theirs = members_of(parent, name), members_of(read, name)
        out[name] = dict(parent=group_rows(mine, read.idx), parent_minus_mirror=paired_rows(mine, theirs, read.idx))
    return out


def run_merge(args) -> int:
    from train_text_to_motion import make_store
    P.check_pins()
    check_instrument(args.instrument)
    check_cause(args.rerun_cause)
    store = make_store(args.store)
    draws = [int(x) for x in args.draws.split(",") if x]
    if not draws:
        raise SystemExit("--draws names no draw")
    with np.errstate(divide="ignore", invalid="ignore"):      # a resample's ratio can hold a zero denominator: ``_quantiles`` leaves what is not finite out
        read = open_read(store, args.prefix, args.instrument, draws, args.composed)
        if args.instrument == "v1" and read.declared and tuple(draws) != DRAWS:
            raise SystemExit(f"v1 reads all eight draws, not {draws}: it is the ruler that matters")
        forms, first, declared, idx, anchors, composed, comp_arrays = read.forms, read.first, read.declared, read.idx, read.anchors, read.composed, read.comp_arrays
        masks = SYM.clip_subsets(store, first["signs"])
        blocks = {kind: form_block(kind, read.built[kind].arrays, read.built[kind].rows, read.built[kind].views, anchors, first, read.built[kind].reads, idx, comp_arrays, masks, read.two_view_real) for kind in KINDS}
        composition = form_block("composition", comp_arrays, read.comp_rows, read.comp_views, anchors, first, {k: read.base[k]["comp"] for k in draws}, idx, comp_arrays, masks, read.two_view_real)
        blocks["oracle"]["by_pick_corpus"] = corpus_rows(read.built["oracle"].arrays, read.built["covered"].arrays, composed)
        groups = {name: dict(seeds=list(SEEDS), **group_rows(members_of(read, name), idx), by_own_coverage=coverage_rows(members_of(read, name)), by_composition_coverage=coverage_rows(members_of(read, name), comp_arrays), hypotheses=hypothesis_row({k: [forms[f"d{k}_{m}"] for m in members] for k in draws}, idx)) for name, members in GROUPS.items()}
        moved = {k: {kind: moved_mask(sorted(composed[k]["rows"][kind], key=lambda r: r["clip"])[:len(first["signs"])]) for kind in KINDS if "shuffled" in kind} for k in draws}
        paired = paired_block(read, moved)
        beside = None if args.instrument == "parent" else parent_on_mirror_draws(store, args, read, draws)
        gap = official_gap(read, idx) if args.instrument == "v1" else None
    real_hypotheses = hypothesis_row({0: [anchors["real_two_view"]]}, idx)      # the instrument's own path
    length_over_monocular = None if args.instrument == "parent" else round(float(np.mean([a / b for a, b in zip(read.base[draws[0]]["comp"]["frames"], forms["real_monocular"]["frames"])])), 4)
    coverage = {str(k): composed[k]["coverage"] for k in draws}
    alerts = [f"draw {k}: its oracle holds no word" for k in draws if not composed[k]["coverage"]["words_kept"]]
    alerts += validity_alerts("composition", {k: read.base[k]["comp"] for k in draws})
    a5 = None
    if args.instrument == "v1" and declared:
        a5, missed = a5_rows(store, forms, draws)
        alerts += missed
    for kind in KINDS:
        alerts += validity_alerts(kind, read.built[kind].reads)
    path_names = dict(real="real_two_view", nearest="nearest_two_view", still="still_two_view", blank="blank") if args.instrument != "mirror" else dict(real="real_monocular", nearest="nearest_monocular", still="still_monocular", blank="blank")
    stage = dict(planned_draws=[int(x) for x in args.planned_draws.split(",") if x] if args.planned_draws else draws, draws=draws, stop_reason=args.stop_reason, elapsed_seconds=args.elapsed_seconds, budget_seconds=args.budget_seconds)
    report = dict(kind=f"{PROBE}-report", report_only=True, note=C.NOT_DECIDED, reading=READING_V1 if args.instrument == "v1" else READING, reading_contrasts=reading_contrasts(paired, args.instrument), official_gap=gap, stored_a5=a5, rerun_cause=args.rerun_cause, declared_read=declared, smoke=None if declared else "SMOKE: composed D20 sentences against c02 train references; these numbers are plumbing, not a read", list=first["list"], code_digest=code_digest(), environment=read.place, instrument=first["instrument"], instrument_name=args.instrument, own_path_anchors=path_names, length_over_monocular=length_over_monocular, draws=draws, draws_note=None if tuple(draws) == DRAWS else f"fewer than eight: {draws}", stage=stage, seeds=list(SEEDS), disclosures=DISCLOSURES, cost_estimate=V1_COST_ESTIMATE if args.instrument == "v1" else COST_ESTIMATE, alerts=alerts, stored=dict(report_sha256=STORED_REPORT_SHA, digest=STORED_DIGEST) if read.stored_composition else None,
                  anchors=anchor_report(forms, args.instrument, idx), real_hypotheses=real_hypotheses, composition=composition, forms=blocks, groups=groups, paired=paired, parent_on_mirror_draws=beside, oracle_coverage=coverage, orders={str(k): composed[k]["orders"] for k in draws}, identity={str(k): composed[k]["identity"] for k in draws},
                  checks=dict(controls=dict(anchors=read.control, **(dict(comp_check=read.comp_check) if needs_comp_check(declared, args.instrument) else {})), blank_identical_to_real=round(float(np.mean([a == b for a, b in zip(anchors["blank"]["hypotheses"], anchors["real_two_view"]["hypotheses"])])), 4)),
                  resources=dict(forms={k: {x: r[x] for x in ("peak_rss_bytes", "gpu_peak_reserved_bytes", "build_seconds", "read_seconds")} for k, r in forms.items()}, compose_seconds={str(k): composed[k]["seconds"] for k in draws}))
    wrote = write_report(store, args.output_key, args.output_file, report)
    print(json.dumps(dict(merged=args.output_key or args.output_file, wrote=wrote, report_only=True, declared_read=declared, instrument=args.instrument, draws=draws, stage=stage, anchors=report["anchors"], composition={k: composition[k] for k in BLOCK_ROWS}, forms={f: {k: blocks[f][k] for k in ("margin", "bleu")} for f in ("oracle", "covered")}, groups={f: {k: groups[f][k] for k in ("margin", "bleu")} for f in groups}, paired={f: {k: paired[f][k] for k in ("margin", "bleu")} for f in ("oracle_minus_covered", "oracle_minus_composition", "covered_minus_composition")}, oracle_coverage=coverage, alerts=alerts)), flush=True)
    return 0


def official_gap(read: SimpleNamespace, idx: np.ndarray) -> dict:
    """Composition's and its shuffled form's shares of the official gap, (form minus blank) over (real minus blank) in BLEU, and the share of composition's gap that shuffling removes, over the draws."""
    composition, shuffled = members_of(read, "composition"), members_of(read, "shuffled")
    draws = sorted(composition[0])
    bleu, bleu_boot = METRICS["bleu"]
    lost, lost_boot = [], []
    for k in draws:
        x = composition[0][k]
        blank = float(_bleu(x["blank_stats"].sum(0)))
        comp, drop = bleu(x), bleu(x) - float(np.mean([bleu(m[k]) for m in shuffled]))
        lost.append(drop / (comp - blank) if comp != blank else float("nan"))
        comps, blanks = bleu_boot(x, idx), _bleu(x["blank_stats"][idx].sum(1))
        lost_boot.append(C.ratio_draws(comps - np.mean([bleu_boot(m[k], idx) for m in shuffled], 0), comps - blanks))
    return dict(note="(form minus blank) over (real minus blank), BLEU, on the composition's blank; all clips, moved or not: paired.moved holds the drop on moved clips", composition=group_rows(composition, idx)["bleu_gap_share"], composition_shuffled=group_rows(shuffled, idx)["bleu_gap_share"], shuffled_minus_composition=paired_rows(shuffled, composition, idx, metrics=("bleu_gap_share",))["bleu_gap_share"], share_of_composition_gap_removed_by_shuffling=C.over_draws(lost, np.stack(lost_boot)))


def a5_rows(store, forms: dict, draws: list[int]) -> tuple[dict, list[str]]:
    """Each draw's composition through v1 against its stored A5 report. A miss is an alert."""
    rows, alerts = {}, []
    for k in draws:
        rows[str(k)], misses = a5_row(forms[f"d{k}_comp"], C.load_top10(store, k), V1_A5_BLEU[k - 1])
        alerts += [f"draw {k}: composition does not reproduce its stored A5 report: {m}" for m in misses]
    return rows, alerts


def run_digest(args) -> int:
    print(code_digest(), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    inputs = sub.add_parser("inputs", help="the job's first step: every pin, the model fitted once; prints no score")
    inputs.add_argument("--store", default="r2")
    inputs.add_argument("--model-file")
    inputs.add_argument("--instrument", default="parent")
    inputs.add_argument("--expect-gpu", help="the GPU type, underscores for spaces; another stops the job")
    sub.add_parser("digest", help="this code's digest, for a rerun's prefix")
    state = sub.add_parser("stage-state", help="exit 0 when a stage's final key holds its local report's bytes, 5 when the key is absent")
    state.add_argument("--store", default="r2")
    state.add_argument("--key", required=True)
    state.add_argument("--file", required=True)
    state.add_argument("--record", help="the stage's record, written when its report is restored")
    state.add_argument("--planned-draws", help="what this run plans, comma list; a restored report must hold the same")
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
    check_comp.add_argument("--instrument", default="parent")
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
    merge.add_argument("--parent-forms", help="the mirror's merge: the parent's forms, for its rows on the mirror's draws")
    merge.add_argument("--planned-draws", help="what the job planned, comma list; default --draws")
    merge.add_argument("--stop-reason", default="not_recorded")
    merge.add_argument("--elapsed-seconds", type=int)
    merge.add_argument("--budget-seconds", type=int)
    merge.add_argument("--rerun-cause")
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
    return dict(inputs=run_inputs, compose=run_compose, preflight=run_preflight, read=run_read, control=run_control, merge=run_merge, check=C.run_check, upload=PP.run_upload, digest=run_digest, **{"comp-check": run_comp_check, "stage-state": run_stage_state, "upload-log": R.run_upload_log, "put-report": run_put_report})[args.command](args)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
