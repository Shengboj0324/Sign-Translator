# Operator guide (owner only, technical)

This guide covers the reviewer sessions: producing clips, running sessions, entering answers and scoring. The study
design and the analysis follow `docs/HUMAN_EVALUATION_PROTOCOL.md`, which stays the reference. Items marked
**NOT BUILT** do not exist in the repo yet; they are collected in §9.

Reviewers never receive this guide, `key/`, any CSV or any script. They never use a terminal or compute a hash. They watch videos and answer on a printed form or in ASL. You generate every reviewer key and enter every answer.

## 0. Rules that apply to every clip

- **Research only, non-commercial.** Clips are shown only to reviewers who signed `CONSENT_FORM.md`. ASLLRP-derived
  renders are shown under the DAI terms only and are never redistributed (protocol §4).
- **Final-test material stays unseen.** How2Sign **test signer 9** and every `test` partition are the untouched final
  check (`docs/NEXT_PHASE_PLAN.md`, R6). Do not show them in the pilot or the main study. Show them only in the final
  study, once (`build_human_eval_packet.py` refuses `test` without `--final-study`), or earlier only by your explicit,
  recorded decision. If shown earlier, they can no longer serve as an untouched test.
- **Which sources may appear as real-signer anchors:**

  | Source | Use as anchor? |
  |---|---|
  | How2Sign val (signers 3, 1/2 and 11), CC BY-NC 4.0 | Yes |
  | ASLLRP / DAI 2 / ASLLVD val | Yes, under the DAI terms |
  | ASLLVD train | For the pilot's `oracle` arm: yes, under the DAI terms, to consenting reviewers, by your confirmation of 2026-11-02, 20:30 UTC |
  | ASL Citizen | **No.** Its consent covers recognition only. |
  | ASL STEM Wiki | **No.** |
  | YouTube-ASL | **No.** It is not redistributable. |
  | WLASL (C-UDA, "computational use") | Not unless you confirm its terms allow it |
  | Sem-Lex | Not unless you confirm its terms allow it |
  | RIT | Not unless you confirm its terms allow it |

  The round-15 oracle draws real words from ASLLVD, RIT, WLASL and Sem-Lex (`compositional.ROUND15_CORPORA`).
  Filter anchors by each record's `corpus` before rendering. The pilot's `oracle` arm (§10) does this itself, and needs your recorded
  note, `--word-format-licence-note`, to name each corpus it shows as covered, ASLLVD train clips included.
- **Avatar only.** Every condition, including the real-signer anchors, is shown through the same avatar renderer.
  Original source video would reveal which clips are real. Optionally, show one original How2Sign val video as an
  unscored warm-up. The Deaf checker, who is not a rater, is the one exception: the checker watches the real recordings and nothing else (§11).

## 1. Environment

- Python: `~/.venvs/sign-translator/bin/python`, run from the repo root. R2 reads need the `R2_*` environment variables.
- Blender 5.2 at `/Applications/Blender.app` (the `render_avatar.py` default; `--blender` to override). An avatar render
  takes about 26 s for a 104-frame clip at 768 px on the Mac, measured.
- Mac outage rule: 1 BLAS thread (`OMP_NUM_THREADS=1`), at most 6 processes, run under `nice`.
- Project rule: every avatar render is also kept in full in the private R2 `renders/` prefix with `INDEX.json`.
  Reviewer answers never go there.

## 2. Producing the clips

**2a. Sentence study, ASLLRP plans (exists).** This is the protocol's Latin-square packet. It builds every system on
the same utterances, hides the system labels and writes the private key:

```
~/.venvs/sign-translator/bin/python scripts/build_human_eval_packet.py \
  --manifest-key <plan manifest key> --manifest-sha256 <sha> \
  --system reference=reference \
  --system plan=plan:<checkpoint key>@<manifest sha> \
  --partition val --count 20 --renderer avatar \
  --output-dir <private dir outside the repo>
```

- Exact keys and the other system kinds (`translate`, `english`, `stitch`) are in protocol §5.
- `--output-prefix` copies the packet and key to a private R2 prefix. Omit it to keep everything local.
- Outputs:
  - `packet/group-<g>/sheet.csv` and the `<item_id>.mp4` files, in presentation order;
  - `key/key.json`, which is private.
- **Do not forward `group-<g>.zip` as it is.** It contains a CSV that only you fill. Play the videos yourself and give
  the reviewer printed `RATING_SHEET.md` pages (first rater read: `RATING_SHEET_TWO_STAGE.md`) numbered 1..N in
  `sheet.csv` row order.

**2b. One real or stored state (exists).**

```
~/.venvs/sign-translator/bin/python scripts/render_avatar.py --key <state_key> --sha256 <state_sha256> \
  --out-dir <new dir> --allow-inferred
```

Use this for an anchor clip of a How2Sign or ASLLRP val record: the manifest record's `continuous.state_key` and
`state_sha256`. The output is `avatar.mp4` plus `verification.json`.

**2c. How2Sign sentences and compositions (built, on test data only so far).** `round15_compose_eval.py --export-prefix` saves
composed states, `--export-word-format` adds the pilot's two report-only arms, and `build_human_eval_packet.py` has `how2sign-draw`,
`how2sign-packet` and `how2sign-stage2` (§10, protocol §5).

**NOT BUILT:**
- The `/rest_only` and `/mismatched_words` controls, and the refiner's forms, as packet systems. The export takes main forms.
- Isolated-word items. The packet builder designs sentence items only.
- Blank (rest pose for the clip's duration) and mismatched systems in the packet builder.

## 3. Study design

- **Two blocks per session.** Isolated words first, because they are easier, then short sentences (3 to 8 signs).
  Order is randomised within each block. Conditions are mixed in each block.
- **Conditions in every block:**
  - generated: each system under test;
  - real-signer anchor: a recorded val utterance or word through the same avatar;
  - blank control: rest pose for the same length. It should rate at the floor.
  - mismatched control: motion of a different item. Scored against the intended text, it should be near 0. This
    checks that the translation judge is blind.
- **Latin square** (`human_study.design_study`). No reviewer sees the same word or sentence twice. Systems are compared
  paired by item. Use at least 2 reviewers per group, working independently.
- **Blinding.** Reviewers see only video numbers. Do not open `key.json` until all sheets are in. In the session,
  give no reaction or hint. Ideally, a second fluent signer who never sees the conditions judges the translations.
- **Session size.** At most 30 to 45 minutes of watching, with a break every ~15 minutes.
  - About 1.5 min per clip on paper: 2 unscored practice clips, then 20 to 24 scored clips.
  - Example of 24 clips:

    | Block | Generated | Anchor | Blank | Mismatched |
    |---|---|---|---|---|
    | Words (10) | 5 | 3 | 1 | 1 |
    | Sentences (14) | 8 | 4 | 1 | 1 |

  - Explain the consent form beforehand, or in its own first 10 minutes.
- **Pilot first.** 2 reviewers, one short session (the first rater read's: 8 reviewers and 60 sentences, §10). Fix every unclear question, then run the main study on fresh items.
- **Optional sentence add-on.** Only for 2a, whose scorer needs it. After Q1 is written, show a large-print card with
  the four sentences (`option_a`..`option_d` of that row), and record the letter. Ask in ASL:
  - statement or question (yes/no, wh, rhetorical)?
  - negative?
  - topic set up?

  This adds 4 items. Leave it out of the word block.

## 4. Running a session (checklist)

1. Signed consent form, kept apart from the answer pages.
2. Reviewer code assigned (R01, R02 ...). The name-to-code list is kept on its own, privately.
3. Instructions given in ASL (by you or an interpreter). `FOR_REVIEWERS.md` and, for the first rater read, the two-stage guide printed in large type.
4. Rating pages pre-numbered with reviewer code, session and video number. First rater read: `RATING_SHEET_TWO_STAGE.md`.
5. Practice clips shown, questions answered, then the scored clips with breaks.
6. Note any clip that did not play (tick box) and anything you had to explain.

## 5. Entering answers into `rating_sheet.csv`

Enter one row per rating page. Use a copy of the template, kept in private storage (§6).

| Column | From |
|---|---|
| `reviewer_code`, `session`, `video_number` | Page header |
| `item_id` | Row `video_number` of that group's `sheet.csv` (row order = presentation order) |
| `viewable` | `no` if the tick box was ticked, else `yes` |
| `translation` | Q1, verbatim (English or gloss) |
| `clarity` | Q2, 1-5 |
| `naturalness` | Q3, 1-5. Same scale as the existing guide: 1 not signing ... 5 fluent and natural. |
| `handshape`, `movement`, `face` | Q4: `yes`, `partly`, `no` or `not_sure` |
| `notes` | Q5 |
| `choice`, `sentence_type`, `negated`, `topic` | Sentence add-on only, else empty |
| `entered_by`, `checked_by` | Initials. A second person checks every row against the paper. |

**For `score_human_eval.py` (2a only):**
- Copy each answered row into that group's `sheet.csv`. Leave `item_id`, `video` and the option columns unchanged.
- `parse_sheet` rejects viewable rows that lack `choice`, `sentence_type`, `negated`, `topic` or `naturalness`.
- `reviewer_sha256` is a key you generate per reviewer code and keep with the name-to-code list: `openssl rand -hex 32`
  gives one, 64 hex characters. It is never the name, and the reviewer never sees or does this.

**For the first rater read (two stages):** enter each page from `RATING_SHEET_TWO_STAGE.md` into that rater's copy of the group's `sheet.csv`, in row order, keeping `item_id` and `video` as they are.

| Sheet column | From the stage 1 page |
|---|---|
| `viewable` | `no` if the box was ticked, else `yes`. If `no`, leave the rest of the row empty. |
| `translation` | Q1, verbatim, English or gloss |
| `sentence_type` | Q2: `statement`, `yes_no_question`, `wh_question` or `rhetorical_question` |
| `negated`, `topic` | Q3, Q4: `yes` or `no` |
| `naturalness` | Q5, 1-5 |
| `notes` | Q6 |
| `reviewer_sha256` | The key you made for this reviewer, on every row |

Then run `how2sign-stage2` (§10), print that rater's stage 2 pages from the sheet it writes, and enter each circled letter in `choice`, leaving every other column as it is. Excel's CSV is fine: the tools read UTF-8 and Excel's Windows encoding. A comma in an answer needs the cell quoted, which Excel does for you.

**NOT BUILT:** a converter from `rating_sheet.csv` to `sheet.csv`, and a scorer for clarity, Q4 and the judged
translations.

## 6. Anonymising and storing

- Keep these four apart:
  - consent forms (they carry names);
  - the name-to-code list;
  - the answer pages (codes only);
  - the entered CSVs.
- Keep paper locked away. Keep files in private storage outside the repo: an encrypted folder on your Mac, or a
  private R2 prefix.
- **Never commit to the public repo:** returned sheets, `key.json`, reviewer keys, rosters, receipts, `draw.json`, the checker's files, consent forms,
  recordings, or free text without quote consent. `reviewer_package/.gitignore` blocks the usual names. Commit only aggregate results.
  **`report.json` is not aggregate:** it holds per-rater rows (`word_format`'s rater balance, the bar read's `raters` and `rater_quality`) keyed by the start of each reviewer key, and `problems` names keys and items. Keep it private; commit a cut-down copy without them.
- Recordings (only with the optional consent): delete them after transcription and log the date.
- Deliver videos in person on your device where possible. Otherwise use an expiring private link without download,
  and ask reviewers to delete the files afterwards.
- Honour deletion requests by reviewer code until the write-up date stated on the consent form.

## 7. Scoring and what the answers measure

Declare the analysis before any sheet returns (EVALUATION_FRAMEWORK §4; in `docs/NEXT_PHASE_PLAN.md` with a UTC
time). Report every measure per condition, beside the anchor's result on the same items. Exclude `viewable = no`
rows and count them.

| Measure | How | Reading |
|---|---|---|
| Comprehension, free answer | A blind judge marks each translation against the intended text: full, partial or wrong. For words: the gloss or an English equivalent from a synonym list fixed in advance. Rate of full (and of full + partial), Wilson 95% CI. | Main endpoint. Generated minus anchor, paired by item (sign-flip permutation). |
| Comprehension, forced choice (2a) | `score_human_eval.py --key key/key.json --sheet ... --output report.json`; first rater read: §10 | Chance is 25%. Conveys meaning only if the lower bound is above 25%. |
| Clarity, naturalness | Mean with bootstrap 95% CI (resample reviewers and items), plus the full 1-5 distribution | Ordinal, so the means are descriptive |
| Hand shape, movement, face | Share of yes / partly / no, Wilson CIs | Shows where the failures are |
| Controls | Blank and mismatched should be near the floor | If not, reviewers are guessing or the judge is not blind. Fix before the main study. |
| Agreement | Fleiss' kappa (existing for choice, sentence type and naturalness) | Below 0.4, a question is unclear |
| Instrument check | Per-item correlation of human scores with back-translation BLEU and the recogniser (protocol §7) | Validates the automatic measures |

Misunderstood items and the notes guide the next round. Free text enters training only with the optional consent.

## 8. Accessibility checklist

- [ ] Instructions and the consent form explained in ASL by you or a qualified interpreter. The printed text is backup.
- [ ] All printed pages in large type (16 pt or more), high contrast, one clip per page.
- [ ] Paper option always offered. Answers may also be given in ASL for you to write down (record only with consent).
- [ ] Captions are not needed. The videos have no audio.
- [ ] Large screen, clips replayable as often as wanted, controls operated by you.
- [ ] Good lighting, no glare, seating at a comfortable viewing distance.
- [ ] Breaks every ~15 minutes, and stopping at any time without loss of payment.
- [ ] Practice clips before scoring, and a chance to ask questions.
- [ ] Gloss answers accepted equally with English.

## 9. Still to build or decide

**Build:**
- Avatar rendering of round-15 controls (§2c). State export, the How2Sign sentence packet mode and the pilot's word-format arms are built for the first rater read (§10), on test data only so far.
- A way to print stage 2 pages from a rater's sheet. Today you copy the four sentences by hand, or from the sheet in a mail merge.
- A per-sentence comparison of the human scores with the instruments (protocol §7). The scorer does not do it.
- Isolated-word packet mode.
- Blank and mismatched systems in the packet builder.
- A `--guide` option, so the packet can carry another page. The guides it carries are plain language now, but keep the zip yourself: it holds the sheet.
- A `rating_sheet.csv` to `sheet.csv` converter, and a scorer for clarity, Q4 and judged translations.

**Decide:**
- Payment amount.
- Ethics advisor or IRB review.
- Which systems go in the study.
- Number of reviewers and groups.
- Interpreter.
- How videos are delivered.
- Whether to use the sentence add-on.
- Whether and when final-test (signer 9) material is shown.
- Which restricted sources (WLASL, Sem-Lex, RIT) may appear as anchors.

## 10. First rater read: notes for the owner (2026-11-01)

This read is diagnostic. It can't clear the release bar, because signers and wrist speed already fail it. Its purpose is to learn what signers actually understand. Details: `reports/rater_package_plan.md`; rules: `docs/HUMAN_EVALUATION_PROTOCOL.md` §6.

**What you need to do:**
- Recruit 9 Deaf raters for the main read, 8 more for the pilot (they can't join the main read), and one Deaf checker who is not a rater. That is 18 people. The checker confirms which real recordings sign a negation, and must not see any generated clip: you lead that session (§11). The builder won't make any packet, pilot included, until the checker's sheet is complete.
- Generate every reviewer key, and the checker's, and enter every answer. Raters and the checker never use a terminal, compute a hash or edit a CSV.
- Set pay. At $40/h, which is an assumption, the pilot is 20 rater hours, about $800, and the main read 75 rater hours, about $3,000, so about $3,800 together. If the 2 minutes per item already covers stage 2, these are 16 hours ($640) and 60 hours ($2,400). Adding stitched real clips as a fourth system to the main read costs about $1,000 more (three more raters), and is the one direct test of whether our scorer undervalues stitching; the pilot's `oracle` arm is a much smaller look at the same thing. Consent, instructions, practice clips, breaks, the checker and any interpreter are on top.
- Take everything to the review board and get its decision: ethics or IRB review, `CONSENT_FORM.md`, `CHECKER_CONSENT_FORM.md`, the pilot's partial disclosure, and the questions listed below. **Raters aren't told some clips are deliberately out of order: the board must be told, and it decides.** The default is a debrief after all sheets return, `DEBRIEF.md`, already drafted; fill in its blanks.
- ASLLVD train clips: confirmed by you, 2026-11-02 20:30 UTC. Pass that record as the export's note. RIT and WLASL aren't covered: each would need your own note naming it, and the export refuses a corpus the note doesn't name.
- Arrange ASL instructions or an interpreter.
- Choose how videos reach raters. Your flow: you play them, or deliver the videos alone, and raters answer on a printed form or in ASL. You hold stage 2 back until stage 1 is in.

**Questions for the review board.** Nothing here is drafted as legal text; each is for the board, and for you to raise:
- **"Computer-made signing."** `CONSENT_FORM.md` says raters watch "computer-made signing", yet the `reference` and `oracle` clips are real signers' motion shown on the figure. Is that accurate enough?
- **The checker's form.** "Computer-made signing" doesn't describe the checker's job at all, so `CHECKER_CONSENT_FORM.md` is a separate draft. Does it fit?
- **Cloud storage and retention.** Sheets, keys and packets can sit in a private R2 prefix. The consent form doesn't say where answers are stored or for how long.
- **Public release.** The consent form says results are shared as totals. It doesn't say whether anything goes in a public repository or write-up beyond that.
- **Independent contact.** Every form lists you alone. Should a rater have someone other than the researcher to contact?
- **Per-rater rows.** `report.json` holds per-rater rows (§6), so it can't be committed as "aggregate", and the forms promise totals.
- **Training use.** The consent form makes training use optional, while protocol §7 says misunderstood sentences become the next round's fine-tuning set. Do the sentences count, apart from the rater's own words?
- **Attribution and the partial disclosure.** `FOR_REVIEWERS.md` names How2Sign and ASLLVD and says motion was estimated from recordings, as the licence's "state changes" asks. Does that fit the disclosure the board settles? The ASLLVD line follows the model card's attribution, which still awaits ASLLRP's review.

**What changed for raters:**
- **Two stages.** They watch, then give the translation (English or gloss), negation, grammar and naturalness answers on the form or in ASL, and you enter that sheet. Only then do they get the four answer options. The options gave negation away: guessing from text alone scored 63.6% on negated sentences. Use `docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md`.
- **Workload, main read.** 200 sentences each: stage 1 about 6.7 hours at an assumed 2 minutes per item, or nine 45-minute sessions, then stage 2 about 1.7 hours at an assumed 30 s per row. About 8.3 hours per rater. Time it in the pilot, with the limits below.
- **Pages.** `FOR_REVIEWERS.md` is written for this read and says some videos may be hard or poor on purpose, as protocol §4 relies on; the old one is `FOR_REVIEWERS_ORIGINAL_SESSION.md`. `RATING_SHEET_TWO_STAGE.md` is the printed form, and `RATING_SHEET.md` stays with the original session format. The guide says about 2.5 hours for the pilot, with a break about every 15 minutes: tell main-read raters their own figure.
- **Face.** The guides say what the renderer does: short dark hair, eyebrows and eyelids that move, moving eyes, and a mouth without separate lips. Check this against the first renders.

**How the result is read:**
- Comprehension must clear 70% on both the Wilson bound and a bound allowing for repeat ratings. "Within 10 points of real signing" uses an upper bound, so a point estimate isn't enough.
- A system is refused if under 95% of its designed answers came back, per system and per group, and fails if over 5% of its rows are unviewable or lost.
- A negated sentence needs an English negation and the checker's confirmation. Recovery needs recall at 70% or better, within 10 points of real signing, and a false-alarm upper bound of 20% or less. If real signing misses its own recall bar, the negation read is void, not passed.
- Each person gets a roster entry. Identical answer sheets block a pass, and low-agreement raters are flagged; a pass must hold with and without them.

**What 9 raters can show (simulated):**

| True state | Chance it passes |
|---|---|
| 80% understood | About 82% |
| 75% understood | About 31% |
| 70% understood (should fail) | About 4% |
| Level with real signing | About 99% |
| 5 points behind real signing | About 58% |
| 10 points behind real signing (should fail) | At most about 5% |

About 15 raters give 81% at a true 5-point gap, and about 45 give 77% at a true 75%. The declared table in the plan says about 65% for the 5-point gap at nine raters; the build's own run, with its final weights, gives 58%.

**The commands, in order** (full export command: `docs/NEXT_PHASE_PLAN.md`, "Rater read 1: tool fixes and the official arm's export"):

```
PY=~/.venvs/sign-translator/bin/python
# a draw: --seed and --name are required, the seed is printed, a copy goes to the store under the name, written once
$PY scripts/build_human_eval_packet.py how2sign-draw --sentences-key <D20 key> --sentences-sha256 <D20 hash> \
  --main 200 --pilot 60 --min-seconds 1.5 --min-hand-coverage 0.7 --seed <declared seed> --name <name> --output-dir <private dir>
# a draw made before this tool: the checker's sheet and your list of recordings, drawing nothing
$PY scripts/build_human_eval_packet.py how2sign-checker-sheet --draw <draw.json> --draw-sha256 <hash> --output-dir <private dir>
# a packet: --renderer is required, avatar for raters. The main set takes the pilot's roster
$PY scripts/build_human_eval_packet.py how2sign-packet --draw <draw.json> --draw-sha256 <hash> --negation-sheet <filled> \
  --negation-sheet-sha256 <hash> --set pilot --renderer avatar --system ... --output-dir <private dir>
# stage 2 of one rater; add --partial only if that rater stopped early
$PY scripts/build_human_eval_packet.py how2sign-stage2 --key <key.json> --roster <roster.json> --stage1-sheet <sheet> \
  --receipts <receipts.json> --output-dir <private dir>
# the read: --roster is required on every two-stage key. Under 95% it writes the report and exits 2, unless --allow-low-coverage
$PY scripts/score_human_eval.py --key <key.json> --sheet <stage 1> ... --stage2-sheet <stage 2> ... --stage1-receipts <receipts.json> \
  --roster <roster.json> --reference <label> --output <private dir>/report.json
```

- **The roster** is JSON, `{"<key>": <group>}`, with at least 2 raters in every group. A key written twice is refused.
- **A stage 1 sheet with rows unanswered** is refused and left untouched. Enter the rest, or pass `--partial` if the rater stopped: the work is kept either way. A rater who never reaches stage 2 still counts for the stage 1 contrasts (`report["stage1"]`, and naturalness and translation in `report["word_format"]`).
- **A build under `--renderer avatar`** keeps each render's `acceptance` on its item, sums it by arm (`avatar_acceptance`, also in the report), keeps every `verification.json` under `key/avatar-verification/`, and stops on a mirror failure. Read the summary before any rater sees a clip: a low pass rate isn't refused, only reported.

**Not ready yet:**
- Everything has run on test data, apart from one dry run of the export on six real D20 sentences that wrote nothing. The packet has not run on real D20, and the plan's 427 eligible and 80 negated sentences are unconfirmed until the real draw's counts are read.
- A draw made before this tool has the old sheet column, `no_generated_clip_seen`, and no list of recordings. `how2sign-checker-sheet` writes both from that draw, without drawing again.
- **The avatar as it stands** (`docs/AVATAR_RENDER.md` section 13; measured on 36 D20 sentences, branch `avatar-fixes`). A plain figure with short dark hair, moving eyebrows and eyelids, and a mouth with no separate lips, in a plain dark top, at 768 px in one camera frame for every clip. Every clip gets the same render. Head shake is down about 7× on real clips and 6× on generated. Hands no longer enter the torso or head; forearms still do in under 1% of frames.
- **What's still weak.** No Deaf signer has seen it, so nothing here says it reads as signing. Generated hands still shake about twice as much as real, and generated motion is visibly fast; the bar's predicted timing rules out slowing it. Fingers don't rest between handshapes. Elbows sometimes swing out where a forearm was moved clear of the body. On c01, a real clip's arms can be mostly interpolated: the arm chain is invalid on a median 37% of frames, over half in 60 of 177 clips, which weakens the reference there. Teeth look grey and lined when the mouth opens. The pilot decides whether it's good enough.
- The tooling gets one more independent check before any sheet goes out.
- Sentences with no selected word are now left out of the draw and listed in `draw.json`, the key and the report. D20 has six. Had they stayed, a pilot of 60 would have held one about two times in three.

**The pilot (declared 2026-11-02, 18:22 UTC, amended 2026-11-02, 19:45 UTC): 60 sentences, 8 raters, 2 per group, all four groups, your decision.**
- **What and why.** All three automatic instruments reward English word order, and they can't say whether a person would follow it. So the pilot adds `composition_shuffled` (the same clips in a seeded random order) and `oracle` (real word clips stitched the same way). Report only: they never count towards the release bar, the bar read refuses a key that holds them, and the builder refuses them in the main set.
- **Composition arm.** The official one: W14b, best of 16, gains on, at predicted timing, `--core-margin 0.1`. The export refuses word-format arms on any other form. A dry run on six real D20 sentences built both new arms from it, reading R2 and writing nothing: 80 s, 3.7 GB at peak, so watch the Mac's 4 GB limit on the full run.
- **Rater load.** 60 sentences each. Stage 1 is 2 hours at an assumed 2 minutes per item, about three 45-minute sessions; stage 2 about 30 minutes at an assumed 30 s per row, which nothing has measured. 2.5 hours per rater, 20 rater hours.
- **Rendering.** 240 clips, about 3 hours on the Mac at the measured 26 s per 104-frame clip (predicted clips run longer, 170 frames on average). A full render, frames and bundle, is about 1 MB per frame, so 240 clips are about 40 GB; upload and delete per clip.
- **What you learn.** With every group filled, every two arms are paired on all 60 sentences, and kappa is back. Under the declared rule, if English order truly costs 20 points of accuracy (60% against 40% over all sentences and raters) the read finds it 79% of the time, and a true 10 points 20%. If 60% and 40% is the median sentence with the median rater, which reads as about 16 points, 55% and 12%. So **a pilot that finds no drop has not shown order doesn't matter**. Forced choice can be right on words alone in any order, so read the free translations and naturalness too.
- **Timing.** Half of what each rater sees is scrambled or stitched real signs, which may be quicker to give up on or slower to puzzle over. The pilot's total times test the session plan. They don't predict the main read's time per item, or fatigue over its much longer run. Sheets hold no time per row: if you want time per arm, note it yourself.
- **Options.** No pilot option holds a negation word, since no pilot sentence does: otherwise an option with one could be crossed out, 29.9% rather than 25% by elimination alone.
- **Pilot raters don't join the main read.** They've seen the scrambled forms. Pass the pilot's roster as `--excluded-raters-roster` for the main set, and the tools refuse them. Once this draw's pilot packet exists, the builder refuses a main packet without it. Before that it warns, because a pilot built on another store can't be seen.
- **ASLLVD train clips: confirmed by you, 2026-11-02 20:30 UTC; RIT and WLASL aren't covered.** Val covers only about 31% of composition's words, against about 90% for train, so the oracle uses train. The export refuses any corpus your note doesn't name as covered, and a sentence like "RIT and WLASL aren't covered" names neither. With ASLLVD alone, about 45% of sentences keep every word (a stored composition without added words; 1 of 6 in the dry run). RIT or WLASL would each need a note of its own.
- **Ethics.** Protocol §4 says raters are told the purpose; this is a partial disclosure. The scrambled clips must not be hidden from the review board. The default is the debrief in `DEBRIEF.md`, given after all sheets return. The board may change it, and what it requires goes in.
- **Reference and timing.** The reference plays at its recorded length and the other arms at the predicted one, which can be more than twice as long on short sentences, so speed may mark it.

**Open items you deferred or should know about:**
- **Consent, deferred until the model finishes (your decision, 2026-10-31).** Recogniser v3d, trained on ASL Citizen, chooses the best-of-16 words that raters would see. ASL Citizen's consent covers sign language recognition. An ethics reviewer may ask about this, so have the answer ready.
- **Pointing signs.** Added words such as you, it, your and this are often pointing in ASL. Ask raters or the checker whether these clips read correctly, and whether a held left hand reads as a sign.
- **Scorer limit.** Raters may understand more or less than our scorer suggests. That's the point of the read, so report the result whichever way it goes.

## 11. The checker's session: your steps

The checker's page is `CHECKER_GUIDE.md`, and the form is `CHECKER_CONSENT_FORM.md`. You lead the session and do everything technical.

**What to play.** `draw.json`'s folder holds `negation_check.csv`, the sheet you fill, and `checker_clips.csv`, your private list, one row per sheet row. Keep both off the screen the checker sees.

| `checker_clips.csv` column | What it is |
|---|---|
| `clip_zip`, `clip_file` | Play this: a member of How2Sign's published clip zip in R2. It's the cut clip the motion states were made from. |
| `seconds` | The clip's length in the manifest |
| `session_video` | The full recording the clip was cut from |
| `start_seconds`, `end_seconds` | Where the clip sits in that recording, from `how2sign_train.csv` or `how2sign_val.csv` |
| `realigned_start_seconds`, `realigned_end_seconds` | Where How2Sign's realigned English sits, from `how2sign_realigned_*.csv`. It can differ from the clip, by about 0.9 s at the median and up to 22 s on D20 |
| `found` | `no` if a sentence wasn't found in the stored CSVs |

The manifest doesn't hold a video or a time range. It has the clip's name, its signer group and its length, so these columns come from How2Sign's own CSVs. All 500 of D20's clips are in the train CSVs and the train zip.

**Getting the clips.** The zip is `acquisition/2026-10-17-v1/how2sign-public/train_rgb_front_clips.zip` in R2, and it's large, so check the free disk first. With it in a private folder:

```
cut -d, -f5 checker_clips.csv | tail -n +2 > clip_names.txt
xargs -I{} unzip -j train_rgb_front_clips.zip {} -d checker_clips < clip_names.txt
```

**A row with `found` = `no`.** Find its `SENTENCE_NAME`, which is the sheet's `sign`, in `how2sign_realigned_train.csv` or `how2sign_realigned_val.csv` under `acquisition/2026-10-17-v1/how2sign-public/`; they're tab-separated. The clip is `raw_videos/<sign>.mp4` in `<split>_rgb_front_clips.zip`. Don't open a test partition's files: no D20 sentence comes from one.

**In the session,** row by row: play the clip with nothing else on the screen, take the checker's first answer, then show the English and the flagged word, then take the final yes or no. Give no reaction or hint.

**The sheet.** Enter these per row:
- `confirmed`: the final yes or no.
- `only_real_recordings_seen`: `yes` only if the checker has seen nothing but real recordings. Anything else stops the packet, and the checker can't check.
- `watched_before_reading`: `yes` if the checker watched before reading. If that slipped, `no`: the row is kept, and listed beside any result as `negation_read_first`.
- `checker_sha256`: the key you made with `openssl rand -hex 32`, never a name.
- `notes`: an answer that changed after reading, a clip that seemed cut off.

Leave `sign`, `english` and `negation_words` as they are. Then hash the filled sheet with `shasum -a 256` and pass it to `how2sign-packet`. The checker's key goes into the study's key, and the tools refuse it as a rater.
