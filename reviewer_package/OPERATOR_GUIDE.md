# Operator guide (owner only, technical)

This guide covers the reviewer sessions: producing clips, running sessions, entering answers and scoring. The study
design and the analysis follow `docs/HUMAN_EVALUATION_PROTOCOL.md`, which stays the reference. Items marked
**NOT BUILT** do not exist in the repo yet; they are collected in §9.

Reviewers never receive this guide, `key/`, any CSV or any script.

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
  | ASL Citizen | **No.** Its consent covers recognition only. |
  | ASL STEM Wiki | **No.** |
  | YouTube-ASL | **No.** It is not redistributable. |
  | WLASL (C-UDA, "computational use") | Not unless you confirm its terms allow it |
  | Sem-Lex | Not unless you confirm its terms allow it |
  | RIT | Not unless you confirm its terms allow it |

  The round-15 oracle draws real words from ASLLVD, RIT, WLASL and Sem-Lex (`compositional.ROUND15_CORPORA`).
  Filter anchors by each record's `corpus` before rendering. The pilot's `oracle` arm (§10) does this itself, and needs your recorded
  note, `--word-format-licence-note`, for any corpus, ASLLVD train clips included.
- **Avatar only.** Every condition, including the real-signer anchors, is shown through the same avatar renderer.
  Original source video would reveal which clips are real. Optionally, show one original How2Sign val video as an
  unscored warm-up.

## 1. Environment

- Python: `~/.venvs/sign-translator/bin/python`, run from the repo root. R2 reads need the `R2_*` environment variables.
- Blender 5.2 at `/Applications/Blender.app` (the `render_avatar.py` default; `--blender` to override). An avatar render
  takes about 13 s for a 147-frame clip at 768 px on the Mac, measured.
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
- **Do not forward `group-<g>.zip` as it is.** It contains the technical `HUMAN_EVALUATION_REVIEWER_GUIDE.md` and a
  CSV. Play the videos yourself and give the reviewer printed `RATING_SHEET.md` pages numbered 1..N in `sheet.csv`
  row order.

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
3. Instructions given in ASL (by you or an interpreter). `FOR_REVIEWERS.md` printed in large type.
4. Rating pages pre-numbered with reviewer code, session and video number.
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
- `reviewer_sha256` is the SHA-256 of a random passphrase that you generate and keep per reviewer code:
  `printf '%s' "$PASS" | shasum -a 256`. It is never the name, and the reviewer does not need to do this.

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
- **Never commit to the public repo:** returned sheets, `key.json`, passphrases, consent forms, recordings, or free
  text without quote consent. `reviewer_package/.gitignore` blocks the usual names. Commit only aggregate results.
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
| Comprehension, forced choice (2a) | `score_human_eval.py --key key/key.json --sheet ... --output report.json` | Chance is 25%. Conveys meaning only if the lower bound is above 25%. |
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
- How the sheets reach raters and how stage 2 is held back (§10). Nothing exists for it, and the two-stage guide assumes a rater edits a CSV and computes a SHA-256.
- A per-sentence comparison of the human scores with the instruments (protocol §7). The scorer does not do it.
- Isolated-word packet mode.
- Blank and mismatched systems in the packet builder.
- A `--guide` option, so the packet carries `FOR_REVIEWERS.md` instead of the technical guide.
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
- Whether ASLLVD train clips may be shown under the DAI terms (§10).

## 10. First rater read: notes for the owner (2026-11-01)

This read is diagnostic. It can't clear the release bar, because timing, signers and wrist speed already fail it. Its purpose is to learn what signers actually understand. Details: `reports/rater_package_plan.md`; rules: `docs/HUMAN_EVALUATION_PROTOCOL.md` §6.

**What you need to do:**
- Recruit 9 Deaf raters for the main read, 8 more for the pilot (they can't join the main read), and one Deaf checker who is not a rater. That is 18 people. The checker confirms which real recordings sign a negation, and must not see any generated clip. The builder won't make any packet, pilot included, until the checker's sheet is complete.
- Set pay. At $40/h, which is an assumption, the pilot is 20 rater hours, about $800, and the main read 75 rater hours, about $3,000, so about $3,800 together. If the 2 minutes per item already covers stage 2, these are 16 hours ($640) and 60 hours ($2,400). Adding stitched real clips as a fourth system to the main read costs about $1,000 more (three more raters), and is the one direct test of whether our scorer undervalues stitching; the pilot's `oracle` arm is a much smaller look at the same thing. Consent, instructions, practice clips, breaks, the checker and any interpreter are on top.
- Take everything to the review board and get its decision: ethics or IRB review, `CONSENT_FORM.md`, and the pilot's partial disclosure. **Raters aren't told some clips are deliberately out of order: the board must be told, and it decides.** The default is a debrief after all sheets return, `DEBRIEF.md`, already drafted; fill in its blanks.
- ASLLVD train clips: confirmed by you, 2026-11-02 20:30 UTC. Pass that record as the export's note. RIT and WLASL would each need your note too.
- Arrange ASL instructions or an interpreter.
- Choose how sheets and videos reach raters. Nothing is built for this. It must hold stage 2 back until stage 1 is returned.

**What changed for raters:**
- **Two stages.** They watch, then give the translation, negation, grammar and naturalness answers and return that sheet. Only then do they get the four answer options. The options gave negation away: guessing from text alone scored 63.6% on negated sentences. Use `docs/HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md`.
- **Workload, main read.** 200 sentences each: stage 1 about 6.7 hours at an assumed 2 minutes per item, or nine 45-minute sessions, then stage 2 about 1.7 hours at an assumed 30 s per row. About 8.3 hours per rater. Time it in the pilot, with the limits below.
- `FOR_REVIEWERS.md` and `RATING_SHEET.md` describe the original session format. For this read, the two-stage guide and generated sheets replace the rating sheet.

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

**Not ready yet:**
- Everything has run on test data, apart from one dry run of the export on six real D20 sentences that wrote nothing. The draw and the packet have not run on real D20, and the plan's 427 eligible and 80 negated sentences are unconfirmed.
- The avatar renderer has not been rerun since the timing change. The avatar is still bald and flat, hands can pass through the body, and shake isn't fixed. The pilot decides whether it's good enough.
- The tooling gets one more independent check before any sheet goes out.
- Sentences with no selected word are now left out of the draw and listed in `draw.json`. D20 has six. Had they stayed, a pilot of 60 would have held one about two times in three.

**The pilot (declared 2026-11-02, 18:22 UTC, amended 2026-11-02, 19:45 UTC): 60 sentences, 8 raters, 2 per group, all four groups, your decision.**
- **What and why.** All three automatic instruments reward English word order, and they can't say whether a person would follow it. So the pilot adds `composition_shuffled` (the same clips in a seeded random order) and `oracle` (real word clips stitched the same way). Report only: they never count towards the release bar, the bar read refuses a key that holds them, and the builder refuses them in the main set.
- **Composition arm.** The official one: W14b, best of 16, gains on, at predicted timing. A dry run on six real D20 sentences built both new arms from it, reading R2 and writing nothing: 80 s, 3.7 GB at peak, so watch the Mac's 4 GB limit on the full run.
- **Rater load.** 60 sentences each. Stage 1 is 2 hours at an assumed 2 minutes per item, about three 45-minute sessions; stage 2 about 30 minutes at an assumed 30 s per row, which nothing has measured. 2.5 hours per rater, 20 rater hours.
- **Rendering.** 240 clips, about an hour on the Mac at the measured 13 s per 147-frame clip (predicted clips run longer, 170 frames on average). The rater plan's 65 GB for 600 clips implies about 26 GB of full render files, so upload and delete per clip.
- **What you learn.** With every group filled, every two arms are paired on all 60 sentences, and kappa is back. Under the declared rule, if English order truly costs 20 points of accuracy (60% against 40% over all sentences and raters) the read finds it 79% of the time, and a true 10 points 20%. If 60% and 40% is the median sentence with the median rater, which reads as about 16 points, 55% and 12%. So **a pilot that finds no drop has not shown order doesn't matter**. Forced choice can be right on words alone in any order, so read the free translations and naturalness too.
- **Timing.** Half of what each rater sees is scrambled or stitched real signs, which may be quicker to give up on or slower to puzzle over. The pilot's total times test the session plan. They don't predict the main read's time per item, or fatigue over its much longer run. Sheets hold no time per row: if you want time per arm, note it yourself.
- **Pilot raters don't join the main read.** They've seen the scrambled forms. Pass the pilot's roster as `--excluded-raters-roster` for the main set, and the tools refuse them.
- **ASLLVD train clips: confirmed by you, 2026-11-02 20:30 UTC; RIT and WLASL aren't covered.** Earlier note: §0's table says "ASLLVD val". Val covers only about 31% of composition's words, against about 90% for train, so the oracle uses train. Please decide whether train clips may be shown under the DAI terms, and record it. The export refuses to run without your note, for any corpus. With ASLLVD alone, about 45% of sentences keep every word (a stored composition without added words; 1 of 6 in the dry run). RIT and WLASL can be named under the same note.
- **Ethics.** Protocol §4 says raters are told the purpose; this is a partial disclosure. The scrambled clips must not be hidden from the review board. The default is the debrief in `DEBRIEF.md`, given after all sheets return. The board may change it, and what it requires goes in.
- **Reference and timing.** The reference plays at its recorded length and the other arms at the predicted one, which can be more than twice as long on short sentences, so speed may mark it.

**Open items you deferred or should know about:**
- **Consent, deferred until the model finishes (your decision, 2026-10-31).** Recogniser v3d, trained on ASL Citizen, chooses the best-of-16 words that raters would see. ASL Citizen's consent covers sign language recognition. An ethics reviewer may ask about this, so have the answer ready.
- **Pointing signs.** Added words such as you, it, your and this are often pointing in ASL. Ask raters or the checker whether these clips read correctly, and whether a held left hand reads as a sign.
- **Scorer limit.** Raters may understand more or less than our scorer suggests. That's the point of the read, so report the result whichever way it goes.
