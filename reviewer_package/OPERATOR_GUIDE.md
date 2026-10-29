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
  | YouTube-ASL | **No.** It is not redistributable. |
  | WLASL (C-UDA, "computational use") | Not unless you confirm its terms allow it |
  | Sem-Lex | Not unless you confirm its terms allow it |
  | RIT | Not unless you confirm its terms allow it |

  The round-15 oracle draws real words from ASLLVD, RIT, WLASL and Sem-Lex (`compositional.ROUND15_CORPORA`).
  Filter anchors by each record's `corpus` before rendering.
- **Avatar only.** Every condition, including the real-signer anchors, is shown through the same avatar renderer.
  Original source video would reveal which clips are real. Optionally, show one original How2Sign val video as an
  unscored warm-up.

## 1. Environment

- Python: `~/.venvs/sign-translator/bin/python`, run from the repo root. R2 reads need the `R2_*` environment variables.
- Blender 5.2 at `/Applications/Blender.app` (the `render_avatar.py` default; `--blender` to override). An avatar render
  takes about 45 s per clip on the Mac.
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

**2c. NOT BUILT: the clips the current milestone needs.**
- Round-15 compositions (core_crossfade, the refiner) and their `/rest_only` and `/mismatched_words` controls.
  `round15_compose_eval.py` and `round15_stage2.render_form` build these states only in memory and never save or
  render them. A state exporter is needed (`--save-states-dir`, or a `compose:` system kind in the packet builder).
- Isolated-word items. The packet builder designs sentence items only.
- How2Sign sentence items. The builder needs an ASLLRP plan manifest (plan plus markers), and How2Sign val has no
  plans, so it needs a mode without grammatical truth.
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
- **Pilot first.** 2 reviewers, one short session. Fix every unclear question, then run the main study on fresh items.
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
- State export and avatar rendering of round-15 compositions and their controls (§2c).
- Isolated-word and How2Sign-sentence packet modes.
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
