# Human evaluation protocol: English -> ASL output judged by Deaf ASL users (plan item 3b)

This is the study team's protocol. Reviewers receive only `HUMAN_EVALUATION_REVIEWER_GUIDE.md` (the first How2Sign read:
`HUMAN_EVALUATION_REVIEWER_GUIDE_TWO_STAGE.md`), their sheet and their videos. Code:
- `signtranslator/eval_framework/human_study.py`: design, sheets, validation and scores;
- `scripts/build_human_eval_packet.py`: builds the packets;
- `scripts/score_human_eval.py`: scores the returned sheets.

## 1. What the study measures

Automatic instruments (the back-translator, the retrieval evaluator, the sign recogniser) are trained on the same kind
of data as the generator and can share its blind spots. This study is the ground truth they are checked against.
- **Comprehension:** does a fluent signer understand the intended sentence? This is a four-way forced choice plus a
  free translation.
- **Grammar:** does the signing carry the sentence type (statement, yes/no, wh or rhetorical question), negation and
  a topic? The reviewer judges these blind, and the plan's ASLLRP markers are the truth.
- **Naturalness:** is it fluent signing (1-5)?

Preference alone is not measured, because a system can look better and still be understood less
(docs/EVALUATION_FRAMEWORK.md §6).

## 2. Systems

Every utterance is signed by every system (`--system`):

| Kind | What it is | Why |
|---|---|---|
| `reference` | The recorded utterance (video estimate), drawn the same way | Human upper reference through the same renderer. A generator must not beat it without independent replication. |
| `plan` | Plan-conditioned generator on the gold ASLLRP plan | Plan -> motion on its own |
| `translate` | Gloss planner -> plan -> the same generator | The full English -> ASL chain |
| `english` | Generator conditioned on English directly | The pre-plan baseline |
| `composition` | A composed state exported by `round15_compose_eval.py --export-prefix` (first How2Sign read) | The composition arm, with its cache and seeds recorded |
| `composition_shuffled` | The same composition's clips in a seeded random order, every word kept, at its frame count (`--export-word-format`). First How2Sign read's pilot only, report only | Does English word order help a Deaf rater follow it, and by as much as it helps the instruments? |
| `oracle` | The same word list as real word clips, by chance pick, stitched the same way through the same tokenizer round trip, at its frame count (`--export-word-format`). Pilot only, report only | What generated clips cost a person, against what the word-by-word format costs |

All systems sign on the recorded utterance's clock, so duration is no cue. Every video is drawn identically:
- four skeleton panels (front, side, each hand) and a schematic face panel (`generation/face_preview.py`: head turn,
  brows, eyelids, gaze, mouth);
- any channel's tracking dropouts are held from the nearest valid frame, for display only, so flicker does not
  reveal the recording.

`--renderer avatar` draws every system on the MakeHuman avatar instead (`scripts/render_avatar.py`; Blender Workbench,
front camera, about 13 s for a 147-frame clip at 768 px on the Mac, measured), the preferred renderer for the study.

**Known limit:** the skeleton and the schematic face show the grammar channels, but not appearance. Results describe
these renders. They do not describe an avatar (item 5), and the study is re-run when the avatar exists.

## 3. Design

- **Utterances:** distinct English sentences from the plan manifest.
  - Pilot: the `val` partition, about 20 utterances (the first How2Sign read's: 60, below).
  - Study: `val`, 60 or more.
  - Final study: `test`, once. It needs `--final-study`, and the result is reported whatever it is.
- **Latin square:** with S systems there are S reviewer groups. Group g sees utterance u from system (u + g) mod S.
  - No reviewer sees the same sentence twice, because a second rendering would be primed by the first.
  - Every (utterance, system) cell is rated, so systems are compared paired by utterance.
- **Reviewers:** at least two per group, working independently. This gives Fleiss' kappa per question. A reviewer
  who answers sheets of two groups is excluded from scoring.
- **Options:** four English sentences per utterance. The distractors are other sentences of the same partition with
  the closest word count, drawn from the whole partition so elimination does not help. Every system of an utterance
  shows the same options.
  - **First How2Sign read:** distractors come only from D20's other sentences, not the whole partition: the
    whole-partition pool would read C20 and fresh text. The builder refuses any wider pool.
- **Two-stage sheet (first How2Sign read; declared 2026-11-01, 03:28 UTC):** the correct option holds a negation word
  31.5% of the time against 16.5% for distractors, and a guess from text alone scores 63.6% on negated rows, so the
  options would give negation away.
  - Stage 1: watch, then give the free translation, negation, sentence type, topic and naturalness, with no options
    shown. It is returned and hashed; the team records the hash when it releases stage 2 (`how2sign-stage2`). Stage 1 is
    then final.
  - Stage 2: the forced choice, on a sheet carrying the rater's key and the stage 1 hash. The scorer refuses a stage 2
    answer without a matching stage 1 hash.
  - The sentence type and topic have no truth on How2Sign (no ASLLRP markers): asked, reported, not scored. Negation
    truth is the Deaf checker's (section 6).
- **Word-format arms (first How2Sign read's pilot only; report only; declared 2026-11-02, 18:22 UTC, amended 2026-11-02, 19:45 UTC, in `docs/NEXT_PHASE_PLAN.md`):**
  - **They count towards nothing.** The bar read refuses a key that holds them, and the builder refuses them in the main set.
  - **The pilot: 60 sentences, 8 raters, 2 per group, all four groups** (owner, 2026-11-02). Four systems: `composition`,
    `composition_shuffled`, `oracle`, `reference`. **The composition arm is the official one: W14b, best of 16, gains on, at predicted
    timing** (`cache-w14b-r10`, `core_crossfade@predicted`). A dry run on six real D20 sentences (reads only, no write, no render) built both arms from it.
  - **They fit the Latin square as it stands.** A rater sees each sentence once and 60 in all, 15 per system. Every pair of arms is paired
    on all 60 sentences, each arm with 2 ratings, and raters of one group rate the same videos, so Fleiss' kappa is available. 240 clips to render.
  - **Load, at the stated assumptions:** stage 1, 60 × 2 minutes = 2 hours per rater. The 2 minutes covers stage 1 only; stage 2 at an assumed 30 s per
    row adds 30 minutes. So 2.5 hours per rater and **20 rater hours; 16 if 2 minutes turns out to cover both.** Consent, instructions, practice clips and breaks are on top.
  - **What 8 raters can show,** simulated by the committed `scripts/rater_pilot_power.py` (4,000 studies, seed fixed) under the declared rule of section 6. "A true 20-point drop" means
    composition understood 60% and shuffled 40% over all sentences and raters: the rule reads it 79% of the time, and a true 10-point drop 20%. If it means
    60% and 40% for a median sentence and rater (about 16 points in the accuracy read), 55% and 12%. Half-width about ±11 points. The sign-flip test alone says 90% and 33%, and it ignores raters.
  - **What the pilot's timing can and can't predict.** Half the clips a rater sees are scrambled or stitched from real signs, so the pilot's minutes don't transfer to a main read
    with other arms: raters may give up on them sooner or puzzle over them longer. The sheet records no time per row, so a session's total is all it gives. It can bound the session plan and show whether 45-minute sessions hold. It can't give a per-arm
    time, unless the operator logs one, or fatigue over the main read's longer run, and the pilot holds no negation candidate.
  - **Forced choice alone cannot settle word order:** a rater who catches two content words can pick the right sentence in any
    order. The stage 1 free translation and naturalness are where it can show.
  - **Oracle against composition mixes real clips with fewer words,** since a word with no permitted real clip is dropped and the rest share the frames under sign-fitted timing, each kept
    word stretched (`core_stretch`, per sentence; 1.03 to 1.53 on the six dry-run sentences). The clean row is `same_words`. **Reference against oracle is not the format alone:** it also holds the
    tokenizer round trip, chance picks, a different signer at every word, the cut and the dropped words.
  - **A moved join keeps the gap planned for other neighbours,** so it covers a different distance in the same frames. Under sign-fitted timing no velocity bound applies to any join, composition's included;
    each row logs how far its joins go past the continuous route's bound, beside the composition's.
  - **Predicted timing and the reference:** the reference plays at its recorded length and the other arms at the predicted one (69 frames against 36, 87 against 30 on two dry-run sentences), so speed may mark it. Each rater sees a sentence once, so no direct comparison of lengths is possible.
  - **Pilot raters do not rate the main read:** they have seen scrambled forms. The main key's `excluded_raters` (`how2sign-packet --excluded-raters-roster`) holds them, checked as the negation checkers are, at stage 2 and at the bar read.
- **Sentences with no selected word** (decided 2026-11-02): the draw leaves them out, under the official arm's selection, and `draw.json` lists them (`excluded_no_selected_word`). D20 has six. Comprehension's weights read the eligible sentences that remain, and this is disclosed beside any result.
- **Blinding:**
  - Item ids and video names are random, and the presentation order is shuffled per group.
  - System labels, correct options and markers exist only in `key/key.json`, which is hashed and never sent.
  - The design seed is drawn at random and kept only in the key.

## 4. Reviewers, consent, payment

Reviewers are Deaf signers fluent in ASL, or hearing signers with an equivalent recognised qualification (for
example NIC/RID certification or ASL teaching credentials).
- Record each reviewer's qualification privately; it does not go in the sheet.
- Reviewers are told the purpose and how their answers are used, and that they may stop at any time.
- They are paid for all work, including the pilot.
- They identify themselves on the sheet by a SHA-256 key of a private passphrase, never by name (as in
  `GLOSS_REVIEW_PROTOCOL.md` §5).
- ASLLRP material is research-only: packets go only to contracted reviewers, under the DAI terms, and are not
  redistributed.
- **The pilot's word-format arms are a partial disclosure:** raters are told the purpose in general, and that some videos may be bad on purpose, but not that some clips are deliberately out of order, or
  that word order is being tested. The review board is told all of it, and it decides. The default is a debrief after all sheets return: `reviewer_package/DEBRIEF.md`, a separate document given to every pilot rater, with the right to withdraw their answers. Nothing about this is hidden from the board.

## 5. Running it

1. Build the packet. The command and its flags:

   ```
   python scripts/build_human_eval_packet.py
   ```

   - `--manifest-key …/plan-manifest-v2/manifest.json --manifest-sha256 b0187725…`
   - `--system reference=reference`
   - `--system plan=plan:<generator best checkpoint>@<plan manifest SHA>`
   - `--system translate=translate:<same>@<same> --planner-run gloss-planner-asllrp-v2`
   - `--system english=english:<v2b checkpoint>@<its manifest SHA>`
   - `--count 60 --output-dir <dir> --output-prefix <private R2 prefix>`
2. Send each group its `group-<g>.zip`, which holds the sheet, the videos and the reviewer guide. Keep `key/` private.
3. Run a pilot (two reviewers, one group; the first How2Sign read's is larger, below) first. Fix anything unclear in the guide, then
   run the full study with fresh items.
4. Score the returned sheets:

   ```
   python scripts/score_human_eval.py --key key/key.json --sheet <returned>.csv … --output report.json
   ```

**First How2Sign read:**

1. `build_human_eval_packet.py how2sign-draw` on D20 gives `draw.json` and the Deaf checker's `negation_check.csv`. The
   checker sees real recordings only and cannot be a rater.
2. `round15_compose_eval.py … --export-prefix … --export-only`, once per arm, on the draw's sentence list.
3. `build_human_eval_packet.py how2sign-packet` builds stage 1 only. A sentence with no selected word is refused and listed.
4. As each stage 1 returns, `how2sign-stage2` hashes it, records the receipt and writes that rater's stage 2.
5. `score_human_eval.py --key … --sheet <stage 1> … --stage2-sheet <stage 2> … --stage1-receipts … --roster … --bar`.

**Its pilot, with the word-format arms** (not yet run for real; the draw and the checker's sheet come first):

1. `how2sign-draw` (`--pilot 60` is the default). Add to the official arm's `round15_compose_eval.py … --export-only --export-forms core_crossfade@predicted` command:
   `--export-word-format --word-format-seed 0 --word-format-licence-note "<the owner's recorded confirmation>"`. It writes `<prefix>/word-format/composition_shuffled/index.json` and
   `<prefix>/word-format/oracle/index.json` for the pilot sentences alone. Real clips are ASLLVD train records; RIT and WLASL by `--word-format-corpora`, under the same note.
2. `build_human_eval_packet.py how2sign-packet --set pilot`, with four `--system`s, each new arm as `LABEL=composition:<its index>@<its hash>`. A sentence the oracle has no word for leaves the pilot, listed in the key.
3. Roster 8 raters, 2 per group. Stage 2 and scoring as above, **without `--bar`**: the scorer adds `report["word_format"]`, and `--bar` is refused.
4. For the main set, pass the pilot's roster as `--excluded-raters-roster`. After all sheets return, give each pilot rater `reviewer_package/DEBRIEF.md`, as the review board decides.

## 6. Analysis (fixed before any sheet returns)

| Endpoint | Statistic | Reading |
|---|---|---|
| Comprehension | Forced-choice accuracy, Wilson 95% interval | Chance is 25%. A system conveys meaning only if its lower bound is above 25%. |
| vs reference | Per-utterance accuracy difference, sign-flip permutation p | Paired by utterance |
| Sentence type | Accuracy, overall and per true type | Statements dominate, so read the per-type rows |
| Negation, topic | Recall on marked sentences, false-alarm rate on unmarked ones | Both reported; recall alone is gameable |
| Naturalness | Mean, bootstrap 95% interval | Ordinal; the means are descriptive |
| Agreement | Fleiss' kappa (choice, sentence type, naturalness) | Below 0.4 the question or the guide is unclear; fix it before the full study |
| Free translation | Indicative BLEU vs the English | The team scores propositions by hand (EVALUATION_FRAMEWORK §6) |
| Word-format contrasts (pilot, report only) | Per sentence paired difference, sign-flip permutation p, bootstrap interval over sentences alone (`ci95_sentences_only`, which ignores raters and decides nothing): shuffled minus composition, also on the sentences whose word sequence moved; oracle minus composition, also where the oracle kept every word; reference minus oracle; reference minus composition. On choice, naturalness, the translation's content words (order-free; a blank translation is 0) and sentence BLEU. On choice also method D with t(R−1) (no interval on zero variance) and rater balance. A pair with no shared sentence gets a within-rater row, which decides nothing | Above or below only with at least 6 raters, if method D's two-sided interval excludes zero and p is under .05 in the same direction, with a rater seeing both arms; otherwise unresolved, never "small". (composition − shuffled) / (composition − chance) on moved sentences stands beside the instruments' 39–51%, and is printed only with a defined interval; it reads against them only by method D's interval, with at least 8 raters and composition under 95%. Neither a large nor a small share is the better outcome. Never a bar condition |

- Every endpoint is reported per system.
- Rows a reviewer marks `viewable = no` are counted and excluded.
- A generator result is credible only beside the reference's result on the same utterances.

**Release bar (owner's rule, 2026-10-31, 22:00 UTC; fixed before any sheet returns).** A lower bound above chance shows
that meaning gets through; it does not allow release. A system may be released only if all of these hold:
- comprehension Wilson 95% lower bound ≥ 70%;
- comprehension within 10 points of the reference on the same utterances;
- negation recovered, with its false-alarm rate reported;
- wrist speed inside the span of real signers on the same motion-recovery path (changed 2026-11-01 from 0.8–1.25× of D20 real, disclosed as a loosening; `docs/NEXT_PHASE_PLAN.md`, "wrist condition moved");
- all of it read at predicted timing, on unseen signers.

How these are read (declared 2026-10-31, 22:39 UTC):
- Intervals are paired by utterance, by two-way cluster bootstrap over utterances and reviewers.
- Comprehension must also have a clustered lower bound ≥ 70%.
- "Within 10 points" means the upper 95% bound of reference minus system is ≤ 10 points.
- A sentence is negated only if its English has a negation and a Deaf checker, not shown any generated clip, confirms the real recording signs it.
- "Negation recovered" means recall Wilson lower bound ≥ 70%, upper 95% bound of reference minus system recall ≤ 10 points, and false-alarm Wilson upper bound ≤ 20%. If the reference's recall lower bound is below 70%, the read is void.
- A system with over 5% of rows marked `viewable = no` does not pass.

The interval method changed on 2026-11-01, 03:28 UTC, before any sheet exists and from simulation only
(`docs/NEXT_PHASE_PLAN.md`, "interval method and sheet order changed"). This loosens the two lines on intervals above;
the plan discloses that. The bar's values are unchanged. As implemented (`human_study.bar_read`):
- **Variance:** V = Σ_u(Σψ)²·U/(U−1) + max(Σ_r(Σψ)²·R/(R−1) − Σψ², 0), the two-way cluster variance of the ratings' influence
  values, by sentence u and rater r. The bootstrap is reported beside it and gates nothing.
- **Comprehension:** Wilson lower bound ≥ 70% and a one-sided 4% logit bound with t(R−1) ≥ 70%. Read at D20's eligible
  shares of three strata: confirmed, rejected candidates, non-candidates. The unweighted value sits beside it. Wilson on a
  weighted rate uses Kish's effective sample size.
- **Within 10 points:** one-sided 5% upper bound of reference minus system with t(R−1) ≤ 10 points, on the sentences both
  have.
- **Negation:** recall by Wilson and by the same bound, both ≥ 70%; the same upper bound on the recall gap ≤ 10 points;
  false-alarm Wilson upper bound ≤ 20%, over every sentence not confirmed. Void if the reference misses either recall
  bound.
- **Coverage:** a read needs designed-row coverage ≥ 95% per system and per group, over the private roster of keys and
  groups; otherwise it is refused and the scorer exits nonzero. Blanks inside a rater's answered range and dropped rows
  count with `viewable = no` against the 5% rule; the reference has the same gate on itself.
- **Raters:** nobody off the roster counts, the Deaf checker cannot be a rater, and identical answer vectors block a pass.
  Raters far below the rest in agreement with the others, computed blind to system, are flagged; every condition must
  hold with and without them.
- **A generator that beats the reference** is flagged, as section 2 requires independent replication; the flag alone does
  not fail it.

The instrument, realism, serving and licensing requirements still apply (`docs/NEXT_PHASE_PLAN.md`).

## 7. Use of the results

- The human scores are the target that the automatic instruments are validated against. Report each instrument's
  correlation with them per utterance.
- Utterances that reviewers misunderstand, or whose markers they miss, become the fine-tuning set for the next round.
- The free translations and notes go to the team, not into training data, unless the reviewers consent to that
  separately.
