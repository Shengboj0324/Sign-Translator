# Human evaluation protocol: English -> ASL output judged by Deaf ASL users (plan item 3b)

This is the study team's protocol. Reviewers receive only `HUMAN_EVALUATION_REVIEWER_GUIDE.md`, their sheet and their
videos. Code:
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

All systems sign on the recorded utterance's clock, so duration is no cue. Every video is drawn identically:
- four skeleton panels (front, side, each hand) and a schematic face panel (`generation/face_preview.py`: head turn,
  brows, eyelids, gaze, mouth);
- any channel's tracking dropouts are held from the nearest valid frame, for display only, so flicker does not
  reveal the recording.

**Known limit:** the skeleton and the schematic face show the grammar channels, but not appearance. Results describe
these renders. They do not describe an avatar (item 5), and the study is re-run when the avatar exists.

## 3. Design

- **Utterances:** distinct English sentences from the plan manifest.
  - Pilot: the `val` partition, about 20 utterances.
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
3. Run a pilot (two reviewers, one group) first. Fix anything unclear in the guide, then run the full study with
   fresh items.
4. Score the returned sheets:

   ```
   python scripts/score_human_eval.py --key key/key.json --sheet <returned>.csv … --output report.json
   ```

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

- Every endpoint is reported per system.
- Rows a reviewer marks `viewable = no` are counted and excluded.
- A generator result is credible only beside the reference's result on the same utterances.

## 7. Use of the results

- The human scores are the target that the automatic instruments are validated against. Report each instrument's
  correlation with them per utterance.
- Utterances that reviewers misunderstand, or whose markers they miss, become the fine-tuning set for the next round.
- The free translations and notes go to the team, not into training data, unless the reviewers consent to that
  separately.
