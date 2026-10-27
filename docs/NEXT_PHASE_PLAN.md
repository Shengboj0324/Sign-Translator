# Next phase: from signing-like motion to deployable ASL (plan log, 2026-10-21)

**Status: planning only.** Nothing here starts until the v2b priors on pod 4 finish and
pass the quality review against v1 and v2 (`scripts/compare_motion_runs.py`). This log
records the user's questions of 2026-10-21, the facts checked, and the plan for each item.

## Where the model stands (inputs to this plan)

| Tokenizer | Joints | Arm/hand points | Fingertips | Wrists |
|---|---|---|---|---|
| v1 (one VQ codebook) | 19.1° | 10.9 cm | 11.7 cm | 36–41° |
| v2 (RVQ + skeleton loss) | 9.6° | 2.2 cm | 2.5 cm | 22° |
| v2b (full frame rate, base blocks) | 8.3° | 1.7 cm | 1.95 cm | 19° |
| data noise floor | 2.0° | 0.9 cm | 1.0 cm | — |

- **Text → motion priors:** v1 was under-articulated (hand speed 0.2–0.5× real). The v2b
  priors are training now.
- **Data:** a 12-hour How2Sign subset, 3,968 training clips.
- **Grammar:** non-manual markers replicate on real data (wh, yes/no, conditional,
  negation). Role shift, pointing → deixis and gaze-follows-point are not validated
  (docs/GRAMMAR_MOTION.md).

## Q1. The sign-to-text back-translation model: train it, or adopt one?

**Recommendation:** train our own, on our own motion representation, with a pretrained
language decoder. Use a published How2Sign translator only as a second opinion.

Reasons:
- **The input must match what the generator emits.** Published How2Sign translators read
  RGB video or 2D keypoints (MediaPipe/OpenPose layouts). Our generator emits canonical 3D
  rotations, face blendshapes and gaze. A pose → text model trained on the same canonical
  states scores generated motion directly, with no rendering step whose own errors would
  be counted against the generator.
- **The cost is moderate.** It is an encoder over the motion channels (it can reuse the
  v2b tokenizer's encoder or codes) plus a pretrained text decoder (T5-class), trained on
  How2Sign pairs.
- **Published How2Sign BLEU is low** (single digits to mid-teens). The metric is therefore
  relative: generated vs real motion through the same translator, never an absolute
  quality claim.
- **Guard against gaming.** The translator is trained on the training partition only and
  frozen before it scores anything. It is never used as a training loss for the
  generator unless a second, independent translator does the scoring.
- **Second opinion.** Project generated motion to 2D keypoints (forward kinematics plus
  camera) and run an openly licensed published translator. Agreement between the two
  raises confidence. Licence and weight availability must be checked per model before
  use.

Is it necessary? For any claim that the output means the English, yes. It is the
standard sign-language-production metric. It is not sufficient: Deaf ASL users'
evaluation (comprehension and naturalness) gates any release.

## Q2–Q4. The BU ASLLRP links: what each is and whether it helps

Facts checked on 2026-10-21. Pages: SignStream 3 download, the ASLLRP index (Sign Bank,
DAI 2) and the DAI terms. Details are in
`research_notes/ASL grammar and spatial reference/datasets_and_licences.md`.

| Link | What it is | Helps? |
|---|---|---|
| SignStream 3 download (`SignStream/3/dnldSS3.html`) | The annotation **software** (SignStream 3.5.1, May 2025), its README and manuals, and an optional "Global Sign Bank" for the app. **No corpus data.** | Only as a tool: its XML documentation (2025) defines the format of the corpus exports, so it is needed for the parser |
| ASLLRP Sign Bank (`#signbank`) | An isolated-sign lexicon: ASLLVD (about 10,000 examples of about 3,000 signs, 1–6 signers each) plus examples segmented from the continuous corpus, with morpho-phonological features (handshapes, etc.). Downloads (citation forms; segmented signs, CSV per Report 24) sit **behind a DAI 2 login**. | **Yes:** a gloss lexicon with video for every sign. It gives lexical supervision for an English → gloss → motion planner, vocabulary for the back-translator, and handshape labels to check hand accuracy |
| DAI 2 (`#dai2`) | The continuous ASLLRP SignStream 3 corpus: 2,048 utterances and 16,659 sign tokens from native signers (2022 report; the site says much more has been added since). Front, side and face close-up views. Frame-level start/end annotation of non-manual markers (brows, gaze, head, eye aperture, mouth) and grammar (wh, yes/no, negation, topic, conditional, relative clause, **role shift**), **locus indices on agreement verbs and pronouns** (`i:GIVE:j`), and English translations. | **Yes, the most valuable item.** It is exactly the gloss and grammar ground truth How2Sign lacks |

**Licence (DAI terms):**
- research and education use only;
- no redistribution without permission;
- commercial use only with explicit permission;
- no patents or copyrights based on the material;
- citation required.

**What this means for us:**
- Storing a copy in our **private** R2 bucket for research is not redistribution and is
  compatible with the terms. The prefix stays private and is never published.
- **A deployed commercial product trained on ASLLRP data needs written permission from
  the ASLLRP (Carol Neidle, Boston University)** before release. Until then, ASLLRP is used
  for training and evaluation in research builds, and the provenance manifest records it,
  so any release build can exclude it or show the permission.
- **Access needs the user:**
  - The Sign Bank downloads need a DAI 2 login, which the user must obtain. Accounts are
    not created on the user's behalf.
  - Downloads go through the site's own cart and download options, at a polite rate,
    without circumventing any access control.

**How each piece plugs in:**
1. **Grammar ground truth.** Validate the motion-grammar detectors on real labels:
   brow/headshake spans against the annotated spans, role shift (not validated on
   How2Sign) against `rs:` spans, gaze-to-locus against gaze-agreement tiers. Report
   per-marker precision and recall instead of English-proxy AUCs.
2. **More 3D data.** The front and side views match our two-view pipeline (Sapiens →
   calibration → triangulation). This produces canonical states with gloss, grammar and
   translation labels, and adds native-signer data to How2Sign.
3. **English → ASL structure.** The (English translation, gloss string with indices, NMM
   spans) triples train the gloss/semantic planner of item 2 below.
4. **Lexicon.** Sign Bank citation forms train isolated-sign motion and give the
   back-translator a gloss vocabulary.

## Additional sources reviewed 2026-10-21 (user-supplied, no login needed)

Read-only review: no download yet. Facts are from the pages and the licence page.

| Source | What it is | Licence | Decision |
|---|---|---|---|
| ASL Citizen: github.com/microsoft/ASL-citizen-code, arXiv 2304.05934, Microsoft Download Center id 105253 (`ASL_Citizen.zip` 42.8 GB, `Demographics.zip` 5 KB, v1.0 of 2024-10-15) | 83,399 webcam videos of 2,731 isolated ASL signs by 52 signers in everyday settings. Collected with consent under IRB approval. Baselines: I3D and pose ST-GCN (code MIT, checkpoints `checkpoints_v1`). Paper: 63% top-1 and 91% recall@10 on unseen signers. | Microsoft Research Licence Terms: "solely for non-commercial, non-revenue generating, research purposes"; "may not distribute the data or your modifications"; use must stay "consistent with the consent". Commercial use: contact ASL_Citizen@microsoft.com. | **Adopt (research builds)** |
| Sign Language MNIST (Kaggle, datamunge) | 28×28 grayscale static images of 24 fingerspelled letters (J and Z excluded: they need motion). 27,455 train + 7,172 test. | CC0, per secondary sources; the Kaggle page itself was not readable | **Reject**: no motion, no 3D, letters only |
| sign-language-translator/sign-language-datasets | **Pakistani** Sign Language: 776 dictionary signs (27 min), a few sentences, MP4 and MediaPipe world/image landmarks | CC BY 4.0 (commercial use allowed with attribution) | **Reject for training**: a different language would contaminate ASL vocabulary and grammar |
| Mendeley "American Sign Language Alphabet Dataset" (DOI 10.17632/jdyksv2jhh.1, 2025-12-04) | Static images of ASL letters (Train/Test, one folder per letter) | CC BY 4.0 | **Reject for training**. Optional later: a probe that the hand pose separates letter handshapes |

**ASL Citizen plan (item 1c; starts only after the current generator review):**
1. **Acquire.** Download from the Download Center into a private R2 prefix (`acquisition/asl-citizen-v1/`): checksums, the licence text archived beside it, never public.
   - **Direct URLs** (read from the page's download data on 2026-10-21):
     - `https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip`: 45,924,134,223 bytes, ETag `0x66C703EABF2B49391CB7B3749773EFF2890BC0510CCD9E0F006D2A34BC975425`;
     - `.../Demographics.zip`: 5,077 bytes, ETag `0xE9872EA784F95815923FDD70903E06A63B317FEBEAAEF620797C3B991D67910B`.

     Both are dated 2025-02-17. No checksum is published, so check whether the 64-hex ETag equals the streamed SHA-256.
   - **Tool:** `scripts/acquisition/acquire_resumable.py` with `--expected-size`, on a pod (HTTP → memory parts → R2 multipart; nothing staged on disk; re-read and re-hashed from R2).
2. **Extract.** Monocular path: Sapiens keypoints → swing IK → canonical states, one isolated sign per clip. Apply the existing quality gates. Monocular 3D is coarser than two-view: arm direction differs by 20–35° on How2Sign. Record the per-clip estimate quality and keep clips that fail the gates out.
3. **Map glosses.** Map ASL Citizen glosses to the ASLLRP Sign Bank and ASL-LEX entries, keeping the mapping table and the unmatched list.
4. **Use for:**
   - (a) an isolated-sign recogniser on our motion representation, for sign-level evaluation of generated motion beside the sentence back-translator;
   - (b) a citation-form lexicon for the English → gloss → motion planner;
   - (c) hand and handshape realism.

   It adds no English-sentence → motion pairs, so How2Sign and DAI 2 remain the training signal for the generator.
5. **Release gate.** Non-commercial terms, like ASLLRP. A commercial build excludes it or holds Microsoft's permission. Signer identity: the raw videos stay private, and the states are used only within the consent terms.

**DAI 2 login:** awaiting the user's confirmation (2026-10-21). The DAI 2 / Sign Bank plan above applies once it is ready.

## Implementation status (2026-10-22)

**Item 2 — English -> ASL structure (code complete; training under way / waits for the exit gate)**
- `signtranslator/planning/asllrp_gloss.py`: ASLLRP surface tokens -> base gloss, agreement/index locus slots
  (renumbered by first placement, the generator's slot order), per-sign multi-hot of the seven grammatical markers;
  split quoted descriptions rejoined (256 pseudo-glosses removed); gloss vocabulary with exact byte fallback.
- `signtranslator/planning/english_gloss.py` + `scripts/train_gloss_planner.py`: English words -> gloss tokens with
  locus and multi-label marker heads; evaluation by gloss WER/BLEU, aligned marker F1 per tier, locus accuracy and
  utterance-level marker presence. Run v1 (old parse) best: gloss BLEU 27.0, WER 0.72, marker F1 0.67 (wh 0.75);
  it memorises the 1,652 training pairs within ~4k steps (data-limited).
- v2 (fixed parse; `gloss-planner-asllrp-v2`, best step 8,000): gloss BLEU 27.7, WER 0.70, marker F1 0.70.
- Locus accuracy is 0.233. On the same predicted glosses, the per-gloss majority locus scores 0.240 and "no locus" 0.
  The locus head has learned only each gloss's usual slot, not the discourse context. Its slots already follow
  canonical order (1 plan in 412 does not), so renumbering gains nothing.
- Referent placement is under-determined from English at this data size. Spatial agreement is a target for the human
  study and the reviewer-driven fine-tuning, not a solved part of the plan.
- `signtranslator/planning/plan_text.py`: the plan as one deterministic text line (`[topic MOTHER ] IX-loc>1
  1>GIVE>2 [neg NOT LIKE ]`), lossless on all 2,403 pairs, so the byte-conditioned generators and their
  classifier-free guidance learn plan -> motion unchanged.
- `scripts/build_asllrp_plan_manifest.py`: plan -> motion manifest v2 (2,403 utterances, train 1,652 / val 412 /
  test 339, SHA `b0187725…`).
- `scripts/translate_english.py`: English -> planner -> plan line -> plan-trained generator -> motion.
- Plan-conditioned generator stage ready (`runpod/fleet/post_gate_plan_generator.sh`, starts after task No.1's exit
  gate):
  - Training: the full-corpus RVQ prior is warm-started (`train_motion_tokens.py --init-key`) on one mixed manifest,
    How2Sign English plus ASLLRP plan lines (`build_mixed_text_manifest.py`), with the full-corpus tokenizer. All
    states are resampled to 24 fps (`--fps`; ASLLRP is 30, and the priors count time in frames).
  - Plan lines take their grammar features from their marker tags, not English words
    (`grammar_motion.text.is_plan_line`). No How2Sign sentence is misread; 2,402 of 2,403 plans are recognised.
  - Evaluation on ASLLRP val: the plan generator vs the English-conditioned full prior on the same utterances.
    Retrieval and back-translation are judged against the English, and both systems are checked against the plan's
    markers.
  - Evaluation on How2Sign val checks for any regression on English input.
- Citation-stitching baseline (EVALUATION_FRAMEWORK §7):
  - Lexicon: `build_citation_lexicon.py`, R2 `citation-lexicon-v1/lexicon.json`, SHA `b326494e…`. 5,439 keys from
    training-partition clips only (2,648 ASLLVD Sign Bank glosses, 2,791 ASL Citizen).
  - ASLLVD clips are cut to the annotated sign: each DAI clip carries exactly 0.833 s of context either side.
  - `generation/citation_stitch.py` normalises gloss decoration, splits compounds and maps indices to the Citizen
    pointing signs. It covers 81.8% of val plan signs; fingerspelling, name signs and classifiers are held.
  - Evaluated beside the plan generator in the post-gate chain, and available as a `stitch:` system in the human
    study.
- ASLLRP data quality: DAI 2 has one camera view per utterance (cam1, 640×480, upscaled to 1280×960). The
  better-tracked hand's validity is 0.94 for Cory and 0.73 for Rachel (train), but 0.65 for Jonathan (val) and
  about 0.6 for Ben (test). Invalid frames are masked in training. Human-study utterances need hand coverage ≥ 0.7,
  so the recorded reference stays a fair upper bound.

**Item 3 — evaluation instruments**
- Back-translator: training on the full corpus (pod C).
- Sign recogniser: v1 failed cross-domain (below). v2 path implemented: `signtranslator/pose/resample.py` (corpora
  are 24 / 25-30 / 60 fps; the recogniser read frames), ASLLVD added through the exact gloss map, speed / trim /
  face-dropout augmentation; v2 training on a pod. `scripts/score_sign_generation.py` scores one-sign plans with a
  recogniser only after it passes validation.

**Item 3b — evaluation by Deaf ASL users (tooling complete; waits for a generator and for reviewers)**
- `docs/HUMAN_EVALUATION_PROTOCOL.md` (team) and `docs/HUMAN_EVALUATION_REVIEWER_GUIDE.md` (sent to reviewers):
  - four-way comprehension choice plus a free translation;
  - blind recovery of sentence type, negation and topic, with ASLLRP markers as the truth;
  - naturalness 1-5.
- Design: a Latin square over reviewer groups, so no reviewer sees a sentence twice and systems are paired by
  utterance. Distractors are drawn from the whole partition and matched in length.
- `signtranslator/eval_framework/human_study.py`:
  - design and blinding: random ids, private hashed key;
  - sheet validation: hashed reviewer keys only, primed reviewers excluded;
  - scores: Wilson intervals, paired permutation against the recorded reference, Fleiss' kappa.
- `scripts/build_human_eval_packet.py` renders every system the same way, on the recording's clock. It shows the
  skeleton plus the new face panel (`generation/face_preview.py`: head turn, brows, eyelids, gaze, mouth). Tracking
  gaps are held for display, so dropouts do not reveal the reference.
- `scripts/score_human_eval.py` scores the returned sheets. Both scripts were smoke-tested end to end on real val
  utterances.

**Item 4 — tuning (sweep ready; runs after the full prior)**
- `runpod/fleet/post_gate_sampling_sweep.sh` evaluates the full prior on validation over:
  - hands temperature {1.0, 1.2, 1.4} × face temperature {0.8, 1.0}, a cooler face because it carries the
    over-active head codes;
  - guidance {1, 1.5, 2}.
- Each setting is scored by retrieval (with its blank control) and back-translation. The setting is selected with
  the same declared exit gate over the core chain's evaluations plus the sweep. The test partition stays untouched.

**Item 6 — serving (path built; measured on the current research release)**
- Pinned, hash-checked releases with a computed licence gate; a translation service; an HTTP API with bounded
  backpressure; a latency benchmark (docs/DEPLOYMENT.md §12).
- Measured on Mac CPU: p50 1.2 s end to end, with generation at 0.4× real time.
- The release is research-only, because the planner is trained on ASLLRP.

**Retrieval evaluator v2 result (2026-10-22, 16:20 PDT).**
- Setup: frozen all-MiniLM-L6-v2 sentence embeddings with a trained projection (`retrieval-minilm-full-v1`), on the
  same data as v1.
- Result: best R@3 **0.256** at step 3,000. v1 (byte text encoder) had 0.264, and chance is 0.094.
- Overfitting moved to the motion side: train batch top-1 0.82 while val R@3 fell to 0.20 by step 7,000. The limit
  is the motion encoder on ~20k clips, not the text encoder.
- v1 stays the instrument.
- v3 (`retrieval-aug-full-v1`) adds motion augmentation (speed 0.85–1.15×, face dropout 0.2) and dropout 0.3 to v1's
  design.
  - Best R@3 **0.264** at step 2,000, the same as v1.
  - Overfitting slowed (train batch top-1 0.31 at step 6k against 0.82 for v2), but validation still plateaued.
- Three designs reach the same ~0.26. That is the ceiling of pose-only retrieval on How2Sign sentences at this data
  scale, not a fixable training fault.
- Generated motion is therefore read relative to real (the ceiling) and blank (chance), never in absolute terms.

## Acceptance rules declared before results (2026-10-22, PDT afternoon)

**Task No.1 review (on top of the declared exit gate).** The exit gate (scripts/exit_gate.py) checks realism,
articulation and marker rates. It does not check meaning. The full-corpus instruments are weak:
- retrieval R@3 is 0.264 (chance 0.094), early-stopped at step 1,500;
- back-translation val BLEU is about 1.5, with val loss flat since step 6k.

So the selected setting is accepted for the next phase only if, on the same validation clips and through the same
frozen instruments:
- generated R@3 is above the duration-only (blank) R@3;
- generated back-translation BLEU is above blank BLEU.

Both are reported beside real, which is the instruments' ceiling. If either fails, the generator round is not done.
The failure is diagnosed (instrument or generator) before anything moves on.

**Strict pass test of the full-corpus prior (declared 2026-10-22, 17:40 PDT, before any prior result).**
`scripts/strict_prior_review.py` runs at the end of the sweep job, on the final gate selection. Task No.1 passes only
if all six rules hold, whatever the gate said:

| Rule | Test | Pass if |
|---|---|---|
| R1 | Retrieval R@3 through the frozen evaluator, per clip, bootstrap 95% CI | generated CI entirely above blank CI |
| R2 | Back-translation corpus BLEU, sentence bootstrap 95% CI | generated CI entirely above blank CI |
| R3 | Nearest-training-clip distance ÷ nearest unseen-real-clip distance (equal set sizes, Fréchet features) | ≤ 5% of generated clips below 0.5. The dry run on v2b gave 0% flagged, median 0.93 against 1.15 for real. A rule based on raw distance would have flagged mode-averaging as copying. |
| R4 | Sentence-length thirds | every third's body, hand and head speed and variation ratios in [0.7, 1.3] |
| R5 | One-word English prompts for the recogniser's reliable glosses | recogniser top-5 CI entirely above blank (real reported) |
| R6 | Test partition, once (92 signer-disjoint clips); heldout_official reported | every gate check that passed on val passes on test |

**Added after the first full-prior validation results (2026-10-22, 20:30 PDT). Disclosed: this is a change made
after seeing results.**
- **Finding:** the full prior's evaluation (guidance 1 and 2, 600 val clips) showed brow furrow at 0.09–0.10× the
  real rate, so the gate's marker check fails. It also showed head speed at 1.53×, which head smoothing already
  targets.
- **Diagnosis:** the tokenizer reconstructs the face blendshapes at 0.89–0.97× their real spread, but the prior
  samples near-neutral faces. Brow-down varies about 3× less than real within a clip.
- **New gate candidates:** `postprocess.calibrate_face`, a per-blendshape affine map estimated on validation that
  never shrinks a range. It was checked on a held-out half of the val clips: brow furrow 0.20× → 0.99×, brow raise
  0.92× → 0.73×, head markers unchanged.
  - The gate rule is unchanged.
  - The calibration is a post-processing setting chosen on validation, like head smoothing.
  - The strict review applies the stored validation calibration unchanged in every rule, and R6 confirms it once on
    the signer-disjoint test partition.
- **Proper fix:** a prior that samples the face's full range, through face-code temperature or a variance-aware loss.
  That belongs to the next generator round.

**Plan-conditioned generator (post_gate_plan_generator.sh), ASLLRP val (Jonathan).** The plan generator is preferred
to the English-conditioned baseline only if all three hold:
1. Retrieval R@3 and back-translation BLEU against the English are not lower than the baseline's, each judged at the
   better guidance of each system.
2. Grammar: for negation and for each question type with at least 10 positives, the marker AUC
   (`grammar.tests[].generated_auc`) is higher than the baseline's. A marker AUC is generated motion's separation of
   marked from unmarked utterances.
3. How2Sign val: Fréchet distance and every articulation ratio stay within 10% of the full prior's own evaluation.

The citation-stitching baseline is reported beside both. The generator should beat it on Fréchet distance and on the
marker AUCs; stitching shows only the citation forms' own non-manuals.

## Mandatory: the sign recogniser must pass as an instrument before it scores generated motion

The first isolated-sign recogniser (`training-runs/sign-recognizer-aslc-v1`, ASL Citizen, best step 27,000: 53.3% top-1 /
76.5% top-5 on a 2,000-clip validation subset) was validated with `scripts/validate_sign_recognizer.py` (rule fixed
before the run; report `training-runs/sign-recognizer-aslc-v1/validation-interim.json`, step-23,000 checkpoint):

| Check | Result | Rule | Pass |
|---|---|---|---|
| In-domain, all ASL Citizen validation signers (10,303 clips, 2,731 classes) | 49.2% top-1, 72.1% top-5 | reference | - |
| Cross-domain, ASLLVD train+val signers, exact Sign Bank map (2,858 clips, 917 glosses) | **13.3% top-1**, 27.7% top-5 | >= 25% top-1 | **No** |
| Blank (duration only) | 0.04% top-1 | <= 2% | Yes |
| No hands | 1.2% top-1 | <= 15% | Yes |
| Hands only | 19.4% top-1 | diagnostic | - |

It reads the hands (no shortcut on duration or face) but does not transfer to another studio, camera set and signers,
so **it must not score generated motion yet**. Required before any sign-level evaluation of the generator:

1. Retrain on ASL Citizen + ASLLVD together (ASLLVD labels mapped through the exact Sign Bank <-> ASL Citizen gloss
   map; ASLLVD test signer Tyler stays unread), with augmentation for camera and signer variation (view rotation,
   scale, speed, small joint noise) - same validation rule, re-run unchanged.
2. Pass all three checks; the published ASL Citizen baselines (~63% top-1 video, ~60% pose) remain the in-domain
   reference.
3. Score generated motion only on the glosses it recognises reliably on real data (top-1 >= 80% over >= 3 clips; 444
   glosses for v1), always beside real-motion and blank scores through the same model.

Until then the generator's meaning-level evaluation rests on the back-translator and the retrieval evaluator.

**v2 result (2026-10-22, 16:57 PDT).**
- Run and report: `sign-recognizer-v2`, best step 27,000; report `training-runs/sign-recognizer-v2/validation.json`.
- Data: ASL Citizen + ASLLVD train signers, 24 fps, augmented.
- Verdict: **passes all three pre-declared checks**. Usable as an instrument on its 474 reliable glosses (top-1 ≥ 0.8
  over ≥ 3 val clips).

| Check | v1 | v2 | Rule |
|---|---|---|---|
| In-domain, all 10,303 ASL Citizen val clips | 49.2% / 72.1% | 50.5% / 72.9% | reference |
| Cross-signer, ASLLVD val signer Naomi (411 clips, 348 glosses) | 13.3%* | **60.8%** / 82.2% | ≥ 25% |
| Blank | 0.04% | 0.03% | ≤ 2% |
| No hands | 1.2% | 1.5% | ≤ 15% |
| In-context ASLLRP signs, unseen studio and signers (diagnostic) | 2.6% | 8.3% / 17.8% (blank 0.17%) | - |

\*v1 was measured on ASLLVD train+val signers with no ASLLVD in training. v2 trains on the ASLLVD train signers, so its
cross check is on the held-out val signer: a new signer in a studio it has seen, not a new studio. The genuinely
unseen domain is the in-context ASLLRP row. It improved 3×, but isolated-sign recognition of coarticulated signs stays
low, so sign-level scoring of the generator stays on one-sign plans, beside real and blank scores.

In-context diagnostic (added 2026-10-22, not part of the rule): `validate_sign_recognizer.py
--continuous-manifest-key` cuts the annotated signs out of ASLLRP continuous utterances of the train/val signers
(Cory, Rachel, Jonathan; unseen by the recogniser), padded by 0.1 s.
- Sample: 5,348 signs from 1,716 utterances, median 0.23 s.
- v1 result: **2.6% top-1, 7.1% top-5** (blank 0.07%).
- Isolated-sign recognisers do not read coarticulated signs inside sentences. Sign-level scoring of the generator
  therefore stays on one-sign plans, and sentence-level meaning rests on back-translation, retrieval and the human
  study.

## Item 7 — YouTube ASL corpora (approved by the owner 2026-10-22, ~18:35 PDT)

**Decision.**
- The owner approved acquiring the YouTube-hosted ASL corpora for research. The candidates are YouTube-ASL (Uthus
  et al. 2023, ~1,000 h, 2,500+ signers) and OpenASL (Shi et al. 2022, ~290 h); the research below confirms or
  amends the list.
- **Release intent:** the full model will eventually be open-sourced as a *research* model. Every corpus therefore
  stays tracked in each release's provenance, and permission to release derived weights is requested from each
  dataset's owners before any public release.
  - Research: `docs/research/YOUTUBE_ASL_RESEARCH.md`, in progress. It covers which corpora and filters give the
    best model, the expected gains, acquisition practice, and contacts and terms for derived-weight permission.
- Data handling is the same as ASLLRP:
  - private R2 prefix `acquisition/2026-10-17-v1/youtube-asl-v1/` and `.../openasl-v1/`;
  - motion states and caption text only; videos are deleted after extraction and never redistributed;
  - per-video provenance (video id, channel, upload date, YouTube licence field, caption kind manual/auto).
  - Because of the licence field, a CC BY-only subset can be separated later if a more permissive release is
    wanted.

**Why.** Pose-only meaning measures are at a data ceiling:
- back-translation BLEU 2.1 (v1), with v2 (T5) training now;
- retrieval R@3 0.26 across three designs.

Published pretraining on YouTube-ASL took pose-based How2Sign translation to about 12 BLEU. Expected effects
(estimates, not results; to be replaced by the research report):
- back-translator BLEU about 6–12;
- retrieval R@3 about 0.35–0.45;
- generator meaning and sign-level scores about 1.5–3×;
- little change to non-manual grammar, which ASLLRP still supplies.

**Research result (2026-10-22, 19:00 PDT; `docs/research/YOUTUBE_ASL_RESEARCH.md`).**
- **Corpora, in priority order:**
  1. YouTube-ASL: 11,093 IDs, 984 h, every video filtered by Deaf annotators; IDs under CC BY 4.0; the list is in R2.
  2. The ASL rows of YouTube-SL-25 add about 410 h but were checked by a non-native signer, so they need stricter
     filters.
  3. DailyMoth-70h and ASL STEM Wiki: consent-based, not on YouTube. They need licence agreements, which only the
     owner can sign. Best suited to the generator.
  4. OpenASL: skipped. Over 72% of its test set is inside YouTube-ASL, and its CC BY-NC-ND licence is the strictest.
- **FLEURS-ASL** (Kaggle, CC BY-SA 4.0) is an independent evaluation set, never training data.
- **Expected gains (pose-only, How2Sign test):**
  - How2Sign alone gives about 1 BLEU, YouTube-ASL pretraining 12.4, all of YouTube-SL-25 15.4, against a human
    ceiling of 19.8.
  - Our target for the back-translator is about 8–14 BLEU, up from 2.1.
  - No published study measures the effect on sign *generation*, so it is A/B-tested here.
- **Filters:** the published ones are implemented in `build_youtube_asl_manifest.py`:
  - 10 s–5 h, at least 360 px high, 15–60 fps, captions covering at least 40% of the video;
  - captions of 300 characters or fewer, at 0.5–4 words per second;
  - face tracked on at least 90% of frames, median head yaw at most 30°, hand coverage at least 0.5;
  - near-duplicates of evaluation sentences removed (exact match, or word-5-gram Jaccard ≥ 0.8);
  - no channel above 5% of the records.
  - Still to calibrate in the pilot: a single-person check and the stricter generation-grade tier, which needs a
    reviewer check of the top channels.
- **Fallback if YouTube blocks the pods** (no circumvention): the UWB YouTube-ASL keypoints (MediaPipe 2D, CC BY 4.0,
  348 GB) or SignVerse-2M (DWPose 2D, CC BY-NC 4.0).
  - Both are 2D, so they can't supply our 3D rotations.
  - They can train a 2D-keypoint back-translator and retrieval evaluator that read generated states projected to 2D.
- **Permission requests (the owner sends them; draft in §4.5 of the report):**
  - Required: Carol Neidle (carol@bu.edu) for ASLLRP, DAI 2, Sign Bank and ASLLVD; Microsoft (ASL_Citizen@microsoft.com)
    for ASL Citizen.
  - Courtesy notices: the YouTube-ASL and YouTube-SL-25 authors at Google, and How2Sign (amanda.duarte@upc.edu).
- **Precedent:** Uni-Sign released YouTube-ASL-pretrained weights under CC BY-NC 4.0.
- **Risk accepted by the owner:** YouTube's Terms restrict downloading.
- **Permission requests sent 2026-10-22, ~19:05 PDT** from the owner's Gmail, in the owner's name (Shengbo Jiang,
  Kehillah High School, https://github.com/Shengboj0324/Sign-Translator), at the owner's instruction:
  - carol@bu.edu (ASLLRP / DAI 2 / Sign Bank / ASLLVD, permission);
  - ASL_Citizen@microsoft.com (permission);
  - the YouTube-ASL / YouTube-SL-25 authors at Google (notice + licence question);
  - amanda.duarte@upc.edu (How2Sign, notice).
  - Replies are tracked by the 20-minute inbox check and recorded in each release's provenance.

**Y2 pilot result (2026-10-22, 20:30 PDT; 95 YouTube-ASL videos, US-TX-3 GPU pod plus one 32-vCPU CPU pod).**
- **Bugs found and fixed during the pilot:**
  - a missing `libEGL`;
  - a broken OpenCV install, where two variants removed `cv2`;
  - English caption tracks filed under the sign-language code `ase` or regional `en-*`;
  - new pods in US-MO-1 and US-NC-1 that never started.
- **No YouTube blocking.** All downloads succeeded from the datacenter.
- **Availability:** 76 of 95 videos converted (80%). Removed and private videos: 11%. 6 videos with no captions
  remain, and 3 have no usable cues.
- **Segments:** 2,633 segments (3.66 h) from 249 min of video. Median hand coverage 0.97, face coverage 1.0.
- **After the manifest filters:** 2,152 segments (3.16 h), 82% kept. The main rejections were caption speed
  (181 too fast, 30 too slow), too few words (98) and hands not visible (83).
- **Throughput:** about 3.4 worker-minutes per minute of video, CPU-bound. Processing all of YouTube-ASL (984 h)
  takes about 3,300 worker-hours: about 2.3 days on two 32-vCPU CPU pods ($0.96/h each), or about 1.2 days on four.
- **Provenance gap:** yt-dlp returned no licence field for any video (the field is null). Channel, upload date and
  caption kind are recorded.

**Blocking event (2026-10-22, ~20:50 PDT).**
- **What happened:**
  - The first full-acquisition pods (two 32-vCPU CPU pods in EU-RO-1) ran 30 parallel downloaders each.
  - YouTube began answering "Sign in to confirm you're not a bot" after about 125 videos.
  - The polite restart still let workers queued on the download lock request once each, about 20 requests.
- **Response, under the no-circumvention rule:**
  - No cookies, sign-in, proxies or address changes.
  - Downloads are serialised to one per machine with a gap between them. The gate re-checks a refusal after taking
    its lock. A refusal makes every worker on the machine wait 1 h, and three refusals stop the shard.
  - Both pods were released during the block.
  - Acquisition resumes at ~22:20 PDT on **one** pod, one download per 24 s (≤150/h, under the ~300/h guest rate).
- **New estimate:**
  - Download-bound: all of YouTube-ASL takes about 3 days at 150 videos/h.
  - Instrument pretraining starts as soon as a useful amount is in. ~200 h of YouTube-ASL arrives in about 0.7 days;
    the published gains already show at a few hundred hours.
  - Fallback for the instruments if refusals continue: the UWB YouTube-ASL keypoints (MediaPipe 2D, CC BY 4.0,
    hosted outside YouTube).

**Stages, cost and time (estimates).**

| Stage | Work | Where | Time | Cost |
|---|---|---|---|---|
| Y0 | Research report: corpora, filters, gains, contacts, terms | agent (running) | ~1 h | - |
| Y1 | Acquisition tool `scripts/acquire_youtube_asl.py` | Mac (code + tests) | ~2 h | - |
| Y2 | Pilot ~50 h: availability rate, caption kinds, hand coverage, extraction throughput, blocking | 1 CPU pod | ~3–6 h | <$10 |
| Y3 | Full acquisition of the selected corpora | ≤4 CPU pods | ~2–4 days | ~$100–300 |
| Y4 | Manifest `youtube-asl-manifest-v1`: quality filters, dedup across corpora, train-only + small channel-disjoint monitor split | Mac | ~2 h | - |
| Y5 | Pretraining and fine-tuning in priority order (meaning instruments first: `runpod/fleet/youtube_pretrain_instruments.sh`, back-translator v2 pre-trained How2Sign+YouTube then fine-tuned on How2Sign, retrieval on the mix; then the generator) | GPU pods (300 GB disk) | ~1–2 days | ~$50–150 |
| Y6 | Same strict rules and gate as task No.1 | GPU pods | ~0.5 day | ~$10 |

Y5 covers, in order:
1. back-translator v2 (T5) and retrieval evaluator, pretrained on YouTube data then fine-tuned on How2Sign;
2. the generator prior on How2Sign + YouTube English (motion tokens from the full-corpus tokenizer, re-checked for
   learnability on the new domain).

**Second blocking event and fallback (2026-10-22, 22:27 PDT).** The polite resume ran on one CPU pod, one
download every 24 s, after a refusal window of more than 80 minutes. YouTube answered its first request with
"Sign in to confirm you're not a bot" on three workers at once. Under the rules, acquisition from YouTube is paused
and the pod was released at once. No cookies, sign-in, proxies or IP hopping. YouTube-ASL stays at the 125 pilot
videos (6.1 h).

Only the owner can unblock it. Acquiring at a low rate from a non-datacenter connection carries Terms-of-Service
implications (research report §4.3), so it is the owner's call. Until then the declared fallback runs:
- **Data:** the **YouTube-ASL Clip Keypoint Dataset** (UWB / JSALT 2024, LINDAT 11234/1-5898, **CC BY 4.0**),
  hosted by LINDAT, not YouTube. It is MediaPipe Holistic 2D, 33 pose + 2×21 hand + 478 face points per frame,
  for 390,547 single-signer sentence clips with English translations (train/dev JSON), 337 GB in 10 zips.
- **Acquisition:** `runpod/fleet/acquire_uwb_keypoints.sh` streams it into R2, MD5-verified, three streams at once.
- **Use:** a **2D "second-opinion" back-translator**, keypoints → English (T5), pretrained on the 390k clips and
  fine-tuned on How2Sign.
  - Both How2Sign and generated motion reach it the same way: canonical 3D state → forward kinematics →
    orthographic front projection → the shared 2D layout (shoulders, elbows, wrists, 2×21 hand points), normalised
    by shoulder centre and width. Fine-tuning and evaluation therefore share one domain.
  - It reads manual channels only. Face meaning stays with the existing instruments.
- **Not covered:** this does not give the generator more 3D training data. That needs the video (blocked).
- **Assumption:** clip names are frame ranges, and the source frame rate is not given, so 30 fps is assumed.
  Pretraining uses time-stretch augmentation of ±25%. The How2Sign fine-tune runs at its known rate.
- **Instrument acceptance before use** (same skepticism as the recogniser): on How2Sign val through the FK path,
  real BLEU must exceed blank and shuffled-text controls by a clear margin. Per-sentence agreement with the T5
  back-translator is reported. If it fails, it is not used.

## Task No.1 verdict: exit gate PASSED on validation, strict review FAILED (2026-10-23, 05:50 PDT)

**Gate.** The sweep over 9 per-part temperature settings × 3 guidance values produced 6 candidates that pass all 12
checks, every one combining head smoothing 0.2 s with face calibration. That fixes both checks the core chain failed
(head speed, brow furrow). The declared rule selected `sweep-h1.4-f1.2-hs0.2-fc` at guidance 2 (Fréchet 280.7): among
passing candidates within 10% of the best Fréchet (257.5), it has the lowest right-hand error.

**Strict review** (rules declared 2026-10-22, 17:40 PDT, before any result): **FAIL.**

| Rule | Result | Pass |
|---|---|---|
| R1 retrieval | generated 0.236 [0.20, 0.27] vs blank [0.07, 0.12]; real 0.290 | yes |
| R2 back-translation | generated 0.83 [0.62, 1.05] vs blank 0; real 0.80 | yes |
| R3 copying | 0% flagged; median ratio 0.92 (real 1.16) | yes |
| R4 length thirds | long and medium pass; short: body variation 1.47, right-hand variation 0.61 | **no** |
| R5 one-word prompts | recogniser top-5: generated 0%, blank 0%, real 99.4% (95 words) | **no** |
| R6 test, once | 92 signer-disjoint clips: 3/11 checks. Head speed 1.83×, nod rate 19.8×, brow raise 3.2×, body variation 1.69×. Official held-out (3,019 clips, reported): 8/12; head 1.35×, nod 2.0×, brow furrow 0.23×. | **no** |

**What this means.** Every partition is signer-disjoint, from fold 0 of the frozen fold plan:
- validation is **one** signer (3, 1,627 clips);
- test is **one** other signer (9, 92 clips);
- training is signers 1, 2, 4, 5, 8 and 11.

So validation was never leaked. It is a single unseen signer, and everything tuned on it (sampling setting, head
smoothing, face calibration) fitted that signer's style. Signer 9 differs. (Correction, 05:58 PDT: a first draft said
validation shares signers with training. It does not.)
- The post-processing tuned on validation did not transfer. Head smoothing brought head speed to 1.05× on val, but
  unseen signers see 1.35–1.83×. The face calibration's brow furrow was 0.67× on val and 0.23× on held-out.
- The carried problem, an over-active head (round 2026-10-21), is not solved; it was masked on validation.
- The generator cannot sign a single word. That is outside its How2Sign sentence-level training, and the recogniser
  reads it from a different capture domain (MediaPipe isolated signs). R5 cannot separate those two causes, but
  generated scores 0 where real scores 99%.
- Meaning at the sentence level is real (R1, R2, and the 2D second opinion).

**The model is not deployable under the declared rules.** The rules are not relaxed; the next generator round
addresses the causes.

**Next generator round (round 3), designed from these failures:**
1. **Multi-signer selection.** Choose sampling and post-processing across several unseen signers, not one.
   - Use the fold plan's leave-one-component-out folds: validation signers 3 and 1/2 (folds 0 and 3), plus signer 11.
   - Require each choice to hold on every validation signer.
   - Estimate face calibration from population statistics of all training signers.
   - R6, test signer 9, stays the untouched final check.
   Then fix head over-activity in the model itself, through head-code temperature and a head-velocity term in the prior
   training, not only by smoothing.
2. **Single signs.** Add ASL Citizen isolated signs (already in R2; research use) to training as one-word English
   prompts. Add ASLLRP citation forms too. R5 then measures a skill that was actually trained.
3. **Mixed tokenizer.** Retrain it on How2Sign + YouTube-ASL + ASL Citizen + ASLLRP. The domain check found ASLLRP
   face and blink encoded 1.5–2.3× worse; ASL Citizen and YouTube are checked the same way.
4. **More signers.** YouTube-ASL home acquisition adds hundreds of signers. This is the direct cure for
   signer-specific style and the main lever for R6.
5. **Short utterances.** Report R4's short-third ratios during training validation.
6. **Recogniser domain.** Validate the recogniser on How2Sign-layout isolated signs before it judges R5, through the
   same FK/conform path, as the plan generator's diagnosis already requires.

**Round 3 addition, declared 2026-10-23 at 11:50 PDT, before any round-3 result: head-excursion damping.**
- **What it is:** `postprocess.damp_rotation_channel` pulls each frame's head rotation toward the clip's mean by a
  factor: R'_t = R̄ · exp(a · log(R̄ᵀ R_t)), a ∈ {0.8, 0.6}. Smoothing removes speed; damping removes amplitude.
- **Why:** the full prior's R6 failure on the unseen signer was amplitude, not only speed (nod rate 19.8×, head
  variation 2.3×).
- **How it is selected:** the variants join the post-processing grid (every head smoothing × damping × pooled face
  calibration). They are chosen by the same multi-signer gate across validation signers, and R6, test signer 9, stays
  the final check.
- **Mechanics:** `runpod/fleet/round3_supplement.sh` runs after stage B. **Round 3's verdict is the strict review of the
  union selection.** Stage B's own strict review, on the undamped grid, is reported as preliminary.
- **Pipeline:** the release spec and service carry `head_damping`, and the strict review applies it in every rule.

**Home route: first soft refusals (2026-10-23, 11:05 PDT).** Two downloads (videos #830 and #956 of about 950) were
answered "Please sign in. This helps protect our community." That is YouTube's soft bot check. It was not in the
detector's patterns, so it was logged as an ordinary error.
- **Fixed:** the wording is now a block pattern.
- **Response, per the rules:** stopped; a 2 h back-off (resume about 13:04 PDT); then half the rate (one download per
  60 s); a 2 h back-off per further refusal; a full stop after 3.
- No sign-in, cookies or address changes, as before.
- 69.1 h collected so far. At the new pace, about 3 days to the 200 h threshold.

**Round 3a: implemented and running (2026-10-23, 06:20 PDT).**
- **Data** (mix `round3a-mixed-v1`, SHA `dc576f4a`; 49,235 training clips):

  | Corpus | Training clips | Manifest |
  |---|---|---|
  | How2Sign, multi-signer re-partition | 18,289 | `how2sign-text-manifest-v2-multisigner`, SHA `dcdb6451` |
  | YouTube-ASL r3a (home route so far; 23.3 h after filters) | 15,560 | |
  | ASL Citizen one-word prompts (capped at 6 per word, 2,289 words) | 13,734 | SHA `50125638` |
  | ASLLRP plans | 1,652 | |

  - Validation is How2Sign signers 3, 1/2 and 11 (3,096 clips), plus the other corpora's own validation splits.
  - Test is signer 9, untouched; held-out is How2Sign's official test.
- **Stage A** (`runpod/fleet/round3_tokenizer.sh`, running): a mixed tokenizer.
  - The full-corpus tokenizer's domain check on the round-3 corpora: ASLLRP fails (blink 1.91×), ASL Citizen passes
    (relative face 2.06×), YouTube passes (blink 1.48×).
  - The new tokenizer must pass the same check on every corpus, or stage B does not start (`after_tok.sh`).
- **Stage B** (`runpod/fleet/round3_prior.sh`): prior from scratch, then:
  - a 3×3 temperature sweep × 3 guidance values, evaluated per validation signer;
  - a post-processing grid with face calibration pooled over signers;
  - the multi-signer gate (every signer must pass the declared checks);
  - the strict review, R1–R6 unchanged, with R6 on test signer 9 once.

  A local dry run (2 signers × 8 clips) exercised every stage.
- **Caveat (instruments):** the retrieval evaluator and back-translator were trained on fold-0 training data, which
  included signers 1/2 and 11, now validation. Their real-clip ceilings on those signers are inflated. The R1 and R2
  rules compare generated with blank, not with real, so they stay valid.

**Second-opinion meaning check of the full prior (2026-10-23, 03:55 PDT).** The accepted 2D back-translator
(`back-translator-kp2d-h2s-v1`, an independent instrument reading FK projections) scored the full prior's core
validation generations on 600 How2Sign val clips:

| Guidance | Generated | Real | Blank | Shuffled | Share of the blank→real gap |
|---|---|---|---|---|---|
| 1 | 1.84 | 2.25 | 1.60 | 1.59 | 37% |
| 2 | **2.16** | 2.25 | 1.60 | 1.59 | **86%** |

Both instruments now find meaning in the generated motion above the text prior, so the signal is not an artefact
of one instrument. The 3D back-translator gave guidance 1 BLEU 0.68 against real 0.80 and blank 0.0. These are
point estimates; the strict review's R2 bootstrap rule uses the 3D translator, as declared.

**Plan-conditioned generator: result against its declared rules (2026-10-23, 03:25 PDT).**
`tokens-rvq-d0b-prior-loci-plan-v1`: warm-started from the full prior and trained on How2Sign + ASLLRP plan lines.
It early-stopped at step 13,000; the best checkpoint is step 8,000 (val loss 3.488).

| Rule | Plan generator | English baseline | Verdict |
|---|---|---|---|
| 1. Retrieval R@3 on ASLLRP val, better guidance | 0.128 | 0.177 | **fail** as declared. The instrument is invalid here: real ASLLRP scores 0.12 against chance 0.094. |
| 1. Back-translation BLEU on ASLLRP | 0.0 | 0.0 | **uninformative.** Real ASLLRP also scores 0.0, because the How2Sign-trained translator does not read ASLLRP. |
| 2. Marker AUC (guidance 2): wh / yes-no / negation | 0.665 / 0.557 / 0.483 | 0.572 / 0.445 / 0.459 | **pass.** Both are below 0.5 on negation (real 0.598). |
| 3. How2Sign Fréchet; articulation within 10% of the full prior | Fréchet 199 vs 290; body speed 0.92 vs 1.06; variation 1.00 vs 1.24 | | **better, but misses the letter:** two ratios moved more than 10%, both toward 1.0 (real). |

Other measurements:
- How2Sign back-translation: 0.71 (guidance 1) against the full prior's 0.68, with real 0.81 and blank 0.0.
- Sign level (recogniser v2, one-sign plans, 219 prompts): top-1 **0.5%**. Real ASLLRP citation clips score 97.3%, the
  stitching pipeline check 88%, blank 0%.
- ASLLRP Fréchet: 7,498 for the plan generator, 6,062 for the English baseline and 219 for stitching. Neither generator
  produces ASLLRP-like motion.

**Verdict: the plan generator is not adopted. The English-conditioned generator stays the selected system.**

**Diagnosis.** On ASLLRP, all three meaning instruments (retrieval, back-translation, recogniser) are either shown
invalid or unvalidated for the generator's output domain.
- **Retrieval and back-translation:** real ASLLRP motion scores at chance and at 0.
- **Recogniser:** validated only on monocular MediaPipe clips. The generators emit How2Sign-layout motion, Sapiens
  two-view, conformed. Stitching passes because it replays real ASLLRP states.
- So the sign-level near-zero cannot yet separate "the generator does not sign the gloss" from "the recogniser does not
  read this domain".
- **The fair sign-level reference is in-context signing.** The plan generator learned glosses from continuous ASLLRP
  utterances, so its one-gloss output resembles coarticulated, in-context signs. On real in-context ASLLRP segments,
  recogniser v2 reached 8.3% top-1, against 60.8% on isolated ASLLVD and 97% on the citation clips. Generated, at
  0.5%, is still about 16× below that in-context reference. That points to the generator as well as the domain.
- The grammar AUCs, which need no instrument, favour the plan generator. The tokenizer domain check found ASLLRP face
  and blink encoded 1.5–2.3× worse, which bounds what any generator on this tokenizer can show on ASLLRP.

**Consequences for the next round:**
- **Tokenizer:** one trained on the mix, as the YouTube round chain already does on a failed domain check.
- **Recogniser:** validated on generator-domain real signs before it judges generated signs. This needs How2Sign-layout
  isolated signs, for example re-estimating ASLLRP or ASL Citizen clips through the two-view-compatible path, or
  testing the recogniser on How2Sign motion conformed to the monocular layout.
- **Meaning on ASLLRP:** the 2D back-translator, which reads both domains through the same FK projection, is the
  candidate meaning instrument. It is only to be used after its own acceptance on ASLLRP real.

**2D back-translator, UWB-pretrained and How2Sign fine-tuned: acceptance PASSED (2026-10-23, 17:11 PDT).**
- **Pretraining** (`back-translator-kp2d-uwb-v2`): 323,978 UWB YouTube-ASL keypoint clips, selected on the UWB
  monitor split. Best step 76,000, monitor BLEU 6.09; it early-stopped at 96,000. Zero-shot How2Sign BLEU stayed
  about 1.1, so the How2Sign fine-tune was needed.
- **Fine-tune** (`back-translator-kp2d-uwb-ft-v1`): initialised from step 76,000 and trained on How2Sign FK
  keypoints. Best val BLEU 5.01, against 2.84 for the How2Sign-only model under the same selection; it early-stopped
  at 15,000. The acceptance scored checkpoint 9,000 on the same 600 val clips and rule as below:

| Control | BLEU | 95% interval |
|---|---|---|
| real | **4.09** | 3.51–4.68 |
| blank (durations only) | 1.95 | 1.56–2.36 |
| shuffled sentences | 1.59 | 1.24–1.94 |

- The real interval lies well above both controls. The gap over blank is 2.1 BLEU, against 1.0 for the How2Sign-only
  model, so the YouTube pretraining made the instrument clearly more sensitive.
- Per-sentence agreement with back-translator v2 (3D T5, BLEU 1.94 on these clips) is Spearman 0.74.
- This model replaces the How2Sign-only one as the 2D second opinion. Its reading of the full prior follows.

**Second opinion of the full prior through the fine-tuned 2D back-translator (2026-10-23, 17:27 PDT).** On the same
600 How2Sign val clips, BLEU is: real **4.26**, blank 1.95, shuffled 1.22, generated **1.43** (guidance 1) and
**1.74** (guidance 2). The full prior's generations score below the duration-only control. The more sensitive
instrument finds no sentence meaning beyond what timing gives, where the 3D back-translator had passed R2 (0.83 vs
real 0.80, within its noise).

**R2b added to round 3's verdict (declared 2026-10-23 17:35 PDT, before any round-3 strict result).** The round-3
union strict review (`round3_supplement.sh`) applies R2's rule through this 2D back-translator as well
(`strict_prior_review.py --second-translator-key`). The generated bootstrap BLEU interval of the selected setting,
pooled over the validation signers, must lie entirely above blank. Like every rule, R2b must pass for round 3 to pass.

**2D back-translator, How2Sign-only model: acceptance PASSED (2026-10-23, 02:32 PDT).**
`back-translator-kp2d-h2s-v1`, best checkpoint step 13,000; it early-stopped at 19,000. Scored on 600 How2Sign val
clips through the FK projection path, beam 4, with sentence-bootstrap 95% intervals:

| Control | BLEU | 95% interval |
|---|---|---|
| real | **2.45** | 2.01–2.90 |
| blank (durations only) | 1.47 | 1.15–1.77 |
| shuffled sentences | 1.53 | 1.19–1.82 |

- The rule passes: the real interval lies entirely above both controls.
- Back-translator v2 (3D T5) scores 1.94 on the same clips.
- Per-sentence agreement with v2 is Spearman 0.85: both instruments find the same sentences legible.
- **Limits:**
  - The margin over the text prior (blank) is modest.
  - Training loss reached 1.4 against a validation loss of 3.9. On How2Sign alone the model overfits early; the UWB
    pretraining stage is meant to fix that.
  - Beam outputs sometimes repeat a phrase.
- It may now be used as the second-opinion meaning instrument. The UWB-pretrained version replaces it only if it is
  better on val BLEU and passes the same rule.

**Home-connection route approved (2026-10-22, about 23:15 PDT).** The owner approved low-rate acquisition from the
home connection. The rules are unchanged: no cookies, sign-in, proxies or CAPTCHA work, and the first bot check
stops the downloader.
- **Home Mac:** `~/.sign-translator-ops/home_youtube.sh` runs `acquire_youtube_asl.py --stage download` in seeded
  random order (spread across channels), one video at a time through the polite gate, 30 s apart. Each video and
  its captions are staged in the private store.
- **CPU pod:** `runpod/fleet/youtube_extract_loop.sh` runs `--stage extract`, writes the reports and deletes each
  staged video. Staged videos are transient: only derived motion and captions are kept.
- **Pace:** Mac upload is 4.1 MB/s and a 720p video is about 85 MB, so roughly 60–80 videos (6–7 h of video) per
  hour. The plan's 200 h threshold for the YouTube pretraining round is about 1.5 days away.
- **Trial:** 3 videos staged, no bot check.

The UWB 2D keypoint instrument continues in parallel. It serves meaning checking, while the home-route video serves
the generator.

**Single-signer filter (2026-10-22, 22:15 PDT).** Acquisition counts people on every 12th frame with a separate
image-mode pose pass. Each segment records `other_person_fraction`, and the manifest drops a segment when a second
person at least half the signer's size appears in more than 10% of its frames (reason `second_person`). In an
interview or group shot the caption may belong to someone other than the tracked signer.
- The signer's own tracker stays single-person. Asking it for two people moved a lone signer's landmarks by a median
  2% of image width (90th percentile 13%).
- Validation: a lone signer scores 0% and a two-person clip 100%. The signer's landmarks are bit-identical, at no
  measurable cost.
- The 125 pilot videos were acquired before this and have no count, so their segments are kept.

**Tokenizer domain check (2026-10-22, 21:58 PDT).** `scripts/tokenizer_domain_check.py`, rule fixed before any
YouTube result: every channel's reconstruction error on a new corpus ≤ 1.5× How2Sign val's. A diagnostic, which
does not change the rule, divides each error by the corpus's own signal scale, separating "moves more" from "encoded
worse". First run, ASLLRP (24 clips) through the frozen full-corpus tokenizer:

| Channel | Error ratio | Signal-scale ratio | Relative-error ratio |
|---|---|---|---|
| body | 0.94 | 1.23 | 0.76 |
| left hand | 0.95 | 1.03 | 0.93 |
| right hand | 0.97 | 0.86 | 1.14 |
| head rotation | 0.94 | 1.00 | 0.93 |
| face | 1.41 | 0.92 | 1.53 |
| blink | **2.09** | 0.93 | **2.25** |

Result: **fail**. Body, hands and head encode as well as How2Sign. Face and blink do not: the motion is the same
size, so this is a real encoding gap, not more movement. Consequences:
- **Plan generator (running):** it uses the frozen tokenizer, so its ASLLRP face and blink output is bounded by
  this. Its grammar results on brows are reported with this limit.
- **YouTube round** (`runpod/fleet/youtube_generator_round.sh`): it checks YouTube-ASL and ASLLRP. A failure on
  either trains a tokenizer on the mix first (How2Sign + YouTube + ASLLRP), then the prior from scratch, rather than
  warm-starting.

**Rules.**
- YouTube data is training-only. No evaluation number may come from it: How2Sign val/test and ASLLRP val stay the
  judges.
- No circumvention of YouTube's bot checks, sign-in requirements or CAPTCHAs. If datacenter downloads are blocked,
  acquisition pauses and the alternatives go to the owner (e.g. pre-extracted public features, a residential
  low-rate path).
- Videos are fetched at ≤720p with English captions. Only segments whose caption timing fits the signing (duration
  limits, hands visible) are kept.

## Round 3 result: the multi-signer gate FAILED (2026-10-23, 22:33 PDT)

- **Union gate** (`tokens-rvq-d0b-prior-r3a-gate-union.json`): 675 candidates (9 temperature settings × 25
  post-processing variants × 3 guidance values). Each was scored on validation signers 1/2 (c01), 3 (c03) and 11
  (c05), with the declared thresholds unchanged.
  - **No candidate passes every check on every signer.**
  - The rule's fallback selects the most checks passed: `sweep-h1.2-f1.0-hs0.2`, guidance 2, **26 of 35 checks**.
    Its mean Fréchet distance is 229.4, against 280.7 for the full prior's selection.
- **Its failures go in opposite directions on different signers:**

  | Signer | Failing checks of the selection |
  |---|---|
  | c01 | brow raise 2.9× |
  | c03 (head too still) | headshake 0.31×, nod 0.26×; hand variation 0.69× / 0.67× |
  | c05 (head too active) | head speed 2.6×, variation 2.9×; headshake 8.9×; brow raise 6.4× |

- Across all 675 candidates, the most common failures are nod, headshake, head speed and variation, and brow raise.
- **Round 3 fails by its declared rules.** This is not reinterpreted after the fact.
- The union strict review (R1–R6 with R2b) on the fallback selection is reported when it finishes.
- **Operational faults fixed on the way:**
  - One sweep evaluation ran out of GPU memory (4 concurrent on 44 GB); it was rerun by the supplement.
  - `GROUPS` is a reserved bash variable: the gates were given signer group `0` (`fbef071`).
  - The strict review lacked `transformers` for R2b (`9d9f08b`).

**Instrument check of the multi-signer gate: real signers fail it (2026-10-23, 22:45 PDT).**
`scripts/cross_signer_baseline.py` scores real validation clips of one signer as if generated, with the gate's own
checks and thresholds, on the clips the evaluation used (`round3a/evaluations/cross-signer-baseline.json`).

| Real clips of | judged by | checks passed |
|---|---|---|
| c01 | c03 / c05 | 6 / 12, 7 / 12 |
| c03 | c01 / c05 | 4 / 11, 2 / 12 |
| c05 | c01 / c03 | 8 / 11, 6 / 12 |
| c01 / c03 / c05 | the other signers pooled | 8 / 12, 4 / 12, 6 / 12 |
| each signer | the other half of its own clips | 11 / 11, 12 / 12, 11 / 12 |

- Between real signers, head speed ratios run 0.29–3.5×, headshake rates 0.05–21×, nod rates 0.09–10.6× and brow
  raise 0–5×.
- Within one signer, nearly every check passes.
- So the per-signer head and face checks measure **who is signing**. A text-conditioned generator cannot pass them on
  signers 3 and 11 at once, and neither can real human signing.
- Round 3's selection (8.7 checks per signer on average) is closer to each signer than the other real signers are
  (5.5 of 12).

**Round 3b gate, declared 2026-10-23 23:07 PDT before any round-3b result.**
- **Envelope rule.** For each validation signer Y and each gate check c, the band is the declared band widened to the
  real cross-signer range: `[min(lo, min r_c(X→Y)), max(hi, max r_c(X→Y))]` over the other real signers X. The
  generated ratio against Y must lie in it.
  - In words: generated motion must be at least as close to each signer as another real signer is.
  - The band is never narrower than the declared one.
- **Population envelope.** The same rule against all validation signers pooled, with the band widened by the real
  signer-vs-others-pooled range.
- A candidate passes only if every check passes in every signer's envelope and in the population envelope.
- Selection, the meaning and copying rules (R1, R2, R2b, R3, R5) and R4 are unchanged.
- **R6** (test signer 9, once): each check that passed on validation must lie in signer 9's envelope. That envelope
  is the declared band widened by the validation signers' real ratios against signer 9.
- The envelopes are computed from real clips only, before any round-3b generation is scored.
- Human Deaf reviewers judge naturalness directly afterwards. The envelope only stops one person's style being
  scored as an error.

**Round 3 union strict review (2026-10-23, 23:36 PDT): FAILED on R2b, R4 and R5.**
The review judged the gate's fallback selection: `sweep-h1.2-f1.0-hs0.2`, guidance 2.

| Rule | Result | Numbers |
|---|---|---|
| R1 retrieval | **pass** | R@3 0.269 [0.241, 0.301] vs blank 0.094 [0.076, 0.113]; real 0.293 |
| R2 3D back-translation | pass | BLEU 0.60 [0.47, 0.74] vs blank 0; real 0.60 |
| R2b 2D back-translation | **fail** | generated 1.67 [1.41, 1.95], **below** blank 2.14 [1.81, 2.46]; real 3.88 |
| R3 copying | pass | 0% flagged |
| R4 length slices | fail | head speed 1.35-1.42x (short, long); head variation 1.46x (short) |
| R5 one-word prompts | fail | top-5 0 / 258; real ASL Citizen 98.9% |
| R6 test signer 9 | pass | no regressions of validation-passing checks |

- Retrieval meaning is close to real (0.269 vs 0.293), up from 0.236 for the full prior.
- The more sensitive 2D instrument still reads generated motion as less informative than durations alone.
- Round 3 is not deployable.

**Why R5 is 0: the ASL Citizen states are mostly missing arms and hands (2026-10-23, 23:40 PDT).**
These were probes on 40 ASL Citizen val clips, conformed to How2Sign's layout as the prior trains on them, with the
validated recogniser `sign-recognizer-v2`:

| Input to the recogniser | top-5 |
|---|---|
| real clips | 0.75 |
| the unit conversion alone (no tokenizer) | 0.75 |
| round-3 tokenizer round trip, quantised or not, 1-4 stages | 0.03-0.05 |
| round trip, with only the hands replaced by the reconstruction | 0.48 |
| round trip, with only the body (arms) replaced | 0.08 |
| **round trip on the frames the real clip observes**, its neutral fill elsewhere | **0.78-0.80** |

- The median ASL Citizen clip has valid **body (arm) rotations on 10%** of its frames and hand rotations on 26-27%.
  These are webcam framings where the elbows and hands leave the MediaPipe pose.
- The recogniser learned to recognise signs from those few observed frames with a neutral fill.
- Complete motion, even a faithful reconstruction of the real clip, is out of its domain: 0.75 -> 0.05.
- So **R5 as declared cannot grade generated motion**, which is always complete. This is the generator-domain
  instrument check that the instrument-skepticism rule demands, and it was missing.
- The tokenizer is not the problem. Where the data exists, its reconstructions are recognised as well as the real
  clips.
- It also means the generator's 13,734 one-word training clips gave it arm supervision on about 10% of their frames.

**Round 3b design, declared 2026-10-23 23:48 PDT before any round-3b result.** It addresses R5 (one-word data
without arms), R2b and R4 (head activity), plus more data.

- **Data** (`round3b-mixed-v1`):

  | Corpus | Training clips | Notes |
  |---|---|---|
  | How2Sign multi-signer | 18,289 | SHA `dcdb6451`, unchanged |
  | YouTube-ASL r3b | about 75k segments, about 140 h after round 3a's unchanged filters | at 200 h captioned |
  | ASL Citizen one-word | 13,734 | SHA `50125638`, kept for hands and face; arm frames are masked as invalid |
  | **ASLLVD one-word** | **5,156** | 2,080 words, 6 signers, arms valid on 66% and hands on 98-100% of frames; publisher partitions; ASLLRP Sign Bank terms; SHA `29a605c4` |
  | ASLLRP plans | 1,652 | |

- **Tokenizer.** Round 3a's tokenizer (`tokens-rvq-d0b-r3a`) is reused only if both checks pass:
  1. the domain check passes on YouTube r3b and ASLLVD (every channel at most 1.5x How2Sign val);
  2. a **recognisability round trip** passes: on ASLLVD val, the recogniser's top-5 on tokenizer reconstructions
     (scored on the frames the real clip observes) is at least 0.9x its top-5 on the real clips.

  Otherwise a tokenizer is trained on the round-3b mix first, and must pass the same two checks.
- **Prior:** from scratch, same architecture and schedule as round 3a, at most 60,000 steps, early stopping
  (patience 5).
- **Sweep:** hands {1.1, 1.2, 1.3} x face {0.8, 1.0, 1.2} x guidance {1, 1.5, 2}, per validation signer, 300 clips each.
  - At most 3 concurrent evaluations, with expandable CUDA segments.
  - Any failed evaluation is rerun before the grid.
- **Post-processing grid:** head smoothing {0, 0.1, 0.2, 0.3} x head damping {1, 0.8, 0.6} x {none, pooled face
  calibration}, per signer and pooled (`--pool-group pool`).
- **Gate:** the envelope rule declared above, per signer and for the population.
- **Strict review:** R1, R2, R2b, R3, R4 and R6 (envelope) as declared.
  - R5's word list, recogniser and pass rule are unchanged.
  - Each one-word prompt is sampled at **the generator's own predicted duration** instead of a fixed 1.2 s. Real
    ASL Citizen word clips run 1.5-5.8 s with their rests, so a fixed 1.2 s was shorter than every real clip.

**Round 3b data built and checked (2026-10-24, 00:25 PDT).**
- **YouTube-ASL r3b** (SHA `55f7e0d3`): 200.7 h captioned, so 97,792 segments and **147.8 h** after round 3a's
  filters. 97,371 train and 421 monitor.
- **Mix `round3b-mixed-v1`** (SHA `13ad9797`): 188,752 records.
  - Training: How2Sign 18,289; YouTube 97,371; ASL Citizen 13,734; ASLLVD 5,156; ASLLRP plans 1,652.
- **Checked:**
  - How2Sign val, test and heldout_official are identical to round 3a's.
  - Each validation signer group's first 300 clips (the evaluated ones) are identical too, so round 3a's real-clip
    baselines carry over.
  - No sign id is in two partitions.
  - Only 2 YouTube training captions of more than two words equal a How2Sign evaluation sentence, both the generic
    "So what does that mean?".
  - All four sources' terms are recorded.
- Launched `runpod/fleet/round3b_all.sh` (smoke run, then the round) at 00:27 PDT.

**Round 3b data rebuilt and every state verified (2026-10-24, 01:30 PDT).**
- **What happened.** The smoke run's prior stopped on a state whose bytes no longer matched `manifest-r3b`.
  - A duplicate extraction started before the claim heartbeat. Its claim had been taken over at the 3 h stale age.
  - It finished at 01:01, after the manifest was built, and rewrote `feXU2ZMBdrI`'s 1,426 states and report.
  - The prefetch had skipped the mismatch silently.
- **Fixed (`1d046c7`):**
  - An extraction never writes over a converted video: states are written only after the "already converted?"
    check, with status `duplicate`.
  - Claims of converted videos are not refreshed.
  - The prefetch exits 3 on any missing or mismatching state, and round 3b stops on it.
  - The last claimless duplicate process was stopped.
- **Rebuilt from the then-consistent reports** (207 h captioned):
  - YouTube `manifest-r3b2` (SHA `f69ad828`): 100,454 segments, **151.8 h** after filters.
  - Mix `round3b-mixed-v2` (SHA `ad8a5876`): 191,414 records, partitions and evaluation clips identical to round 3a.
- **All 100,454 YouTube states were downloaded and SHA-verified** (0 mismatches) before round 3b was relaunched.

## ASLLRP attribution requirements (Carol Neidle, email 2026-10-24 11:14 PDT)

ASLLRP's permission for a non-commercial weights release (2026-10-23) stands. Before any distribution:
1. **Cite each individual dataset used**, with the citation given on its DAI download page:
   - the ASLLRP continuous signing corpus (2,123 plan utterances);
   - the RIT sentence collection (280);
   - ASLLVD via the Sign Bank (5,156 one-word training clips in round 3b, plus the recogniser's validation).

   These citations are to be copied verbatim from the DAI pages (login) before release.
2. **List the DAI and Sign Bank websites:** https://dai.cs.rutgers.edu/dai/s/dai and https://dai.cs.rutgers.edu/dai/s/signbank.
3. **Give the full ASLLVD citation:** C. Neidle, A. Thangali and S. Sclaroff (2012), "Challenges in Development of the American
   Sign Language Lexicon Video Dataset (ASLLVD) Corpus", 5th Workshop on the Representation and Processing of Sign
   Languages: Interactions between Corpus and Lexicon, LREC 2012, Istanbul, 27 May 2012.
4. **The attribution wording is revisited with ASLLRP** when results are ready to distribute.

`scripts/build_model_card.py` now carries the full citations, both URLs and the attribution paragraph, marked as
awaiting ASLLRP's review.

She also points to C. Neidle (2023), "Challenges for Linguistically-Driven Computer-Based Sign Recognition from
Continuous Signing for American Sign Language", arXiv:2311.00762. Her point: recognising signs within continuous
signing is a prerequisite for translation.
- Our sign-level checks today (R5) use isolated one-word prompts.
- Recognition of signs inside generated continuous signing is not measured. It is recorded as a limitation and a
  candidate check for a later round, using the ASLLRP segmented continuous-corpus signs from the Sign Bank.

## Speech input: audio -> English text -> ASL (owner's request, logged 2026-10-24 00:00 PDT; after the generator passes)

The deployed model takes English text (`TranslatorService.translate(text)`). To translate spoken English, such as a
video's soundtrack, a speech-to-text stage goes in front of it:
1. **Audio -> English sentences.** Either the repo's speech front-end (`signtranslator/speech/`: streaming
   features, alignment, decoding, latency model) or a standard ASR model (for example Whisper, licence checked). It
   outputs timed sentences and punctuation for the planner.
2. **Acceptance before integration**, declared when the stage is built:
   - word error rate on held-out English speech against a stated bar;
   - median and p95 streaming latency;
   - sentence segmentation checked against the generator's input limits (`max_text_chars`).
3. **Integration:** `TranslatorService.translate_audio(audio)` runs ASR, then the existing text path. The release spec
   pins the ASR model by SHA-256, like every other component.
4. **End to end:** the strict review's meaning rules run on ASR transcripts of spoken How2Sign-like prompts, and the
   human Deaf reviewers judge audio -> sign output as well.

This starts once a generator round passes its strict review. The text path is the product's core, and speech only
feeds it.

## Plan for the six items (order and gates)

| # | Item | Plan | Gate before it starts |
|---|---|---|---|
| 1 | More data | (c) ASL Citizen isolated signs (section above). (a) The full How2Sign (≈80 h): run the existing Sapiens fleet and two-view chain over the remaining sessions, plus the take-5 re-cut stage (B80). (b) ASLLRP DAI 2 corpus through the same pipeline, under a private R2 prefix with provenance. | v2b priors reviewed; DAI 2 login from the user for (b) |
| 2 | English → ASL structure | Train an English → (gloss with locus indices + NMM spans) planner on ASLLRP. Condition the motion prior on the planned gloss and NMM sequence instead of raw English. The existing semantic planner (docs/SEMANTIC_PLANNER.md) is the base. | Item 1b |
| 3 | Meaning-level evaluation | (a) Pose → text back-translator on our representation (Q1), frozen on the training partition. (b) A protocol for evaluation by Deaf ASL users: comprehension, naturalness, grammar; recruited and paid evaluators; done by people, not automated. | (a) after the v2b review; (b) needs the user to engage evaluators |
| 4 | Model scale and tuning | Larger prior, longer training, sweeps of guidance strength and duration on the larger corpus. Use the merged 2.2× faster step. | Item 1a |
| 5 | Avatar rendering | Retarget body, hands, face blendshapes and gaze onto a rigged signer (docs/AVATAR_RENDER.md). Face and gaze matter for grammar. | Any time; independent |
| 6 | Serving | Export (ONNX/TensorRT), cached text encoding, versioned checkpoints, an API with latency targets, monitoring (docs/DEPLOYMENT.md). | After items 3 and 4 show a model worth serving |

**Licence gate for release:** a release build lists every corpus in its training
provenance. ASLLRP (and any other research-only data) is either excluded or covered by
written permission.

## Execution order after the generator round (recorded 2026-10-21 06:20 UTC)

These start when the generator round's exit gate closes (docs/CONTINUOUS_TRAINING.md). Per the user's instruction of 2026-10-21, they proceed without waiting for input, under the same quality rules:
- every pod stops when its chain ends;
- every stage's log goes to R2;
- data stays in R2 under private prefixes.

1. **ASL Citizen acquisition** (item 1c). One small pod runs `acquire_resumable.py` for both files (≈ 46 GB, about 30 min on a datacentre link). Then:
   - verify the ZIP members;
   - extract monocular states;
   - map glosses.
   - **Status (2026-10-21 12:55 UTC).**
     - Acquired and verified: `ASL_Citizen.zip`, 45,924,134,223 B, SHA-256 `349c0e33…8eab`, re-read from R2.
     - Contents: 83,399 videos, 2,731 glosses, every row with an ASL-LEX code. Splits: train 40,154 (35 signers), val 10,304 (6), test 32,941 (11).
     - Extraction runs on 16 shards (`runpod/fleet/extract_asl_citizen.sh`, about 73 clips/min per 13.6-core pod).
   - **Smoke-test findings (fixed before the full run):**
     - The pod image lacked `libEGL`, and the extractor recorded the failure as a refusal of every clip. Now each worker probes the MediaPipe runtime at start and fails the job.
     - Webcam framing cuts the body above the hips: hips are visible in 2% of frames. The hip-based chest frame then invalidated every arm joint, so the body was available in 3 of 48 clips.
     - The opt-in `--upright-chest` prior takes the chest up axis from the camera vertical. It keeps chest yaw and roll and drops its lean. How2Sign is unchanged.
     - On 512 clips, the body is now available in 84%. 77% of clips have arm rotations in at least 20% of their frames (median 41%), and hands are seen in a median 38% of frames. Isolated-sign clips start and end at rest.
   - **Manifest:** `scripts/build_asl_citizen_manifest.py`.
     - Publisher partitions, with signer-disjointness checked.
     - Participants as pseudonyms.
     - Gloss and ASL-LEX code with status `publisher_label`.
     - Per-split quality counts.
   - **Done 2026-10-21 15:26 UTC.**
     - All 83,399 clips converted: no refusals, no unresolved errors.
     - Nine shards died at 13:29 on a transient R2 failure and were resumed after the retry fix.
     - Manifest: `derived/asl-citizen-v1/manifest-v1/manifest.json`, SHA-256 `708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6`, built on the Mac.
     - Clips with arm rotations in at least 20% of frames: train 77%, val 70%, test 79%. Median arm fraction 0.41, 0.35, 0.41.
     - Clips with hands in at least 20% of frames: 91–93% in every split.
     - Downstream uses filter on the per-clip `quality` counts.
   - **Gloss mapping:** 887 of the 2,731 glosses match SAPA's ASL-LEX phonology entries exactly. Normalised matching is not used, since it would merge variants such as `SOCCER2`. The ASLLRP Sign Bank join waits for DAI 2.
2. **Full How2Sign** (item 1a).
   - **Sapiens:** keypoints for every remaining clip. `runpod/pipeline.sh` without `SUBSET`; R2 claims skip the ≈ 13k frontal and side clips already done. About 57k clips at ≈ 40 clips/min on ten L40S pods: ≈ 24 h, ≈ $260.
   - **Take-5 sessions:** re-cut first (B80, `scripts/recut_how2sign_take5.py`).
   - **3D:** calibration and triangulation per session.
   - **Data:** manifest v2.
   - **Retrain:** the selected generator family and the retrieval evaluator on manifest v2.
3. **Meaning-level evaluation** (item 3a). The retrieval evaluator exists (`text_motion_retrieval.py`; R@3 ≈ 0.29 on real validation motion against chance 0.094 at 4k clips; it overfits early). Retrain it on manifest v2 with stronger regularisation. Score every generator with R-precision and keep their generated states for the human reviewers.
4. **DAI 2 / Sign Bank** (items 1b and 2).
   - **Access:** the user's account was activated on 2026-10-21. The user signs in; Claude never types passwords.
   - **Acquisition** (`~/.sign-translator-ops/dai_uploader.py`): downloads from the signed-in browser session into private R2 at `asllrp-dai2-v1/<collection>/`.
     - Each file is identified by its exact server size.
     - It is SHA-256 hashed while uploading, re-read from R2 and re-hashed, and written with a provenance record (source URL, version, licence).
     - The local copy is deleted once verified.
   - **Scope:** 121 current-version files, about 204 GB. The older archived releases (about 180 GB) are skipped: the site asks that the CSV and videos come from the same version. ASLLVD sign batch 17 is a 281-byte stub on their server.
   - **Done (2026-10-21 22:20 UTC):**
     - every annotation CSV and Excel file;
     - the ASLLRP continuous-corpus sign, utterance and source videos;
     - all RIT sentences files;
     - all RIT citation files.
   - **Remaining:** the ASLLVD videos (about 176 GB). Waiting for the user to sign in again, because the DAI session expired at about 22:05. Downloads then return HTTP 500 (`login_id` null).
   - **Sign Bank outputs** (`scripts/build_signbank_lexicon.py`, under `derived/signbank-v1/`):
     - lexicon: 4,037 main entries;
     - ASL Citizen gloss map, by tier: 977 exact, 17 variant, 147 letters, 421 letters-no-number (non-exact matches need review), 1,169 unmatched;
     - 2,410 ASLLRP and RIT utterances, holding 19,510 signs in time order.
   - **Utterance motion:** extracted on the Mac. The ASLLRP studio framing shows hips, elbows and wrists in every frame, so the upright-chest prior is not used.
   - **Gloss -> motion manifest** (`derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json`, SHA `f8c40aea…`): all 2,407 extracted
     utterances (2,127 ASLLRP, 280 RIT; 3.1 h) joined to their gloss sequences. Signer-disjoint: train 1,655, val 413 (Jonathan),
     test 339 (Ben); 3 rows without a readable signer left out. Median hand coverage 81% of frames (p10 46%), face 100%.
     Some glosses carry classifier descriptions in backslashes (`ICL:X-over-thumb\eating out of a bowl\`); the planner must
     normalise them before use.
   - **DAI 2 transfer complete (2026-10-22 06:20 PDT):** all 121 files (200.8 GB) are in private R2 under
     `asllrp-dai2-v1/`, each re-read and re-hashed after upload with a provenance record (source URL, SHA-256, licence).
     That includes the ASLLVD citation-form source and sign videos (93 files, about 172 GB). Their motion extraction is next.
   - **English translations (acquired 2026-10-22):** the DAI 2 utterance search export (2,407 utterances: English translation and
     SignStream rough-gloss tiers) is in R2 (`asllrp-dai2-v1/asllrp-continuous/utterance-search-2026-10-22.json`, SHA `cc9102d8…`).
     `scripts/build_asllrp_english_gloss.py` wrote the planner's pairs (`derived/asllrp-dai2-v1/english-gloss-v1/pairs.json`,
     SHA `91c9f83f…`): 2,403 English -> gloss pairs (train 1,652 / val 412 / test 339, same signer split), each with its
     grammatical markers mapped to the signs they span: topic/focus 982, negation 600, conditional/when 428, wh 325,
     rhetorical 249, yes-no 238, relative clause 43.
5. **Avatar and serving** (items 5 and 6) follow, with a model selected on R-precision and the realism gates.
6. **Human Deaf reviewers** (on their way): they evaluate the stored generated states and the rendered avatar. Their findings drive fine-tuning.

## Smoothness rule R7 and head-translation smoothing (declared 2026-10-24, 17:55 PDT, before the round-3b verdict)

**Finding.** The owner saw the first round-3b avatar render's head shake. Measured on 30 validation sentences
(sweep h1.3-f0.8, c03, guidance 2) against the same sentences' real clips:
- The share of motion energy at ≥ 6 Hz for head translation was 6.2% vs 0.6% real. The tokenizer's round trip alone gives 2.9%.
- For head rotation it was 3.6% vs 0.8%.
- Body, hands and root were at or below real.
- In absolute amplitude (`motion_metrics.high_frequency_rms`), head translation is 2.0× real and head rotation 1.05×. The excess is broadband above 3 Hz, not a token-rate line.

The post-processing grid smooths and damps only head *rotation*; head translation passes through untouched. No gate
or review check measures jitter, because the articulation ratios compare mean speeds.

**R7 smoothness (new rule, blocks release like R1–R6).**
- For body, both hands, head rotation, head translation and root translation, the median over clips of the generated ≥ 6 Hz amplitude must be at most 1.5× the median of the same sentences' real clips.
- It applies after the candidate's own post-processing, in every validation signer group, and again on test.
- Script: `scripts/smoothness_check.py`.

**Head-translation smoothing.**
- `postprocess.smooth_vector_channel` smooths head translation with the same centred window as the head rotation.
- On the sample above, 0.2 s on both head channels gives at most 0.73× real on every channel, with slow motion kept (unit tests).

**Procedure for round 3b.**
- The pod's chain runs as declared.
- If its strict review passes, the selected candidate gets head-translation smoothing equal to its head-rotation smoothing (at least 0.2 s).
- The strict review is then rerun with it, plus R7. Both must pass before release.
- If the review fails, round 3c adds translation smoothing to the grid and R7 to the review from the start, and treats the tokenizer's head-translation reconstruction (2.9% vs 0.6%) as a training target.

## Round 3b result: FAILED (2026-10-24, 23:45 PDT)

- **Gate (envelope, multi-signer): 46 of 47.** Selected sweep h1.2-f1.0 at guidance 1.5, head smoothing 0.1 s, damping 0.8, face calibration on, Fréchet 226. The one failure is c03 right-hand variation, 0.664 against the 0.70 floor. Head checks pass in every group.
- **Strict review:**
  - R1 retrieval PASS: 0.221 [0.193, 0.248] vs blank 0.094, real 0.293.
  - R2 PASS: BLEU 0.78 vs blank 0.
  - R3 PASS: no copies.
  - **R2b FAIL:** 1.77 vs blank 2.14 (real 3.88).
  - **R4 FAIL:** head speed 1.45 (short sentences) and 1.33 (long); short-sentence body variation 1.38.
  - **R5 FAIL:** 0 of 111 words recognised (real 98.8%).
  - **R6 FAIL:** test brow-raise 2.63 against an envelope maximum of 2.58.
- **R5 root cause (measured on the Mac):**
  - The generated single words have plausible durations (2.3–2.9 s), but the right hand moves at 29–77 °/s against 125–254 °/s real. The recogniser reads the real clips correctly, so it is not at fault.
  - The prior's single-word targets come from two corpora. The 13,734 ASL Citizen clips have the body tracked on 14% of frames, and the tokenizer's neutral fill takes their median right-hand speed to **0 °/s**. The 5,156 ASLLVD clips keep 135 °/s.
  - The prior learned that a single word means static hands. Round 3's 0 of 258 has the same cause.
- **R2 vs R2b:** both translators are weak readers, and R2 rates generated motion above real (0.78 vs 0.60). R2b's instrument is checked before round 3c treats it as a target.

## Round 3c design (declared 2026-10-25, 00:05 PDT, before any round-3c result)

1. **Data:** mix `round3c-mixed-v1` (SHA `004bf6596a23…`, 125,015 training clips). It is round 3b's mix without ASL Citizen's motion: How2Sign, the YouTube r3b2 snapshot (unchanged, so data is not a second variable), ASLLRP plan lines and ASLLVD. ASL Citizen stays only in the recogniser, an evaluation instrument covered by its consent. That also removes it from the generator's provenance.
2. **Prior:** same tokenizer (r3a), architecture, schedule and 60k-step cap as 3b, plus `--corpus-weight asllvd-text-v1=3`, so single words are about 11.5% of the samples.
3. **Sweep:** hands {1.2, 1.25, 1.3} × face {0.8, 1.0} × guidance {1, 1.5, 2}, the hands and face settings 3b found best, plus 1.25 for c03's right-hand variation.
4. **Grid:** head smoothing {0.1, 0.2, 0.3} × damping {1, 0.8, 0.6} × face calibration. Head-translation smoothing equals the head-rotation smoothing (at least 0.1 s). The grid runs on CPU pods.
5. **Gate:** the round-3b envelope gate, plus R4's sentence-length slices as gate checks (per slice, head and body speed and variation in [0.7, 1.3]), plus R7 smoothness, so the selection cannot pick a candidate that the review then fails on those.
6. **Strict review:** R1–R7 as declared (R7 per 2026-10-24 17:55), and the R6 test confirmation once. Every rule must pass.
7. **Operations:** the GPU pod runs only GPU stages; the sweep uses two GPU pods; the review's memory uses `expandable_segments`.

## Round 3c amendment: tokenizer stage (declared 2026-10-25, 00:20 PDT, before any round-3c result)

**Finding.** Real How2Sign validation clips (c01/c03/c05) passed through round 3a's tokenizer and back, measured on the frames each clip observed (`scripts/tokenizer_fidelity_check.py`):
- head rotation speed 1.58–1.93× real;
- hand speed 0.71–0.80× and hand variation 0.67–0.74×;
- body speed 0.67–0.72×;
- R2b on 200 clips: round trip 3.05 [2.27, 3.90] vs blank 1.74 [1.04, 2.32]. The intervals overlap.

This is round 3b's remaining motion failure profile before any prior is involved: head too active (R4), right-hand variation too low (the gate). Head rotation is observed on 100% of frames, so this is the tokenizer's quantisation, not gap filling. Round 3b's prior job was stopped during its smoke run's data check; nothing was trained.

**Amendment.**
- Stage A trains three tokenizer candidates in parallel on `round3c-mixed-v1`, with round 3a's recipe otherwise (hidden 512, code dimension 128, 40,000 steps maximum, early stop):
  - s6: 6 residual stages;
  - s8: 8 residual stages;
  - c1024: 4 stages, 1024-entry codebook.
- Each candidate must pass all three checks:
  1. the domain check (YouTube, ASLLVD);
  2. the recognisability round trip (ASLLVD val);
  3. the new fidelity check (300 val clips): body, both hands and head rotation speed and variation ratios in [0.85, 1.15], and the R2b round trip's interval entirely above blank's.
- The passing candidate with the fewest codes per step is used (c1024 16, s6 24, s8 32).
- If none passes, the round stops before the prior (TOKENIZER-FAILED) and is redesigned.
- Everything else in the round-3c design is unchanged.

## Round 3c amendment 2: a velocity-matched candidate (declared 2026-10-25, 02:45 PDT, before any tokenizer verdict)

**Audit of the step-6000 checkpoints (s8, s6)** on real val clips (`scripts/tokenizer_audit.py`, How2Sign c01/c03/c05):
- head speed 1.8–3.2×;
- hand speed 0.67–0.86×;
- right-hand variation 0.59–0.66×;
- head-translation jitter 2.5–5.4×.

They are no better than round 3a's tokenizer, although their validation loss is lower. The tokenizer's loss is per-frame error, plus velocity and acceleration only through the body and hand skeleton. Head rotation, head translation and the hand rotations carry no velocity term, so minimising per-frame error damps fast motion and leaves noise on near-still channels.

**Amendment:**
- `motion_rvq.velocity_matching_loss` (opt-in `--velocity-weight`) adds a relative frame-to-frame motion error per channel.
- A fourth candidate, **s8v** (the s8 recipe plus velocity weight 1.0 on body, both hands, head rotation and head translation), trains on its own GPU pod (`runpod/fleet/round3c_tokenizer_v.sh`) and is judged by the same three checks.

**Precedence:**
- If any of s6, s8 or c1024 passes, it is used.
- Only if all three fail and s8v passes does s8v's pod train the prior, with the same recipe.
- If no candidate passes, the round stops before the prior.

## Round 3c amendment 3: s8v3 (declared 2026-10-25, 03:25 PDT, before any tokenizer verdict)

The audit of s8v at step 8000 (How2Sign c01/c03/c05):
- head speed 0.97–1.02×, head-translation jitter 0.80–0.93×, now in band;
- right-hand speed 0.81–0.83× and variation 0.73–0.80×;
- body speed 0.64–0.71× and variation 0.55–0.70×, still damped. YouTube and ASLLVD are at 0.83–0.94×.

A fifth candidate, **s8v3**, uses s8v's recipe with velocity weight 3 on body and both hands (1 on head rotation and translation). It runs on its own pod (`runpod/fleet/round3c_tokenizer_v3_all.sh`) with the same three checks. Precedence: c1024, s6, s8, then s8v, then s8v3. A candidate trains the prior only if every earlier one failed and it passes.

## Round 3c amendment 4: s8v5 (declared 2026-10-25, 04:30 PDT, before s8v3's and the original candidates' verdicts)

**s8v failed its domain check**, and correctly. Its per-frame pose error is far above round 3a's:
- How2Sign: body 8.3° vs 3.5°, head 9.8° vs 2.6°;
- YouTube and ASLLVD: body 26–27° vs 4°.

At weight 1, the velocity term, which is relative and of order 1, outweighed the per-frame loss (of order 0.01) by 10–100×. Velocity matching ignores a constant offset, so the tokenizer kept the motion and lost the pose. Speeds and variation were close (head 0.93–1.13, YouTube and ASLLVD 0.83–0.99) while positions drifted. The deep audit (`tokenizer_audit.py`) now also reports absolute per-frame error, so a speed gain cannot hide a pose error.

The per-batch normaliser also diverged on near-still batches (validation loss 14.6 at step 22000). Fixed in 418fb9e: a running energy scale.

**New candidate s8v5** (`runpod/fleet/round3c_tokenizer_v5_all.sh`):
- 8 stages, the stable velocity matching at weight 0.05 on the same five channels;
- the same three checks;
- precedence after c1024, s6, s8, s8v and s8v3.

## Round 3c amendment 5: noise check and a declared amplitude fallback (2026-10-25, 05:00 PDT, before any pending fidelity verdict)

**Measurement check.** Real clips smoothed over 0.1 s, against themselves:
- speed drops to 0.54–0.78 (body 0.54–0.71); variation stays at 0.84–0.98;
- so raw speed is noise-sensitive.

Recomputed with both sides smoothed (How2Sign, 60 clips):

| tokenizer | body speed | hand speed | variation |
|---|---|---|---|
| r3a | 0.80 | 0.74–0.76 | 0.72–0.94 |
| s8 at step 20k | 0.80 | 0.78–0.84 | 0.73–0.92 |
| s8v at step 20k | 0.77 | 0.80–0.83 | 0.69–0.95 |

No verdict would change, so **the declared fidelity rule stands unchanged**. Hand and body amplitude is genuinely damped by about 20% in every candidate.

**Fallback, declared before the verdicts of c1024, s6, s8, s8v3 and s8v5.** It applies only to a candidate whose every failure is speed or variation of the body or hands below the band, while head rotation is in band and the domain and recognisability checks pass.
- **Correction:** a per-channel excursion gain after decoding, `R' = R̄ exp(g log(R̄ᵀ R))` (`postprocess.damp_rotation_channel` with a factor above 1). For each channel, g is fitted on 300 *training* clips as real variation over round-trip variation, capped at [1, 1.5].
- **Rule:** the fidelity check is rerun on validation with the gains applied and must pass in full (all ratios in [0.85, 1.15], R2b above blank). The recognisability check must also still pass.
- **Downstream:** the gains become part of that tokenizer's decoder and are applied to every generated state downstream (sweep, grid, review), recorded in the release spec.
- **Precedence:** a candidate that passes without gains outranks one that needs them.

## Round 3c amendment 6: s8v6 (declared 2026-10-25, 05:30 PDT, before s8's and s8v5's verdicts)

**Verdicts so far:**

| candidate | domain | recognisability | fidelity |
|---|---|---|---|
| c1024 | pass | pass (0.996) | fail: head 2.04, hand speed 0.74–0.76, hand variation 0.64–0.67, body speed 0.68 |
| s6 | pass | pass (1.0) | fail: head 1.90, hands 0.70–0.82, body speed 0.68 |
| s8v | **fail** (pose error) | — | — |
| s8v3 | **fail** (pose error; validation loss diverged to 102 at step 25000, best at 20000) | — | fail, damping only: body 0.76/0.84, hands 0.74–0.79; R2b round trip 3.23 vs blank 2.09 |

**s8v5 audit at step 8000** (How2Sign):
- head speed 1.09–1.30, head-translation jitter 0.72–0.82;
- body pose error 6.8–9.9° (s8 at step 20k: 5.7–8.2°);
- hand variation 0.64–0.76.

**New candidate s8v6:** velocity weight 0.3 on head rotation and head translation, 0.05 on body and both hands, otherwise s8v5's recipe. The head barely moves, so its per-frame loss cannot see jitter and needs a stronger velocity term than the articulators. Precedence: after s8v5.

## Round 3c amendment 7: s8v7 (declared 2026-10-25, 06:00 PDT, before s8's, s8v5's and s8v6's verdicts)

The s8v5 audit at step 16000 (How2Sign):
- head speed 1.13–1.53, up from 1.09–1.30 at step 8000;
- body pose error 6.3–8.6°, close to s8's;
- hand variation 0.71–0.77.

At weight 0.05 the head drifts out of band as training continues. **s8v7** uses velocity weight 1.0 on head rotation and translation (the weight that put s8v's head in band) and 0.05 on body and hands (where s8v's weight 1 cost pose accuracy). Precedence: after s8v6.

## Round 3c amendment 8: quantiser dropout, s8v8 and s8v9 (declared 2026-10-25, 07:00 PDT, before s8v6's and s8v7's verdicts)

**s8v5 verdict:** rejected, on two checks.
- **Domain:** ASLLVD body 1.507× How2Sign, against a 1.5 limit.
- **Fidelity:** body speed 0.70, head 1.19, hand variation 0.77–0.78.
- **R2b round trip:** 3.16, against blank 2.09; the intervals overlap.

**Diagnosis of the damping, run on s8v6 at step 19000 (all diagnostics only, no verdict):**
1. **The shrinkage is genuine, not noise.** On How2Sign, body speed is 0.65–0.77× at every percentile from p10 to p99. Smoothing both sides by 0.1 s leaves body 0.74, hands 0.73–0.77.
2. **It is not signer generalisation.** How2Sign *training* clips (seen signers) give the same ratios as validation: body 0.64, hand variation 0.69–0.76.
3. **It is not the quantiser.** Decoding the continuous latents, with the quantiser bypassed, gives the same ratios as 8 stages: body 0.63, hands 0.70–0.79. Fewer stages shrink further: 4 stages give 0.67–0.70, 1 stage 0.51–0.57.
4. **Reference noise is corpus-specific:**
   - How2Sign torso joints (Pelvis, Chest, thighs, upper arms) have negative lag-1 velocity autocorrelation (−0.19 to +0.03), so they are jitter-dominated. Their speed halves under 0.1 s smoothing.
   - How2Sign fingers are smooth (+0.31 to +0.56).
   - YouTube and ASLLVD fingers are jitter-dominated (−0.22 to +0.03).
   - The fidelity rule is unchanged; these numbers are recorded for the next design.

**Hypothesis.** `quantizer_dropout = 0.5` trains half of all batches on 1–8 random stages, with a decoder not told how many. The decoder that minimises the loss for coarse codes shrinks toward the mean, and the same weights then shrink full and continuous latents too. The shrinkage gets worse as stages are removed, which fits.

**Candidates:** s8v6 at step 19000 (EMA weights, SHA-256 `b52fa16b…ac83`) is fine-tuned with its velocity weights unchanged.
- Settings: lr 1e-4, warmup 200, up to 10000 steps, validation every 500, patience 5.
- **s8v8:** quantiser dropout 0.
- **s8v9:** quantiser dropout 0.1, so coarse stages keep some standalone use.
- Judged by the same three checks and rules, smoke-tested first.
- `train_motion_tokens.py --init-key` now warm-starts a tokenizer whose config differs only in the loss-only fields `quantizer_dropout` and `velocity_weights`.

**Precedence:** s8v6, s8v7, s8v8, s8v9. A candidate that passes without gains outranks one that needs the amendment-5 gains.

## Round 3c tokenizer results (2026-10-25; s8v7 added at 08:50 PDT)

Every judged candidate is rejected:

| candidate | domain | recognisability | fidelity (body speed, hand variation) | R2b round trip (blank 2.09) |
|---|---|---|---|---|
| s8v5 | fail (ASLLVD body 1.507) | — | fail (0.70, 0.77–0.78; head 1.19) | 3.16 |
| s8v6 | pass (1.466) | pass (1.008) | fail (0.66, 0.77) | 3.17 |
| s8v8 | fail (ASLLVD body 1.508) | pass (0.988) | fail (0.65, 0.77) | 3.46 |
| s8v9 | pass (1.451) | pass (0.988) | fail (0.67, 0.77–0.78) | 3.28 |
| s8v7 (verdict 08:50 PDT) | pass (1.417) | pass (1.004) | fail (0.61, 0.66–0.67; head 1.02) | 3.21 |

**Round 3c result: the tokenizer stage FAILED.** All nine candidates were rejected, so the round stops before the prior, as declared. s8v7 ran its full 40000 steps without a spike, but its head weight of 1.0 damps the hands most of all.

None qualifies for the amendment-5 gains: in each, R2b also fails, which is not a speed or variation failure.

**What the damping is, from the diagnostics recorded in amendment 8 and these:**
1. **Not a bug.** The representation is lossless: real clip → unit space → raw gives every ratio exactly 1.0.
2. **Not EMA weights.** The raw weights damp the same.
3. **Not the network's capacity.** Overfitting s8v6 on 8 How2Sign clips for 1200 steps gives body speed 1.05, variation 0.99, hands 0.93–0.98 and body error 1.6°. So the network can reproduce full amplitude.
4. **It is underfitting**, a regression toward the mean across 125k clips, and it shrinks steadily with training. For s8 at steps 8k, 20k and 34k:
   - hand variation 0.69–0.70, 0.73–0.79, 0.81–0.84;
   - hand speed 0.72–0.77, 0.81–0.89, 0.88–0.94;
   - body variation 0.81, 0.83, 0.88;
   - head speed 1.78, 1.64, 1.50.
5. **The cap on training length is the late spikes.**
   - s8: validation loss 0.16 → 1.15 at step 35k.
   - s8v6: 0.27 → 443 at step 30k, in head translation, face and blink.
   - Both came as the cosine learning rate fell toward zero (1.2e-5 and 4.6e-5), while the codebooks kept updating at full EMA rate with random dead-code re-seeding.
   - s8v6 at step 30k with its live weights explodes on rare validation clips: head translation error 0.375 m, a face value of 3.26.
6. **Body speed over the torso measures jitter.** `scripts/joint_reliability.py` on 200 How2Sign training clips (report `round3d/joint-reliability-how2sign-train.json`):
   - jitter-dominated (lag-1 velocity autocorrelation ≤ 0.1, speed about halved by 0.1 s smoothing): Pelvis, Chest, upper arms, thighs;
   - signal: forearms, wrists and every finger joint.

## Round 3d design (declared 2026-10-25, 08:20 PDT, before any round-3d result)

**Tokenizer changes:**
- **Codebooks follow the learning rate** (`--codebook-follows-lr`). The EMA rate is the full configured rate through warmup, then lr/peak; no dead-code re-seeding below a quarter of the peak.
- **Longer training with patience 10.**
- **Velocity weight 0.3** on head rotation and head translation only. That is s8v6's head term, which put head speed in band (1.055); body and hands get none, because velocity terms cost ASLLVD/YouTube pose accuracy.

**Candidates**, each on its own pod with a smoke run first (`runpod/fleet/round3d_tokenizer_{d1,d2,d3}.sh`):
- **d1:** s8 at step 34000 (EMA weights, SHA-256 `36f2c84e…06e1`, the best round-3c reconstruction) warm-started. lr 1.5e-4, cosine over 60000 steps, warmup 500.
- **d2:** d1 with How2Sign sampled at weight 2. How2Sign is the damped corpus and the one the fidelity check judges.
- **d3:** from scratch, the s8 recipe over 100000 steps at lr 3e-4.

**Acceptance:** round 3c's three checks and limits, with two measurement corrections.
- **Fidelity speed ratios** are taken over the reliable joints of the joint-reliability report (body: forearms and wrists; hands: all). All-joint speeds are still reported; variation ratios use every joint.
- **R2b runs on 1000 How2Sign validation clips** instead of 300. At 300, even real clips (4.09 [3.16, 5.08]) barely clear blank (2.09 [1.44, 2.74]).

No other rule changes. The domain check stays as declared, even though s8v8 showed that it penalises a better How2Sign reconstruction.

**Precedence:** among passing candidates, the one with the highest R2b round-trip BLEU. The round-3c prior recipe (60000 steps, ASLLVD weight 3) then trains on it. The evaluation chain and the gate are those of round 3c, under round-3d run names.

**If no round-3d candidate passes:** stop before the prior and report.

## Round 3d amendment 1: d4, gradient-spike skipping (declared 2026-10-25, 09:35 PDT, before any round-3d verdict)

Both warm-started candidates went unstable at mid learning rate:
- **d1:** validation loss 1.95 at step 8k and 13.2 at 10k; best is still step 6k (0.242).
- **d2:** 77.8 at 8k and 4500 at 9k; best is step 5k (0.205).

d3, trained from scratch, is stable so far (0.411 at step 5k).

d2's training log shows the trigger:
- The gradient norm is normally 3–6, and every step is clipped to 1.0.
- At step 7000 it jumped to 6998, with a head-translation velocity error of 7.9 and face error of 0.28.
- Clipping bounds the step size, but the spike step still moves the weights in a corrupted direction.
- Head-translation data have no glitches: How2Sign's largest frame jump is 3 cm, and no other corpus carries translations.

**d4:** d3's recipe plus `--skip-grad-spikes 8`.
- A step whose gradient norm exceeds 8× the median of the last 200 accepted steps (after step 500) applies neither the optimizer update nor the EMA update.
- More than 100 consecutive spikes stop the run as diverging.
- d4 is judged by the same rules and joins the same precedence (the passing candidate with the highest R2b round trip).

## Round 3d amendment 2: domain check against a fixed reference (declared 2026-10-25, 09:45 PDT, before d2's, d3's and d4's verdicts)

**d1 verdict:** rejected under the declared domain rule. ASLLVD right hand came out at 1.58 against the 1.5 limit (left hand 1.44, body 1.34); YouTube passed. d1's recognisability and fidelity checks are still running.

The rule itself conflicts with the fidelity check. Its denominator is the candidate's own How2Sign error, which fidelity demands be driven down.

| errors (unit space) | How2Sign body | How2Sign right hand | ASLLVD body | ASLLVD right hand |
|---|---|---|---|---|
| s8 (passed: 1.24, 1.29) | 4.16 | 1.67 | 5.15 | 2.14 |
| d1 (failed: 1.34, 1.58) | 3.96 | 1.32 | 5.29 | 2.09 |

d1 encodes ASLLVD hands better than s8 did and fails only because its How2Sign error fell 21%. s8v8 failed the same way.

**Amended rule (for d2, d3 and d4):**
- Each channel's denominator is the larger of the candidate's own How2Sign error and round 3c's s8 reference error (`round3c/tokenizer-domain-check-s8.json`).
- Every channel of every new corpus must stay within 1.5× that denominator.
- So a better How2Sign reconstruction can no longer fail the check, and a corpus encoded much worse than How2Sign still fails as before.

*Correction (09:50 PDT, before any d2–d4 result).* The first wording fixed the denominator at s8's errors alone. A plumbing run on d1 (40 clips, diagnostic only) showed that form failing head rotation at 2.2. The head velocity term raises head per-frame error on every corpus alike, How2Sign included (5.60 against s8's 3.10). That is a uniform accuracy shift, not the domain question, so the larger of the two errors is the denominator.
- The check runs from the Mac orchestrator after each pod's checks, with the same clips, limit and corpora (`tokenizer-domain-check-<n>-a2.json`), via `tokenizer_domain_check.py --fixed-reference-key`.

d1 stays judged by the rule declared before its result.

## Round 3d amendment 3: amendment 5's gains fallback applies to round 3d (declared 2026-10-25, 10:00 PDT, before d2's, d3's and d4's fidelity verdicts)

**d1's fidelity (diagnostic, d1 is already rejected on the domain check):**
- R2b on 1000 clips now separates: round trip 3.56 [3.08, 4.05] against blank 1.86 [1.54, 2.18].
- In band: head speed 1.10, head variation 0.92, body variation 0.86, left-hand speed 0.87.
- Out of band, all below: reliable-joint body speed 0.846, hand variation 0.73–0.75, right-hand speed 0.82.
- Recognisability 1.004.

The design's "round 3c's three checks and limits" includes round 3c's amendment-5 fallback. Stated explicitly here, it applies unchanged. A candidate qualifies when:
- every fidelity failure is speed or variation of the body or hands below the band;
- head rotation is in band;
- R2b is above blank;
- the domain check (amendment 2) and the recognisability check pass.

For a qualifying candidate:
1. Per-channel gains are fitted on 300 How2Sign *training* clips (`fit_tokenizer_gains.py`, cap [1, 1.5]).
2. The fidelity check (1000 validation clips, reliable-joint speeds) and the recognisability check are rerun with `--gains-key`, and both must pass in full.
3. The accepted gains are then written to `decoder-gains-<step>.json` beside the checkpoint, and every generated state downstream uses them (`load_generator`).

**Precedence:**
- A candidate passing without gains outranks any that needs them.
- Within each class, the highest R2b round trip wins.
- The gained checks run on a GPU pod (`runpod/fleet/round3d_gains.sh`), started by the orchestrator.

## Round 3d note: d4's divergence stop (2026-10-25, 11:59 PDT)

**What happened:** d4's spike guard stopped the run at step 31230, after more than 100 consecutive spikes (173 skipped in all).
- Its best checkpoint, step 31000 at validation loss 0.2227, was validated before the blow-up.
- The gradient norm then rose to 5098 and 17892 while every update was skipped.
- The network's weights were therefore frozen, and the only state still changing was the RVQ codebooks: EMA update and dead-code re-seeding run in the forward pass, whatever the optimizer does.
- This is direct evidence that the codebook dynamics, not optimizer steps, drive the tokenizers' spikes. It is the design input for any later tokenizer: freeze or slow the codebooks when the gradient norm spikes, or update them only on accepted steps.

**How d4 is judged:** amendment 1 says the run "stops as diverging", not that the candidate is rejected. Every candidate is judged on its best checkpoint, early-stopped runs included. So d4's three checks run on step 31000 with the same rules (`CHECKS_ONLY=1 runpod/fleet/round3d_tokenizer_d4.sh`).

## Round 3d tokenizer result (2026-10-25, 14:36 PDT): d2 with decoder gains

No candidate passed without gains. All three that qualified for the amendment-3 fallback passed with them:

| candidate | gains (body, left hand, right hand) | gained fidelity (all ratios in [0.85, 1.15]) | gained recognisability | gained R2b (blank 1.86 [1.54, 2.18]) |
|---|---|---|---|---|
| **d2** (s8 warm start + 5000 steps, How2Sign ×2) | 1.17, 1.28, 1.27 | pass: body 1.00 / 1.04, hands 0.94–1.15, head 1.08 / 0.93 | 0.980 | **3.16 [2.74, 3.57]** |
| d3 (from scratch, best at step 39000) | 1.21, 1.45, 1.41 | pass | 0.971 | 2.97 [2.54, 3.42] |
| d4 (from scratch with spike skipping, best at 31000) | near the 1.5 cap | pass | — | 2.78 |

**Choice:**
- d2 wins on R2b, per the declared precedence. It also needs the smallest gains.
- Its accepted gains are at `training-runs/tokens-rvq-d0b-r3d-d2/decoder-gains-step-00005000.json`. `load_generator` applies them to every generated state.
- Its prior, `tokens-rvq-d0b-prior-r3d-d2g`, started speculatively at 11:10 PDT while d3 and d4 were still being judged, and becomes round 3d's prior (step 17000, validation loss 3.59).
- The evaluation chain (sweep, grid, review and gate) follows automatically once the prior finishes.

**Recorded observations:**
- Applying gains lowers the R2b back-translation BLEU for every candidate (d3 3.48 → 2.97, d2 3.37 → 3.16): the 2D back-translator reads damped motion more easily, an instrument property worth knowing at the gate.
- d2's deep audit with gains (`round3d/tokenizer-audit-d2-gained.json`):
  - How2Sign hands at 0.91–1.02 variation and 1.03–1.24 speed (c05's left hand 1.24);
  - no added jitter (jitter ratios 0.83–1.12);
  - one carried flaw: c05's browInnerUp is 45× its nearly static real variation. Every candidate adds brow motion there, an R6 risk for the gate.

## Round 3d amendment 4: the sweep's face temperatures (declared 2026-10-25, 17:22 PDT, before any round-3d evaluation result)

The sweep's face temperatures {0.8, 1.0} were chosen when round 3b's head moved 1.45× too much. Head rotation is part of the face token part, so face temperature sets the head's motion.

**Diagnostics** (d2's prior at step 32000, signer group c03, 24 validation clips, guidance 1.5, d2's gains, no post-processing):

| hand / face temperature | head speed, variation | headshake / nod rate vs real | brow-raise z (real 2.29) |
|---|---|---|---|
| 1.0 / 1.0 | 0.42, 0.47 | — | — |
| 1.25 / 1.0 | 0.63, 0.48 | 0 / 0 | 2.59 |
| 1.25 / 1.25 | 1.25, 1.12 | 1.75× / 3.1× | 2.76 |
| 1.25 / 1.5 | 2.01, 1.12 | 5.1× / 6.7× | 1.97 |

At hand 1.0 the hands are far too still (variation 0.38–0.50); at 1.25 they are near (0.83–0.84).

- With this tokenizer the head is too still at face temperatures up to 1.0, and 0.8 can only make it stiller.
- The grid can only smooth or damp the head (factors ≤ 1), so it cannot rescue a still head.
- The declared hand range {1.2, 1.25, 1.3} is right.

**Amendment:**
- Face temperatures {1.0, 1.1, 1.2} replace {0.8, 1.0} (`R3_FACES`). Hand temperatures, guidance, the grid, the envelopes, the gate and the strict review R1–R7 are unchanged.
- That makes 27 sweep evaluations, run on 3 GPU pods (`R3_NPARTS=3`), and 675 evaluation files after the grid.

This widens the declared search where the diagnostics show the optimum lies. It loosens no rule. The strict review still judges the selected setting, R6 included on the held-out test states.

## Round 3d sweep: over-active at every declared temperature (2026-10-26, 07:21 UTC; all times UTC from here)

Sweep results so far: 15 of 27 at 07:08, guidance 1.5, ratios of generated to that signer group's real motion:

| setting | c01 body speed / variation | c05 body speed / variation | c05 hand speed | c03 body speed / variation |
|---|---|---|---|---|
| hand 1.2 / face 1.0 | 1.26 / 1.41 | 1.28 / 1.41 | 1.63–1.68 | 0.97 / 1.12 |
| hand 1.2 / face 1.2 | 1.42 / 1.62 | 1.50 / 1.66 | 1.75–1.77 | 1.13 / 1.31 |
| hand 1.25 / face 1.2 | 1.54 / 1.72 | 1.61 / 1.70 | 1.85–1.95 | — |

**What the gate will do with these:**
- Its articulation bands are [0.7, 1.3], widened by real cross-signer ratios (round 3b's cross-signer baseline, the same real clips). Body variation, for example, is about [0.70, 1.30] for judges c01 and c05.
- The post-processing grid adjusts only the head (smoothing, damping) and face calibration.
- So no round-3d setting can pass the body and hand checks on c01 and c05. The declared chain still runs to its verdict, because its strict review measures R2b, R5 and R6 on generated motion.

**Why the sweep overshoots:**
- The hand temperatures {1.2, 1.25, 1.3} were tuned to offset round 3b's damped tokenizer.
- d2's accepted decoder gains (body 1.17, hands 1.27) already restore amplitude, so the two compound.
- c03, a more energetic signer, sits near range. c01 and c05 do not.

**Lower-temperature diagnostics** (final prior, face 1.0, 24 clips per group, guidance 1.5):

| group, temperature | body speed / variation | hand speed | hand variation | head speed / variation |
|---|---|---|---|---|
| c01, 1.0 | 0.73 / 0.86 | 0.98–0.99 | 0.77–0.85 | 0.95 / 0.82 |
| c01, 1.1 | 1.02 / 1.00 | 1.29 | 1.00–1.13 | 1.22 / 0.81 |
| c05, 1.0 | 0.80 / 0.69 | 0.94–1.09 | 0.73–0.86 | 1.31 / 1.37 |
| c05, 1.1 | 1.04 / 0.94 | 1.19–1.29 | 0.87–1.06 | 1.84 / 2.04 |

**Sweep pod imbalance:** the round-3d split gave one pod every c03 evaluation, and c03's clips are longer, taking about 2 h per evaluation. A helper pod computed c03's three hand-1.3 evaluations with identical flags, and the sweep script's "already stored" check skips them.

## Round 3e design: an evaluation-only round (declared 2026-10-26, 07:21 UTC, before any round-3e result)

**What stays the same:** round 3d's tokenizer (d2 with its accepted decoder gains) and its prior (`tokens-rvq-d0b-prior-r3d-d2g` at step 60000), with no retraining. Also unchanged:
- guidance {1, 1.5, 2};
- the post-processing grid (head smoothing {0.1, 0.2, 0.3} × damping {1, 0.8, 0.6} × face calibration, head-translation smoothing);
- the cross-signer envelopes and the gate;
- the strict review R1–R7, including the held-out test set.

**What changes:**
- **Sweep temperatures:** hands and body {1.0, 1.05, 1.1} (`R3_HANDS`), face {0.9, 1.0} (`R3_FACES`). That is 18 evaluations on 3 GPU pods.
- **Balanced split:** signer groups are rotated per setting (`R3_ROTATE=1`), so each pod gets a share of every group.
- **Location:** results go under `round3e/` (`R3_D`).

Every other rule is as declared for rounds 3c and 3d. The round-3d verdict is recorded when its chain finishes; it does not gate round 3e.

## Round 3e result: the exit gate passed, the strict review FAILED (2026-10-26, 14:39 UTC)

**Gate: passed, 95 of 95 checks**, the first setting of any round to do so.
- Selected setting: hand and body temperature 1.1, face 1.0, guidance 1, head smoothing 0.1 s, face calibration (Fréchet 381.5).
- Only 1 of 324 settings passed. Its nearest rivals failed only c01's nod rate (guidance 1.5 and 2) or brow-raise fraction (without face calibration).

**Strict review on that setting:**

| rule | result |
|---|---|
| R1 retrieval | pass: R@3 0.186 [0.161, 0.212] vs blank 0.094 (real 0.294) |
| R2 text back-translation | pass: BLEU 0.80 vs blank 0.0 |
| **R2b 2D back-translation** | **fail: generated 1.60 [1.32, 1.87] below blank 2.14 [1.81, 2.46]** (real 3.88) |
| R3 no copying | pass: copy rate 0.0 |
| R4 length slices | pass |
| **R5 single-sign words** | **fail: top-5 1.35% [0, 3.2%]** (real 99%, 111 words) |
| **R6 test confirmation** | **fail** |
| R7 smoothness | pass, validation and test |

R6 in detail:
- The test signer passed all 11 articulation checks within its envelope.
- How2Sign's official held-out set had too still a head (speed 0.62, variation 0.65) and too few headshakes (0.30×) and nods (0.40×).
- The R7 checks were flagged as test regressions within R6.

**The motion statistics are now right. The content is not.** R2b and R5 failed the same way in round 3b, so the tokenizer and temperature work of rounds 3c–3e did not touch them.

**Diagnosis:**
1. **The text encoder.** The prior conditions on raw UTF-8 bytes (`text_to_motion.encode_texts`) through a 4-layer transformer trained from scratch on about 125k captions. It has no lexical semantics, and a rare or single word gives almost no usable signal. The prior's validation token accuracy is 24.5% (stage 1: 41.5%).
2. **Lexical coverage.**
   - Of R5's 150 candidate words, only 94 appear as isolated ASLLVD signs in training, at about 2.6 clips each among 125k clips.
   - The other 56 appear only inside How2Sign sentences (median 12 sentences per word).
   - ASL Citizen, the large isolated-sign source, was removed in round 3c because its extracted states were mostly invalid (round 3b's R5 diagnosis).

**Guidance is not the fix.** R2b on 300 round-3e validation clips (selected setting, saved generations, the same post-processing):

| | R2b BLEU |
|---|---|
| real | 3.82 [3.08, 4.61] |
| blank | 2.20 [1.68, 2.71] |
| generated, guidance 1 | 1.44 [1.02, 1.82] |
| generated, guidance 1.5 | 1.80 [1.28, 2.28] |
| generated, guidance 2 | 1.83 [1.35, 2.29] |

Every guidance stays below blank. The content failure lies in the prior.

## Round 4 pilot: a pretrained text encoder for the prior (declared 2026-10-26, 15:08 UTC, before any round-4 result)

**Hypothesis.** The byte-level text encoder, trained from scratch, gives the prior almost no lexical meaning. That would explain R2b below blank and R5 near zero, which motion statistics cannot fix.

**Unchanged:** round 3d's tokenizer (d2 with its accepted decoder gains), the mix, and the prior recipe (60000 steps, ASLLVD ×3, locus conditioning).

**Arms:**
- **A (control):** round 3d's byte-encoder prior `tokens-rvq-d0b-prior-r3d-d2g`, already trained on the same 60000-step schedule. Its checkpoints at matching steps are the control.
- **B:** a frozen `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) feeding the trainable text layers (`--text-model`).
- **C:** the frozen `google/flan-t5-base` encoder (Apache-2.0), likewise.

**Pilot measure** (`scripts/pilot_content_check.py`). At step 20000 of each arm, at round 3e's selected sampling setting (hands and body 1.1, face 1.0, guidance 1.5, head and head-translation smoothing 0.1 s):
- **meaning:** R2b-style BLEU on 300 fixed How2Sign validation clips, generated against real and blank;
- **lexicon:** R5-style top-5 on the review's word list.

**Decision rule:**
- An arm is promising if its generated BLEU exceeds the control's at step 20000 and its lexicon top-5 is at least the control's.
- The promising arm with the highest generated BLEU continues to 60000 steps; the other is stopped.
- That full prior is then judged by the full declared chain (sweep, grid, gate, strict review).
- If neither arm is promising, the text-encoder hypothesis is rejected and the next lever is lexical data (a re-extracted ASL Citizen).
- No rule of the gate or the review changes.

## Round 3d result, recorded (2026-10-26, 15:36 UTC): FAILED

**Gate:** the best of 324 settings passed 78 of 95 checks: hand 1.2, face 1.0, head smoothing 0.1 s, damping 0.8, face calibration, guidance 2. It failed body variation and hand speed for c01 and c05, the over-activity found at 07:21.

**Strict review on that setting:**
- R2b: fail, 1.62 below blank 2.14.
- R4: fail.
- R6: fail.
- R5: passed narrowly. Top-5 was 2.25% [0.45%, 4.5%] against blank 0, at guidance 2, where round 3e's 1.35% [0, 3.15%] at guidance 1 just failed.
- Both rounds sit around 2% against real signing's 99%. That is consistent with the content diagnosis, not progress on it.

## Round 4 pilot result, recorded (2026-10-26, 20:28 UTC): HYPOTHESIS REJECTED

Pilot checks at step 20000 (`round4/pilot-<arm>-step20000.json`; same 300 validation clips, same 150 words × 2 samples, same setting for every arm):

| Arm | Meaning BLEU [95% CI] | Lexicon top-5 | Verdict |
|---|---|---|---|
| A control (byte encoder) | 1.406 [0.98, 1.82] | 0.0167 (5/300) | reference |
| B MiniLM-L6 | 1.541 [1.02, 2.05] | 0.0133 (4/300) | not promising (lexicon below control) |
| C flan-T5-base | 1.666 [1.16, 2.16] | 0.0033 (1/300) | not promising (lexicon below control) |
| blank | 2.09 [1.44, 2.74] | 0 | |
| real | 4.09 [3.16, 5.08] | 0.99 (review R5) | |

**Validation loss** tracked the control within 0.01 at every step for both arms (step 20000: MiniLM 3.567, flan-T5 3.558, control 3.562).

**Decision, as declared:** neither arm is promising, so neither continues to 60000 steps. Both prior pods were stopped at 20:28 UTC with checkpoints to step 23000–24000 in R2, and the pods were deleted after their logs were verified in R2.

**Reading:**
- The pretrained encoders move meaning in the right direction, by +0.14 and +0.26 BLEU. Neither move is significant, and every arm stays below blank.
- Lexicon is at the floor in all three arms (1–5 hits in 300 against real signing's 99%). The differences between arms are a few samples and carry no information.
- A frozen English encoder alone does not teach the prior what a sign looks like. The binding constraint is lexical data: only 94 of the 150 review words have any isolated clips in the mix, about 2.6 clips each.
- The next lever is therefore lexical data, as declared. The next round is designed and declared before any of its results. flan-T5 remains a candidate encoder for it, as the arm with the best meaning point estimate.

## Round 5: lexical data within its terms, and a right-dominant recogniser (declared 2026-10-26, 20:55 UTC, before any round-5 result)

**Findings that set this round** (2026-10-26, 20:30–20:55 UTC):
1. **The R5 recogniser is not mirror-invariant.** On 240 ASL Citizen validation clips, `sign-recognizer-v2` scores top-5 0.767 on the stored states and 0.304 on the same states mirrored (`signtranslator.pose.mirror`).
2. **ASL Citizen stores the signing hand on the left in about 60% of clips.** These are left-handed participants or mirrored webcams: 184 of 300 sampled train clips have more left-forearm tracking. How2Sign and the generator sign right-handed (37 of 40 How2Sign clips have the faster right hand). So the instrument that grades the generator's one-word signs learned partly from left-dominant signing. The low tracking fractions found earlier (median arm 0.40, hand 0.39) are mostly the resting hand out of frame, not failed tracking of the signing hand.
3. **Consent.** ASL Citizen's consent covers sign language recognition, so its motion stays out of the generator, as round 3c decided. It may train the recogniser.
4. **Lexical data within the ASLLRP terms** (already part of the generator's sources):
   - The RIT citation-form signs of DAI 2 were never extracted. They hold 12,197 tokens, 592 of them for 86 of the 150 R5 words, against 249 from ASLLVD train.
   - ASLLRP's continuous signing has 19,510 annotated sign tokens, 683 of them for 50 R5 words in the training signers.

**A. Instrument: `sign-recognizer-v3`.**
- **Recipe:** v2's exactly (same arguments, ASL Citizen plus the ASLLVD train signers, 24 fps, augmented), except that ASL Citizen comes from the canonical manifest (`scripts/canonicalize_asl_citizen.py`). That manifest is right-dominant: left-dominant clips are mirrored by the exact mirror, which `tests/test_pose_mirror.py` checks against mirrored landmarks and reflected forward kinematics. Each clip is trimmed to its active signing ±0.2 s.
- **Validation:** v2's three checks, unchanged (blank top-1 ≤ 2%, no-hands top-1 ≤ 15%, ASLLVD val signer top-1 ≥ 25%), plus two that keep it at least as good as v2:
  - canonical ASL Citizen val top-5 ≥ 0.70 (v2: 0.729 on its own manifest);
  - ASLLVD val signer top-1 ≥ 0.58 (v2: 0.608).
- **If v3 passes:** it is the R5 and pilot-lexicon instrument from round 5 on. Its reliable glosses and the R5 word list are re-derived by the unchanged rules (top-1 ≥ 0.8 over ≥ 3 val clips; reliable glosses that occur as words in How2Sign training sentences). If it fails, v2 stays.
- Every round-5 lexicon number, control included, is read by the same instrument.

**B. Generator data: mix `round5-mixed-v1`.** It is `round3c-mixed-v1` unchanged plus:
- **RIT citation signs**, one record per annotated token:
  - text: the main-entry gloss, lowercased, with variant numbers and punctuation removed (ASLLVD's convention);
  - partition: signer-disjoint, one RIT signer for validation, fixed in the builder;
  - weight: ×3, like ASLLVD.
- **ASLLRP in-context sign segments** from the gloss manifest's training utterances:
  - lexical signs only;
  - the annotated frames ±0.1 s, at least 0.2 s long;
  - text as above; weight ×1.

ASL Citizen motion is not in the generator.

**C. Tokenizer:** d2 with its accepted gains, unchanged. Before any prior trains, an instrument check: the tokenizer round trip of RIT validation clips must keep at least 0.8 of the real clips' recogniser top-5 (v3, or v2 if v3 is not ready). If it fails, the prior does not start and the cause is diagnosed first.

**D. Prior pilot.** Round 4's recipe (60,000 steps, ASLLVD ×3, locus conditioning) on `round5-mixed-v1`:
- **A5:** the byte text encoder.
- **C5:** frozen flan-T5-base, the round-4 arm with the best meaning point estimate.
- **Control:** round 4's control (byte encoder, `round3c-mixed-v1`) at step 20000, re-measured with the round-5 instrument.
- **Pilot measure:** `scripts/pilot_content_check.py` at step 20000, round 3e's sampling setting, with the round-5 instrument.
- **Decision rule:**
  - An arm is **promising** if three things hold:
    - its lexicon top-5 is at least 0.05 above the control's;
    - the paired bootstrap 95% interval of that difference (resampling words) excludes 0;
    - its meaning BLEU is not below the control's.
  - The promising arm with the higher lexicon continues to 60,000 steps; the other is stopped and deleted. The winner is then judged by the full declared chain: sweep, grid, gate, and the strict review with the round-5 R5 instrument.
  - If neither arm is promising, the hypothesis "isolated lexical data within our terms fixes the lexicon" is rejected. The next lever is structural: a gloss-planned path (the English → gloss planner plus a per-sign motion lexicon), designed and declared separately.
- **Unchanged:** every other gate and strict-review rule.

**Cost and pods.** Two CPU pods now (ASL Citizen canonicalisation, RIT extraction). Then one GPU pod for the recogniser, and two for the priors once the mix and the tokenizer check are ready: at most 5 pods. Every pod is deleted after use, once its log is verified in R2.

**Round 5 amendment 1 (2026-10-26, 20:59 UTC, before any round-5 result):**
- **The ASLLRP in-context sign segments leave the generator mix.** Their signs are coarticulated fragments: median 0.234 s (recogniser v2's in-context diagnostic), so about 0.43 s with the ±0.1 s margins. Isolated citation forms run about 1.5 s with their holds. As one-word records for the same English words, the fragments would pull the prior's duration for one-word prompts toward a fragment's length. RIT's citation signs are the clean source.
- **RIT clips are canonicalised to right-dominant** by the ASL Citizen rule (`scripts/canonicalize_asl_citizen.py`): dominant side by tracked forearm and hand frames; left-dominant clips mirrored by the exact mirror. They are not trimmed: the DAI sign clips are already cut to the sign.
- **The mix** is `round3c-mixed-v1` plus RIT citation one-word records (×3, the text builder's cap of 6 training clips per word). Nothing else changes.

**Round 5 amendment 2 (2026-10-26, 23:00 UTC, before any round-5 result; no instrument has graded anything):**
- **The per-clip dominance rule was wrong for ASL Citizen.** Within a single signer it mirrored between 52% and 100% of the clips with a tracked arm (all but two signers). A signer's handedness and recording do not change between clips, so `canonical-v1` mixed left- and right-dominant versions of the same people. Recogniser v3 trained on it reached validation top-5 0.72 at step 8000, against v2's 0.78 at the same step.
- **v3 is stopped.** Its run (`training-runs/sign-recognizer-v3`, stopped at step 8500, no validation report) is not an instrument.
- **New rule: dominance per signer.** All of a signer's clips are mirrored if that signer's left forearm is tracked on more frames, summed over their clips.
  - Hand frames are left out: the resting hand at the bottom of a webcam frame is detected through most of a clip while its forearm is not.
  - ASL Citizen: 50 signers mirrored, 2 kept (78,897 and 4,502 clips). This fits mirrored webcams and about 4% left-handed participants.
  - Trimming and the generator flags are unchanged. They now follow the signer's dominant forearm, and a clip where it is never tracked is kept whole.
- **New ASL Citizen manifest:** prefix `derived/asl-citizen-v1/canonical-v2`.
- **New instrument: `sign-recognizer-v3b`.** v2's recipe on canonical-v2, under round 5's unchanged instrument checks and rule.
- **RIT** is studio video with both forearms in view, so tracking counts do not show its signers' dominance. Its per-signer side is the side whose wrist moves faster: mean speed in the canonical skeleton, over up to 100 clips per signer. All of a signer's clips are mirrored when it is the left.
- **Amendment 2, RIT detail (23:05 UTC, before any RIT result):** wrist speed was rejected as the RIT cue before use. On ASLLVD's right-dominant studio signers it differed by only 5–20% between sides and called one of six left. RIT's per-signer side is instead the side whose wrist is raised higher: the median over up to 100 clips of each clip's 90th-percentile wrist height above the shoulder (`scripts/signer_dominance_motion.py`). On ASLLVD it picks the right for all six signers, by 0.05–0.19 in median.

**Round 5 amendment 3 (2026-10-26, 23:58 UTC, before any round-5 result; no prior has trained):**
- **The tokenizer check passed, but the data underneath it failed.** The ratio was 1.13 on 219 RIT val clips; the RIT states themselves are not usable:
  - hands are tracked on a median 20–27% of frames, against about 100% for ASLLVD;
  - in the conformed validation states, hands are valid on 2–4% of frames;
  - the recogniser reads only 14% top-5 on the real clips.
- **Cause.** RIT is a wide 1280×720 studio shot, so the hands are small (about 40 px) and often blurred. The extractor's hand crops (0.9 shoulder widths around each wrist) cut the fingers off.
- **Wider crops fix it.** On three RIT clips, crops of 1.8 shoulder widths find both hands on 69%, 98% and 94% of frames, against 12%, 67% and 39% at 0.9.
- **The two pilot priors were stopped and deleted during setup (23:56 UTC), before any step.** Nothing trained on the hand-less states.
- **Re-extraction:** RIT is re-extracted with `--hand-scale 1.8` into `derived/asllrp-dai2-v1/rit-citation-sign-videos-v2`. Its manifest, signer sides, canonical states, text records and mix are rebuilt as v2: `round5-mixed-v2`, which is round 3c plus RIT v2, nothing else.
- **A data gate before any prior trains** (new, declared now), on RIT validation:
  - the median clip's frames with both hands tracked must be at least 0.6;
  - the recogniser's real top-5 must be at least 0.30;
  - the tokenizer ratio of 0.8 stays.
  - If the gate fails, RIT does not enter the mix.

**Round 5 amendment 4 (2026-10-27, 00:46 UTC, before any instrument result):**
- **v3b trailed v2 at the same steps as well.** Validation top-5 was 0.286 at step 2000 and 0.446 at step 3000, against v2's 0.374 and 0.592.
- **The trim, not the mirror, is the visible cost.** v2 scores 0.721 on 240 original val clips and 0.683 on the same clips trimmed and mirrored back. The trim takes the median clip from 60 to 41 frames.
- **The instrument must read generated motion,** which keeps its rest before and after the sign. So trimming is the wrong domain for the recogniser. It stays only in the generator-quality flags.
- **New instrument candidate: `sign-recognizer-v3c`.** v2's exact data and recipe, with the per-signer reflection only (no trim).
  - Records keep their stored states and carry `"mirror": true` for every clip of a left-dominant signer (`scripts/build_mirror_flag_manifest.py`). `corpus_manifest.load_sample` mirrors those clips on load, after channel masking. A test checks that loading equals `mirror_state`.
  - Validation and checks are unchanged (round 5 A).
- **v3b is stopped and is not an instrument.** The instrument is v3c if it passes, otherwise v2.

## New lexical sources: conditions and plan (2026-10-27, 02:06 UTC)

The owner pointed to five sources and approved MS-ASL. Each one's conditions, as read from its own pages and files:

| Source | What it holds | Terms (as stated by the source) | Use here |
|---|---|---|---|
| **PopSign ASL v1.0** (Georgia Tech) | 250 signs, 200,686 phone videos, 47 signers, 1.1 TB; per-sign tar files by split (`signdata.cc.gatech.edu/data/popsign_v1_0/{game,non-game}/{train,val,test}/<sign>.tar`, about 2.1 GB per training tar) | "Popsign v1.0 is licensed under CC BY 4.0" | Generator and recogniser, with attribution. About 30 of the 150 R5 words. |
| **ASL-LEX 2.0** (OSF `zpha4`) | `signdata.csv` with 60+ lexical and phonological properties for 2,723 signs, plus `signdataKEY.csv`; about 10 example videos | Database CC BY 4.0 (body text; the site footer says CC BY-NC 4.0). "The sign reference videos are © ASL-LEX.org." | The lexical database only: gloss mapping, frequency and phonology for the gloss-planned path. The videos are not taken. |
| **Sem-Lex** (BU; github.com/leekezar/SemLex) | 91,148 isolated videos, 3,149 signs, 41 deaf signers, aligned with ASL-LEX and Sign Bank. Pose files "will be available soon". | Download links come after agreeing to the terms in a Google Form. Participants consented to sharing "for the purpose of developing sign language technology". | **Needs the owner to submit the form** (personal details and acceptance of terms). |
| **WLASL v0.3** (cloned) | 2,000 glosses, 21,083 instances. Only 5,135 instances (3,250 videos) are on YouTube; the rest are on dictionary sites (signingsavvy, handspeak, aslpro, spreadthesign, asldeafined, signstock, aslsearch). 113 R5 words, 1,303 instances. | C-UDA 1.0: "solely for Computational Use"; trained models are unrestricted "Results". The provider gives no warranty of rights in the videos. Pre-processed videos are available through a Google Form that accepts their terms of use. | The YouTube subset through the approved home route. The dictionary-site videos and the form **wait for the owner's decision**. |
| **MS-ASL** (Microsoft, `MS-ASL.zip`, 1.9 MB of JSON) | 1,000 classes, 25,513 clips cut from 7,213 YouTube videos, with times and boxes. 61 R5 words, 1,562 clips. | C-UDA 0.1 (computational use). Approved by the owner (2026-10-27). | YouTube downloads through the approved low-rate home route (no circumvention), then extraction on pods. |

**Sequencing:**
- Round 5 stays as declared (RIT only), so its result is not confounded.
- The new sources are acquired and extracted now, in parallel, into private R2 prefixes with their terms in each manifest.
- They enter the generator in a round 6 that is declared before any of its results. PopSign and MS-ASL clips are canonicalised per signer by the round-5 rule.

**Acquisition status (2026-10-27, 02:12 UTC):**
- **ASL-LEX 2.0:** `signdata.csv` and `signdataKEY.csv` are in R2 (`acquisition/2026-10-17-v1/asl-lex-2.0/`), with provenance, licence and both required citations.
- **PopSign v1.0: probed and held.** One validation tar (zebra, 42 clips) is in R2 with provenance, and 8 clips were extracted.
  - The phone selfie framing is a close-up: the face fills a 1944×2592 portrait frame, and the hands are at or beyond the bottom edge.
  - With the upright-chest prior, forearms are tracked on 0–10 of 54–72 frames per clip and hands on 18–43%.
  - The motion the generator must learn is largely out of frame. Bulk acquisition (~525 GB of training tars) is held, pending another use such as a hand-shape or face model. `scripts/acquire_popsign.py` and `runpod/fleet/acquire_popsign.sh` are ready.
- **MS-ASL:** `MS-ASL.zip` is in R2 with provenance. `scripts/acquire_msasl.py` (home download plus pod extraction of each labelled clip window) is ready. YouTube refused the very first home request with "Sign in to confirm you're not a bot" (02:10 UTC). By the no-circumvention rule the downloader stopped; the gate holds 2 h before any retry, and no cookies, sign-in or proxies are used.
- **WLASL:** the YouTube subset faces the same refusal. The dictionary-site subset (about 16,000 instances on signingsavvy, handspeak, aslpro, spreadthesign and others) and the authors' pre-processed-video form both wait for the owner.
- **Sem-Lex:** the form waits for the owner.
- **WLASL dictionary sites** (owner-approved 2026-10-27, 02:14 UTC; C-UDA). `scripts/acquire_wlasl_sites.py` runs on the Mac: robots.txt is respected, each host gets one request every 3 s with an honest user agent, a host that keeps refusing is stopped, and nothing is circumvented.
  - **Disallowed by robots.txt, so skipped:** handspeak.com (2,211) and signingsavvy.com (2,668).
  - **Not fetched:** aslpro.com is Flash (1,736). aslsignbank.haskins.yale.edu fails its TLS certificate (hostname mismatch) and its paths are "protected_media" (1,071); certificate checks are not turned off.
  - **Fetched:** the other seven hosts, about 8,260 instances.
  - **Next:** `after_wlasl.sh` packs per-host zips into R2 with provenance and extracts them on one CPU pod (`runpod/fleet/extract_wlasl.sh`).
  - `WLASL_v0.3.json` and `C-UDA-1.0.pdf` are in R2 with provenance.

**Round 5 amendment 5 (2026-10-27, 02:55 UTC, before any reading by the round-5 instrument):**
- **RIT v2 failed the declared data gate, so it stays out of the round-5 mix.** On the median validation clip, both hands are tracked on at most 0.485 of frames, against 0.6.
  - For the record, and not as a gate: the wider crops raised the dominant hand's tracking from 0.12 to 0.71 (median max(L,R)) and the right hand's from 0.06 to 0.60.
  - The both-hands measure was declared before the result and is kept as declared. RIT is reconsidered for round 6 under a gate declared then, and that gate will say openly that the 0.71 is already known.
- **Without RIT, round 5's arms would train on exactly round 4's data.** A5 is the round-4 control's recipe; C5 is round 4's arm C. So no new prior is trained.
- **Instead, the round-5 decision rule (D) is applied to the round-4 checkpoints at step 20000,** each read by the round-5 instrument (v3c if it validates, otherwise v2):
  - MiniLM (`tokens-rvq-d0b-prior-r4-minilm`) and flan-T5 (`tokens-rvq-d0b-prior-r4-flant5`) are the arms;
  - the byte prior `tokens-rvq-d0b-prior-r3d-d2g` is the control.
- **A promising arm resumes training** from its latest checkpoint (step 23000–24000, same run, config and manifest) to 60000, then goes through the full declared chain.
- If neither is promising, round 6 (new lexical data) is next.
- **WLASL audit (2026-10-27, 04:26 UTC).** All 8,258 downloads are packed into seven zips in R2, each verified by re-read. The local copies were deleted at 04:09. Checking content signatures before trusting the extraction showed that aslsearch.com's 1,875 "videos" are 114-byte HTML pages: the site is dead and answers every URL with a page. They are excluded; the pod refused all of them. **Usable: 6,382 real MP4 clips** from six hosts:

  | Host | Clips |
  |---|---|
  | signstock | 1,968 |
  | asldeafined | 1,833 |
  | spreadthesign | 1,584 |
  | startasl (s3) | 619 |
  | aslbricks | 217 |
  | elementalaslconcepts | 161 |

  The downloader now rejects any payload without a video signature (`not_a_video`).

**Round 5 instrument result (2026-10-27, 06:25 UTC): `sign-recognizer-v3c` fails a declared check, so v2 stays the instrument.**

| Check | v3c | v2 | Rule |
|---|---|---|---|
| ASL Citizen val, top-5 (10,303 clips; v3c reads them mirrored) | 0.718 | 0.729 | ≥ 0.70: pass |
| ASLLVD val signer, top-1 (411 clips) | **0.511** | 0.608 | ≥ 0.58: **fail** |
| Blank / no hands / hands only, top-1 | 0.0003 / 0.012 / 0.240 | 0.0003 / 0.015 / 0.226 | v2's rules: pass |
| Reliable glosses | 469 | 474 | — |
| In-context ASLLRP, top-1 (diagnostic) | 0.063 | 0.083 | — |

- Best checkpoint step 32000 (early stop at 40000). The pilot checks therefore read the round-4 checkpoints with v2.
- **Audit: an anomaly.** ASLLVD signers are right-dominant (wrist-height check: all six signers), so a right-dominant ASL Citizen should have helped on ASLLVD, yet the cross-signer score fell 10 points. The leading hypothesis is that **reflecting a stored state is not the same as estimating a real right hand.**
  - The mirror is exact on the stored rotations (`tests/test_pose_mirror.py`).
  - But MediaPipe's hand and pose estimators are not mirror-equivariant: handedness-specific models, depth conventions. A mirrored left hand therefore carries the estimator's left-hand biases into the right-hand channel.
  - v2 kept ASL Citizen and ASLLVD on their native sides, and v3, v3b and v3c all lagged it.
- **Consequences:**
  - (a) Generated motion is still graded by v2. Its bias, real left-dominant ASL Citizen, is a known limitation of R5, declared in round 5.
  - (b) **Round 6 does not mirror generator data.** Signers whose dominant side is left are left out instead.
  - (c) A cleaner right-dominant instrument would need re-estimation from horizontally flipped video, not mirrored states. ASL Citizen re-extraction on flipped frames is noted as a later option.

## Round 6: more isolated signs within their terms (declared 2026-10-27, 06:29 UTC, before any round-5 verdict or round-6 result)

**Goal:** lexicon. Single words recognised clearly above the control at step 20000, as in round 5 D, with the same instrument: `sign-recognizer-v2`, the round-5 outcome.

**Data.** Mix `round6-mixed-v1` = `round3c-mixed-v1` (unchanged) plus two corpora.
- **RIT citation signs v2** (`rit-sign-manifest-v2`, 1.8-shoulder hand crops).
- **WLASL dictionary-site clips** (`wlasl-sites-v1`, 6,381 clips, C-UDA; training split only).
- Both enter as one-word records through the text builder: glosses → English, variant numbers dropped, at most 6 training clips per word. Each is sampled at weight 3, like ASLLVD.

**No mirroring** (lesson of the v3c audit). A signer whose dominant side is left is left out.
- **Dominant side:** RIT and WLASL are studio or dictionary video with both forearms in view, so each signer's side comes from raised-wrist height (`signer_dominance_motion.py`). RIT is already known: all 12 signers right.

**Per-clip quality filter** for generator use, declared now. The corpus medians are already known (RIT v2: right hand 0.60, dominant hand 0.71; WLASL first two hosts: dominant hand 0.58–0.60), so the threshold is not blind. It is set before any model result.
- Keep a clip if its right forearm (with the upper arm) is tracked on at least 0.5 of its frames **and** its right hand on at least 0.5.
- This replaces round 5's corpus-level both-hands gate, which penalised one-handed signs.

**Tokenizer check, per new corpus,** on its validation clips, before any prior trains:
- the round trip keeps at least 0.8 of the real clips' v2 top-5;
- the real clips' v2 top-5 is at least 0.30.

A corpus that fails stays out. If both fail, the round stops.

**Prior arms.** Round 4's recipe (60000 steps, locus conditioning, tokenizer d2 with its gains):
- **B6:** the byte encoder.
- **C6:** frozen flan-T5-base.
- **Control:** round 4's control at step 20000 (byte, `round3c-mixed-v1`), read by v2 with per-word hits (round 5's file).

**Decision at step 20000** (round 5 D's rule, unchanged):
- An arm is promising if its lexicon top-5 is at least 0.05 above the control's, the paired-by-word bootstrap 95% interval excludes 0, and its meaning BLEU is not below the control's.
- The best promising arm continues to 60000 steps and then goes through the full chain.
- If neither is promising, the next lever comes from the within-architecture list: an auxiliary word-identity loss, one-word curriculum sampling, per-prompt guidance and duration. It is declared before use; there is no structural change, as the owner asked.
- **Round 6 build note (2026-10-27, 06:36 UTC, before any round-6 result).** The first WLASL text manifest (`wlasl-text-r6`) came out empty: the gloss → English rule accepted only uppercase glosses, and WLASL's are lowercase English. It was caught before the mix was written. The rule now accepts lowercase and spaced glosses, with uppercase labels unchanged (checked on ten cases and the builder's test). WLASL is rebuilt as `wlasl-text-r6b`; the empty manifest is unused. RIT's text records (`rit-text-r6`: 5,295 training records, 1,546 words) are unaffected.
- **Round 6 data checks (2026-10-27, 06:42 UTC), as declared:**
  - **RIT v2** (`rit-text-r6`, 141 val clips): real v2 top-5 0.553 (it was 0.14 with the old crops), round trip 0.553, ratio 1.00. **It enters.**
  - **WLASL** (`wlasl-text-r6`, 279 val clips): real top-5 0.685, round trip 0.677, ratio 0.989. **It enters.**
    - The first attempt could not run: WLASL's English labels have no Sign Bank gloss to map through. It was not a failed check, and the two prior pods that had started were deleted before any step.
    - Labels are now matched by exact uppercase spelling (`--label-upper`), which pairs 279 of 478 val clips; variants such as ABOUT1/ABOUT2 are not guessed.
  - Mix `round6-mixed-v1`: 143,042 records, sha `3c8024e3`. That is round 3c's 134,606 + RIT 5,949 + WLASL 2,487, nothing else.
  - The priors B6 and C6 launch with `--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3` (plus ASLLVD ×3 as before).

## Round 5 result, recorded (2026-10-27, 07:17 UTC): NOT PROMISING (rule D, instrument v2)

The round-4 checkpoints at step 20000, re-measured with per-word hits. Meaning scores reproduce round 4 exactly (deterministic seeds).

| Arm | Meaning BLEU | Lexicon top-5 | Paired difference vs control [95% CI] | Promising |
|---|---|---|---|---|
| Control (byte) | 1.406 | 0.0167 (5/300) | — | — |
| MiniLM | 1.541 | 0.0133 | −0.0033 [−0.0167, 0.0100] | no |
| flan-T5 | 1.666 | 0.0033 | −0.0133 [−0.0300, 0.0033] | no |

- Neither arm reaches the +0.05 gain, and neither interval excludes 0; both meaning scores are at or above the control's.
- The text-encoder lever is closed for this data.
- Round 6 (declared 06:29 UTC, both priors training since 07:05–07:07) is the next test: more isolated signs, RIT v2 and WLASL.
- **Round 6 pilot note (2026-10-27, 10:56 UTC, before any round-6 pilot number).** The first B6 pilot check stopped at load: the check was hard-wired to round 3c's manifest, and the round-6 priors are bound to `round6-mixed-v1`. It produced no numbers; the C6 check was stopped and its pod deleted before it ran.
  - The check now takes the prior's training mix (`--mix-key/--mix-sha256`). Round 3c's and round 6's mixes yield the same 300 How2Sign validation clips and the same 150 R5 words (checked), so the arms stay comparable with the round-5 control file.
  - Both checks were relaunched at 10:56 UTC.

## Round 6 result, recorded (2026-10-27, 11:38 UTC): NOT PROMISING (rule D, instrument v2)

| Arm | Meaning BLEU [95% CI] | Lexicon top-5 | Paired difference vs control [95% CI] |
|---|---|---|---|
| Control (round 4, byte, round3c mix) | 1.406 | 0.0167 (5/300) | — |
| B6 (byte, round6 mix) | 1.460 [0.89, 2.10] | 0.0067 (2/300) | −0.010 [−0.027, 0.003] |
| C6 (flan-T5, round6 mix) | 1.543 [1.01, 2.04] | 0.000 (0/300) | −0.017 [−0.033, −0.003] |

- Both priors were stopped and their pods deleted at 11:38, after the logs were verified in R2. Checkpoints are kept, to step 25000 (B6) and 24000 (C6).
- **Audit.** About three times the clean isolated-sign data (RIT v2 and WLASL, both with real top-5 above 0.55 through the tokenizer) moved lexicon not at all. C6 is significantly below the control. So the bottleneck is no longer data quantity.
- **Next, before any new training lever:** a diagnostic on the round-6 checkpoints.
  - Can the prior produce words it **was trained on**: the most frequent one-word training texts, read by v2?
  - Are its one-word durations in the range of real citation signs?
  - If trained words also fail, the fault is in how one-word conditioning is learned or sampled, not in vocabulary coverage, and the next lever targets that.
- **Round 6 audit, first findings (2026-10-27, 12:20 UTC):**
  - **B6 cannot sign its own training words.** On the 60 most frequent one-word training texts the recogniser knows, v2 reads real training clips at top-5 0.925, but B6's samples at 0.000 (own duration) and 0.017 (real duration). Durations match the real clips (median 2.15 s vs 2.20 s). So the lexicon failure is not vocabulary coverage, and not length.
  - **Locus mismatch is ruled out for one-word prompts.** Motion-measured training loci are non-empty for only 2–7% of one-word clips, and the text-planned ones are empty.
  - **Noted for later:** How2Sign sentences have text-planned loci on 52% of texts but motion-measured ones on 12%, a train/generation mismatch for sentences.
  - **Running:** teacher-forced token loss with real vs empty text (does the prior condition on the word at all?), plus sampling with oracle loci and at temperature 0.7.

**Round 6 audit, root cause (2026-10-27, 12:57 UTC): the prior barely conditions on its text.**

The B6 checkpoint at step 20000, 256 training clips per corpus: teacher-forced token cross entropy with the real text vs with the empty text (the same input as training's text dropout).

| Corpus | CE, text | CE, empty | Text gain (nats) | Token accuracy, text / empty |
|---|---|---|---|---|
| RIT | 3.156 | 3.168 | 0.011 | 0.2946 / 0.2933 |
| WLASL | 2.639 | 2.654 | 0.015 | 0.4116 / 0.4098 |
| ASLLVD | 3.653 | 3.668 | 0.015 | 0.2130 / 0.2114 |
| How2Sign | 3.509 | 3.514 | 0.005 | 0.2420 / 0.2415 |

- **What it means.** The text moves the code prediction by about 0.01 nats, so the prior is close to an unconditional motion model. Its previous codes predict the next ones (motion continuity), and it learned to rely on those rather than on the text.
- **This explains every content failure.** R2b is below blank, R5 is near 0, training words come out at 0.000 against 0.925 for real clips, more data did not help, and guidance helped only a little.
- **Sampling is ruled out.** Real duration gave 0.025, oracle loci 0.025, temperature 0.7 gave 0.0125.
- **Next: a training-procedure fix inside the current architecture,** as the owner asked. During training, a random fraction of the previous-code inputs is corrupted (the T2M-GPT strategy), so the prior cannot coast on continuity and must use the text. It is declared as round 7, with the text-use gain as its early check.

## Round 7: make the prior read its text (declared 2026-10-27, 12:58 UTC, before any round-7 result)

**Hypothesis** (from round 6's audit). The prior ignores the text because its previous codes already predict the next ones. Corrupting that history in training forces it to condition on the text. The architecture, data and instrument stay the same; only the training procedure changes.

**Arms.** B6's recipe exactly: byte encoder, `round6-mixed-v1`, RIT and WLASL at weight 3, ASLLVD at weight 3, 60000 steps, locus conditioning, tokenizer d2 with its gains. Plus `--input-corruption`: each sequence's previous-code inputs are replaced by random codes at a rate drawn uniformly in [0, x].
- **A7:** x = 0.5.
- **D7:** x = 1.0.

**Early check, at step 5000** (`text_use_check.py` on the training clips of RIT, WLASL, ASLLVD and How2Sign). An arm continues only if its text gain (empty-text CE minus text CE) is at least 0.10 nats on at least two of the three one-word corpora. B6 had 0.011–0.015. An arm that fails is stopped and its pod deleted.

**Pilot, at step 20000.**
- The round-5 D rule, unchanged: v2 lexicon on the R5 words, at least 0.05 above the control's, with the paired bootstrap 95% interval excluding 0, and meaning BLEU not below the control's.
- The training-word diagnostic (`lexicon_diagnostic.py`) and the text gain are reported beside it, not gated.
- The best promising arm continues to 60000 steps and then goes through the full declared chain.

**If neither arm passes the early check or the pilot:** the next within-architecture lever, declared before use. Candidates: a stronger text pathway (more text layers or a larger text width), an auxiliary text–motion contrastive loss, or classifier-free guidance trained with a higher text-dropout rate.

## Round 7 early check, recorded (2026-10-27, 14:27 UTC), and amendment 1: the early gate was invalid (declared 14:31 UTC, before any further round-7 result)

**Recorded result, step 5000** (`text_use_check.py`, 256 training clips per corpus):

| Arm | RIT | WLASL | ASLLVD | How2Sign | Declared gate (≥ 0.10 on 2 of 3 one-word corpora) |
|---|---|---|---|---|---|
| A7 (x = 0.5) | 0.0072 | 0.0110 | 0.0091 | 0.0039 | fail |
| D7 (x = 1.0) | 0.0062 | 0.0143 | 0.0085 | 0.0036 | fail |

Both arms were stopped at 14:27 UTC as declared, and their pods were deleted (logs verified in R2). Their step-5000 checkpoints are in R2.

**Audit: the gate could not be passed by any prior, and the round-6 root cause is withdrawn.**
- **The bound.** The cross entropy is a mean per code, and tokenizer d2 has no time downsampling. That gives 32 codes per frame at 24 fps, so about 1,650–1,860 codes per one-word training clip (mean 2.15–2.42 s).
- The text carries at most the word: I(word; codes) ≤ H(word). That is 5.4 nats over the 256 sampled clips, and at most about 7.6 nats over the whole training vocabulary.
- So the largest possible mean text gain per code is about H(word) / codes per clip:

  | Corpus | Bound (sample H) | Bound (full vocabulary) |
  |---|---|---|
  | RIT | 0.0033 | 0.0044 |
  | WLASL | 0.0029 | 0.0038 |
  | ASLLVD | 0.0031 | 0.0043 |
  | How2Sign | 0.0012 | — |

  The declared 0.10 was 25–35 times above what a perfect use of the text can give.
- **What the measured gains do show.** B6, A7 and D7 measure 0.006–0.015 per code. That is already at or above the bound, which is possible only because the empty text is not the marginal over words: training drops the text only 10% of the time.
  - Per clip, the gain is about 0.011 × 1,700 ≈ 15–25 nats.
  - So the measure cannot tell "ignores the text" from "uses it fully". The round-6 conclusion ("the prior barely conditions on its text") rested on it and is withdrawn.
  - The lexicon failures themselves stand: training words are generated at 0.000–0.033 against real clips at 0.908–0.925. Their cause is open again.
- **Lesson, recorded:** a per-token gate must be checked against its information bound before it is declared.

**Amendment 1 (declared before any further round-7 result):**
1. **The early gate is void.** A7 and D7 resume from their step-5000 checkpoints, with the same run, config and manifest, to the declared step-20000 pilot. The pilot rule (round-5 rule D, unchanged) decides round 7.
2. **A valid text-use instrument: teacher-forced word retrieval** (`word_retrieval_check.py`). Per one-word corpus, 64 training clips with 64 distinct words are drawn with a fixed seed.
   - Each clip's codes are scored under each of the 64 words by summed teacher-forced log-likelihood, with no corruption and the clip's own loci and duration.
   - It reports top-1, top-5, mean rank (chance: 1/64, 5/64, 32.5), and the summed per-clip gain over the empty text.
   - It also reports the score margin: true word vs the mean of the other words, in nats per clip.
   - It runs on the control (`r3d-d2g` step 20000), B6 step 20000, and A7 and D7 at steps 5000 and 20000.
   - It is a diagnostic. It is reported, not gated, until it is validated.
   - Its validation requirement: on a sanity case with shuffled words (each clip scored against a permuted candidate list), it must read at chance.
3. **Reading it.**
   - If retrieval is high (top-1 well above chance) while generated training words fail the recogniser, the prior does read its text, and the fault lies in autoregressive sampling: error accumulation over about 1,700 codes, which input corruption is meant to reduce.
   - If retrieval is near chance, the text pathway is the fault, and the next lever targets it.

## Word retrieval result, and the code audit's guidance bug (recorded and declared 2026-10-27, 14:59 UTC)

**Teacher-forced word retrieval** (64 training clips with 64 distinct words per corpus, rank of the own word; chance top-1 0.016, mean rank 32.5; the clip's own measured loci):

| Checkpoint | RIT top-1 / mean rank | WLASL top-1 / mean rank | ASLLVD top-1 / mean rank |
|---|---|---|---|
| Control (r3d-d2g step 20000; never saw RIT or WLASL) | 0.016 / 29.9 | 0.031 / 28.3 | 0.953 / 1.17 |
| B6 step 20000 | 0.672 / 3.39 | 0.750 / 3.12 | 0.906 / 1.50 |
| A7 step 5000 | 0.109 / 17.2 | 0.219 / 15.0 | (pending) |

The shuffled control reads at chance in every case (top-1 0.00–0.03). **So the priors do read their text:** on their own training corpora, teacher-forced likelihood picks the clip's word out of 64 in 67–95% of cases. The control reads at chance only on the corpora it never trained on.

**Code audit (2026-10-27, 6 reviewers): a guidance bug found independently by two of them.**
- In the unguided branch of classifier-free guidance, every sampler (`motion_rvq.py` single and batched, `motion_tokens.py`, `text_to_motion.py`) encoded the empty text **without the locus projection**.
- In training, a dropped text always passes zero slots through that projection, bias included (`training_loci`). So at any guidance ≠ 1, sampling pushed away from an unconditional distribution computed on an input never seen in training.
- Every locus-conditioned prior was sampled at guidance 1.5 in the pilots and diagnostics: round 5 and later, the control included.
- Fixed in 8664f5c, with a regression test.
- **The other audit findings** (minor, or outside the generator's path) are recorded separately when all six reviews are in.

**Identification test (declared before its result).** B6 step 20000, `lexicon_diagnostic.py` on its 60 most frequent training words (pre-fix: generated 0.000 at own duration, 0.017 at real duration; real 0.925):
1. the fixed sampler at guidance 1.5;
2. guidance 1.0, which has no unguided branch, so the bug cannot act;
3. word retrieval with zero loci (the condition one-word generation uses).

**Reading.**
- If (1) or (2) is well above the pre-fix 0.000–0.017 (for example at least 0.15), the guidance bug was the main cause of the content failure. In that case every earlier pilot is re-scored with the fixed sampler, the control included, before any verdict is revised.
- If both stay near 0 while retrieval stays high, the cause is in autoregressive sampling itself. The round-7 pilot (input corruption) then tests that.

## Full code audit, recorded (2026-10-27, 15:02 UTC; six independent read-only reviews at 802219c–fe739b0)

Each finding was verified by the reviewer with a numerical check or an exact derivation. Status: **fixed**, **test running**, or **queued** (with the stage that needs it).

**Affects the generator or its grading now**
1. **Guidance used an unseen empty-text input** (found by two reviewers). The unguided branch skipped the locus projection. **Fixed** (8664f5c); the identification test is running.
2. **The prior learns "flat hand" codes for untracked frames.** `diffusion_targets` fills invalid cells with the neutral value, and `rvq_prior_loss` supervises every frame-valid token without a channel-validity check. Clips may lose a hand for up to 50% of frames under the round-6 filter, so the prior is taught to emit static identity hands there.
   - **Queued** as the next training-procedure lever: mask each code's cross entropy by its part's validity.
3. **Generated states are graded in a form the recogniser never saw.** `assemble_state` marks every trained channel valid in every frame. Real training clips have legs (and, in ASL Citizen, pelvis and chest) almost always invalid, which the recogniser reads as neutral.
   - **Queued** as an instrument-parity check before any verdict: re-score generated motion with the channels masked as in real data.
4. **The BLEU half of the pilot rule is a coin flip.** It compares point estimates (±0.5 BLEU noise), so an arm with equal meaning passes about half the time. **Amendment 2 below.**
5. **The real reference is inflated.** The recogniser trained on the ASLLVD train signers, and the lexicon diagnostic's real clips come from the same partition. The held-out ASLLVD val signer reads 0.608 top-1 / 0.822 top-5, against 0.908–0.925 here.
   - The real reference in future reports uses held-out clips.
6. **`lexicon_diagnostic.py` prompts spaced texts without their spaces** ("thank you" becomes "thankyou").
   - The pre-fix and identification runs share this bug, so they stay comparable. **Queued:** prompt with the record's own text.
7. **Text-use measures mixed in an unseen input** (empty text with measured loci). `--zero-loci` was added; the information bound in amendment 1 is loose, but its conclusion stands (the gate needed about 170 nats per clip).

**Data quality (queued for the next data rebuild)**
8. **WLASL duplicates:** 827 URLs are shared by 1,838 instances; 804 of them carry different glosses, and 380 cross train and val/test. **Dedupe** by URL and state SHA, and drop multi-gloss videos.
9. **Extraction can put one physical hand in both hand channels** (two ROI trackers locking on the same hand), more likely at `--hand-scale 1.8`. **Measure** how often first: frames where both assigned wrists coincide.
10. **`--upright-chest` (ASL Citizen, MS-ASL) stores the arms against a chest frame it doesn't save.** Recogniser data only. **Fix** before MS-ASL extraction.
11. **Mirrored ASL Citizen canonical states keep the unmirrored `available_channels`,** so loading masks the dominant hand. This was v3/v3c recogniser data; v2, the instrument, is unaffected. **Fix** before any v4.
12. **Minor:** `english()` artefacts (trailing spaces, dropped "(1)" suffixes); `load_state` skips masking and mirroring; and the last frame after resampling is needlessly invalid.

**Tokenizer and serving**
13. **Serving crashes** on a v2 prior (decoder gains need `store.exists`). **Fix** before deployment.
14. **Minor:**
   - Unused codebook vectors decay toward zero late in training (no smoothing in mass division).
   - Padding leaks into the last ~0.25 s of shorter clips through the conv receptive field.
   - Blink gets no gradient beyond ±1.
   - The phase input is one frame off between training and generation.

**Avatar and grammar (queued for avatar v2)**
15. **Forearm twist half-share flips by 180°** for twists in [−180°, −90°) (quaternion sign not fixed before atan2).
16. **The face baseline (clip median) erases clause-long non-manual markers.** Use pooled baselines, as `nmm.py` already does.
17. **The text planner counts pronoun tokens, not referents,** and treats complementizer "that" as deixis. This is a likely cause of the 52% vs 12% planned-vs-measured locus gap.
18. **Minor:** the role-shift gaze is measured from the chest forward, not the camera; the left/right of face blendshapes is unverified.

**Checked OK by the reviewers:**
- **Pose maths:** rotation conversions and composition, mirroring (exact to 7e-16 against re-extraction), smoothing, unit conversions, and the body-frame handedness.
- **Tokenizer:** the RVQ maths (residuals, straight-through estimator, EMA, quantizer dropout, masks), the part column partition, and normalisation round trips.
- **Prior and sampling:** the sampler matches teacher forcing exactly (−295.9077 both ways), and the code order, CFG formula, EMA loading, LR schedule and resume are correct.
- **Data:** corpus weighting, partitions, cache integrity, manifest SHAs, and locus slot conventions.
- **Evaluation:** recogniser frame rate and label casing, the paired bootstrap, ranks, and corpus BLEU.

**Amendment 2 to round 7 (declared 15:02 UTC, before the round-7 pilot result):**
1. The pilot runs every arm and the control on the fixed sampler. The control is re-scored, and that re-score is the paired reference.
2. The BLEU condition of rule D becomes non-inferiority: the paired-bootstrap 95% lower bound of BLEU(arm) − BLEU(control), over the same resampled clips, must be above −0.5 BLEU. The lexicon condition is unchanged.

## Identification test 1 result, and test 2 declared (2026-10-27, 15:30 UTC)

**Test 1 result** (B6 step 20000, its 60 most frequent training words; real clips 0.925):

| Condition | Own duration | Real duration |
|---|---|---|
| Pre-fix sampler, guidance 1.5 | 0.000 | 0.017 |
| Fixed sampler, guidance 1.5 | 0.033 | 0.017 |
| Guidance 1.0 | 0.000 | 0.000 |

Word retrieval with zero loci equals the measured-loci result (top-1 0.672 / 0.750 / 0.906). **The guidance bug was real but not the main cause.** The prior reads its text, and the sampler matches teacher forcing exactly, so the failure lies in free-running generation, or in how the recogniser reads generated motion.

**Test 2 (declared before its result): `exposure_diagnostic.py`** on B6 step 20000, the same 60 words, one real clip each, prompted with each record's own text. The recogniser reads six forms:
- `real`: the real clip;
- `recon_full`: its own codes decoded, in the generator's form;
- `recon_masked`: the same, with the real clip's channel validity;
- `onestep_sample` and `onestep_argmax`: each step drawn (or taken as the mode) from the prior given the real history and the text;
- `free_run`: ordinary sampling at the real duration, guidance 1.5.

**Reading.**
1. If `recon_masked` is well above `recon_full` (for example by at least 0.2), the instrument misreads the generator's form. Grading then masks generated motion as real data is masked, declared before any re-score.
2. If `recon_full` is itself low, no generator can pass through this tokenizer-to-instrument path, and the tokenizer is the bottleneck.
3. If one-step samples read close to `recon_full` while `free_run` stays near 0, exposure bias (errors accumulating over the sampled history) is the cause. Round 7's corruption arms test exactly that; the next levers would be scheduled sampling or a coarser token rate.
4. If one-step samples are also near 0, the prior's per-step distributions themselves lose the sign.

## Identification test 2 result: the cause is exposure bias (recorded 2026-10-27, 15:44 UTC)

B6 step 20000, 60 training words, recogniser v2:

| Form | Top-5 | Top-1 | Hits / 60 |
|---|---|---|---|
| Real clip | 0.917 | 0.900 | 55 |
| Own codes decoded, generator form | 0.883 | 0.817 | 53 |
| Own codes decoded, masked like real | 0.933 | 0.900 | 56 |
| One step from the real history, sampled | 0.817 | 0.633 | 49 |
| One step from the real history, mode | 0.867 | 0.717 | 52 |
| Free run (guidance 1.5) | 0.017 | 0.017 | 1 |

**Reading 3 of the declaration applies.**
- **The tokenizer is not the bottleneck:** reconstruction reads 0.883 against 0.917 for real clips.
- **The recogniser reads the generator's form:** masked minus full is 0.05, below the declared 0.2, so grading stays as it is.
- **The prior's per-step distributions keep the sign** when the history is real (0.82–0.87), even though only 8–11% of the individual codes match the real ones.
- **Free-running sampling loses it** (0.017). The failure is exposure bias: errors accumulate over a sampled history of about 50 steps × 32 codes, which the model never saw in training.
- **Per word:** 46 of 60 words are readable from the real history but lost in free run, and 1 survives.
- Every content result since round 3 (R2b, R5, the training-word diagnostics) is explained by this.

**What follows.**
- Round 7 (input corruption, A7 and D7) is the direct test of the cause. Its step-20000 pilot is already declared.
- **Early read, not gated:** `exposure_diagnostic.py` on A7 and D7 at step 10000. If their free-run form rises above B6's 0.017 while the one-step forms hold, corruption is working.
- **If round 7 fails the pilot, the next within-architecture levers, in order:**
  1. **Scheduled sampling:** part of the training history comes from the model's own one-step samples instead of random codes.
  2. **A coarser autoregressive rate:** a tokenizer with time downsampling, so 4–8× fewer sequential steps per sign.
  3. Both are training changes with the same architecture. Each is declared before use.

## Round 8: scheduled sampling against exposure bias (declared 2026-10-27, 15:45 UTC, before any round-8 result)

**Hypothesis** (from identification test 2). The prior loses the sign only when it conditions on its own sampled history. Training on histories drawn from its own one-step samples closes that gap more directly than round 7's random corruption. It runs in parallel with round 7, so a failure of round 7 does not cost another half day.

**Arm S8.** B6's recipe exactly (byte encoder, `round6-mixed-v1`, RIT/WLASL/ASLLVD at weight 3, locus conditioning, tokenizer d2 with its gains, 60000 steps), plus `--scheduled-sampling 0.5`:
- a gradient-free pass draws each step's 32 codes given the real history (one parallel depth pass);
- each sequence replaces its history by those codes at a rate drawn uniformly in [0, 0.5];
- no random corruption. Smoke first.

**Checks, on the fixed sampler:**
- **Early read at step 10000, not gated:** `exposure_diagnostic.py`, free-run form against B6's 0.017.
- **Pilot at step 20000:** round-5 rule D with amendment 2, i.e. lexicon +0.05 over the re-scored control with the paired bootstrap 95% interval excluding 0, and paired BLEU non-inferiority (lower bound > −0.5). The training-word diagnostic and the exposure diagnostic are reported beside it.

**Selection.** If more than one of A7, D7 and S8 is promising, the one with the highest lexicon difference continues to 60000 steps and then the full declared chain. A tie within 0.02 goes to the higher paired BLEU.

## Hold controls, the research diagnosis, and round 9 (recorded and declared 2026-10-27, 17:03 UTC, before any round-9 result)

**Hold controls** (B6 step 20000, 60 training words, top-5). Each pair compares a real prefix followed by the model's continuation against the same prefix followed by a held pose:

| Real prefix | Model continues | Held pose |
|---|---|---|
| 25% | 0.067 | 0.017 |
| 50% | 0.650 | 0.517 |
| 75% | 0.867 | 0.817 |

- The model's continuations add only 0.05–0.13 over a held pose.
- From a mostly-rest prefix, it cannot introduce the sign.
- Coarse-only sampling (finer stages at their mode) gives 0.033.
- **Round 7 early read at step 10000:** A7 free run 0.017, D7 0.033 (B6 0.017).

**Research diagnosis** (`docs/research/exposure-bias-fixes-2026-10-27.md`; four literature reviews). The failure matches the **copycat shortcut**:
- Wen et al. 2020, arXiv 2010.14876; ChauffeurNet, arXiv 1812.03079; acRNN freezing, arXiv 1707.05363; Progressive Transformers' drift, arXiv 2004.14874.
- At 24 fps with all 32 codes of the previous frame as input, the history nearly determines the next frame, so the text matters only where history cannot predict: the onset of the sign.
- Published discrete-token generators run at about 5 steps/s with ≤ 6 codes per step; ours runs at 768 codes/s.
- The two cheapest fixes ranked highest are both training changes within the architecture:
  1. hide whole history spans, so the model must start the sign from the text;
  2. feed only the coarse residual stages back as history.

**Round 9 arms.** B6's recipe exactly, each with a smoke run first:
- **H9:** `--history-dropout 0.5`. Per sequence with probability 0.5, every history step before a uniformly random cut is replaced by the start code, as at generation's first step.
- **HC9:** `--history-dropout 0.5 --history-stages 1`. The same, and the temporal transformer sums only stage 1 of each part of the previous step, in training and generation alike.

**Checks, on the fixed sampler:**
- **Early read at step 10000, not gated:** `exposure_diagnostic.py`, free run against B6's 0.017.
- **Pilot at step 20000:** rule D with amendment 2, against the re-scored control.
- **Selection:** among the promising arms of rounds 7–9 by the highest lexicon difference (tie within 0.02 goes to the higher paired BLEU). The winner continues to 60000 steps and the full declared chain.

## Round 7 pilot, A7, recorded (2026-10-27, 18:06 UTC): NOT PROMISING (rule D, amendment 2)

A7 (input corruption 0.5) at step 20000 against the control re-scored on the fixed sampler:

| Measure | A7 | Control | Paired difference [95% CI] | Rule | Result |
|---|---|---|---|---|---|
| Lexicon top-5 | 0.0033 | 0.0167 | −0.0133 [−0.0267, −0.0033] | ≥ +0.05, CI above 0 | fail |
| BLEU | 2.119 | 1.592 | +0.527 [−0.121, 1.186] | lower bound > −0.5 | pass |

- Real motion reads BLEU 4.089, blank 2.09. The BLEU gain only brings A7 to the blank level.
- A7 stays stopped; its checkpoints are kept in R2.
- This matches the step-10000 early read (free run 0.017): random corruption does not break the copycat shortcut. The coarse stages of the history survive it.

## Round 10: coarser tokenizers (declared 2026-10-27, 18:09 UTC, before any round-10 result)

**Why** (research report, rank 4).
- Every published discrete-token motion or sign generator runs at about 5 steps/s with ≤ 6 codes per step. Ours runs at 768 codes/s.
- Measured sweeps (MoTok; LMR) show the full-rate tokenizer reconstructs best but generates worst.
- MoMask's generation quality is best at 5–6 residual levels.
- This is the remedy for the copycat shortcut at its source, and a hedge if rounds 8–9 fail. It is a hyperparameter change of the same residual-VQ tokenizer, not a new architecture.

**Candidates.** Round 3d's tokenizer stage and checks (`round3d_tokenizer.sh`, now `round10_tokenizer.sh`): d2's recipe from scratch, 60000 steps, smoke first.
- **t2x:** `--down-layers 1` (12 Hz), `--stages 4`: 16 codes per step, 192 codes/s.
- **t4x:** `--down-layers 2` (6 Hz), `--stages 4`: 16 codes per step, 96 codes/s. It starts when a pod is free.

**Gates (unchanged from round 3d):** the domain check, the recognisability check, and the fidelity check, with amendment 5's gains fallback.

**If a candidate passes:**
- **Learnability check first:** a smoke prior.
- **Then a prior** on B6's recipe, with the best round 8 or 9 training option if one is promising.
- **The same pilot** (rule D with amendment 2), and the exposure diagnostic's free-run form reported beside it.

## Round 7 result, recorded (2026-10-27, 18:21 UTC): NOT PROMISING

D7 (input corruption 1.0) at step 20000 against the re-scored control:

| Measure | D7 | Control | Paired difference [95% CI] | Rule | Result |
|---|---|---|---|---|---|
| Lexicon top-5 | 0.0033 | 0.0167 | −0.0133 [−0.030, 0.0033] | ≥ +0.05, CI above 0 | fail |
| BLEU | 2.03 | 1.592 | +0.437 [−0.381, 1.242] | lower bound > −0.5 | pass |

- With A7, round 7 is not promising. Both arms stay stopped; their checkpoints are kept.
- Rounds 8 (S8), 9 (H9, HC9) and 10 (t2x; t4x launched 18:21 UTC) continue as declared.

## Early reads at step 10000, rounds 8–9 (recorded 2026-10-27, 19:29 UTC; not gated)

`exposure_diagnostic.py` on the same 60 training words (B6 step 20000 for reference):

| Arm | Free run | One step (sampled / mode) |
|---|---|---|
| B6 (step 20000) | 0.017 | 0.817 / 0.867 |
| A7 | 0.017 | 0.700 / 0.817 |
| D7 | 0.033 | 0.633 / 0.767 |
| S8 (scheduled sampling 0.5) | 0.017 | 0.717 / 0.833 |
| H9 (history dropout 0.5) | 0.000 | 0.667 / 0.800 |
| HC9 (+ stage-1 history) | 0.033 | 0.517 / 0.667 |

**Reading (not a verdict; the pilots at step 20000 decide).**
- No training-procedure remedy moves the free run at step 10000.
- The one-step forms are high for every arm. At 24 fps the next frame given the real previous one is mostly a small change, so a real history all but guarantees a readable decode. The one-step result therefore shows copying ability, not knowledge of the sign's trajectory.
- The working interpretation is now: **at one code step per frame, no step carries enough of the sign for the text to determine it.** The token rate is the lever, which is round 10 (t2x, t4x).
- **Change of priority** (declared): as soon as a round-10 tokenizer passes its gates, its prior is trained with history dropout 0.5, the guard against copying that stays valid at a coarser rate. Rounds 8–9 still run to their declared pilots.

## Root-cause analyses (2026-10-27, four independent agents) and round 11 (declared 19:54 UTC, before any round-11 result)

**Mechanism (free-run trace; B6 step 20000, 60 training words, CPU; scripts kept with the agent's notes).**
- **Text influence per step is small throughout,** and about equal under teacher forcing and in free runs.
  - KL(text ‖ empty) over the 4 stage-1 codes is 0.02–0.07 nats per step after step 0, against a stage-1 entropy of 2.0–2.4 nats.
  - Summed over all 32 codes it is about 0.2 nats per step, mostly in the fine stages.
- **One-step forms come from the history, not the text:**

  | Form | Top-5 |
  |---|---|
  | Copy of the previous real frame | 0.883 |
  | One-step with the true text | 0.817 |
  | One-step with an empty text | 0.733 |
  | One-step with a wrong word's text | 0.800 |

- **Free runs are generic signing motion.**
  - Arm speed, acceleration and jitter are real-like, and motion starts on schedule.
  - There is no return to rest (0.64 m/s in the last 0.25 s against 0.16 real), and hand articulation is about 30–35% lower.
  - By wrist-path DTW they rank their own word at chance (27.6 of 59), and two seeds of a word are as unlike as two words.
  - The recogniser reads them as scattered labels.
- **No inference lever helps:** guidance 0–5, temperature 0.7, the duration convention, or real prefixes of up to 12 frames.
- **No remaining train/generation mismatch.**
- **Reading:** the prior learned generic signing dynamics but not the word-to-trajectory mapping. The text controls fine-stage detail, which is why teacher-forced retrieval works, but not the movement the recogniser reads. Exposure bias is a symptom: nothing pulls a free run back to the word.

**Data timing (one-word clips, 80 per corpus).**
- The lexical movement lies in about 0.3–0.7 of each clip. Rest before and after is about 11–17 frames each (RIT, ASLLVD). WLASL's arm is untracked on 23% of frames (median), filled with an identity pose (arm horizontal).
- One-word clips are 20% of samples but only 9.5% of trained frames. The median is 5 clips per word, and 32% of words have ≤ 2.
- Duration prediction (5.7% error), phase, loci and the text encoder (discriminative: centred cosine −0.02 between words) are consistent and not the cause.

**History (why this was found only now).**
- The failure was visible from 2026-10-23 (R5 0% against real 99.4%). It was attributed in turn to real-but-insufficient confounds.
- The full-rate tokenizer (v2b, `down_layers 0`, 10-20/21) was chosen on sentence Fréchet, a texture statistic. Round 3c's fidelity-only selection raised it to 8 stages (768 codes/s).
- No gate measured "can it sign its own training words in free run". The cheap decomposition (free run vs real-prefix continuation vs held pose, on trained words) was first run on 10-27.

**Standing gates from now on** (declared):
1. **Generation gap, on every prior checkpoint that is evaluated.** On trained words, free-run top-5 against reconstruction, and each real-prefix continuation against its hold control. Reported with every pilot, gated from round 11 (below).
2. **Before attributing a content failure to data or vocabulary,** free run on trained items is measured.
3. **Tokenizer learnability and token rate** are reported before any prior is trained: codes/s, and a smoke prior's free run on trained words.
4. **Every per-token threshold is checked against its information bound;** conditioning is measured per clip, with a shuffled control.
5. **After one null round, decompose before adding arms.**
6. **Instrument rules:**
   - Generated reading at or above real voids an instrument.
   - Real references use held-out signers.
   - BLEU conditions are paired.
   - The guided and unguided branches are tested for parity with training's dropped text.

**New training options (opt-in, tested):**
- `--part-valid-loss`: a code's cross entropy counts only where its part was observed.
- `continuous.window_seconds` in manifests: frames outside are dropped on load.
- `build_trimmed_word_manifest.py`: one-word clips cut to the dominant wrist's active span ± 0.25 s. Building now: `round10-mixed-v1`.

**Round 11: can the prior learn the word-to-trajectory mapping at all?** An isolated-sign learnability test on the current tokenizer d2.
- **Arm L11:** B6's recipe on `round10-mixed-v1` (trimmed), with these changes:
  - `--corpus-weight` rit-text-r6=60, wlasl-text-r6=60, asllvd-text-v1=60, so one-word clips dominate the samples;
  - `--history-dropout 0.5`;
  - `--part-valid-loss`;
  - smoke first.
- **Gate at steps 5000 and 10000** (`exposure_diagnostic.py`, the same 60 trained words, prompted with each record's text):
  - **Pass:** free run ≥ 0.20 top-5 at step 10000, above B6's 0.017. The mapping is learnable, and the round-10 prior (coarse tokenizer) uses this recipe.
  - **Fail:** free run < 0.05 at step 10000. Even focused on isolated signs, this prior cannot produce its own training words from text. That is the evidence a structural change needs (for example, retrieval of real dictionary clips for isolated words), and it is brought to the owner with the numbers.
  - **Between 0.05 and 0.20:** it continues to step 20000, with the same reading.
- **Launch:** when the trimmed manifest is in R2 and a pod is free. To free the slot, S8 is stopped then, its checkpoints kept. Its step-10000 early read showed no gain (free run 0.017), and its declared pilot would not finish before 22:10 UTC.

## Round 10 note: the domain check's declared form (2026-10-27, 20:11 UTC, before the amended check's result)

- **Which form applies.** Round 10's gates are "unchanged from round 3d", and round 3d's domain check is the amendment-2 form: s8's fixed reference errors as the denominator, run by `after_tok_3d.sh`.
- **What went wrong.** The pod stage copied from `round3d_tokenizer.sh` runs the plain form. It reported t2x (best at step 13000; early stop at step 23000) as failing on ASLLVD, with left hand 1.62 and right hand 1.57 against 1.5.
- **What runs instead.** The amendment-2 check runs on the same clips, limit and corpora as round 3d (`domain_a2_r10.sh`). Its result decides. Both reports are kept.
- **Recognisability: passed.** Round trip top-5 0.763 against 0.817 for real (ratio 0.93, at least 0.9 required).
- **Fidelity:** still running on the pod.

## Round 10: t2x verdict, and amendment 1, t2xb (recorded and declared 2026-10-27, 20:21 UTC, before any t2xb result)

**t2x: REJECTED (fidelity).** Best checkpoint at step 13000; early stop at step 23000 after gradient spikes up to 214.
- **Fidelity ratios** (band 0.85–1.15):

  | | Body | Left hand | Right hand | Head rotation |
  |---|---|---|---|---|
  | Speed | 0.694 | 0.558 | 0.542 | 1.03 |
  | Variation | 0.814 | 0.604 | 0.566 | 0.878 |

- **Recognisability passed** (0.763 against 0.817 real), and the R2b round trip is above blank (2.91 against 1.86).
- **Amendment 5's gains fallback cannot pass.** Gains are capped at 1.5, so the right hand would reach 0.542 × 1.5 = 0.81 speed and 0.566 × 1.5 = 0.85 variation, both below the band. It is not run.
- The amendment-2 domain check is still running and is recorded as a report.

**Amendment 1: t2xb** (`round10_tokenizer_t2xb.sh`). The same 2× tokenizer (12 Hz, 4 stages), and the same gates as t2x, fixed against the two causes seen:
- **Instability:** `--lr 1.5e-4 --warmup 2000 --skip-grad-spikes 8` (round 3d amendment 1's rule).
- **Damped hands:** velocity matching at 0.3 on body and both hands (round 3c's s8v used 1.0, which damped the hands).
- **Run:** smoke first, then 60000 steps with patience 10, launched in t2x's freed slot.
- **t4x continues as declared** (best validation still improving). t4xb is prepared and launches only by a further amendment.
