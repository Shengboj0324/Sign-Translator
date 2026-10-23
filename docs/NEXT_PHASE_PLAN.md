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
