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
   - **Still needed for the planner:** English translations of the utterances. They are in the DAI utterance XML, which needs the signed-in session.
5. **Avatar and serving** (items 5 and 6) follow, with a model selected on R-precision and the realism gates.
6. **Human Deaf reviewers** (on their way): they evaluate the stored generated states and the rendered avatar. Their findings drive fine-tuning.
