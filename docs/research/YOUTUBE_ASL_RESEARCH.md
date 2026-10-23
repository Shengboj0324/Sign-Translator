# YouTube-hosted ASL corpora: which to use, expected gains, acquisition, and permission to release weights

Research report for plan item 7 (`docs/NEXT_PHASE_PLAN.md`, Y0). Compiled 2026-10-22 (PDT) from primary sources:
arXiv papers (full text read), official GitHub repositories, dataset pages, and public bucket and Hugging Face
metadata. **No video, caption, or dataset file was downloaded.** The one exception is the first lines of the public
OpenASL TSV, streamed to read its column header. Bracketed numbers such as [P1] refer to the source
list in §7. Anything I could not verify is marked **(unverified)**.

> **Not legal advice.** §4 summarises licences, the YouTube Terms of Service, and published legal commentary so the
> owner can ask the right questions. It is not a legal opinion. Get qualified counsel before any public release.

---

## 0. One-page recommendation

**Corpora, in priority order**

| # | Corpus | Use | Why |
|---|---|---|---|
| 1 | **YouTube-ASL** (11,093 video IDs, 984 h, 610,193 captions, ≥2,519 channels) [P1][R1] | Main pretraining set for the back-translator, the retrieval evaluator, and the generator prior | Native Deaf ASL annotators filtered every video for correct, well-aligned English captions. This is the cleanest large ASL set, and nearly every published system is pretrained on it. |
| 2 | **YouTube-SL-25, ASL rows only** (16,724 videos, 1,394 h; about 5,600 videos and 410 h beyond YouTube-ASL) [P2][R2] | Extra pretraining, with stricter automatic filters | The rows beyond YouTube-ASL were triaged by one non-native signer over four days, so they are lower-assurance. Refreshed to May 2024. Pretraining on the full set gives +0.9 BLEU on How2Sign over the ASL subset (15.4 vs 14.5) [P2]. |
| 3 | **Non-YouTube consented corpora**: DailyMoth-70h (CC BY-NC, video included, one native Deaf signer) [P4][R5] and ASL STEM Wiki (300+ h, 64,266 sentences, 37 certified interpreters, research licence) [P13] | Generator fine-tuning (consistent, fluent signing) and evaluator robustness | These need no YouTube download and have explicit consent or licence agreements. Both authors warn that YouTube's signer variety limits its use for *generation* [P1 §6][P2 §4]. |
| 4 | **OpenASL** (98,417 pairs, 288 h, about 220 signers) [P3][R3] | Optional. Use only the parts not in YouTube-ASL. Never use for evaluation after YouTube pretraining. | OpenASL is "largely a subset of YouTube-ASL" [P1], and more than 72% of its test samples are in YouTube-ASL [P8]. Its annotations are **CC BY-NC-ND 4.0**, the most restrictive licence in the stack. Leave it out of release builds unless TTIC permits. |
| — | **FLEURS-ASL** (5 Certified Deaf Interpreters, 1,749 sentences, 7.49 h, CC BY-SA 4.0, on Kaggle) [P5] | **Evaluation only.** Use it as the independent zero-shot judge for the back-translator and recogniser. | It is not on YouTube, it is professionally translated, and it is built for zero-shot use. Do not train on it, because CC BY-SA would attach to anything adapted from it. |

**Filtering recipe** (details in §1.3). Each rule comes from published work unless it is marked as our proposal.

1. Video level, as in YouTube-ASL and YouTube-SL-25:
   - manually uploaded English captions only;
   - duration 10 s to 5 h; width ≥ 480 and height ≥ 360; 15–60 fps;
   - captions cover ≥ 40% of the duration.
2. Caption level, as in YouTube-ASL:
   - ≤ 300 characters; duration 0.2–60 s;
   - exactly one person in the span (YOLO person or pose detector; multi-person clips dropped [P7][P10]).
3. Pose level (our proposal, to be calibrated in the Y2 pilot):
   - dominant-hand landmarks in ≥ 70% of frames, face in ≥ 90%, wrists inside the frame;
   - roughly frontal (|yaw| < 30°);
   - signing-space crop and normalisation, as in Železný et al. [P10].
4. Alignment plausibility (our proposal): words per second 0.5–4, and drop a caption whose start falls in a stretch with no hand motion.
5. Generation-grade subset (our proposal, motivated by [P2 §2.2]):
   - channels of Deaf-produced content with no speech track;
   - exclude sign-along songs, live-interpreted events and learner channels;
   - cap DailyMoth and Sign1News so these two largest channels do not dominate.
6. Deduplication:
   - by video ID and time overlap across YouTube-ASL, YouTube-SL-25 and OpenASL (SHuBERT removed OpenASL-intersecting clips [P8]);
   - drop any caption that is a near-duplicate of a How2Sign or FLEURS-ASL validation or test sentence;
   - if DailyMoth-70h is used for evaluation, remove DailyMoth channel videos from **YouTube-SL-25**, because its May 2024 refresh can contain the post-January-2022 DailyMoth-70h test period (our inference from [P2][P4]).

**Expected gains** (literature, How2Sign test, sacreBLEU; details in §2)

- **Pose-only ASL→English:**
  - training on How2Sign alone gives about 1 BLEU;
  - pretraining on YouTube-ASL gives **12.4 BLEU** (46.6 BLEURT) [P1];
  - pretraining on all of YouTube-SL-25 gives 15.4 BLEU (47.9) [P2];
  - private noisy data of about 6,600 h gives 21.1 BLEU (55.7) [P6].
- Zero-shot (no How2Sign) is only **3.6–4.2 BLEU** [P1][P2].
- The human sentence-level ceiling on How2Sign is **19.8 BLEU / 56.6 BLEURT** [P9].
- For our back-translator (currently 2.1 BLEU), a realistic target after YouTube-ASL pretraining and How2Sign fine-tuning is **about 8–14 BLEU, 40–47 BLEURT**. That is consistent with the plan's 6–12 estimate.
- **No published study measures how YouTube pretraining affects sign *production* quality.** Treat the generator gain as unproven and A/B-test it.

**Permission requests to send** (contacts in §4.2). Send all of them before any public weight release.

| To | About | Contact |
|---|---|---|
| Carol Neidle (BU ASLLRP) | ASLLRP, DAI 2, Sign Bank and ASLLVD. **Required:** the terms forbid redistribution and any "patents and copyrights based on this material". | carol@bu.edu |
| Microsoft Research ASL Citizen team | ASL Citizen and ASL STEM Wiki. **Required:** the licence forbids distributing the data "or your modifications". | ASL_Citizen@microsoft.com |
| TTIC (Shi, Livescu, Shakhnarovich, Brentari) | OpenASL, if used. Needed because of the ND licence. | bshi@ttic.edu, klivescu@ttic.edu, greg@ttic.edu, dbrentari@uchicago.edu |
| Google authors (Uthus, Tanzer, Georg; Zhang for YouTube-SL-25) | YouTube-ASL and YouTube-SL-25. A courtesy notice. **They cannot grant rights in creators' videos.** | duthus@, gtanzer@, mgeorg@, biaojiaxing@google.com |
| Amanda Duarte (How2Sign, UPC) | How2Sign. A courtesy notice; CC BY-NC 4.0 already allows non-commercial sharing with attribution. | amanda.duarte@upc.edu |

**Main risks**

1. **YouTube ToS.** The Terms forbid downloading and automated access without permission [L1]. This applies to acquisition itself, whatever happens to the weights. YouTube's CEO called training on YouTube videos without permission a "clear violation" [L3].
2. **Creators' copyright and consent.** The videos belong to thousands of creators who were not asked. Since December 2024, YouTube's third-party AI-training setting is off by default [L4].
3. **Datacenter blocking.**
   - YouTube blocks datacenter IPs [L7].
   - PO-token workarounds "no longer bypass the bot check for the majority of cases" [L6].
   - Under plan rules we do not circumvent. Pre-extracted 2D keypoints are the fallback.
4. **Representation gap.** Both public keypoint releases are **2D** (MediaPipe 2D with 104 points [P10][R6]; DWPose 2D [P11][R7]). Our canonical 3D rotations and face blendshapes need the video, so the keypoint releases serve the instruments, not the generator.
5. **Contamination.** OpenASL test sits inside YouTube-ASL, and DailyMoth-70h test may sit inside YouTube-SL-25. Evaluation must stay on How2Sign, FLEURS-ASL and ASLLRP val.
6. **Signer variety.** YouTube data may teach the generator a blend of skill levels and dialects [P1][P2].
7. **Release licence.** The strictest component terms (OpenASL ND, ASLLRP, ASL Citizen) decide what can be released. The likely outcome is a **non-commercial research licence** on a release build that leaves out any corpus without written permission.

---

## 1. Which YouTube ASL data gives the best model

### 1.1 Candidates at a glance

| Corpus | Size | What is released | Where | Licence of release | Quality notes | Availability / link rot |
|---|---|---|---|---|---|---|
| **YouTube-ASL** (Uthus, Tanzer, Georg; NeurIPS 2023 D&B) [P1] | 11,093 videos, 984 h; 610,193 captions totalling 813 h; ≥2,519 channels (signer lower bound); 60K vocabulary. Captions average 8.8 words and 4.8 s; videos average 319 s. | **Video IDs only.** No captions, timestamps or splits: "Each instance is the video id" [P1 App. B.2]. | `gs://gresearch/youtube-asl/youtube_asl_video_ids.txt` (133,151 bytes, dated 2023-06-07; public bucket listing) [R1]. The pointer README is at `google-research/google-research/tree/master/youtube_asl`. | IDs under **CC BY 4.0**. Videos and captions are "subject to the YouTube Terms of Service" [P1 App. B.6, D]. | 88,002 candidates went to 3 native ASL annotators, leaving 11,093 (12.6%). Annotators sometimes labelled whole channels, so a few outlier videos may slip through [P1 §3.2]. Skill levels and signing styles were kept on purpose, which the authors say "may limit … usefulness for generation" [P1 §3.2]. | No official rate. SignMusketeers retrieved **601,995 of 610,193 caption clips (98.7%)** in 2024 [P7 fn. 4]. Železný et al. kept 390,547 clips (64%) after their own single-person filter and availability losses [P10][R6]. |
| **YouTube-SL-25** (Tanzer & Zhang; ICLR 2025) [P2] | 39,197 videos, 3,207 h (2,980 h of captions, 2.16M captions) across 25+ sign languages. **ASL rows: 16,724 videos, 2,523 channels, 1,394 h.** | Video IDs plus metadata. The paper says IDs; the bucket file is named `metadata.csv`, so it probably holds language labels **(unverified; not opened)**. | `gs://gresearch/youtube-sl-25/youtube-sl-25-metadata.csv` (667,188 bytes, dated 2024-06-18) [R2]. README at `google-research/google-research/tree/master/youtube_sl_25`. | **Not stated** in the README or arXiv v1 **(unverified; ask)**. Video terms are the same as YouTube-ASL. | A superset of YouTube-ASL. The non-YouTube-ASL rows were triaged by the first author (a non-native hearing signer) over a four-day weekend, channel by channel. His audit of 100 random videos found 1 that should not be included, 1 borderline, and 3 sign-along songs. A duration-weighted sample found 0.76% speech instead of signing and 0.96% uncaptioned interstitials [P2 §2.2]. Adds a caption-coverage rule (≥ 40%) and **keeps multi-signer videos**, so single-signer filtering is up to us. | Refreshed May 2024 with newer videos from accepted channels [P2 fn. 3]. SignVerse-2M lists 39,196 entries [P11]. |
| **OpenASL** (Shi, Brentari, Shakhnarovich, Livescu; EMNLP 2022) [P3] | 98,417 sentence pairs, 288 h, about 220 signers, 33K vocabulary. Sources: TheDailyMoth, Sign1News and NAD, up to June 2021. | **Caption text, timestamps and splits.** The TSV columns are `vid, yid, start, end, raw-text, tokenized-text, gloss, split`, with gloss for val/test. Person bounding boxes are in `data/bbox-v1.0.json`, from face detection and heuristics, error "below 1%". | `github.com/chevalierNoir/OpenASL` (`data/openasl-v1.0.tsv`, `prep/download.py`, `prep/crop_video.py`) [R3]. | **CC BY-NC-ND 4.0** (`LICENSE.md`) [R3]. *A 2026 survey calls it "Apache-2.0" [P14]. That is wrong; the repository licence file is CC BY-NC-ND.* | Captions are "self-generated", not interpreted, so they are aligned to signing. Most shifts are under 2 s; fewer than 5% are larger. Original captions score 81.0 BLEU against the interpreter-corrected ones. Val (966) and test (975) were corrected by professional interpreters. About 10.6% of test sentences are duplicates such as "Hello" and "Thank you" [P3 §3, §5.4]. | The README says some videos "may no longer be publicly available". No rate is published. |
| **FLEURS-ASL** (Tanzer; NAACL 2025) [P5] | 1,749 FLORES sentences in 495 discourses, 7.49 h, 5 Certified Deaf Interpreters. Median sentence lasts 13.3 s, against 5.3 s for How2Sign. | **Video and text**, with zero-shot, signer-independent and signer-dependent splits. Caption alignments were annotated manually. | Kaggle `googleai/fleurs-asl` (link from the paper PDF) [P5]. | **CC BY-SA 4.0** [P5]. | Interpreters gave informed consent for machine-learning use and were paid market rates. Human baseline: 13.0 BLEU / 64.6 BLEURT. | Self-contained (not on YouTube). |
| **DailyMoth-70h** (Rust et al., ACL 2024; Meta) [P4] | 496 raw videos (76.9 h) cut into 48,386 clips (75.8 h); 1 native Deaf signer; January 2021 to April 2023. | **Video** (raw, blurred clips, unblurred clips) plus manifests with timestamps and text. | `github.com/facebookresearch/ssvp_slt` (`DATASETS.md`, `scripts/download_dm70.sh`) [R5]. | **CC BY-NC 4.0**, "obtained through a license agreement with TheDailyMoth's host" [P4]. | Burned-in captions are blurred. Val and test come from videos posted after January 2022, to avoid leakage from YouTube-ASL and OpenASL [P4 App. B]. | Self-contained. |
| **ASL STEM Wiki** (Microsoft, EMNLP 2024) [P13] | 254 Wikipedia articles, 64,266 sentences, 300+ h, 37 certified interpreters. | Video and sentence alignment. | Microsoft Download Center; `github.com/microsoft/ASL-STEM-Wiki`. | "A license that permits use for research purposes". The exact text is on the project's licence tab **(unverified; assumed similar to ASL Citizen)**. | Consented and IRB-approved. Interpreted from English, so the signing is professional but stays close to English. | Self-contained. |
| *Larger private sets* (Google) [P6][P8] | YT-ASL about 2,800 h and YT-Full about 6,600 h of "noisy captioned" video | Not released | — | — | These show how far scale goes: 21.1 BLEU on How2Sign. | — |

**No newer *public* ASL video and English caption corpus larger than YouTube-SL-25 turned up** (searched October 2026). The 2025–2026
additions are derived releases of YouTube-SL-25 or YouTube-ASL (SignVerse-2M, the UWB keypoint set; see §3.3) and
pipeline tools (SignDATA [P15]). A 2026 survey lists the same ASL corpora [P14].

### 1.2 What the published systems actually did

| Step | YouTube-ASL [P1] | YouTube-SL-25 [P2] | FLEURS-ASL / Scaling SLT [P5][P6] | Železný et al. (pose T5) [P10] | SHuBERT [P8] |
|---|---|---|---|---|---|
| Candidate retrieval | Knowledge Graph tags for sign language / ASL, public listed videos, January 2022 | Same, July 2023, plus a May 2024 refresh | — | — | — |
| Caption kind | **Manually uploaded only** (speech-derived captions are not aligned to signing) | Same | — | — | — |
| Video heuristics | 10 s–5 h; ≥ 480 × 360; 15–60 fps; ≥ 1 caption span with **exactly one person** | Same, plus **caption coverage ≥ 40%**; multi-signer videos kept | — | — | — |
| Human filter | 3 native signers, per video or per channel | 1 author, per channel, sorted by hours; outlier audit | — | — | — |
| Caption filter | ≤ 300 characters, 0.2–60 s, one person in the span | — | **Random multi-caption clips** (34 s window, 512 frames at half rate) with timestamp tokens, which tolerate misalignment | Clips with several people (YOLOv8-nano) dropped | — |
| Pose features | MediaPipe Holistic, 85 points (21 per hand, 6 body, 37 face), x/y/z, unit-box normalisation per clip, every second frame (about 12–15 fps) | Same | Same | 104 points, 2D (21 + 21 + 25 body + 37 face); **SignSpace** normalisation lifted zero-shot How2Sign BLEU from 0.73 to 2.17; interpolate gaps of 2–3 frames | MediaPipe, then hand and face crops (DINOv2) plus body pose; every second frame (14.89 fps) |
| Dedup / leakage | — | — | — | Own 90:10 split | **Removed clips intersecting OpenASL**, back-filled from YouTube-SL-25 to keep 984 h |

### 1.3 Concrete selection and filtering recipe for this project

Rules marked **[pub]** come from the papers above. **[ours]** marks our proposals, to be calibrated on the Y2 pilot of about 50 h. Each threshold should be reported with its keep rate.

1. **Video gate** [pub]
   - YouTube-ASL heuristics: manual English captions only; 10 s–5 h; ≥ 480 × 360; 15–60 fps.
   - YouTube-SL-25 caption coverage ≥ 40%.
   - Record the YouTube licence field (Standard / CC BY) and the caption kind in provenance, as the plan already requires.
2. **English language check** [ours]
   - The caption track language must be `en`, and the caption text must pass a text language-ID check (fastText or CLD3, p ≥ 0.9).
   - For YouTube-SL-25 rows outside YouTube-ASL, also require the YouTube-SL-25 metadata label `ase`.
   - The metadata labels come from heuristics on titles, descriptions and caption language, with manual correction [P2 §2.2].
3. **Caption gate** [pub + ours]
   - ≤ 300 characters; 0.2–60 s [pub].
   - For sentence-level training of the back-translator, keep 1–20 s [ours].
   - Also keep the FLEURS-ASL-style random multi-caption windows (up to about 30 s), which are robust to misalignment [pub, P5].
4. **Single signer** [pub]: exactly one detected person for the whole span (YOLO or RTMPose person count). Drop the clip otherwise [P1][P10].
5. **Pose quality** [ours]
   - Dominant-hand landmarks in ≥ 70% of frames.
   - Face mesh in ≥ 90% of frames.
   - Both wrists inside the frame in ≥ 90% of frames.
   - Median hand-landmark confidence ≥ 0.5. Uni-Sign uses hand-keypoint confidence as a reliability score [P12 §3].
   - Signer height ≥ 35% of frame height.
   - Interpolate gaps of up to 3 frames [pub, P10].
6. **Frontal view** [ours]: head yaw from the face landmarks |yaw| < 30°, and shoulder-width ratio ≥ 0.6 of the clip's maximum.
7. **Alignment plausibility** [ours]
   - Rate between 0.5 and 4 English words per second of caption.
   - Hand motion energy inside the span ≥ 0.3× the video median, which rejects captions over pauses or interstitials. YouTube-SL-25 found about 1% interstitials [P2].
8. **Generation-grade tier** [ours, motivated by P1 §3.2 and P2 §2.2 with fn. 6–7] for generator fine-tuning, a strict subset:
   - channels with no speech track (Deaf-produced; captions "virtually always well-aligned");
   - none of: sign-along songs, live-interpreted events, obvious learners;
   - a reviewer check of the top 200 channels by hours, done by a Deaf ASL reviewer under the human-evaluation protocol.
9. **Channel balancing** [ours]: cap any one channel at 5% of the hours sampled per epoch. The two largest YouTube-ASL channels are the OpenASL news channels [P1 §3.3].
10. **Dedup and leakage** [pub + ours]
    1. Union YouTube-ASL, YouTube-SL-25 (ASL) and OpenASL by `yid`, then by caption-span overlap ≥ 50%.
    2. Remove captions that are MinHash near-duplicates (Jaccard ≥ 0.8 on word 5-grams) of any How2Sign, FLEURS-ASL, ASLLRP or DailyMoth-70h val or test sentence.
    3. Never evaluate on OpenASL test or val after YouTube-ASL pretraining; more than 72% overlap [P8].
    4. If DailyMoth-70h val or test is ever used, remove DailyMoth channel videos uploaded from February 2022 on from YouTube-SL-25.
    5. Keep a **channel-disjoint** monitor split of about 1% for YouTube data. It is training-time monitoring only, as the plan already says.

---

## 2. Expected gains reported in the literature

All scores are How2Sign test, sacreBLEU BLEU-4 (and BLEURT-20 where reported), unless noted. "→ H2S" means fine-tuned on How2Sign.

### 2.1 Pose-only inputs (closest to this project)

| System | Pretraining data | How2Sign only | Zero-shot (no H2S) | With YouTube data → H2S |
|---|---|---|---|---|
| YouTube-ASL baseline: MediaPipe + T5.1.1-Base [P1] | YouTube-ASL, 984 h | 1.22 (29.98); 0.86 from scratch | **3.95** (34.98) | **12.39** (46.63); 11.89 if mixed |
| YouTube-SL-25: T5-v1.1-Small [P2] | ASL subset, 1,394 h / full set, 3,207 h | 2.6 (31.2) with text-only T5 | 3.6 (30.2) / 4.2 (29.7) | **14.5** (46.6) / **15.4** (47.9) |
| Uni-Sign, pose-only (GCN + mT5) [P12] | YouTube-ASL | — | — | **14.5** (48.6) |
| Zhang et al., Scaling SLT (MediaPipe, ByT5/mT5) [P6] | about 6,600 h private plus MT and augmented data | — | — | **21.1** (55.7), reported via [P8]. Going from YT-ASL to YT-Full gained about 7 BLEURT. |
| Tanzer 2024a (fingerspelling) [P8 table] | about 2,800 h private | — | — | 18.1 (50.8); modality as in [P6] **(unverified)** |
| Železný et al. (pose T5, 2D 104 points) [P10] | YouTube-ASL keypoints | — | 2.17 zero-shot after normalisation study | — |
| POSESTITCH-SLT (pose, synthetic pretraining) [P16] | Template sentences (no YouTube) | 1.97 → 4.56 | — | — |
| Human, sentence-level [P9] | — | **19.8** (56.6); 21.5 with context | — | — |

**Other evaluation sets**

- **FLEURS-ASL zero-shot, pose:**
  - T5-Small on YouTube-SL-25 ASL: 3.4 BLEU / 33.7 BLEURT; on the full set: 4.4 / 40.1 [P2].
  - The FLEURS-ASL random-clip baseline: 3.7 / 37.2, against 2.9 / 33.6 for sentence-level clips [P5].
  - The human baseline is 13.0 / 64.6.
- **OpenASL** without YouTube data: 6.72 (I3D [P3]) and 13.21 (C2RL, via [P12]).

### 2.2 RGB or hybrid inputs (for comparison)

| System | Input | H2S only | YouTube-ASL → H2S | Notes |
|---|---|---|---|---|
| Tarrés et al. 2023 [via P1] | I3D video features | 8.03 | — | — |
| **SSVP-SLT** [P4] | Face-blurred RGB; MAE (SignHiera) plus T5 | 7.0 (39.3) | 14.7 (49.3); **15.5 (49.6)** with language-supervised pretraining; **7.1 zero-shot** | Longest run took about two weeks on 64 A100s. Zero-shot RGB (7.1) beats zero-shot pose (4.0). |
| SignMusketeers [P7] | DINOv2 face and hand crops plus body pose | — | **14.3** (+1.9 over the pose baseline) | Retrieved 98.7% of the clips |
| **SHuBERT** [P8] | Self-supervised, multi-stream (hand and face crops plus pose) | — | **16.2 (49.9)**; OpenASL 23.2 (60.6); FLEURS-ASL 4.7 (41.2) | Pretraining on 10% of the data cut BLEU from 2.2 to 0.7 in a frozen-encoder ablation, so data size clearly matters. |
| Uni-Sign, RGB plus pose [P12] | RTMPose plus RGB crops; mT5 | — | 14.9 (49.4); OpenASL 23.14 (60.40), *contaminated* [P8] | — |
| AttnSign (2026) [P17] | Vision-language model (InternVL-2.5-1B), raw video | 9.45 | **15.54** (50.44); OpenASL + YT 20.75, *probably contaminated* | Gemini-2.5-Pro zero-shot: 1.41 BLEU |

### 2.3 Reading for this project

- **The gain comes from the data, not the architecture.** Pose-only and RGB systems all land at 14–16 BLEU on the same public data. Pose-only reaches 15.4 with YouTube-SL-25 [P2]. Only private data at 3–6× the scale goes past 18 [P6][P8].
- **Instruments.**
  - Pretraining our back-translator on YouTube-ASL and fine-tuning it on How2Sign should move it from 2.1 to roughly the 8–14 BLEU band. The literature's 12–15 used a pretrained T5 with MediaPipe landmarks, and our canonical 3D rotations differ.
  - Retrieval (R@3) has no direct published analogue, so treat the plan's 0.35–0.45 as an estimate.
  - Following the instrument-skepticism rule, validate on FLEURS-ASL zero-shot. It is out of domain for How2Sign and YouTube alike.
- **Generation.**
  - No paper ablates YouTube pretraining for text-to-sign. SignLLM's ASL data is How2Sign only.
  - SignVerse-2M (DWPose of YouTube-SL-25) reports text-to-pose back-translation of 9.65–10.10 BLEU-4 on its ASL test split, but **with no ablation** [P11]. Its back-translator (SignX) reportedly scores 25–28 BLEU-4, which is unusually high against the numbers above. That is a single source, so treat it with caution.
  - YouTube-ASL's and YouTube-SL-25's own authors expect their data to be less suitable for generation [P1][P2].
  - The plan's "generator 1.5–3×" is therefore a hypothesis. Test it with a held-out A/B on How2Sign and ASLLRP judges.
- **Non-manual grammar.** None of these corpora has NMM annotation; ASLLRP still supplies it.

---

## 3. Practical acquisition

### 3.1 How others downloaded

- **OpenASL**
  - `prep/download.py` reads the unique `yid`s from the TSV.
  - It calls **yt-dlp**: `-f bestvideo+bestaudio[ext=m4a]/bestvideo+bestaudio/best --merge-output-format mp4`.
  - It can shard over Slurm (default 100 shards, `slurm_array_parallelism` 100, 240-minute timeout).
  - It writes `missing.txt` for videos that failed.
  - It switched from youtube-dl to yt-dlp on 2025-07-22 [R3].
- **SHuBERT (TTIC):** `dataset/download_youtubeasl.sh` starts **256 Slurm jobs** of `download_youtubeasl.py`, with batch size 3,000, a 25,200 s limit and a `problem_file`. Clip cutting, cropping, MediaPipe, and hand and face feature scripts follow [R4].
- **SignVerse-2M (Rutgers)**
  - A manifest-driven pipeline: metadata, then subtitles and video in parallel, then DWPose at 24 fps from the ffmpeg stdout stream.
  - It writes `poses.npz` and `caption.json` plus a `.complete` marker per video, deletes raw video once extraction finishes, and uploads shards to Hugging Face [P11].
  - This matches our plan's "delete after extraction" rule.
- **Throughput.** **No paper reports download throughput or failure rates beyond clip counts.**
  - yt-dlp's documentation gives YouTube's own limits: about **300 videos per hour for a guest session** and about 2,000 per hour for a signed-in account. It recommends **5–10 s sleeps** and warns that signed-in scraping risks the account being banned "temporarily or permanently" [L5].
  - At the guest rate, YouTube-ASL's 11,093 videos take about 37 IP-hours, and the YouTube-SL-25 ASL rows (16,724) take about 56 IP-hours. This is arithmetic from the stated limits, not a measurement.
  - Raw size at 720p is very roughly 0.5–1.5 GB per hour of video, so on the order of 1 TB in transit for YouTube-ASL. That is an estimate; measure it in Y2.

### 3.2 Known problems

1. **Removed or private videos and edited captions.** By design, the ID-only release reflects deletions [P1 App. B]. The 2024 retrieval got 98.7% of caption clips [P7]. Expect lower in late 2026 (unmeasured), and record the rate in Y2.
2. **Captions through the official API are not possible.**
   - `captions.download` requires the caller to "have permission to edit the video" and costs 200 quota units [L8].
   - So every group fetches captions with yt-dlp's subtitle options. Ask for manual subtitles only: `--write-subs --sub-langs "en.*"`, and never `--write-auto-subs`, to match YouTube-ASL's rule.
   - SignVerse-2M's `caption.json` already holds the subtitle segments for all YouTube-SL-25 IDs, so it avoids fetching captions at all [R7].
3. **Bot checks on datacenter IPs.**
   - Invidious documentation states that YouTube blocks datacenter IP addresses [L7].
   - yt-dlp now needs **PO tokens** for most clients. Without them requests "may return HTTP Error 403, or result in your account or IP address being blocked" [L6a].
   - The main PO-token provider says passing PO tokens "no longer bypass[es] the bot check for [the] majority of cases" [L6].
   - Under this project's rules (no circumvention of bot checks, sign-in walls or CAPTCHAs), a datacenter block means **stop and choose an alternative** (§3.3). Do not rotate proxies, spoof tokens, or use account cookies.
4. **Unofficial re-hosts.** Hugging Face holds several repositories that look like re-uploaded YouTube-ASL or OpenASL video archives, for example `ClaireTAN/YouTubeASL`, `PhoenixHu/youtube_asl_clips_224` and `ClaireTAN/OpenASL`. They have **no licence or dataset card and no stated provenance** [R8]. **Do not use them.** They redistribute creators' video, which is what the ID-only releases were designed to avoid.

### 3.3 Pre-extracted features that avoid re-downloading video

| Release | Coverage | Content | Size | Licence | Fit for this project |
|---|---|---|---|---|---|
| **YouTube-ASL Clip Keypoint Dataset** (Železný, Hrůz, Straka, Gueuwou; UWB, JSALT 2024) [R6][P10] | **390,547** YouTube-ASL sentence clips that passed their single-person filter | **MediaPipe 2D**, 104 points (208-dimensional) per frame, in JSON, plus annotation JSON (caption text and timing are likely but **unverified**) | **348 GB** (336.7 × 10⁹ bytes), 13 files | **CC BY 4.0** | Good for pretraining the 2D "second-opinion" back-translator and for pre-screening clips. No z, no blendshapes, no rotations. |
| **SignVerse-2M** (Fang, Zhong, Zhang, Metaxas; Rutgers, 2026) [P11][R7] | All of YouTube-SL-25 (39,196 videos, about 2M clips, 55+ languages, including `ase`) | **DWPose 2D** body, hands and face with confidence, per person, in `poses.npz`; **`caption.json` with subtitle segments and provenance**; a variant without face landmarks | **599 GB**, `.tar` shards | **CC BY-NC 4.0** (derived keypoints and subtitle metadata only). Creators can request removal, handled within 7 business days. | The cheapest route to captions plus a person count and hand confidence for **every** YouTube-SL-25 ASL video. Use it to decide what to download, so we fetch only videos with usable clips. It is 2D only and not our canonical representation. |
| **Uni-Sign** Hugging Face model repo (USTC) [R9] | How2Sign, OpenASL, WLASL, CSL-Daily | Pose-format zips (`how2sign_pose_format.zip.*`, `openasl_pose_format.zip.*`) plus checkpoints | — | Repository field **cc-by-nc-4.0** | OpenASL keypoints without crawling. Note the tension with OpenASL's ND licence. |
| How2Sign official [R10] | How2Sign | OpenPose 2D (B-F-H) keypoints | — | CC BY-NC 4.0 | Already in use |

**Suggested acquisition path** (it amends Y1–Y3 without changing the plan's rules):

1. Ingest the SignVerse-2M ASL shards' `caption.json` files and pose confidences (CC BY-NC) for metadata only.
2. Apply the §1.3 filters to rank and select clips before any video download.
3. Pilot downloading the ranked set from one CPU pod (Y2).
4. If the pilot is blocked, pretrain the instruments on the UWB 2D keypoints (CC BY 4.0) and ask the owner whether to acquire at a low rate from a non-datacenter connection. That is a policy decision with ToS implications (§4.3); this report does not recommend it either way.

---

## 4. Permission to release derived weights

### 4.1 What each licence says about models and derived works

| Dataset | Licence / terms (quoted or closely paraphrased) | Trained models addressed? | Implication for releasing weights |
|---|---|---|---|
| **YouTube-ASL** [P1] | IDs under CC BY 4.0. "The actual videos/captions on YouTube are preexisting and subject to the YouTube Terms of Service." | No | The ID licence asks only for attribution. The real rights holders are the creators, and the ToS governs access (§4.3). The authors cannot license the videos. |
| **YouTube-SL-25** [P2][R2] | Not stated **(unverified)**. Video terms as above. | No | Ask Tanzer what licence the ID list carries. |
| **OpenASL** [R3] | **CC BY-NC-ND 4.0.** You may "produce and reproduce, but not Share, Adapted Material". | No | Whether trained weights count as "Adapted Material" of the annotations is legally unclear. Get explicit permission, or leave OpenASL out of release builds. |
| **ASLLRP / DAI 2 / Sign Bank** [R11] | "Can be used for research and education purposes, but cannot be redistributed without permission. Commercial use, without explicit permission, is not allowed, nor are any patents and copyrights based on this material." Requires citations and URLs. | No | Releasing weights under any licence asserts rights in material derived from ASLLRP. **Get written permission.** Our SignStream form already declared that derived data stays private. |
| **ASLLVD** [R12] | Same terms (now distributed through the Sign Bank); "do not use the 2008 data" | No | Same as ASLLRP |
| **ASL Citizen** [R13] | "Solely for non-commercial, non-revenue generating, research purposes". "You may not distribute the data or your modifications to the data". Publications may include no "material or substantial portion". Personal data must not be used to contact anyone and must be destroyed when the research ends. | No | Whether weights are "modifications" is ambiguous. **Ask Microsoft.** Note that the MSR team released its own baseline checkpoints [R14]. |
| **ASL STEM Wiki** [P13] | "Permits use for research purposes" (full text on the MSR licence tab) | **Unverified** | Ask together with ASL Citizen (same contact) |
| **How2Sign** [R10] | **CC BY-NC 4.0**: credit, link the licence, indicate changes, "may NOT use the material for commercial purposes". The How2 text has its own terms. | No | Non-commercial sharing of adapted material with attribution is allowed. Send a courtesy notice. |
| **DailyMoth-70h** [P4][R5] | **CC BY-NC 4.0** | No | Non-commercial release is fine with attribution |
| **FLEURS-ASL** [P5] | **CC BY-SA 4.0** | No | Evaluation only. Training would bring share-alike into play. |
| **SignVerse-2M** [R7] | **CC BY-NC 4.0**, research use; no re-identification; no safety-critical use | No | Weights released NC if it is used |
| **UWB keypoints** [R6] | **CC BY 4.0** | No | Attribution only. This does not cleanse the underlying creators' rights. |

### 4.2 Contacts (only publicly listed ones)

- **YouTube-ASL** (Google)
  - David Uthus, **duthus@google.com**; Garrett Tanzer, **gtanzer@google.com**; Manfred Georg, **mgeorg@google.com** (listed in the paper).
  - The datasheet route is "By contacting any of the authors listed on the publication"; updates go via the GitHub page [P1 App. B.7].
- **YouTube-SL-25** (Google / Google DeepMind): Garrett Tanzer, correspondence address **gtanzer@google.com** [P2]. Biao Zhang, Google DeepMind, **biaojiaxing@google.com** (listed in [P6]).
- **FLEURS-ASL:** Garrett Tanzer, **gtanzer@google.com** [P5].
- **OpenASL** (TTIC / UChicago)
  - Bowen Shi **bshi@ttic.edu** (now at FAIR, Meta, per [P4]); Karen Livescu **klivescu@ttic.edu**; Greg Shakhnarovich **greg@ttic.edu**; Diane Brentari **dbrentari@uchicago.edu** [P3].
  - Repository issues: `github.com/chevalierNoir/OpenASL/issues`.
- **ASLLRP, DAI 2, Sign Bank and ASLLVD** (Boston University, with Rutgers): Prof. Carol Neidle, **carol@bu.edu**, BU ASLLRP, Linguistics Dept, Boston, MA 02215 [R15].
- **ASL Citizen and ASL STEM Wiki** (Microsoft Research)
  - **ASL_Citizen@microsoft.com**, which the project page names as the route for commercial or other licensing questions [R13].
  - PI Danielle Bragg (MSR). Boston University co-leads include Naomi Caselli.
- **How2Sign** (UPC / BSC / CMU / Gallaudet): Amanda Duarte, **amanda.duarte@upc.edu** (written "amanda.duarte[at]upc.edu" on the site); data issues on the GitHub repository [R10].
- **DailyMoth-70h** (Meta FAIR): `github.com/facebookresearch/ssvp_slt` issues. The underlying rights sit with the host of The Daily Moth, under the licence agreement described in [P4].
- **UWB keypoints:** Tomáš Železný **zeleznyt@kky.zcu.cz**; Marek Hrůz **mhruz@ntis.zcu.cz** (repository metadata [R6]).
- **SignVerse-2M:** removal and permission requests through the Hugging Face dataset discussions [R7].

### 4.3 Precedent: public models trained on these datasets

| Model | Trained on | Weights | Licence |
|---|---|---|---|
| **Uni-Sign** (USTC, ICLR 2025) [P12][R9] | YouTube-ASL pretraining; How2Sign and OpenASL fine-tuning; CSL-News | `ZechengLi19/Uni-Sign` on Hugging Face (`how2sign_pose_only_slt.pth`, `openasl_pose_only_slt.pth`, …) | **cc-by-nc-4.0** |
| **SHuBERT** (TTIC, ACL 2025) [P8][R4] | YouTube-ASL (minus OpenASL) plus a YouTube-SL-25 back-fill | Google Drive link in the README | Code MIT. Weights licence **not stated**. |
| SSVP-SLT (Meta) [P4][R5] | YouTube-ASL, How2Sign, DailyMoth-70h | No checkpoint link found **(unverified)** | Code CC BY-NC 4.0 |
| ASL Citizen baselines (Microsoft) [R14] | ASL Citizen | GitHub release `checkpoints_v1` | Code MIT; weights not stated |
| SignVerse-2M data (Rutgers) [R7] | Derived from YouTube-SL-25 | Pose data, not weights | CC BY-NC 4.0 with a takedown process |
| YouTube-ASL / YouTube-SL-25 baselines (Google) | — | **Not released** (none found) | — |
| SignGemma (Google DeepMind, announced May 2025) [L9] | Undisclosed | Release **not confirmed** as of this search | — |
| ASLLRP | — | SignX (2025, with Neidle as co-author) exists. No public weights confirmed **(unverified)**. | — |

**Reading.** Academic groups do release YouTube-ASL-trained weights, mostly under **CC BY-NC 4.0**, with no public objection on
record. That is precedent, not permission. No precedent was found for public weights trained on ASLLRP data without the
group's involvement.

### 4.4 YouTube Terms of Service and creator copyright (not legal advice)

- **ToS (effective December 15, 2023) [L1]**
  - The Terms forbid users to "access, reproduce, download, distribute, … alter, modify" content except as the Service authorizes or with YouTube's prior written permission.
  - They forbid accessing the Service "using any automated means (such as robots, botnets or scrapers)" except for public search engines under robots.txt or with prior written permission.
  - They forbid collecting "any information that might identify a person".
  - These clauses bind the *person doing the acquisition*. They are a separate issue from whether the weights may be released.
- **How the dataset papers handle it.**
  - YouTube-ASL releases IDs only, so that "changes to the underlying videos are automatically reflected", and says the videos "fall under YouTube's Terms of Service" [P1 §7, App. B].
  - It had no IRB review, and creators were "not notified" [P1 App. B.3].
  - Its baselines train on reduced poses "as a form of anonymization" [P1 §7]. YouTube-SL-25 does the same [P2 Ethics].
  - SSVP-SLT pretrains on face-blurred video [P4].
- **YouTube's stated position.**
  - In April 2024 YouTube's CEO said training a model (OpenAI's Sora) on YouTube videos would be a "clear violation" of the terms [L3].
  - In December 2024 YouTube added a creator setting for *third-party AI training*. It is **off by default** and not retroactive [L4].
  - Whether this setting can be read publicly per video is **unverified (likely not)**.
  - The 2024 "YouTube Subtitles" reporting shows the reputational exposure of training on YouTube content without permission [L10].
- **Copyright.**
  - Creators own their videos. The YouTube **CC BY** licence option exists only for content the uploader can license and is shown in video metadata [L2]. The plan's provenance field makes a CC BY-only subset possible; how large it is, is unknown.
  - In the US, two June 2025 district-court rulings (*Bartz v. Anthropic*, *Kadrey v. Meta*) found LLM training to be fair use on their records. Commentators stress that the rulings are narrow and fact-specific, and that market harm is decisive [L11].
  - In the EU, the DSM Directive's Art. 3 text-and-data-mining exception for *research organisations* cannot be overridden by contract. The general Art. 4 exception can be opted out of by rights holders [L12]. An individual may not qualify as a "research organisation" **(unverified)**.
- **Personal data.** Signers' faces and bodies carry the language and also identify them. Pose reduces but does not remove this [P1 §7][P4].
- **Practical mitigations** (based on these sources):
  - release **weights only**, never video, captions or keypoints;
  - a **non-commercial research licence** with use restrictions (no re-identification, no safety-critical interpretation), modelled on SignVerse-2M's terms;
  - a model card listing corpora, channel counts, filters and known biases (YouTube-SL-25 estimates 1.9% of hours at Monk skin tone 6–10 [P2 §2.3]);
  - a **takedown or opt-out route** for creators, as SignVerse-2M does (7 business days);
  - re-check that each ID is still available before each training run, and drop deleted videos from future runs;
  - test generator outputs for memorisation of individual signers.

### 4.5 Draft permission email (for the owner to send)

> **Subject:** Permission request — releasing research model weights trained in part on [DATASET]
>
> Dear [Name],
>
> I am [owner name], working on an open research project that generates American Sign Language motion from
> English text (pose-based: 3D body and hand rotations plus facial blendshapes; no video is generated or
> redistributed). Our training data includes [DATASET], used under its terms of use [licence / terms URL] for
> non-commercial research, and stored privately with access restricted to the project.
>
> We intend to release the trained **model weights** (not the data, videos, captions or extracted keypoints) as a
> **non-commercial research model**, with a model card that cites [DATASET] and its papers, documents how it was
> used, and states restrictions (no re-identification of signers, no safety-critical use). Because [DATASET]'s terms
> [quote the relevant clause, e.g. "cannot be redistributed without permission"], we would like to ask:
>
> 1. May we publicly release weights trained in part on [DATASET] under a non-commercial research licence?
> 2. Are there conditions you would like us to follow (attribution wording, licence choice, a review before release,
>    or excluding particular subsets)?
> 3. If you prefer that [DATASET] not be included in a public release, we will release a variant trained without it.
>
> We are glad to share the evaluation results and the model card before release, and to credit your group as you
> prefer. Thank you for creating and maintaining [DATASET].
>
> Best regards,
> [Name] · [affiliation, if any] · [project URL] · [contact email]

Variants: for YouTube-ASL, YouTube-SL-25 and How2Sign, phrase it as a *notice and courtesy request*, since
attribution-only or NC terms already apply. For ASLLRP and ASL Citizen, it is a *permission request*. For
OpenASL, it is a permission request citing the ND clause.

---

## 5. Open questions to settle in the Y2 pilot

- Current availability rate for YouTube-ASL and YouTube-SL-25 ASL IDs, and the share whose manual English captions still exist.
- Keep rates for each §1.3 filter, and the size of the generation-grade tier in hours.
- Whether datacenter acquisition is blocked at all from the pod region; stop on the first bot check.
- How many hours carry the YouTube **CC BY** licence field.
- Whether our 3D canonical extractor's hand coverage on YouTube video matches How2Sign's. It may not, given frame rates as low as 15 fps and varied framing.

---

## 6. Unverified items (summary)

- The licence of the YouTube-SL-25 ID list, and the columns of `youtube-sl-25-metadata.csv` (file not opened).
- Whether the UWB keypoint annotation JSON includes caption text.
- The full licence text of ASL STEM Wiki, and whether weights trained on it may be released.
- SSVP-SLT, SignX and SignGemma checkpoint availability.
- The input modality of Tanzer 2024a (18.1 BLEU).
- Whether YouTube's third-party training setting is publicly visible per video.
- Whether an individual qualifies under EU DSM Art. 3.
- The throughput and storage estimates in §3.1 are arithmetic, not measurements.

---

## 7. Sources

**Papers (full text read)**

- [P1] Uthus, Tanzer, Georg. *YouTube-ASL.* NeurIPS 2023 D&B. https://arxiv.org/abs/2306.15162
- [P2] Tanzer, Zhang. *YouTube-SL-25.* ICLR 2025. https://arxiv.org/abs/2407.11144 · https://openreview.net/forum?id=nFVsK3QLgs
- [P3] Shi, Brentari, Shakhnarovich, Livescu. *Open-Domain Sign Language Translation Learned from Online Video* (OpenASL). EMNLP 2022. https://arxiv.org/abs/2205.12870
- [P4] Rust, Shi, Wang, Camgöz, Maillard. *Towards Privacy-Aware Sign Language Translation at Scale* (SSVP-SLT, DailyMoth-70h). ACL 2024. https://arxiv.org/abs/2402.09611
- [P5] Tanzer. *FLEURS-ASL.* NAACL 2025. https://arxiv.org/abs/2408.13585 · data: https://www.kaggle.com/datasets/googleai/fleurs-asl
- [P6] Zhang, Tanzer, Firat. *Scaling Sign Language Translation.* NeurIPS 2024. https://arxiv.org/abs/2407.11855
- [P7] Gueuwou, Du, Shakhnarovich, Livescu. *SignMusketeers.* https://arxiv.org/abs/2406.06907
- [P8] Gueuwou, Du, Shakhnarovich, Livescu. *SHuBERT.* ACL 2025. https://arxiv.org/abs/2411.16765 · http://shubert.pals.ttic.edu
- [P9] Tanzer, Shengelia, Harrenstien, Uthus. *Reconsidering Sentence-Level Sign Language Translation.* https://arxiv.org/abs/2406.11049
- [P10] Železný et al. *Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights.* 2025. https://arxiv.org/abs/2507.01532 · code: https://github.com/zeleznyt/T5_for_SLT
- [P11] Fang, Zhong, Zhang, Metaxas. *SignVerse-2M.* 2026. https://arxiv.org/abs/2605.01720 · https://signerx.github.io/SignVerse-2M
- [P12] Li et al. *Uni-Sign.* ICLR 2025. https://arxiv.org/abs/2501.15187
- [P13] *ASL STEM Wiki.* EMNLP 2024. https://arxiv.org/abs/2411.05783 · https://www.microsoft.com/en-us/research/project/asl-stem-wiki/
- [P14] Ni, Cheng, Li, Cheng. *Sign-Language Datasets at Scale: A Comprehensive Survey.* 2026. https://arxiv.org/abs/2606.19352
- [P15] Chen, Lin. *SignDATA: Data Pipeline for Sign Language Translation.* 2026. https://arxiv.org/abs/2604.20357
- [P16] Joshi et al. *POSESTITCH-SLT.* 2025. https://arxiv.org/abs/2511.00270
- [P17] Hu et al. *Attention-Steered Vision-Language Models for Sign Language Translation* (AttnSign). 2026. https://arxiv.org/abs/2608.00235
- Also consulted: Javorek et al., *Beyond a Single Reference* (2026), https://arxiv.org/abs/2601.21128 · Fang et al., *SignLLM* (ICCV-W 2025), https://arxiv.org/abs/2405.10718

**Repositories and dataset pages**

- [R1] https://github.com/google-research/google-research/tree/master/youtube_asl · bucket listing `gresearch/youtube-asl/youtube_asl_video_ids.txt`
- [R2] https://github.com/google-research/google-research/tree/master/youtube_sl_25 · bucket listing `gresearch/youtube-sl-25/youtube-sl-25-metadata.csv`
- [R3] https://github.com/chevalierNoir/OpenASL (README, `LICENSE.md`, `prep/download.py`, TSV header)
- [R4] https://github.com/ShesterG/SHuBERT (README, `DATASETS.md`, `dataset/download_youtubeasl.sh`)
- [R5] https://github.com/facebookresearch/ssvp_slt (README, `DATASETS.md`)
- [R6] YouTube-ASL Clip Keypoint Dataset, http://hdl.handle.net/11234/1-5898 (LINDAT item metadata) · mirror https://data-dspace.zcu.cz/items/ec23a038-4a87-46ae-aff2-29665739afb9
- [R7] https://huggingface.co/datasets/SignerX/SignVerse-2M (card: licence, takedown policy, size)
- [R8] Hugging Face dataset search API results for "youtube-asl" and "openasl" (metadata only)
- [R9] https://huggingface.co/ZechengLi19/Uni-Sign (file list, licence field) · https://github.com/ZechengLi19/Uni-Sign
- [R10] https://how2sign.github.io/
- [R11] https://www.bu.edu/asllrp/dai-terms.html
- [R12] https://www.bu.edu/asllrp/av/dai-asllvd.html
- [R13] https://www.microsoft.com/en-us/research/project/asl-citizen/ · licence tab https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
- [R14] https://github.com/microsoft/ASL-citizen-code
- [R15] https://dai.cs.rutgers.edu/dai/s/dai

**Terms, legal and acquisition sources**

- [L1] YouTube Terms of Service (effective 2023-12-15). https://www.youtube.com/static?template=terms
- [L2] YouTube Help, Creative Commons. https://support.google.com/youtube/answer/2797468
- [L3] Engadget, *YouTube CEO warns OpenAI that training models on its videos is against the rules* (April 2024). https://www.engadget.com/youtube-ceo-warns-openai-that-training-models-on-its-videos-is-against-the-rules-121547513.html
- [L4] TechCrunch, *YouTube will let creators opt out of third-party AI training* (2024-12-16). https://techcrunch.com/2024/12/16/youtube-will-let-creators-opt-out-of-third-party-ai-training · Tubefilter, https://www.tubefilter.com/2024/12/17/youtube-generative-ai-training-creators-opt-in-consent/
- [L5] yt-dlp wiki, Extractors (YouTube rate limits, cookies and ban risk). https://github.com/yt-dlp/yt-dlp/wiki/Extractors · FAQ https://github.com/yt-dlp/yt-dlp/wiki/FAQ
- [L6] bgutil-ytdlp-pot-provider (PyPI; PO tokens no longer bypass the bot check in most cases). https://pypi.org/project/bgutil-ytdlp-pot-provider · [L6a] yt-dlp PO Token Guide, https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
- [L7] Invidious documentation (YouTube blocks datacenter IPs). https://git.hackliberty.org/iv-org/documentation/commit/3cac3e4b5bba9dadbf820a14bda92ba37b91e1cd
- [L8] YouTube Data API, captions.download. https://developers.google.com/youtube/v3/docs/captions/download
- [L9] Slator / MultiLingual coverage of SignGemma (May 2025). https://multilingual.com/google-signgemma-on-device-asl-translation/
- [L10] Proof News, *Apple, Nvidia, Anthropic Used Thousands of Swiped YouTube Videos to Train AI* (2024). https://proofnews.org/apple-nvidia-anthropic-used-thousands-of-swiped-youtube-videos-to-train-ai/
- [L11] Debevoise, *Anthropic and Meta Decisions on Fair Use* (June 2025). https://www.debevoise.com/insights/publications/2025/06/anthropic-and-meta-decisions-on-fair-use · Manatt, https://www.manatt.com/insights/newsletters/client-alert/making-sense-of-the-first-two-often-contradictory-sometimes-confounding-gen-ai-fair-use-rulings
- [L12] Kluwer Copyright Blog, *The New Copyright Directive: Text and Data Mining (Articles 3 and 4)*. https://legalblogs.wolterskluwer.com/copyright-blog/the-new-copyright-directive-text-and-data-mining-articles-3-and-4/
