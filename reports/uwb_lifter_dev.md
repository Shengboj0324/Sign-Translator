# UWB 2D-to-world lifter: development report (2026-10-30, 02:50 UTC; review fixes 03:30 UTC; re-review fixes 04:30 UTC; final-review fixes 05:30 UTC)

Branch `uwb-lifter` (from main `ab7df81`). Nothing ran on a pod; no training beyond a local smoke; R2 was read, and
written only under `scratch/uwb-lifter-dev/`.

## What was built

| Piece | File |
|---|---|
| z-free regressor features `world-regressor-features-xy-v1` (hand 40 inputs, pose 54); bundles record their feature version, v1/v2 behaviour and SHAs unchanged | `signtranslator/data_engineering/world_regressor.py` |
| `--features xy` training (same pairs; z dropped by the features; pseudo-world baseline at zero depth) | `scripts/train_world_regressors.py` |
| UWB clip adapter (parse, clock, normalise, learned or pseudo world, visibility, blendshapes, optional affine head) | `signtranslator/data_engineering/uwb_keypoints.py` |
| Converter: ranged reads, units of 1,000 members, shards, resumable, per-clip provenance | `scripts/convert_uwb_clips.py` |
| Paired gate: per-video fps estimate, cue pairing, lag alignment, bone errors, validity agreement, mirror check | `scripts/uwb_lifter_gate.py` |
| Framing-invariant features `world-regressor-features-xy-v2` (pose 50 inputs, no context); `--features xy2`; `--augment-framing K` for xy; `input_zscores` | `signtranslator/data_engineering/world_regressor.py`, `scripts/train_world_regressors.py` |
| Training-record selection that refuses unchecked clips and drops leaks and duplicates | `signtranslator/data_engineering/uwb_manifest.py` |
| Outside-frame accuracy of visibility rules on How2Sign two-view truth | `scripts/uwb_outside_frame_accuracy.py` |
| Development videos excluded from the declared gate (70) | `reports/uwb_gate_smoke_videos.txt` |
| Tests (synthetic, offline): see the review section for counts | `tests/test_de_world_regressor_xy.py`, `tests/test_de_uwb_keypoints.py`, `tests/test_convert_uwb_clips_script.py`, `tests/test_uwb_lifter_gate.py`, `tests/test_uwb_outside_frame_accuracy_script.py`, `tests/uwb_synthetic.py` |

## The real UWB format (checked before coding)

Sample: 120 random clips, 12 from each zip, 15,944 frames.

- 10 zips, 390,547 members, all `<video>.<first>-<last>.json` with no directories. 9,564 videos.
- Keys: `keypoints`, `cropped_keypoints`, `bbox_left_hand`, `bbox_right_hand`, `bbox_face`, `yolo_sign_space`, `sign_space`, `size`, `cropped_size`.
- Per frame: `pose_landmarks` (33), `left_hand_landmarks` (21), `right_hand_landmarks` (21), `face_landmarks` (478).
  - Every point is `[x, y]` in source pixels.
  - A part that was not detected is an empty list. There are no NaNs and no partial blocks.
  - Pose is missing on 75 of the 15,944 frames, face on 732.
- **`size` is `[height, width]`.** In 119 of 120 clips the face lies inside x ≤ size[1], y ≤ size[0]. With the bounds swapped, only 34 fit. `yolo_sign_space` agrees in 120 of 120.
- `cropped_keypoints` is a linear map of the same points into a 512² crop of `sign_space`. It is not used.
- **Frame count:** `last - first + 1` in 95 clips, `last - first` in 23, other in 2. The adapter tolerates ±2 frames and records the difference.
- **Hand sides:** the `left_hand_landmarks` wrist is nearer pose 15 on 99.3% of 10,886 frames. The `right_hand_landmarks` wrist is nearer pose 16 on 99.8% of 13,843. So the slots are the subject's own sides in the unmirrored upload, the same as in Sem-Lex. Pose 11 is image-right in 120 of 120 clips.
- **Translations:** 391,494 clip entries over 9,422 videos. 1,073 member clips have no translation. The converter keeps them with `text: null`.

## Frame rate: estimated, not assumed

`uwb_lifter_gate.py --fps-only` was run on every video we extracted that UWB also holds. That is 2,605 converted videos.

- Method: text-matched caption cues, then a least-squares fit of `frame = fps·t + c`, then a snap to standard rates.
- **The UWB frame index is `floor(start × fps)` at the video's native rate.** The fit offset is −0.5 frames, and the median per-video residual is 0.24 frames.
- Coverage:
  - 2,568 videos got an estimate: 2,547 by text fit, 21 by time grid.
  - 37 did not; they have too few cues.
  - The estimated videos hold **217,458 of the 390,547 UWB clips (55.7%)**.
- **Rates are not 30.** Among the estimates: 29.97 in 1,262 videos, 30 in 574, 23.976 in 461, 25 in 79, 24 in 55, plus some at 15, 6 and 20. 137 are unsnapped, mostly VFR averages around 29.8–29.9, with residuals as small as the snapped ones.
  - **26.9% of videos are not 30/29.97 fps.** An assumed 30 fps stretches those clips by up to 25%.
- Agreement with yt-dlp's rate in our reports: 2,339 of 2,568 within 0.2%.
  - The 153 videos we downloaded at 50/60 fps are at 25/30 in UWB. UWB downloaded a different stream than we did.
- Cue pairs (IoU ≥ 0.5, one-to-one): **214,209** across these videos. The plan's estimate was about 100k.
- Outputs:
  - table: `scratch/uwb-lifter-dev/uwb-fps-table-v1.json` (sha256 `ceaf7675949b99c6a7c797dc36229a21ed280456522e85e28d446ac5f9e07be0`)
  - full report: `scratch/uwb-lifter-dev/2026-10-29-gate-fps-only.json`

## Local smoke results (indicative only: the smoke bundle is not the deliverable)

### Smoke xy training

- Data: 230 of the 1,199 verified pairs files. Train signers 3, 5 and 8 (60 files each); validation signers 1 and 2 (25 each).
- Settings: 10 epochs, legacy Holistic inputs.
- Bundle: `scratch/uwb-lifter-dev/world-regressors-xy-smoke.pt`, sha256 `8200d463afd2db18078b0edde7b7fe460be4b4c08f82509ccbae80ecb9dd1557`.

Validation medians in degrees, through our IK, on the same frames:

| | xy smoke | v2 (with z, full training) | pseudo-world, z = 0 |
|---|---|---|---|
| Fingers L / R | 16.6 / 13.5 | 12.5 / 9.4 | 46.4 / 41.7 |
| Palm L / R | 15.5 / 11.8 | 12.2 / 7.9 | 69.3 / 62.8 |
| Upper arm L / R | 6.4 / 8.8 | 4.4 / 5.1 | 31.6 / 45.7 |
| Forearm L / R | 16.1 / 12.4 | 11.0 / 8.1 | 131 / 156 |

### Paired gate smoke

Real data, smoke bundle, `--head affine`: 30 videos, 119 cue pairs, all evaluated.

Median / p90 in degrees against our own Tasks-extracted states, on each method's own valid frames:

| Method | Upper arms | Forearms | Wrists | Fingers |
|---|---|---|---|---|
| learned (visibility head) | 15.0 / 27.3 | 22.8 / 48.8 | 15.3 / 51.9 | 18.2 / 58.1 |
| learned + inside-image visibility | 12.5 / 24.6 | 18.7 / 43.2 | 15.5 / 53.0 | 18.2 / 58.1 |
| pseudo-world (x, y/a, 0) | 19.3 / 32.7 | 41.2 / 64.4 | 63.5 / 99.8 | 57.4 / 109.3 |

95% CIs by video, for the learned method: upper arms [13.0, 17.2], fingers [17.3, 19.2].

- **After one global rotation per clip**, the learned arms drop to 10.5° and 17.0°. This is optimistic, since the rotation is fitted on the same frames. It points to a systematic frame offset between the How2Sign-crop-trained regressor and YouTube full frames.
- **Hand-valid agreement:** 0.90 left, 0.91 right.
- **Forearm-valid agreement** (frames where our chest is valid):
  - visibility head: 0.93 left, **0.41 right**. On 5,077 frames the head drops the right forearm where our extraction keeps it.
  - inside-image rule: 0.96 left, 0.85 right.
  - So `--visibility` is now a required converter flag; the full gate decides it.
- **Mirror check** (clips where the mirrored hypothesis fits better): 1 of 81 for learned-inside, 2 of 81 for pseudo. **UWB is not mirrored relative to our extraction.** The visibility-head variant shows 18 of 79, which follows from its bad right-arm validity.
- **Head rotation** (affine fit vs our face-geometry head): 3.97° median, 7.8° p90, over 17,903 frames. Promising, but it stays off by default until the full gate.
- **Alignment:**
  - 80 pairs aligned on arms, 39 had no arms in our state (the original solver needs both hips).
  - The lag minimum is shallow. With a 0.5° margin, 90 of 119 pairs keep lag 0.
  - Per-video lag drift is mostly under 1 frame/min. Two videos show about 3 and one about −9.6, worth a look in the full run.

### Converter smoke

- Ran 10 real clips from zip 1 into `scratch/uwb-lifter-dev/convert-smoke-2026-10-29/`. These are partial reports and do not mark units finished.
- Provenance, text, partition, fps source, frame-count difference and channel coverage were all recorded.
- The state clock id reads e.g. `youtube_asl_uwb_keypoints:51HD1pS4lE0:frame-index-at-30fps-assumed`.
- Speed: about 1.4 s per clip per worker with blendshapes, including R2 reads. The full set is roughly 150 worker-hours.

## Open risks

1. **FPS for the remaining 44% of clips.** These are in videos we never extracted, so they get `fps_source: assumed` (30). By our sample, about 27% of those are 24/25/15 fps.
   - Option: train only on `estimated` clips.
   - Option: accept stretched time with augmentation.
   - Option: get rates from video metadata. This would mean YouTube requests and is the owner's call.
2. **Holistic version.** UWB does not state its MediaPipe/Holistic version. Our inputs are legacy 0.10.14 at complexity 1, on Sem-Lex-framed How2Sign crops: 8 studio signers. YouTube framing, resolution and signers differ.
   - The global-rotation gain above suggests adding full-frame pairs to the training set, for example `collect_world_pairs.py`'s YouTube-staged branch and full How2Sign frames, before the real training.
3. **Visibility head domain shift.** See the right-forearm agreement of 0.41 above.
4. **Reference is not truth.** "Errors" are disagreement with our own monocular Tasks states.
   - Our arms exist only on hip-visible videos.
   - The Sem-Lex validation put our extraction at about 19° arm error against two-view truth.
5. **±1 frame ambiguity.** Frame counts can be `last - first`. Lag estimation is shallow, so small offsets are absorbed and barely move the metrics.
6. **UWB stream differs from ours.** 50/60 fps videos appear at 25/30 in UWB, so pixel sizes and encodes differ. Normalised coordinates absorb resolution, not encoding differences.
7. **Mirroring:** checked, none found (see above). Hand slots are anatomical.
8. **Not filtered:** evaluation-sentence leaks (How2Sign/ASLLRP val and test) and text quality. Manifests must do this, as `build_uwb_keypoint_shards.py` does.
9. **The pairs** (1,199 files plus their legacy twins, every SHA equal to the v2 training manifest) existed only in a `/tmp` scratchpad. They are now at `scratch/uwb-lifter-dev/world-pairs-v2/`:
   - `pairs.tar`: sha256 `6c4d3ecc18e66f2bbc885aacda9a86499d7f5ce48618fa2438540ca958526758`
   - `legacy-pairs.tar`: sha256 `7042e0a8bcd7452d13743fcac5f6b37be1854fafb06c417ab54fa8aa24c65fa0`
   - `INDEX.json`

## Review fixes (2026-10-30)

The independent review of `e186d62` approved with fixes and blocked conversion and training until items 1–4 were
fixed.

**Tests:** 52 UWB lifter tests after the final-review fixes (49 at 4c67724, 46 at 6625c83). The whole suite passes under `python -W error -m pytest`, as CI runs it: 3,972 passed.

Status of every item:

1. **Visibility (major).**
   - Done:
     - New declared rules: `learned`, `inside`, `margin:<m>`, plus the diagnostic `all`. `uwb_keypoints.visibility_margin` parses them.
     - The converter refuses a run without an explicit `--visibility`, and refuses `all`.
     - The gate takes a declared `--visibility` (the primary method) and scores candidate rules beside it.
     - Gate validity now covers all frames, split by framing (both hips inside the UWB frame or not). It reports hand agreement, kappa and UWB-only rate on every frame, forearm survival on every frame, and forearm agreement where our chest is valid.
     - `scripts/uwb_outside_frame_accuracy.py` scores each rule against How2Sign two-view truth. It reports survival against MediaPipe's, and the accuracy of what a rule keeps and drops, inside and outside the frame. Signer 9 is refused.
   - The "visibility head retrained with crop/translation augmentation" candidate is the xy-v2 head. It cannot see the frame at all (item 2).
   - Dev results (smoke bundles, 30 new videos, smoke videos excluded):

| Rule (xy2 smoke) | Forearm survival L / R, all frames | Same, hips not visible | Forearm agreement L / R (our chest) | UWB-only R | Arms, own frames (upper / fore) |
|---|---|---|---|---|---|
| `learned` (framing-blind head) | 0.78 / **0.25** | 0.75 / 0.27 | 0.80 / **0.33** | 0.02 | 13.9 / 18.2 |
| `inside` | 0.63 / 0.77 | 0.53 / 0.70 | 0.93 / 0.82 | 0.17 | 11.4 / 15.5 |
| `margin:0.25` | 0.92 / 0.98 | 0.90 / 0.97 | 0.94 / 0.82 | 0.18 | 11.4 / 15.7 |

   - The right-forearm loss of the learned head is not caused by framing: the framing-blind xy2 head shows it too. It comes from the How2Sign-crop targets, where MediaPipe rarely calls the right arm visible (right forearm scored on 653 of 7,212 smoke val frames, the left on 4,011).
   - On How2Sign two-view truth (12 videos of signers 1 and 2; 56 crops; 6,364 frame-sides; elbow outside the crop on 97%):

| Keep decision | Survival | vs MediaPipe | Kept forearm error | Dropped forearm error |
|---|---|---|---|---|
| MediaPipe (Tasks visibility) | 0.146 | 1.0 | 27.3 | 42.2 |
| `learned` (xy2 head) | 0.221 | 1.51 | 26.4 | 46.3 |
| `inside` | 0.010 | 0.07 | 27.9 | 40.0 |
| `margin:0.1` | 0.202 | 1.38 | 33.8 | 45.8 |
| `margin:0.25` | 0.362 | 2.47 | 33.0 | 60.1 |
| `margin:0.5` | 0.617 | 4.21 | 35.2 | 90.1 |

     Forearm errors are median degrees of the ungated smoke lifter against truth (Tasks image inputs, not legacy Holistic). Upper-arm errors stay at 9–11° under every rule.
   - Reading: on chest-up crops, `margin:0.25` keeps 2.5× MediaPipe's arms at about 6° worse forearm accuracy. `learned` matches MediaPipe's accuracy but loses the right arm on YouTube. **No rule is declared here.** The choice belongs to the full gate with the full bundle. Candidate: `margin:0.25` (or 0.1).
2. **Pose context out of range (major).**
   - Done:
     - `world-regressor-features-xy-v2` drops the four context inputs. The pose model and its visibility head are invariant to image translation, scale and frame shape by construction (tested).
     - `--augment-framing K` re-frames training pose frames for xy-v1: scale 0.3–1.2, shoulder midpoint uniform in x 0.2–0.8 and y 0.1–0.7, frame shapes 16:9, 4:3, 1:1 and 9:16. It is refused for xy2, where it would be a no-op.
     - `input_zscores` and the gate's `inputs` block report p99 |z| per input.
   - On 30 new YouTube videos:
     - xy smoke bundle: `context.mid_y` p99 |z| = 9.07, `context.mid_x` = 5.78.
     - xy2 smoke bundle: no context inputs. Its arms improve from 13.5 / 20.2 to 11.4 / 15.7 (`margin:0.25`, own frames).
   - **Still above 4 under xy2:**
     - head-relative y inputs `p0.y` (nose) and `p9.y`/`p10.y` (mouth) reach 5.1–5.7;
     - hand y inputs reach 6.2.
     These are pose-distribution shifts (YouTube vs 3 How2Sign signers in the smoke). The full training on 5 signers widens the statistics. If the bar still fails, YouTube-framed pairs are needed. Full-frame pairs are not available now: `collect_world_pairs.py` has a YouTube branch, but needs staged videos.
3. **Evaluation leaks (major, critical).**
   - Done:
     - `convert_uwb_clips.py --eval-manifest KEY@SHA` is required and repeatable.
     - Every clip's text is checked with `build_youtube_asl_manifest.EvalIndex.leaks` against val/test/heldout sentences (exact match, or 5-gram Jaccard ≥ 0.8).
     - `eval_leak` and the index SHA-256 are stored per clip. Leaks are counted per unit (`checks.eval_leak`).
     - `signtranslator/data_engineering/uwb_manifest.select_training_records` refuses any converted record without the leak or overlap fields, refuses one checked against another index, and drops leaks, duplicates and text-less clips. Any manifest builder must call it.
   - Manifests to pass (as `youtube_pretrain_instruments.sh` does): `derived/full-v1/how2sign-text-manifest-v1/manifest.json` and `derived/asllrp-dai2-v1/plan-manifest-v2/manifest.json`, with their SHAs.
4. **Resume and settings (major).**
   - Done:
     - Each unit report stores its settings and a settings fingerprint: adapter, world, visibility, face, head, flags, default fps, the SHAs of the fps table, overlap table, eval index and regressors, chunk size, prefixes.
     - A finished unit with different settings is never skipped silently. The unit returns `settings-mismatch` naming the keys, and the run exits 2.
     - Runs write only unit reports and `shards/shard-i-of-n.json`.
     - `--merge-index`, run once after all shards, writes `index.json`. It refuses when units disagree on settings and records each unit's fingerprint.
5. **Overlap with our own YouTube states (major).**
   - Done:
     - The gate's `--overlap-table-out` lists every cue pair (UWB clip → our sample id; 212,291 pairs) and every video we extracted with our channel-hashed partition (2,605 videos: 2,587 train, 18 monitor, `monitor_share` 0.01).
     - The 79 videos we hold but could not time are marked `whole-video`: all their UWB clips count as duplicates.
     - The converter (`--overlap-table`, required) stores `duplicate_of_our_cue`, `our_cue_iou`, `our_partition`, `uwb_partition`, and `partition` (ours wins for our videos). The selection helper drops the duplicates.
   - Table: `scratch/uwb-lifter-dev/uwb-overlap-table-v1.json`, sha256 `645bc6459d2679a633fd4f4845cbf66b6c05cee1a052d0ea00188fa4af32a7f9`.
   - **Superseded** by the re-review fixes below: v1 had no near-duplicate overlaps, `whole-video` was never read, and
     the dedup and partition are now decided at manifest-build time.
6. **Gate method (major).**
   - Done:
     - `--exclude-videos reports/uwb_gate_smoke_videos.txt` lists all 70 videos any development run touched.
     - `--visibility` is a required declared rule.
     - The summary's `primary` is the declared lifter on `common` frames (held by ours and every compared method). Strata, CIs and `proposal_check` use it.
     - Strata: hips visible or not, fps ≈ 30 or not, stream match or not, frame-count difference 0 / −1.
   - The dev smoke's **stream-mismatch stratum** (UWB rate ≠ our download's) is far worse: wrists 61.7°, fingers 37.7° against 15.9° / 17.1°, on 15 pairs. Watch it in the full run; it may need excluding.
7. **fps estimation (minor).**
   - Done:
     - Theil–Sen start (median pairwise slope).
     - Refusal when the offset is outside [−1, 0] frames or the rate is outside 10–70 fps.
     - The time-grid fallback needs ≥ 50% of the video's clips.
     - A full gate run drops videos with |drift| > 1 frame/min from the table it writes (`drift_checked`).
   - Rerun on all 2,605 videos:
     - 2,526 estimated.
     - 40 refused for offset, 1 implausible (6 fps), 38 with too little evidence.
     - 2,298 agree with yt-dlp within 0.2%; 26.8% of the 2,526 are not 30/29.97.
   - New table: `scratch/uwb-lifter-dev/uwb-fps-table-v2.json`, sha256 `157dd673f0bbb7db2963299ef86c6b14284c491d3a45dc6ccc8ab50d889eca76`. This one is not yet drift-checked; the full gate rewrites it.
8. **Reporting (minor).**
   - Done:
     - Lags by frame-count difference.
     - Kappa and UWB-only rate in every validity table.
     - Median CIs from 2,000 video resamples (default).
     - The licence in every clip record and in every state's inference method.
   - Dev smoke:
     - |lag| ≤ 1 on 70% of arm-aligned pairs. The lag minimum is shallow, so this criterion may need the margin raised or the signal changed.
     - Lag 0 dominates for both frame-count classes.

## Re-review fixes (2026-10-30, re-review of 6625c83: APPROVE WITH FIXES)

1. **`whole-video` is now applied, and unpaired clips are counted (major).**
   - Every clip of a video we hold but could not time carries `duplicate_of_our_video`.
   - Every clip records `our_video`, `our_channel_id` and `our_cue_overlaps`: our cues overlapping it, with cover (intersection / clip span) and IoU, up to 3 with cover ≥ 0.05.
   - The overlap table is now `uwb-overlap-table-v2` and the converter refuses v1. Per video it gives the paired, unpaired and unpaired-but-overlapping counts.
   - Real data (2,605 of our videos; 218,651 UWB clips of them):
     - 212,291 paired;
     - **3,953 unpaired** in timed videos, of which 560 overlap one of our cues with cover ≥ 0.05;
     - 79 untimed videos holding **2,407 clips**, all `duplicate_of_our_video`.
   - The largest unpaired counts per video are 171, 168, 116 (43 of them overlapping), 77 and 58.
   - Table: `scratch/uwb-lifter-dev/uwb-overlap-table-v2.json`, sha256 `c617a2af0e696591f8b47bc8f1e02fda44a84d77dc2aad3d20167aeed0cfd6d3`.
2. **Dedup at manifest-build time (major, design).**
   - `uwb_manifest.select_training_records(records, eval_index_sha256=..., our_records=...)` takes the records of our YouTube manifests that the same build uses. It drops a UWB clip only when the cue of ours it duplicates is **selected**:
     - its pair (IoU ≥ 0.5), or a selected cue covering ≥ 50% of its span;
     - for an untimed video, any selected cue of that video.
   - So a UWB clip is kept where our cue was dropped, for example the 51k arm-untracked cues.
   - The partition of our videos is **read from our manifest records**: the video's records, else its channel's (our split is channel-disjoint). A clip of our video whose channel has no selected record is dropped (`our_partition_unknown`), never re-hashed.
   - The converter no longer overrides `partition`. It keeps UWB's own, and the hashed `our_partition` is informational.
   - Tests cover our cue selected vs not selected, near-duplicates, untimed videos, channel fallback and unknown partitions.
3. **Selection hardening (minor).**
   - `eval_index_sha256` is a required keyword.
   - Partitions outside {train, monitor} are refused, ours and UWB's alike, as are our records putting one video or channel in two partitions.
   - With the builder's `EvalIndex`, a stale `eval_leak=False` on leaking text is refused.
   - Test: a leaked clip never reaches the selection output.
4. **Index and resume (minor).**
   - `--merge-index` recomputes each unit's fingerprint from its recorded settings and refuses a mismatch with the stored one.
   - It computes `units_total` from the source zips and sets `complete`.
   - Every run checks one existing unit report against its settings before converting anything (exit 2).
5. **Drift (minor).**
   - A drift is measured only from ≥ 8 arm-aligned cues spanning ≥ 3 minutes. Otherwise the video is `unchecked` and keeps its rate.
   - The fps table carries per-video `drift` (ok / dropped / unchecked) and the rule.
   - With `--pairs-per-video 5` almost every video is `unchecked`. A drift check needs `--pairs-per-video 12` or more (see the commands).
6. **Input z (note).**
   - `summary.inputs` reports p99 and median |z| for every input.
   - `proposal_check` shows it as `inputs_diagnostic` (max p99 |z|, the reference 4), not as a pass/fail item.
7. **Outside-frame check.**
   - Defaults are exactly {learned, inside, margin:0.1, margin:0.25}.
   - It reports per side, per bone (upper arm, forearm) and per elbow bin (inside, 0–0.1, 0.1–0.3, > 0.3 frame heights): survival, and the kept frames' median and p90 ungated lifter error, each with a by-video 95% CI upper bound (1,000 resamples).
   - UWB-weighted survival uses UWB's own elbow-distance distribution. The gate reports it (since the final-review fixes: per bone, `summary.uwb_arm_distance`); dev weights came from 300 random clips of videos we never extracted: elbow inside on 74% (left) / 77% (right), 0–0.1 out on 17% / 15%, 0.1–0.3 on 7%, > 0.3 on 1%. The weighting covers only bins with ≥ 200 frames and reports the UWB weight covered.
   - **Inputs:** legacy MediaPipe Holistic 0.10.14 (model complexity 1, refined face), run locally on the same crop boxes of signers 1 and 2. All 56 crops used legacy inputs; none fell back to Tasks. Arrays: `scratch/uwb-lifter-dev/legacy-val.tar`, sha256 `77ac7d757230eb784388ba5877f3b4c66db5122ce12b256a8dae6c006992401a`.
   - Results with the xy2 smoke bundle, 12 videos, 6,364 frame-sides. Median / p90 is the kept forearm error, by-video CI upper in brackets.

| Rule | Side | Survival (all) | 0–0.1 (n 508 / 846) | 0.1–0.3 (n 1,055 / 846) | > 0.3 (n 1,574 / 1,292) | Kept forearm, all |
|---|---|---|---|---|---|---|
| MediaPipe | L / R | 0.286 / 0.007 | 0.11 / 0.01 | 0.63 / 0.00 | 0.12 / 0.00 | 22.1 / 53.8 [26.8 / 109] · 50.1 / 71.2 |
| `learned` | L / R | 0.421 / 0.024 | 0.12 / 0.03 | 0.72 / 0.03 | 0.33 / 0.01 | 22.3 / 40.3 [25.7 / 47.8] · 35.0 / 44.7 |
| `inside` | L / R | 0.000 / 0.023 | 0 / 0 | 0 / 0 | 0 / 0 | – · 32.2 / 42.0 |
| `margin:0.1` | L / R | 0.020 / 0.226 | 0.12 / 0.74 | 0 / 0.00 | 0 / 0 | 28.6 / 45.3 · 33.7 / 41.8 |
| `margin:0.25` | L / R | 0.208 / 0.475 | 0.16 / 0.78 | 0.55 / 0.84 | 0 / 0.01 | 23.0 / 44.8 [30.6 / 61.6] · 32.4 / 42.8 [34.7 / 48.9] |

   - Upper-arm kept medians are 8–13° for every rule.
   - **Coverage limits:**
     - These crops put the elbow inside on only 45 (left) / 198 (right) frame-sides. The bins with ≥ 200 frames cover only **26% / 23% of UWB's weight**.
     - UWB-weighted survival over the covered bins: MediaPipe 0.26 / 0.005, `learned` 0.31 / 0.03, `margin:0.1` 0.08 / 0.48, `margin:0.25` 0.27 / 0.77.
     - Ratios against MediaPipe are meaningless on the right: MediaPipe Tasks itself keeps 0.5% of right forearms on these crops. That is the source of the learned head's right-arm loss.
   - The proposed "chest-up survival ≥ 80% of MediaPipe's" is not well defined on this truth set: the denominator is near zero on the right, and inside-frame framings are missing.
   - Next step: add How2Sign crops with the elbows inside (a wider framing, nose at 0.2 and shoulders at 0.45), so that UWB's dominant bin has truth.

## Final-review fixes (2026-10-30, final review of 4c67724: APPROVE WITH FIXES)

1. **Bones binned by their farther end.**
   - The outside-frame check bins each forearm by max(elbow, wrist) distance outside the frame and each upper arm by max(shoulder, elbow). A bone's swing joint needs both ends (`video_landmarks.BODY_JOINTS`).
   - MediaPipe's keep decision follows the bone's chain: shoulder and elbow for the upper arm, plus the wrist for the forearm.
   - Each rule's keep decision is that bone's validity in the rule's state.
   - The gate's UWB distribution is now per bone (`summary.uwb_arm_distance`). On 300 clips of videos we never extracted, with the forearm both ends in frame:
     - forearm inside on 64% (left) / 75% (right), out by 0–0.1 on 17% / 15%, 0.1–0.3 on 10% / 7%, > 0.3 on 9% / 4%;
     - upper arm inside on 74% / 77%.
2. **Anchor and the declared selection.**
   - **Anchor:** the ungated lifter's error on the **full How2Sign frames** of the same signers (shape `full`, legacy Holistic inputs, run locally on the full frames), on frames where shoulder, elbow and wrist are all in view. It reports median and p90 per side and bone, with by-video 95% intervals.
   - Per rule, side, bone and bin (crops only), it reports:
     - frames and survival;
     - kept frames and videos;
     - kept median and p90, each with its CI upper bound;
     - `admitted`: the rule keeps ≥ 5% of the bin;
     - `unmeasured`: under 200 kept frames or under 5 videos.
   - UWB-weighted survival uses the covered bins, with coverage.
   - **Selection, as declared:**
     - A rule is admissible iff, for every side, bone and admitted bin, the CI-upper kept median ≤ anchor median + 3° and the CI-upper kept p90 ≤ anchor p90 + 5°. Unmeasured counts as a fail.
     - The admissible rule with the largest UWB-weighted forearm survival (mean of the sides) is chosen; rules within 2 points go to the more conservative (inside < margin:0.1 < margin:0.25 < learned).
     - `learned` is chosen only conditional on the YouTube gate's validity: all-frame forearm agreement ≥ 0.85 per side and UWB-only valid ≤ 10%.
     - If no rule is admissible, `inside`.
   - Dev run, xy2 smoke bundle: 12 videos, 68 clips (all legacy inputs), 15,452 frame-side-bone rows. Report: `scratch/uwb-lifter-dev/2026-10-30-outside-frame-xy2-smoke-v3.json`, sha256 `2edda1dc5c435f41758693236366785a99f79e6e3a8878dea528f63d27e1c5e6`.
   - Anchor (median / p90 [by-video median CI]):

| Side | Upper arm | Forearm |
|---|---|---|
| Left (460 frames, 11 videos) | 11.4 / 16.8 [10.1, 12.5] | 28.3 / 67.0 [22.6, 36.6] |
| Right (474 frames, 11 videos) | 9.5 / 20.2 [5.7, 15.4] | 32.0 / 40.7 [28.2, 34.1] |

   - The smoke bundle's forearm error is large even with the arm in full view (about 30°). A full-data bundle should lower the anchor; the bars are relative to it.
   - Selection result: **no rule admissible, so `inside` (the declared fallback)**. Every rule fails on unmeasured admitted bins, because these crops are thin where UWB's frames are:
     - left elbow-and-wrist-inside forearm frames: none;
     - right inside forearm frames: 74 (65 kept by `inside`, 3 videos);
     - right upper arm under `inside`: 182 kept frames in 5 videos.
   - `margin:0.1` and `margin:0.25` also fail the p90 bar on the left upper arm: CI upper 22.6–22.8° against 21.75°.
   - UWB-weighted forearm survival (left / right) over the covered bins, which hold only 19% / 25% of UWB's forearm weight:

     | Rule | Left | Right |
     |---|---|---|
     | `margin:0.25` | 0.39 | 0.82 |
     | `margin:0.1` | 0.00 | 0.57 |
     | `learned` | 0.60 | 0.03 |
     | MediaPipe | 0.46 | 0.01 |

   - A real decision needs truth crops with the arm in view (wider How2Sign framings) and the full-data bundle. The script then decides mechanically.
3. **Summed covers.** `uwb_manifest` drops a UWB clip when the selected cues' covers **sum** to ≥ 0.5, or when a selected cue is its IoU ≥ 0.5 pair. It is tested: two selected cues covering 0.3 each drop the clip; one alone keeps it.
4. **Known residual (recorded here and in `uwb_manifest`).** UWB stores no channel ids, so a UWB-only video (one we never extracted) from one of our **monitor** channels keeps UWB's partition (usually `train`). Our channel-disjoint monitor split holds for our videos and for UWB clips of our videos, not for other uploads of a monitor channel. With `monitor_share` 0.01 (18 of our 2,605 videos are monitor), the exposure is small but not zero.

### Proposed gate (to be declared by the coordinator)

- Full xy bundle, smoke videos excluded, all estimated-fps videos ×5 pairs, common frames, 95% CI by video.
- Accuracy: medians upper arm ≤12°, forearm ≤18°, wrist ≤15°, fingers ≤16°, each CI upper ≤ bar+1°; p90 upper arm ≤25°, forearm ≤40°.
- Relative: each group's YouTube median ≤1.6× the same bundle's How2Sign-val median; arms gain ≤3° after one global rotation per clip.
- Strata (hips-visible/not, fps≈30/not, stream-match/not, fc-diff 0/−1): no stratum with ≥200 pairs worse than the global bar by >3°.
- Validity on all frames: hand-valid agreement ≥0.88 per side; forearm-valid agreement ≥0.85 per side on hip-visible frames; UWB-only-valid ≤10%; chest-up forearm survival ≥80% of MediaPipe-visibility survival.
- Inputs: p99 |z| ≤4 for every regressor input.
- Mirror-better ≤3% clips; |lag| ≤1 on ≥90% of arm-aligned pairs; |drift| <1 frame/min on ≥95% of videos (failures dropped from the fps table).
- Downstream: back-translator top-5/BLEU on lifted states ≥90% of on our states for paired cues, and clearly above blank.

`uwb_lifter_gate.py` evaluates the computable items in `summary.proposal_check`. Two items are not computed there:

- How2Sign-val relative ratios: the training report scores local rotations, not the gate's direction errors.
- The back-translator read.

Chest-up survival versus MediaPipe comes from `uwb_outside_frame_accuracy.py`.

Dev smoke, informative only, not a verdict (smoke bundle; the declared rule is not chosen):

- Hand agreement, UWB-only rate and strata pass.
- Accuracy medians fail by 2–6° (upper arms 13.9, forearms 18.2, wrists 17.7, fingers 18.5 on common frames).
- Rotation gain fails: 4.0 upper arms, 3.2 forearms.
- Inputs fail (p99 |z| 6.2).
- Mirror-better is 2/69 (2.9%) for `inside` and `margin:0.25`, and 14/67 for `learned`. The table above shows the `learned` primary.
- Lag within 1 fails (0.70).

## Commands

All are CPU jobs. Run them from a checkout of this branch with the R2 secrets in the environment.

### (a) Train the regressors

Two bundles: the framing-invariant xy2 (recommended primary), and xy with framing augmentation (the alternative), so
that the gate can compare them. About 1 h each on CPU.

```bash
mkdir -p /workspace/uwb && cd /workspace/uwb
python - <<'EOF'
import os, hashlib, tarfile, sys; sys.path.insert(0, "/workspace/sign-translator")
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
for name, sha in (("pairs.tar", "6c4d3ecc18e66f2bbc885aacda9a86499d7f5ce48618fa2438540ca958526758"),
                  ("legacy-pairs.tar", "7042e0a8bcd7452d13743fcac5f6b37be1854fafb06c417ab54fa8aa24c65fa0")):
    b = s.get(f"scratch/uwb-lifter-dev/world-pairs-v2/{name}"); assert hashlib.sha256(b).hexdigest() == sha
    open(name, "wb").write(b); tarfile.open(name).extractall(".")
b = s.get("acquisition/2026-10-17-v1/sem-lex-mirror-v1/adapter-dev/world-regressors-v2/world-regressors-v2.pt")
assert hashlib.sha256(b).hexdigest() == "0953cd2cb2b48506808dff2d7199061eac3218ad555ea35327ff34edac3acab1"
open("v2.pt", "wb").write(b)
EOF
T=/workspace/sign-translator/scripts/train_world_regressors.py
python $T --pairs pairs --legacy-dir legacy-pairs --features xy2 --epochs 30 --seed 0 --val-signers h2s-1,h2s-2 \
  --baseline-bundle v2.pt --out world-regressors-xy2-v1.pt --report world-regressors-xy2-v1-training.json
python $T --pairs pairs --legacy-dir legacy-pairs --features xy --augment-framing 3 --epochs 30 --seed 0 \
  --val-signers h2s-1,h2s-2 --baseline-bundle v2.pt --out world-regressors-xy-aug-v1.pt \
  --report world-regressors-xy-aug-v1-training.json
# upload the four files (e.g. under .../youtube-asl-uwb-keypoints-v1/lifter-dev/) and record their SHA-256
```

### (b) Gate

Declare the visibility rule and the bar first. Then run the paired gate, all estimated-fps videos × 5 pairs, with the
smoke videos excluded. It also writes the drift-checked fps table:

```bash
python scripts/uwb_lifter_gate.py --regressors <key of world-regressors-xy2-v1.pt> --regressors-sha256 <sha> \
  --visibility <declared rule, e.g. margin:0.25> --candidate-visibility learned,inside,margin:0.1,margin:0.25 \
  --exclude-videos reports/uwb_gate_smoke_videos.txt --videos 0 --pairs-per-video 5 --head affine \
  --workers 30 --out gate-xy2-v1.json
# the same with the xy-aug bundle if both are candidates (declare which is primary before running)
# drift check for the fps table (needs >= 8 arm-aligned cues over >= 3 min per video; the lifter is irrelevant here):
python scripts/uwb_lifter_gate.py --regressors <key> --regressors-sha256 <sha> --visibility <declared rule> \
  --candidate-visibility "" --exclude-videos reports/uwb_gate_smoke_videos.txt --videos 0 --pairs-per-video 12 \
  --workers 30 --out gate-drift.json --fps-table-out uwb-fps-table-v3.json
```

Then run the outside-frame check (minutes). Its data is in:

- `scratch/uwb-lifter-dev/outside-frame-data.tar` (sha256 `598ab2ce80f05d663cdb1a0805fb4c56e412ff5708e76810c886d9eb028cc1a3`), extracted to `ofd/`;
- `scratch/uwb-lifter-dev/legacy-val.tar` (sha256 `77ac7d757230eb784388ba5877f3b4c66db5122ce12b256a8dae6c006992401a`), extracted beside it.

The UWB weights come from the gate report:

```bash
python scripts/uwb_outside_frame_accuracy.py --crop-dirs ofd/val ofd/val-tight --two-view-dir ofd/twoview \
  --legacy-dir legacy-val --require-legacy --uwb-weights gate-xy2-v1.json \
  --regressors world-regressors-xy2-v1.pt --regressors-sha256 <sha> --report outside-frame-xy2-v1.json
# report["selection"]["chosen"] is the rule to declare (learned only with its YouTube-gate condition met)
```

Use `scratch/uwb-lifter-dev/legacy-val-v2.tar` (sha256 `b131e878a8a7a48e9ccf41f2e35de3d8d71e56f37535e7ab270cf6ef0da058fb`) for `legacy-val/`. It adds the full-frame arrays that the anchor needs; `legacy-val.tar` holds the crops only.

The overlap table (v2), if regenerated, needs every video (no `--videos`):

```bash
python scripts/uwb_lifter_gate.py --fps-only --overlap-table-out uwb-overlap-table-v2.json --out fps-only.json
```

### (c) Convert all 390,547 clips (only after the gate passes)

The output prefix is the owner's to approve. Example with 8 pods × 30 workers:

```bash
ACQ=acquisition/2026-10-17-v1
python scripts/convert_uwb_clips.py --out-prefix $ACQ/youtube-asl-uwb-keypoints-v1/states-xy2-v1 \
  --regressors <key> --regressors-sha256 <sha> --visibility <declared rule> \
  --fps-table <key of the drift-checked table>@<sha> \
  --overlap-table scratch/uwb-lifter-dev/uwb-overlap-table-v2.json@c617a2af0e696591f8b47bc8f1e02fda44a84d77dc2aad3d20167aeed0cfd6d3 \
  --eval-manifest $ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json@<sha> \
  --eval-manifest $ACQ/derived/asllrp-dai2-v1/plan-manifest-v2/manifest.json@<sha> \
  --face blendshapes --head none --shard $I --shards 8 --workers 30
# after every shard has finished (one process):
python scripts/convert_uwb_clips.py --out-prefix $ACQ/youtube-asl-uwb-keypoints-v1/states-xy2-v1 --merge-index
```

- Rerunning a shard with the same settings resumes it.
- Any other settings on a finished unit exit 2.
- Training manifests must select records through `uwb_manifest.select_training_records(records,
  eval_index_sha256=<the converter's index SHA>, our_records=<the YouTube manifest records of the same build>,
  eval_index=<the builder's EvalIndex>)`. This applies the dedup against our selected cues and reads our partitions.
