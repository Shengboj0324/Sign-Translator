# UWB 2D-to-world lifter: development report (2026-10-30, 02:50 UTC)

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
| Tests (synthetic, offline; 37 new; 98 with the related regressor and Sem-Lex suites, all passing under `-W error`) | `tests/test_de_world_regressor_xy.py`, `tests/test_de_uwb_keypoints.py`, `tests/test_convert_uwb_clips_script.py`, `tests/test_uwb_lifter_gate.py`, `tests/uwb_synthetic.py` |

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

## Commands

### (a) Train the xy regressors

CPU only, about 1 h.

```bash
set -a; source .env.r2.local; set +a          # pod: secrets already in the environment
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
python /workspace/sign-translator/scripts/train_world_regressors.py --pairs pairs --legacy-dir legacy-pairs \
  --features xy --epochs 30 --seed 0 --val-signers h2s-1,h2s-2 --baseline-bundle v2.pt \
  --out world-regressors-xy-v1.pt --report world-regressors-xy-v1-training.json
# upload both files to R2, e.g. under .../youtube-asl-uwb-keypoints-v1/lifter-dev/world-regressors-xy-v1/
```

### (b) Run the gate

CPU pod, all about 2,568 videos, 5 pairs each:

```bash
python scripts/uwb_lifter_gate.py --regressors <key of world-regressors-xy-v1.pt> --regressors-sha256 <sha> \
  --videos 0 --pairs-per-video 5 --head affine --workers 30 \
  --out gate-xy-v1.json --fps-table-out uwb-fps-table-v1.json
```

Before the full run, declare a bar. Proposal:

- learned arms and fingers at least 10° below pseudo, on common frames;
- forearm-valid and hand-valid agreement at least 0.85 per side;
- mirror-better in at most 5% of clips;
- then a recogniser read of lifted against our states on paired cues.

### (c) Convert all 390,547 clips

The output prefix is the owner's to approve. Example with 8 pods × 30 workers, about 40 min:

```bash
python scripts/convert_uwb_clips.py --out-prefix acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/states-xy-v1 \
  --regressors <key> --regressors-sha256 <sha> --visibility inside \
  --fps-table scratch/uwb-lifter-dev/uwb-fps-table-v1.json@ceaf7675949b99c6a7c797dc36229a21ed280456522e85e28d446ac5f9e07be0 \
  --face blendshapes --head none --shard $I --shards 8 --workers 30
```

- `--visibility inside` is what the smoke supports. Confirm it with (b).
- Rerunning the same command resumes.
- `index.json` summarises the finished units.
