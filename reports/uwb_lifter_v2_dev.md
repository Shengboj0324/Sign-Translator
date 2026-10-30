# UWB lifter v2 and gate v2: development report

Branch `uwb-lifter-v2` (from main 4db7a96). Binding spec: `docs/NEXT_PHASE_PLAN.md`, "UWB lifter v2 and gate v2: declared (2026-10-30, 11:12 UTC)", with the gate v1 declaration (03:34 UTC) and amendment 1 (03:59 UTC). No bar was changed. Nothing has been trained or run on a pod yet. Every result below comes from synthetic tests.

## What was built

| Piece | File |
|---|---|
| Channel-disjoint split, SHA-printed and deterministic | `scripts/build_uwb_lifter_split.py` |
| Training and lifter-val pairs (npz shards with `pairs-index.json`) | `scripts/build_uwb_lifter_pairs.py` |
| Tight crops of How2Sign training signers (fetch, then legacy Holistic jobs) | `scripts/uwb_lifter_v2_crops.py` |
| Pose arms, selection on lifter-val, validity head and thresholds | `scripts/train_uwb_lifter_v2.py` |
| Optional validity head; `validity_features()` | `signtranslator/data_engineering/world_regressor.py` |
| Visibility rule `head` | `signtranslator/data_engineering/uwb_keypoints.py` |
| `--split-file`, `--lag-curves`, `--secondary-visibility`; `pooled_video_lags()` | `scripts/uwb_lifter_gate.py` |
| Rule `head` accepted; `--tight-summary` | `scripts/uwb_outside_frame_accuracy.py` |
| `--gate-version 2` with the declared rows | `scripts/uwb_gate_verdict.py` |
| Pod phases | `runpod/fleet/uwb_lifter_v2.sh` |
| Tests (19) | `tests/test_uwb_lifter_v2.py` |

### Split

- **Held-out:** channels where `sha256("uwb-lifter-v2:" + channel)`, read as an integer over 2^256, is below 0.20. This is an exact integer comparison. Every monitor channel is added (`monitor_channel`, share 0.01).
- **Lifter-val:** of the remaining channels, those where `sha256("uwb-lifter-v2-val:" + channel)` is below 0.10.
- **Train:** the rest.
- **Dropped:** the 70 development videos are dropped everywhere.
- **Composition**, per part: channels, videos, UWB clips, our cues, and candidate pairs (from the gate's own fps estimate and pairing). It also gives the monitor channels and the largest channel's share of videos, clips and candidate pairs.

### Pairs

The builder uses the gate's own functions (`estimate_fps`, `pair_clips`, `choose_lag` with max lag 6 and margin 0.5, `frame_map`, `state_items`). It refuses held-out ids three times: on the video list, on the jobs, and again in each worker before anything is read.

A pair is kept only if:
- the stream matches (within 0.2%);
- the alignment is `arm-2d`;
- the chosen lag and the best lag are both in {-1, 0, 1};
- it has ≥ 400 bone-frames;
- the lag-0 score is ≤ 20°.

Each refusal is counted by reason.

**Stored per frame:**
- **Inputs:** UWB pose and hands, normalised.
- **Pose targets:** the 4 arm bones and the shoulder line, in MediaPipe axes, with their validity.
- **Validity labels:** elbow = our upper arm is valid; wrist = our forearm is valid, labelled only where the upper arm is valid. Frames without our chest are masked.
- **Forearm validity:** our forearm validity, which is the same flag the gate's forearm row reads.
- **Fingers:** camera-frame finger directions, used only by the hands comparison.

### Training (CPU)

**Pose arms:**
- **Fine-tuned:** starts from xy2 (its normalisation is kept), 15 epochs at lr 3e-4.
- **From scratch:** 30 epochs at lr 1e-3.
- Batches are 50/50:
  - How2Sign: v2 pairs of signers 3, 4, 5, 8, 11, plus the training signers' tight crops, with the original loss.
  - YouTube: the train part, with 1 − cos on the 4 arm bones and the shoulder line.
- One epoch is one pass over the larger source.
- Hands stay frozen. `--finetune-hands` is the comparison arm and is never selected.

**Selection:** the arm with the lowest lifter-val forearm median; a tie goes to the fine-tuned arm.

**Validity head:**
- MLP(68 → 256 × 2 → 6), masked BCE, 50/50 batches of YouTube train frames and tight crops, 15 epochs at lr 1e-3.
- Thresholds are set per side on lifter-val, over every frame where our chest is valid. A side's forearm counts as kept when its shoulder, elbow and wrist all reach the threshold.
- The threshold maximises agreement subject to UWB-only ≤ 0.07. Ties go to the higher threshold. A grid value of 1.01 keeps nothing, so the constraint can always be met.

**Signers:** signer 9 and How2Sign val signers 1 and 2 are refused by name before anything is read.

### Gate v2 verdict (`--gate-version 2`)

**Kept from gate v1:**
- every v1 bar;
- the accuracy, relative, offset and strata rows on `lifter[inside]` common frames;
- the mirror row and the downstream row.

**New rows:**
- **Setup:** `inside`; the v1 candidate set; `head` as a secondary method; the declared v2 bundle; 12 pairs per video; 2,000 resamples; 70 development videos excluded; lag curves present; the declared split.
- **Population:** every gated video is held out; the split is disjoint by channel and by video; the pairs index and the bundle's training report are both on this split.
- **v1 reference:** must be present and on an identical sample. Its medians and its pooled-lag value are reported beside v2's.
- **How2Sign non-inferiority:** v2 ≤ v1 + 0.5° per group, each side measured on ≥ 200 values from ≥ 5 videos.
- **Truth anchor:** ≤ 28.2° (left) and ≤ 29.1° (right).
- **Chest-up survival of `head`.**
- **Pooled-lag timing.**

**Reported only:** head-kept accuracy, and the held-out minus lifter-val gaps.

**Cross-check:** only the proposal_check keys whose rows keep their v1 definition on the same method (accuracy, offset, strata, mirror, drift).

**Gate v1 mode** is unchanged; the byte-identity check is below.

## Byte-identity checks

On a synthetic store (3 videos, 15 pairs), I ran `uwb_lifter_gate.py` (v1 mode, with fps table) and `uwb_gate_verdict.py` (v1 mode) twice into the same paths: once from a `git archive` of main 4db7a96, once from this branch. **All four outputs are byte-identical**: `gate.json`, `fps.json`, `verdict.json` and `verdict.md`.

Three tests pin the rest:
- A bundle without a head saves the same bytes after loading, with no `validity` entry.
- Attaching a head changes no gate v1 output except the bundle SHA.
- `--secondary-visibility head` leaves `lifter[inside]`'s common-frame statistics identical.

## Tests

Run warning-strict (`-W error`, `OMP_NUM_THREADS=1`):
- `tests/test_uwb_lifter_v2.py`: 19 passed;
- related UWB suites, with the new tests included: 105 passed;
- full suite (4,081 tests collected): exit 0, no failures.

## Interpretations and deviations (to ratify)

1. **Videos without a `channel_id`** are dropped everywhere, as the development videos are. No channel can hold them out.
2. **Rule `head`:** points 11–16 come from the head. Every other point uses `inside`.
3. **Gate v2 validity rows are judged on `lifter[head]`**, the rule the head becomes. The rows are hand agreement, forearm agreement and UWB-only. `inside`'s values are reported, not gated. The accuracy rows stay on `inside` common frames, as declared.
4. **`head` runs as a secondary method outside the common frames.** This keeps v2's common frames defined exactly as v1's, with the candidate set `learned,inside,margin:0.1,margin:0.25`.
5. **Chest-up row:**
   - It reads tight crops only (`--tight-summary`) and the cell for out-of-frame forearms, i.e. distance > 0.
   - MediaPipe counts as measured at ≥ 200 kept frames from ≥ 5 videos, as in v1's ruling.
   - Admissibility uses the amendment-1 rule on those crops.
6. **v1 reference** is a gating setup row. The verdict is not-evaluable unless the reference is xy2, on the same split and settings, with the identical (video, clip) sample.
7. **Pooled lag:**
   - The pooled curve is the mean of the per-pair median curves over the lags that every pair scored.
   - Ties go to the smaller |lag|, then the earlier lag.
   - A video with no common lag counts as outside ±1.
   - A video whose common lags lack 0 counts as an offender.
8. **Not declared, chosen here:**
   - pose-arm selection metric: the lifter-val forearm median;
   - the scratch arm's lr: 1e-3;
   - the epoch definition: one pass over the larger source;
   - the validity head's epochs and lr;
   - the threshold grid;
   - crops: 40 clips per signer × (4:3, 16:9), seed 0.
9. **Extra gating provenance rows** (pairs index and training report on the split) make the verdict not-evaluable when those inputs are missing.
10. **The lifter-val gap mixes two error measures**, so it is a rough comparison:
    - lifter-val: the trainer's error over every frame where our bone is valid;
    - held-out: the gate's own-frame error, for `inside` and for `head`.
11. **`convert_uwb_clips.py` now also accepts `--visibility head`**, through `visibility_margin`. Conversion stays blocked until gate v2 passes.

## Pod commands (CPU unless noted)

Each phase runs on its own pod, in this order. Each refuses an existing `lifter-v2/<phase>/INDEX.json` and checks the SHA of every input. Pinned inputs: xy2 `02a90e4f…`, `pairs.tar` `6c4d3ecc…`, `legacy-pairs.tar` `7042e0a8…`, outside-frame data `598ab2ce…`, `legacy-val-v2.tar` `b131e878…`.

```bash
bash runpod/fleet/uwb_lifter_v2.sh split                     # prints SPLIT_SHA; publish the composition before training
SPLIT_SHA=<..> bash runpod/fleet/uwb_lifter_v2.sh pairs       # prints PAIRS_INDEX_SHA
bash runpod/fleet/uwb_lifter_v2.sh crops                      # prints CROPS_SHA
SPLIT_SHA=<..> PAIRS_INDEX_SHA=<..> CROPS_SHA=<..> bash runpod/fleet/uwb_lifter_v2.sh train      # prints BUNDLE_SHA
SPLIT_SHA=<..> PAIRS_INDEX_SHA=<..> BUNDLE_SHA=<..> bash runpod/fleet/uwb_lifter_v2.sh gate      # prints GATE_SHA
SPLIT_SHA=<..> PAIRS_INDEX_SHA=<..> BUNDLE_SHA=<..> GATE_SHA=<..> bash runpod/fleet/uwb_lifter_v2.sh downstream   # GPU
```

The gate phase runs:
- gate v2 for the v2 bundle (`--secondary-visibility head`);
- gate v2 for the xy2 reference on the same split and settings;
- How2Sign-val medians for both bundles;
- the outside-frame report (v2 bundle, rules incl. `head`, `--tight-summary`);
- the verdict without the downstream row.

The downstream phase adds that row and writes the final verdict.

## Risks not covered by tests

- **The crops phase has never run on a pod.** It installs legacy Holistic (Python 3.11, mediapipe 0.10.14) in a uv venv and needs all three Tasks models in `/root/work/models`.
- **Memory in the train phase:** it runs the three pose arms in parallel, each holding every YouTube frame. Size the pod's memory for that.
- **Outside-frame tar layouts:** the script locates the directories by content, because the tars' layouts are not recorded.
