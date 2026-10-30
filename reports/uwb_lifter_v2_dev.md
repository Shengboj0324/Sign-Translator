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
| Tests (25) | `tests/test_uwb_lifter_v2.py` |

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
  - How2Sign: v2 pairs of signers 3, 4, 5, 8, 11, with the original loss. The tight crops do not enter the pose regressor (amendment 2); they train only the validity head.
  - YouTube: the train part, with 1 − cos on the 4 arm bones and the shoulder line.
- One epoch is one pass over the larger source.
- Hands stay frozen. `--finetune-hands` is the comparison arm and is never selected.

**Selection** (amendment 1, 11:51 UTC): the arm with the lowest lifter-val mean of the upper-arm and forearm medians; a tie goes to the fine-tuned arm.

**Validity head:**
- MLP(68 → 256 × 2 → 6), masked BCE, 50/50 batches of YouTube train frames and tight crops, 15 epochs at lr 1e-3.
- Thresholds are set per side on lifter-val, over every frame where our chest is valid. A side's forearm counts as kept when its shoulder, elbow and wrist all reach the threshold.
- The threshold maximises agreement subject to UWB-only ≤ 0.07. Ties go to the higher threshold. A grid value of 1.01 keeps nothing, so the constraint can always be met.

**Signers:**
- Signer 9 is refused by name before any of its files is read.
- The v2 pairs files of How2Sign val signers 1 and 2 are read by `load_pairs` with the rest, then filtered out before training.
- Crops of signers 1, 2 or any non-training signer are refused by name.

### Gate v2 verdict (`--gate-version 2`)

**Kept from gate v1:**
- every v1 bar;
- the accuracy, relative, offset and strata rows on `lifter[inside]` common frames;
- the mirror row;
- the downstream row, which now reads `lifter[head]` states, the conversion rule (amendment 2). It runs with `uwb_downstream_check.py --visibility head`, and the verdict checks that rule.

**New rows:**
- **Setup:** `inside`; the v1 candidate set; `head` as a secondary method; the declared v2 bundle; 12 pairs per video; 2,000 resamples; 70 development videos excluded; lag curves present; the declared split.
- **Population:** every gated video is held out; the split is disjoint by channel and by video; the pairs index and the bundle's training report are both on this split.
- **v1 reference:** must be present, with the same evaluated pairs and every gate setting the same except the bundle and `head` as secondary (amendment 2). Its medians and its pooled-lag value are reported beside v2's.
- **How2Sign non-inferiority:** per group, v2's How2Sign-val median ≤ v1's + 0.5°. Both medians of a group must be measured on ≥ 200 values from ≥ 5 videos.
- **Truth anchor:** ≤ v1's anchors, read unrounded from v1's pinned selection report (`lifter-v1/selection/outside-frame-xy2-v1.json`, sha `3e26180a…`): 28.24° left, 29.07° right (amendment 2). A test checks that xy2 passes its own anchor.
- **Chest-up survival of `head`:** see item 5 below.
- **Pooled-lag timing.**

**Reported only:**
- a 95% interval clustered by channel (2,000 resamples of channels) beside each gated by-video interval (amendment 2);
- head-kept accuracy;
- the held-out minus lifter-val gaps.

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
- `tests/test_uwb_lifter_v2.py`: 25 passed (after amendment 2);
- related UWB suites, with the new tests included: 111 passed (after merging main af4174d);
- full suite: 4,090 passed (after amendment 2).

## Interpretations and deviations

Amendment 1 (2026-10-30, 11:51 UTC) settled these. It confirmed items 1–10 below except the selection metric, which it changed (see Training).

1. **Videos without a `channel_id`** are dropped everywhere, as the development videos are. No channel can hold them out.
2. **Rule `head`:** points 11–16 come from the head. Every other point uses `inside`.
3. **Gate v2 validity rows are judged on `lifter[head]`**, the rule the head becomes. The rows are hand agreement, forearm agreement and UWB-only. `inside`'s values are reported, not gated. The accuracy rows stay on `inside` common frames, as declared.
4. **`head` runs as a secondary method outside the common frames.** This keeps v2's common frames defined exactly as v1's, with the candidate set `learned,inside,margin:0.1,margin:0.25`.
5. **Chest-up row:**
   - It reads tight crops only (`--tight-summary`) and the cell for out-of-frame forearms, i.e. distance > 0.
   - MediaPipe counts as measured at ≥ 200 kept frames from ≥ 5 videos, as in v1's ruling.
   - Admissibility uses the amendment-1 rule on those crops.
   - When MediaPipe is unmeasured, amendment 2 sets the fallback:
     - `head` must keep **more** out-of-frame forearms than `inside`, and must admit at least one measured outside bin; otherwise the row is not-evaluable, never pass.
     - An inadmissible `head` fails, as declared.
6. **v1 reference** is a gating setup row. The verdict is not-evaluable unless the reference is xy2 with every gate setting the same: split, prefixes, head, excluded videos, text matches and the rest. Only the bundle, `head` as secondary, workers, output and members cache may differ. The sample must also be identical, both the (video, clip) pairs and which of them were evaluated.
7. **Pooled lag:**
   - The pooled curve is the mean of the per-pair median curves over the lags that every pair scored.
   - Ties go to the smaller |lag|, then the earlier lag.
   - A video with no common lag counts as outside ±1.
   - Offenders are counted only among videos whose pooled lag lies beyond ±1. Such a video is an offender when its pooled lag beats lag 0 by more than 2°, or when its common lags lack 0.
8. **Not declared, chosen here:**
   - pose-arm selection metric: superseded by amendment 1, which uses the mean of the upper-arm and forearm medians;
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
- **Memory in the train phase:** the two primary pose arms run one after another by default. `PARALLEL_ARMS=1` runs them together. Each arm prints its peak RSS and records it in its report (`peak_rss_mb`), so the pod can be sized from a first run.
- **Hands comparison:** it runs after selection and after the validity head. A failure is recorded in `pose-ft-hands.FAILED` and does not stop the primary pipeline.
- **Outside-frame tar layouts:** the script locates the directories by content, because the tars' layouts are not recorded.

## Independent review of 0ec5362 and amendment 2 (2026-10-30, 12:03 UTC)

Verdict: APPROVE WITH FIXES. Main af4174d is merged.

1. **Chest-up fallback:** `head` must keep more out-of-frame forearms than `inside` and admit ≥ 1 measured outside bin, otherwise not-evaluable. An inadmissible `head` fails. The review's vacuous pass is now a test.
2. **Downstream on `head`:** `uwb_downstream_check.py --visibility head` records `visibility` and `gate_visibility`. `downstream_rows(expected_visibility="head")` makes an `inside` read not-evaluable. The pod's downstream phase passes `head`.
3. **Crops removed from the pose arms:** `pose` no longer accepts `--crops`, and the crops docstring is fixed. The crops train only the validity head.
4. **Truth-anchor bars** are read unrounded from v1's pinned report (SHA-checked, `--outside-frame-v1`). xy2 passes its own anchor.
5. **Channel-clustered 95% intervals:** in split runs the gate carries each pair's channel and adds `median_ci95_by_channel` to the primary common-frame medians. The verdict reports them beside the by-video intervals. Expected composition (amendment 2): 160 held-out channels including 8 monitor channels, 507 videos, 38,610 candidate pairs, split SHA `27c35505…`. The split phase flags a different SHA.
6. **Thresholds refused below 500 chest-valid lifter-val frames per side.**
7. **v1 reference identity:** every gate setting and the evaluated pair sets must match.
8. **Pod script:**
   - the chosen `.pt` is checked against `selection.json`'s SHA;
   - `uv venv` runs only if no venv exists;
   - the legacy Holistic model SHAs (`check-models`, against the venv that produced the v2 legacy pairs) are checked in the crops and train phases;
   - the hands comparison runs after selection, and its failure is non-fatal and recorded.
9. **Wording:**
   - offenders are only videos whose pooled lag is beyond ±1;
   - non-inferiority is per group;
   - signers 1 and 2 are read, then filtered out.

The review's adversarial file now fails on findings 1, 4 and 6, as intended. Its cases are branch tests (`test_review_adversarial_cases` and the threshold-floor test). Gate v1 and verdict v1 remain byte-identical to main on the synthetic store.
