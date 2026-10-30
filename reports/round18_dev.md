# Round 18 (UWB 4×-data test): development report

Branch `round18` (from main 812fa93), written 2026-10-30 13:40 UTC. Binding spec: `docs/NEXT_PHASE_PLAN.md`, "Round 18: the UWB 4×-data test (declared 2026-10-30, 13:03 UTC)", with the UWB lifter v2 / gate v2 declaration (11:12 UTC) and its amendments 1–2.

Nothing has run on a pod. Round 18 runs only if gate v2 passes. Everything below was tested on synthetic stores or read from R2; nothing was written to R2.

## What was built

| Piece | File |
|---|---|
| Converter: `--fps-sources estimated`, `--stream-check KEY@SHA`, a `head` bundle check; left-out members are listed as `skipped` and never read | `scripts/convert_uwb_clips.py` |
| Conversion pod job: shards, merge, manifest | `runpod/fleet/round18_convert.sh` |
| Mix builder: `round18-mixed-v1` = round14b-mixed-v1 + `youtube-asl-uwb-lifted-v2` | `scripts/build_round18_manifest.py` |
| R18 training job | `runpod/fleet/round18_train.sh` |
| R18 pilot, comparisons and verdict job | `runpod/fleet/round18_pilot.sh` |
| The declared rule, exit 0 PASS / 4 FAIL / 2 not evaluable | `scripts/round18_verdict.py` |
| Tests | `tests/test_convert_uwb_clips_script.py` (+3), `tests/test_build_round18_manifest.py` (4), `tests/test_round18_verdict.py` (7) |

### Converter additions

The existing flags already covered most of the spec: `--eval-manifest` (required), `--overlap-table` (v2 required), `--fps-table`, `--visibility head`, `--shard/--shards`, `--merge-index`, and resume with a settings fingerprint. Added:
- **`--fps-sources estimated`**: only videos with an fps-table entry are converted.
- **`--stream-check`**: reads the gate report that wrote the fps table (`gate-drift.json`, sha `dd130f78…`). A table video is left out when its stream does not match by the gate's `stratum_of` rule (`|fps / meta_fps − 1| ≥ 0.002`) or has no `meta_fps`. The run is refused unless every table video is in that report at the table's rate.
- Both settings are in the resume fingerprint.
- `--visibility head` with a bundle that has no validity head is refused before any worker starts. Before this change it raised inside the pool.

### Conversion job

`round18_convert.sh` runs in three modes:
- **Shard mode** pins fps table v3 (`b3d2a80c…`), the stream check (`dd130f78…`), the v2 overlap table (`c617a2af…`) and three evaluation manifests. Visibility is `head`, faces use blendshapes, head rotation is off and the flags are `generator`: the converter defaults the gate-v2 downstream row read.
- **Shard refusals:** `BUNDLE_KEY`/`BUNDLE_SHA` must be set. A shard whose summary says it finished is refused (exit 3). Interrupted shards resume.
- **Workers:** from `sched_getaffinity` and cgroup memory (2 GB each), never `nproc`.
- **Other modes:** `merge` writes the index and refuses unless it is complete; `manifest` runs the builder.

### Builder

- **Input check:** it reads the complete index and checks every declared conversion setting: bundle SHA, `head`, fps table, stream check, overlap table, `estimated`, the eval-manifest list, world, face, head, flags, minimum frames and adapter. Every converted clip must be estimated-fps with that bundle and rule.
- **Dedup:** `uwb_manifest.select_training_records` against round14b's `youtube-asl-v1` records, using its own `EvalIndex`, whose SHA must equal the converter's.
- **Rejected cues stay out (amendment 2):**
  - Each lifted clip is joined to our cue: its `duplicate_of_our_cue`, else its largest overlap covering at least half the clip.
  - The cue's fate in `youtube-asl-v1` (`manifest-r3b2`, sha `f69ad828…`) is recomputed from our extraction reports with that manifest's own filters. `our_cue_reasons` mirrors `build_youtube_asl_manifest.build` and is tested against it, channel cap included.
  - **Kept:** clips with no cue of ours (`never_extracted`) and cues rejected as `hands_not_visible`.
  - **Dropped:** every other rejection, plus cues our filters keep but the snapshot lacks.
- **Partitions:** a UWB dev clip of one of our training videos takes our video's partition and trains. No evaluation data is exposed. Other UWB dev clips, our monitor videos and `monitor_channel` channels stay out.
- **Segment rule:** `youtube-asl-v1`'s rule, applied where the field exists. Cleaned-text leaks are dropped (`evaluation_leak_cleaned`). One text is kept per video, counting our own selected texts of that video.
- **Refusals:**
  - fewer than 20,000 lifted records: `ROUND18-NO-MEANINGFUL-GAIN`, exit 5, nothing written;
  - any kept text that is a How2Sign or ASLLRP val/test/heldout sentence, from the eval manifests or from the base (an assertion after the filter);
  - a moved validation prefix (1,536 / 2,048 records);
  - a changed 1,000-clip pilot sample;
  - any existing output key.
- **Output:** it prints counts per corpus and partition, lifted hours, and `LIFTED_SHA` / `MANIFEST_SHA`.

## Amendment 2: expected lifted records (real R2 metadata, read-only, 2026-10-30 ~14:30 UTC)

The builder's own `select_lifted` was run before any conversion, assuming every eligible clip converts and full hand coverage:
- **Population:** 200,737 candidate clips.
- **`select_training_records`:**

  | Outcome | Clips |
  |---|---|
  | Kept | 96,640 |
  | Duplicate of our cue | 80,812 |
  | Our partition unknown | 19,419 |
  | Evaluation leak | 3,764 |
  | Near duplicate | 102 |

- **Rejected cues** (amendment 2):

  | Fate of our cue | Clips dropped |
  |---|---|
  | Passes our filters but not in snapshot | 45,897 |
  | Channel cap | 18,329 |
  | Too fast | 11,405 |
  | Face not visible | 5,637 |
  | Few words | 1,642 |
  | Not frontal | 1,192 |
  | Too slow | 1,148 |
  | Repeated text | 586 |
  | Caption coverage | 517 |
  | Note | 497 |
  | Resolution | 138 |
  | Second person | 49 |
  | Evaluation leak | 1 |

  Left after this step: 7,551 `hands_not_visible` and 2,051 `never_extracted`.
- **Segment rule:** duration 1,647, few words 290, repeated text 88, monitor 50, note 45, too fast 24, too slow 8, cleaned-text leak 1.

**Expected lifted records: 7,449 (7.7 h; 706 videos, 250 channels).** Of these, 7,416 are `hands_not_visible` and 33 are `never_extracted`.

That is below the 20,000 floor, so the manifest stage would print `ROUND18-NO-MEANINGFUL-GAIN` and exit 5. Round 18 as declared would not train. Even re-admitting the caption and timing rejections, re-judged on UWB's own caption, gives only 8,361.

**Snapshot check:**
- Our reports now number 2,922. That is not the report list `manifest-r3b2` was built from.
- The recomputation reproduces 98,553 of the snapshot's 100,454 cues. The 1,901 it misses are channel-cap shifts on the larger set.
- Our filters now keep **75,982 more cues (117.0 h, 1,317 videos) that round14b does not contain.** Their Tasks states exist (20/20 sampled).
- These are our own states, not lifted. A rebuilt YouTube manifest would add about 77% more YouTube data with no lifter.

## Findings that need a decision or a note

1. **The data gain is about 1.8×, not 4×.**
   - The estimated-fps population is only the 2,509 videos we downloaded ourselves, because the fps estimate needs our cues. 2,287 of them are stream-matched, which gives 200,737 UWB clips.
   - The 176,264 clips of videos we never downloaded have no estimated rate. They are excluded until the held-out ablation the plan names.
   - An upper-bound simulation on real metadata (before conversion refusals, assuming full hand coverage) keeps **≤ 79,666 lifted train records, ≤ 104.7 h**. Ours are 99,918 records, 151.8 h.
   - What drops them: duplicates of our cues 80,812; our partition unknown 19,419; evaluation leaks 3,764; the segment rule (mostly too fast) 16,346; monitor 628.
   - The kept clips are, by construction, cues our own pipeline did not select. Their quality may be lower than ours.
2. **The evaluation index needed round14b.**
   - 1,460 of round14b's 3,096 How2Sign val sentences, which are the pilot's population, are *not* val/test in `full-v1/how2sign-text-manifest-v1`.
   - The conversion and the builder therefore also pass `round14b-mixed-v1@710a3941…` as an `--eval-manifest`.
   - Separately, 3 of our existing `youtube-asl-v1` train records match those texts. R16C and W14b share them, so the effect is not differential.
3. **The pilot cannot change only the checkpoint.**
   - `load_generator` refuses a checkpoint trained on a manifest other than `--mix-sha256`, so the pilot names `round18-mixed-v1` and its SHA.
   - Its How2Sign records are byte-identical to round14b's, and the builder refuses a changed pilot sample.
   - `--output-key` is `round18/pilot-r18.json`; R16C's key would overwrite R16C's pilot.
   - Every other flag is byte-identical to `job_r16c-pilot.sh` (checked by diff). The training command differs from `job_r16c-train.sh` only in MIX/MSHA and PRUN (checked by diff).
4. **Segment rule (my choice, not in the declaration):** lifted clips get `youtube-asl-v1`'s caption and duration rule, plus hand coverage ≥ 0.5 read as the larger per-side hand fraction. Face, yaw and second-person fields do not exist for lifted clips. One text is kept per video. Without the rule, 0.3-s or "[music]" clips would train.
5. **Still to run on a GPU:** the pilot smoke (`SMOKE=1`, 8 clips, on R18's step-300 smoke checkpoint). The plan requires it before the chain is declared ready.

## Verification

- **Tests:** 116 passed in the touched and adjacent suites (warning-strict). The full suite (`tests/`, warning-strict) also passes (exit 0, 13.7 min).
- **Converter `--dry-run` against R2** with the real pinned inputs and shard 0/8: 10 zips, 400 units, fps table 2,509 videos, stream check 2,287 match / 222 mismatch (consistent with the table), overlap 2,605 videos, 9,890 evaluation sentences. Nothing was written.
- **Comparisons and verdict:** the comparisons run under `pipefail` into local files, each status is checked, and the verdict reads those files. The compare and verdict keys are write-once.
- **Verdict on real data:** `compare_pilot_lexicon.py` was run locally, without `--output-key`, with R16 as the arm. The verdict reproduces round 16 exactly:
  - vs R16C: +0.109 [−0.203, +0.427];
  - vs W14b: −0.018 [−0.333, +0.295];
  - lexicon: +0.020 [−0.020, +0.060];
  - result FAIL, exit 4.
- **Pinned keys exist, with these SHAs (R2, 2026-10-30):**
  - R16C pilot `round16/pilot-r16c.json` `b90c537a…`
  - W14b v3 read `v3read/pilot-w14b-step30000-v3d.json` `d90daf72…`
  - W14b step 30000 `0a689471…` (as pinned in R16C)
  - tokenizer d2 step 5000 `021ee00a…`
  - word list `round14/r5-word-split.json` `676f56cb…`
  - round10 vocab mix `ccfd10b8…`
  - round14b `710a3941…`
  - fps table v3 `b3d2a80c…`
  - gate-drift `dd130f78…`
  - overlap v2 `c617a2af…`
  - How2Sign full-v1 `0529f229…`
  - ASLLRP plan v2 `b0187725…`
  - **Free:** `training-runs/tokens-rvq-d0b-prior-r18-uwb`, `round18-mixed-v1` and `round18/` do not exist yet.
- `bash -n` passes for all three jobs.

## Code bundles

- **R16C training:** the bundle of main at launch (00:25–00:27 UTC). That is `7876a0d`, which holds the rollout fixes `52965cc`. This is inferred from the commit times: `run.json` and the job log do not record it.
- **R16C pilots:** ran `f96ad7f` / `965382c` (plan, round 16 result).
- **R18:** trains on the main that holds this branch. `make_bundle.sh` prints its commit at launch, and that commit is recorded here then.
- **Trainer changes since R16C:** only round 17's opt-in plan-prior flags, all off by default.
- **Recipe check:** `round18_train.sh` checks R18's smoke `run.json` against R16C's smoke `run.json` before the full run. They must have the same `config_sha256` (`6a921061…`), the same arguments except run name and manifest, and new arguments at their defaults. Tested with a shim on R16C's real `run.json`.

## Launch sequence (only after gate v2 PASSES)

From the lifter-v2 chain output (`CHAIN-SHAS … BUNDLE_SHA=…`):

```bash
cd ~/.sign-translator-ops
export BUNDLE_KEY=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v2/train/files/world-regressors-v2-uwb.pt
export BUNDLE_SHA=<BUNDLE_SHA from uwb_lifter_v2_cpu.sh>
# 0. inputs and unit count only (no writes)
CPU=8 DISK=40 bash run_job_pod.sh r18-convert-dry "BUNDLE_KEY=$BUNDLE_KEY BUNDLE_SHA=$BUNDLE_SHA DRY_RUN=1 bash runpod/fleet/round18_convert.sh 0 4"
# 1. conversion: 4 CPU pods (32 vCPU -> 30 workers), shards 0..3; rerun any shard that exits 1 (resumes)
for i in 0 1 2 3; do CPU=32 DISK=60 bash run_job_pod.sh r18-convert-$i "BUNDLE_KEY=$BUNDLE_KEY BUNDLE_SHA=$BUNDLE_SHA bash runpod/fleet/round18_convert.sh $i 4" >> job_r18-convert-$i.log 2>&1 & sleep 60; done; wait
# 2. merge (refused unless complete), then the mix (prints MANIFEST_SHA=...)
CPU=8 DISK=40 bash run_job_pod.sh r18-merge "bash runpod/fleet/round18_convert.sh merge"
CPU=16 DISK=60 bash run_job_pod.sh r18-manifest "BUNDLE_SHA=$BUNDLE_SHA bash runpod/fleet/round18_convert.sh manifest"
# exit 5 + ROUND18-NO-MEANINGFUL-GAIN (< 20,000 lifted records): stop here, round 18 is not trained (amendment 2)
export MANIFEST_SHA=<MANIFEST_SHA from r18-manifest>
# 3. training (GPU, as R16C: prefetch + SHA check, smoke 300 steps, then 10,000 steps)
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r18-train "MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round18_train.sh"
#    the pilot smoke can run as soon as the smoke checkpoint exists (8 clips, before the chain is declared ready):
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r18-pilot-smoke "SMOKE=1 MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round18_pilot.sh"
# 4. pilot + comparisons + verdict (exit 0 PASS / 4 FAIL); verdict at youtube-asl-v1/round18/verdict-r18.json
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r18-pilot "MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round18_pilot.sh"
```

`GDCS` is round 16's data-centre list (`r16_chain.sh`). The jobs must run from a code bundle that holds this branch.

## Estimated compute

- **Conversion:**
  - 200,737 clips at about 1.4 s per clip per worker (the lifter-v1 smoke, with blendshapes and R2 reads; the `head` MLP adds little) is about 78 worker-hours.
  - On 4 × 32-vCPU pods with 30 workers each, that is about 40–50 min per pod, about 3 CPU pod-hours.
  - Skipped members are not read.
  - States in R2: about 46 KB per second of motion × about 240 h is about 40 GB.
- **Manifest:** minutes. It reads the 181 MB base and 400 unit reports.
- **Training:**
  - R16C's 10,000 steps at batch 64, so the GPU time is R16C's.
  - The prefetch grows from about 36 GB to at most about 53 GB of states (round14b 220 h + ≤ 105 h lifted). That is within the trainer's 100 GiB cache and the default 120 GB disk.
- **Pilot:** as R16C's, plus the two comparisons (CPU, minutes).
