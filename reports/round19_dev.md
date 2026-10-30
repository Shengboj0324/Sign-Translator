# Round 19 (refresh our own YouTube data): development report

Branch `round19` (from main 26e7660), written 2026-10-30. Binding spec: `docs/NEXT_PHASE_PLAN.md`, "Round 19: refresh our own YouTube data (declared 2026-10-30, 14:16 UTC)".

The manifests are built and stored in R2 (new keys, write-once). Nothing has trained. Times are UTC.

## How round14b's YouTube component was built (verified)

- **Source manifest:** round14b-mixed-v1 (`710a3941…`) holds 100,454 `youtube-asl-v1` records. Every one is byte-identical, apart from the `corpus` tag, to `youtube-asl-v1/manifest-r3b2/manifest.json` (`f69ad828…`).
- **Who built it:** `build_youtube_asl_manifest.py` built manifest-r3b2 on 2026-10-24 (plan, "Round 3b data rebuilt") from the extraction reports of that time. Its filters are recorded in the manifest:
  - manual captions only;
  - 2+ words, 1–20 s, 0.5–4 words/s, ≤ 300 characters;
  - hand coverage ≥ 0.5, face coverage ≥ 0.9, yaw ≤ 30°, second person ≤ 0.1;
  - video gate, channel cap 0.05, monitor share 0.01;
  - leak filter from How2Sign full-v1 and ASLLRP plan v2.
- **Path into round14b:** round3b-mixed-v2 → round3c → round6 → round10 (trims only the one-word corpora) → round14b.
- **round14b itself** = round10-mixed-v1 + `semlex-text-v2`.
  - `semlex-text-v2` is Sem-Lex from `states-video-v1`, with image-right signers 11, 23, 46 and 76 excluded (`round14_manifests.sh`).
  - That exclusion concerns Sem-Lex only, which round 19 keeps byte-identical.
- **Corpus weights** are training flags, and they are unchanged.

## What was built

| Piece | File |
|---|---|
| The rebuild (YouTube component, then the mix) | `scripts/build_round19_manifest.py` |
| Verdict generalised: `--arm R18/R19` (rule text, next step, `R19-VERDICT` marker) | `scripts/round18_verdict.py` |
| R19 training: R18's job with only the manifest and run names changed | `runpod/fleet/round19_train.sh` |
| R19 pilot and verdict: R18's job with only the names changed | `runpod/fleet/round19_pilot.sh` |
| Tests | `tests/test_build_round19_manifest.py` (3), `tests/test_round18_verdict.py` (+1) |

### `round19_train.sh`

- It requires `MANIFEST_SHA`, checks the mix's identity (`inputs` and `replaces`) and refuses an existing run.
- **Prefetch:** `prefetch_cache.py` fetches every state and checks its SHA before the smoke run. `round5_prior.sh` checks again before each stage.
- **Training:** R16C's exact command (diff-checked against R18's), then the smoke. After the smoke, R19's smoke `run.json` must match R16C's: `config_sha256` `6a921061…`, every argument except run name and manifest, and the new round-17 flags at their defaults. Then the full run.

### `round19_pilot.sh`

- **Pilot:** R16C's pilot command with checkpoint, mix and output key changed.
- **Comparisons:** run under `pipefail` into local files, and each status is checked.
- **Verdict:** `round18_verdict.py --arm R19` reads those local files.
- **Outputs:** written once to `youtube-asl-v1/round19/`.

### The rebuild

- **YouTube component:** `build_youtube_asl_manifest.build` runs over all 2,922 current reports with manifest-r3b2's own filters. Its leak filter also includes round14b's own val, test and heldout texts. `build` compares the cleaned caption, which is the text that trains. The result is asserted leak-free afterwards.
- **Mix:** round14b with `youtube-asl-v1` replaced. Every other top-level field is kept, and `replaces` records the change.
- **Refusals:**
  - round14b's YouTube records are not manifest-r3b2's;
  - any non-YouTube record is not byte-identical;
  - a YouTube record falls outside train or monitor;
  - the validation prefix (1,536 / 2,048) moved;
  - the 1,000-clip pilot sample changed;
  - any existing output key.

## Result (run for real, 2026-10-30)

| | Key | SHA-256 |
|---|---|---|
| YouTube component | `acquisition/2026-10-17-v1/youtube-asl-v1/manifest-r19/manifest.json` | `1c23e7aeacae70aafff4dd6fd6c1e45d397cfd335d70cdbc6813fa94041f3496` |
| Mix | `acquisition/2026-10-17-v1/youtube-asl-v1/round19-mixed-v1/manifest.json` | `99508be68ce97d9696d74691718e852da82133bc3b8f12da3905af414ea0b6af` |

The dry run and the real run gave identical SHAs.

- **Records:** 177,946 → **251,868**. YouTube grows from 100,454 to **174,376**:
  - train +73,343 (+113.3 h);
  - monitor +579 (+0.8 h).
- **YouTube delta against round14b:**
  - 98,458 old records kept, all byte-identical;
  - 1,996 dropped (2.8 h): 1,908 by the channel cap recomputed over the larger set, and 88 by round14b's own val/test texts;
  - 75,918 new (116.9 h): 75,338 train, 580 monitor.
- **Build reasons over all cues:**
  - too fast 16,241; hands not visible 15,190; face not visible 8,974; second person 6,259; few words 5,713;
  - not frontal 2,118; too slow 1,927; note 1,028; repeated 797;
  - evaluation leak 300; channel cap 27,062.
- **Checks:**
  - all non-YouTube records are byte-identical;
  - validation prefixes 1,536 and 2,048 are unchanged;
  - the pilot sample is unchanged;
  - both stored manifests load (canonical, `usable` consistent);
  - 200 of the 75,918 new states were sampled from R2 and all 200 matched their SHA;
  - the training job's mix check passes on the stored mix.

## Tests

The new and adjacent suites and the full suite pass under warning-strict pytest (see the commit message).

## Launch commands

These run from a code bundle of main that holds this branch (`make_bundle.sh`). `GDCS` is round 16's data-centre list (`r16_chain.sh`).

```bash
cd ~/.sign-translator-ops
export MANIFEST_SHA=99508be68ce97d9696d74691718e852da82133bc3b8f12da3905af414ea0b6af
# 1. training (GPU; prefetch + SHA check of every state, smoke, recipe check vs R16C, 10,000 steps)
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r19-train "MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round19_train.sh" >> job_r19-train.log 2>&1
#    pilot smoke as soon as the smoke checkpoint exists (8 clips; plan: before the chain is declared ready)
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r19-pilot-smoke "SMOKE=1 MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round19_pilot.sh" >> job_r19-pilot-smoke.log 2>&1
# 2. pilot + comparisons + verdict (exit 0 PASS / 4 FAIL / 2 not evaluable); youtube-asl-v1/round19/verdict-r19.json
BAD_DCS="US-NC-1 US-MO-1" DCS="$GDCS" bash run_job_pod.sh r19-pilot "MANIFEST_SHA=$MANIFEST_SHA bash runpod/fleet/round19_pilot.sh" >> job_r19-pilot.log 2>&1
```

**Prefetch stays inside the GPU job.**
- R16C's prefetch of 177,946 records took about 8 minutes on the GPU pod. Round 19 adds 75,918 states (about 18 GB), so roughly 4–5 more minutes.
- A separate CPU prefetch pod would not share the GPU pod's disk cache.
- The cache is about 333 h × 46 KB/s ≈ 55 GB. That fits the default 120 GB disk and the trainer's 100 GiB cache.

**Compute:** as R16C. The step count and batch are the same, so the GPU time is the same; only the prefetch grows.
