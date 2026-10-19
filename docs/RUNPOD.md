# Running on Runpod

Everything reads and writes the R2 bucket; no data is baked into images. Credentials
are Runpod **secrets**, exported into the pod environment as `R2_ENDPOINT_URL`,
`R2_BUCKET`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Never put them in a
command line, a file in the repository, or a log.

## 1. Pods

| Job | Pod | Why |
|---|---|---|
| Video extraction | CPU pod, many vCPUs (e.g. 32+) | MediaPipe Tasks runs on CPU in Python |
| Training | GPU pod (e.g. 1× A100/H100 80 GB or L40S 48 GB) | the text-to-motion transformer |

Use a PyTorch 2.13 image (CUDA build for GPU pods). Mount a network volume at
`/workspace` so the content cache and models survive restarts; set
`SIGN_CACHE_DIR=/workspace/cache`.

## 2. Bootstrap (every new pod)

```bash
export WORK=/workspace
curl -fsSL https://raw.githubusercontent.com/Shengboj0324/Sign-Translator/main/runpod/bootstrap.sh | bash
```

This clones the repository, installs `runpod/requirements-runpod.txt`, downloads the
three MediaPipe models from R2 (`models/mediapipe-tasks-0.10.21/`), verifies their
pinned SHA-256, and runs the smoke tests.

## 3. Extract video states (CPU pods; one command per shard)

The six clip zips are in R2 and verified (`scripts/verify_how2sign_zips.py` wrote a
member manifest beside each). Their SHA-256 values, re-checked by every job against
the acquisition provenance before it reads a byte:

| Zip (`acquisition/2026-10-17-v1/how2sign-public/`) | SHA-256 |
|---|---|
| `train_rgb_front_clips.zip` | `98e36aa4377dfb3b5e59de1ba5af94ade0a77e9565cacb0c7183267480b95cd3` |
| `train_rgb_side_clips.zip` | `dbcd63cf43bdfcb53534db7caec2b867ddd0fd652ed785faf21f8217d8446f6b` |
| `val_rgb_front_clips.zip` | `6cb48170131fdb7fc64d40381a3c4be41112ba4454444170f5a31af2227b6d36` |
| `val_rgb_side_clips.zip` | `333db1b9f187c5ebd98f1d8daff1c0e8bd2b99c858f46456ca39b003e5e07c48` |
| `test_rgb_front_clips.zip` | `cb63633542f1c3f703ad76ec5e1420bfe9bf028afdb120508ed7cded16ed1b55` |
| `test_rgb_side_clips.zip` | `ec5013b24ed561c46e65ec60ce3a1669c8ad218269c0496aea5d0fefb67a8e8e` |

Monocular states of the frontal view (face, head and blink channels of the two-view
states come from here; also the monocular comparison in the gate). Run shard `i` of
`n` for each split (start with `--limit 20` to check one shard end to end):

```bash
H=acquisition/2026-10-17-v1/how2sign-public; D=acquisition/2026-10-17-v1/derived
declare -A FRONT=([train]=98e36aa4377dfb3b5e59de1ba5af94ade0a77e9565cacb0c7183267480b95cd3 [val]=6cb48170131fdb7fc64d40381a3c4be41112ba4454444170f5a31af2227b6d36 [test]=cb63633542f1c3f703ad76ec5e1420bfe9bf028afdb120508ed7cded16ed1b55)
for split in train val test; do
  python scripts/extract_video_states.py --zip-key $H/${split}_rgb_front_clips.zip --zip-sha256 ${FRONT[$split]} --prefix $D/how2sign-video-estimate-v1 --models-dir /workspace/models --shard 0 --shards 8 --workers 30
done
```

Shards are independent and resumable: re-running skips clips already written. Clip
names are unique across splits (How2Sign splits by source video), so all splits share
one prefix. Each shard writes `reports/<zip>-shard-XXX-of-NNN.json`, so reports of
different zips never overwrite each other.

## 3b. Sapiens 2D on both views (GPU pods) and two-view triangulation

The checkpoint is mirrored in R2 at
`acquisition/2026-10-17-v1/models-sapiens/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2`;
copy it to `/workspace/models/`; its pinned SHA-256 is `d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3` (4,677,003,899 bytes).

Flip test-time augmentation is on by default (two forward passes per frame); the
session script then self-calibrates on rigid-bone joints only and refits each hand
with a rigid, temporally smooth two-view skeleton (`DATA_QUALITY_DECISION.md` §6).

```bash
declare -A SIDE=([train]=dbcd63cf43bdfcb53534db7caec2b867ddd0fd652ed785faf21f8217d8446f6b [val]=333db1b9f187c5ebd98f1d8daff1c0e8bd2b99c858f46456ca39b003e5e07c48 [test]=ec5013b24ed561c46e65ec60ce3a1669c8ad218269c0496aea5d0fefb67a8e8e)
CK=/workspace/models/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2; CS=d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3
for split in train val test; do
  python scripts/sapiens_keypoints_runpod.py --zip-key $H/${split}_rgb_front_clips.zip --zip-sha256 ${FRONT[$split]} --prefix $D/how2sign-sapiens-front-v1 --checkpoint $CK --checkpoint-sha256 $CS --shard 0 --shards 4
  python scripts/sapiens_keypoints_runpod.py --zip-key $H/${split}_rgb_side_clips.zip --zip-sha256 ${SIDE[$split]} --prefix $D/how2sign-sapiens-side-v1 --checkpoint $CK --checkpoint-sha256 $CS --shard 0 --shards 4
done
python scripts/triangulate_how2sign_sessions.py --front-keypoints $D/how2sign-sapiens-front-v1 --side-keypoints $D/how2sign-sapiens-side-v1 --monocular $D/how2sign-video-estimate-v1 --prefix $D/how2sign-two-view-v1 --shard 0 --shards 8
```

**Gate before training.** Read every `how2sign-two-view-v1/reports/*.json`: each session
lists its gate result and reasons. Train only on the two-view states (the manifest
below is built from `how2sign-two-view-v1/reports/`). If too few sessions pass, stop
and report — do not fall back to monocular states silently.

## 4. Build the training manifest (once, after all shards)

```bash
python scripts/build_how2sign_text_manifest.py --reports-prefix $D/how2sign-two-view-v1/reports/ --corpus-id how2sign-two-view-text-v1 --sentences $H/how2sign_realigned_train.csv $H/how2sign_realigned_val.csv $H/how2sign_realigned_test.csv --fold-plan "Sign Translator Stage Documentation/evidence/w1-fold-plan-2026-10-17/fold-plan.json" --fold-plan-sha256 00ea2bb9305bb78d96b25e4dbbc8eb2d1393ab862cd79857db56d73c11a34778 --manifest-key $D/how2sign-text-manifest-v1/manifest.json
```

It prints the manifest SHA-256 used in the next step.

## 5. Train (GPU pod)

```bash
python scripts/train_text_to_motion.py --manifest-key acquisition/2026-10-17-v1/derived/how2sign-text-manifest-v1/manifest.json --manifest-sha256 <MANIFEST_SHA256> --run t2m-v1 --steps 100000 --batch-size 32 --workers 8
```

Checkpoints go to `training-runs/t2m-v1/checkpoints/` every 2,000 steps. If the pod
stops, run the same command again: it resumes from the newest checkpoint and refuses
one made with a different configuration or manifest.

## 6. What you need to provide later

Only the four R2 secrets in Runpod. Nothing else in this workflow needs your
identity or a licence acceptance.
