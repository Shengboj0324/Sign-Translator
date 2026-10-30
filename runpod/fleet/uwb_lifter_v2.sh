#!/usr/bin/env bash
# uwb_lifter_v2.sh — UWB lifter v2 and gate v2 (plan "UWB lifter v2 and gate v2: declared", 2026-10-30 11:12 UTC).
# One phase per run, each on its own pod run; every phase writes to a new prefix (refused if its INDEX.json exists) and
# checks the SHA-256 of every input it reads (pinned constants, or the SHA the previous phase printed, passed in env).
#
#   uwb_lifter_v2.sh split                                         (CPU)  -> lifter-v2/split/      prints SPLIT_SHA
#   SPLIT_SHA=.. uwb_lifter_v2.sh pairs                            (CPU)  -> lifter-v2/pairs/      prints PAIRS_INDEX_SHA
#   uwb_lifter_v2.sh crops                                         (CPU)  -> lifter-v2/crops/      prints CROPS_SHA
#   SPLIT_SHA=.. PAIRS_INDEX_SHA=.. CROPS_SHA=.. uwb_lifter_v2.sh train      (CPU)  -> lifter-v2/train/  prints BUNDLE_SHA
#   SPLIT_SHA=.. PAIRS_INDEX_SHA=.. BUNDLE_SHA=.. uwb_lifter_v2.sh gate      (CPU)  -> lifter-v2/gate/   prints GATE_SHA
#   SPLIT_SHA=.. PAIRS_INDEX_SHA=.. BUNDLE_SHA=.. GATE_SHA=.. uwb_lifter_v2.sh downstream (GPU) -> lifter-v2/downstream/
#
# The split's composition (split/files/uwb-lifter-split-v1.json "composition") is published before pairs or training.
# Each phase ends with UWB-V2-<PHASE>-DONE. Workers: sched_getaffinity and cgroup memory, never nproc.
set -euo pipefail
PHASE=${1:?usage: uwb_lifter_v2.sh split|pairs|crops|train|gate|downstream}
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
K1=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1
V2=${V2_PREFIX:-acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v2}
XY2=$K1/files/world-regressors-xy2-v1.pt; XY2_SHA=02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde
PAIRS_TAR=scratch/uwb-lifter-dev/world-pairs-v2/pairs.tar; PAIRS_TAR_SHA=6c4d3ecc18e66f2bbc885aacda9a86499d7f5ce48618fa2438540ca958526758
LEGACY_TAR=scratch/uwb-lifter-dev/world-pairs-v2/legacy-pairs.tar; LEGACY_TAR_SHA=7042e0a8bcd7452d13743fcac5f6b37be1854fafb06c417ab54fa8aa24c65fa0
OFD_TAR=scratch/uwb-lifter-dev/outside-frame-data.tar; OFD_SHA=598ab2ce80f05d663cdb1a0805fb4c56e412ff5708e76810c886d9eb028cc1a3
LEGVAL_TAR=scratch/uwb-lifter-dev/legacy-val-v2.tar; LEGVAL_SHA=b131e878a8a7a48e9ccf41f2e35de3d8d71e56f37535e7ab270cf6ef0da058fb
CANDIDATES=learned,inside,margin:0.1,margin:0.25
DEV=reports/uwb_gate_smoke_videos.txt
N=$(python -c "import os
cpus = len(os.sched_getaffinity(0)) - 2
try: mem = int(open('/sys/fs/cgroup/memory.max').read()) / 2**30
except (OSError, ValueError): mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
print(max(1, min(30, cpus, int(mem // 2))))")   # not nproc (it honours OMP_NUM_THREADS=1: gave --workers -1); 2 GB per worker
W=/root/v2-$PHASE; rm -rf $W && mkdir -p $W/out

fetch() {   # fetch KEY SHA DEST: the object, refused unless its SHA-256 is SHA
  python - "$1" "$2" "$3" <<'PY'
import hashlib, os, sys
from signtranslator.data.text_motion_dataset import R2Store
key, sha, dest = sys.argv[1:4]
b = R2Store(os.environ).get(key)
if hashlib.sha256(b).hexdigest() != sha:
    sys.exit(f"{key}: SHA-256 {hashlib.sha256(b).hexdigest()} is not the pinned {sha}")
open(dest, "wb").write(b)
print("verified", key, flush=True)
PY
}
fetch_phase() {   # fetch_phase PHASE DIR: every file of an earlier phase, each checked against that phase's INDEX
  python - "$V2/$1" "$2" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
prefix, out = sys.argv[1:3]
s = R2Store(os.environ)
for f in json.loads(s.get(f"{prefix}/INDEX.json"))["files"]:
    b = s.get(f["key"])
    if hashlib.sha256(b).hexdigest() != f["sha256"]:
        sys.exit(f"{f['key']}: SHA-256 differs from {prefix}/INDEX.json")
    (Path(out) / f["path"]).parent.mkdir(parents=True, exist_ok=True)
    (Path(out) / f["path"]).write_bytes(b)
print("fetched", prefix, flush=True)
PY
}
need() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "$1: SHA-256 is not the pinned $2"; exit 1; }; }
refuse_existing() {
  python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(f'{sys.argv[1]} exists: a rerun writes elsewhere (V2_PREFIX)') if R2Store(os.environ).exists(sys.argv[1]) else None" "$V2/$1/INDEX.json"
}
store() {   # store PHASE NOTE
  python scripts/store_local_files.py --dir $W/out --prefix $V2/$1 --source "uwb_lifter_v2.sh $1 (branch uwb-lifter-v2 $(git rev-parse --short HEAD))" \
    --licence "derived (UWB CC BY 4.0; our YouTube-ASL states research-only; How2Sign CC BY-NC 4.0)" --note "$2"
}
split_file() { fetch $V2/split/files/uwb-lifter-split-v1.json "${SPLIT_SHA:?SPLIT_SHA from the split phase}" $W/split.json; }
xy2() { fetch $XY2 $XY2_SHA $W/xy2.pt; }
h2s_pairs() {
  fetch $PAIRS_TAR $PAIRS_TAR_SHA $W/pairs.tar && fetch $LEGACY_TAR $LEGACY_TAR_SHA $W/legacy-pairs.tar
  tar -xf $W/pairs.tar -C $W && tar -xf $W/legacy-pairs.tar -C $W && rm $W/pairs.tar $W/legacy-pairs.tar
}
yt_pairs() {
  mkdir -p $W/yt && fetch_phase pairs $W/yt
  need $W/yt/pairs-index.json "${PAIRS_INDEX_SHA:?PAIRS_INDEX_SHA from the pairs phase}"
}
pairs_index() { fetch $V2/pairs/files/pairs-index.json "${PAIRS_INDEX_SHA:?PAIRS_INDEX_SHA from the pairs phase}" $W/pairs-index.json; }
refuse_existing $PHASE

case $PHASE in
split)
  python scripts/build_uwb_lifter_split.py --exclude-videos $DEV --workers $N --out $W/out/uwb-lifter-split-v1.json
  store split "channel-disjoint split; composition published before training"
  echo "SPLIT_SHA=$(sha256sum $W/out/uwb-lifter-split-v1.json | cut -d' ' -f1)" ;;
pairs)
  split_file
  python scripts/build_uwb_lifter_pairs.py --split-file $W/split.json --workers $N --out-dir $W/out
  store pairs "train and lifter-val pairs (stream match, arm-2d, lags in {-1,0,1}, >= 400 bone-frames, lag-0 <= 20 deg)"
  echo "PAIRS_INDEX_SHA=$(sha256sum $W/out/pairs-index.json | cut -d' ' -f1)" ;;
crops)
  python scripts/uwb_lifter_v2_crops.py fetch --clips-per-signer 40 --seed 0 --video-dir $W/videos --list $W/videos.txt
  python scripts/elbow_outside_accuracy.py extract --videos $(cat $W/videos.txt) --out-dir $W/c/crops \
    --models-dir /root/work/models --workers $N
  uv venv -q -p 3.11 /root/legacy-mp       # legacy Holistic, as the v2 pairs' inputs (mediapipe 0.10.14, complexity 1)
  VIRTUAL_ENV=/root/legacy-mp uv pip install -q mediapipe==0.10.14 av==12.3.0 numpy==2.4.6 protobuf==4.25.9
  python scripts/uwb_lifter_v2_crops.py legacy-jobs --video-dir $W/videos --crop-dir $W/c/crops --legacy-dir $W/c/legacy \
    --jobs $W/legacy-jobs.json
  mkdir -p $W/c/legacy
  /root/legacy-mp/bin/python scripts/legacy_holistic_arrays.py --jobs $W/legacy-jobs.json --workers $N \
    --hashes $W/c/legacy-holistic-models.json
  [ "$(ls $W/c/crops | wc -l)" -eq "$(ls $W/c/legacy | wc -l)" ]
  tar -cf $W/out/crops.tar -C $W/c crops legacy legacy-holistic-models.json
  store crops "tight crops of How2Sign training signers 3, 4, 5, 8, 11 (Tasks) + legacy Holistic 0.10.14 arrays"
  echo "CROPS_SHA=$(sha256sum $W/out/crops.tar | cut -d' ' -f1)" ;;
train)
  split_file; xy2; h2s_pairs; yt_pairs
  fetch $V2/crops/files/crops.tar "${CROPS_SHA:?CROPS_SHA from the crops phase}" $W/crops.tar
  tar -xf $W/crops.tar -C $W && rm $W/crops.tar
  T="python scripts/train_uwb_lifter_v2.py"
  COMMON="--split-file $W/split.json --yt-pairs $W/yt --h2s-pairs $W/pairs --h2s-legacy $W/legacy-pairs \
    --crops $W/crops --crops-legacy $W/legacy --init-bundle $W/xy2.pt --init-sha256 $XY2_SHA --seed 0"
  # Pose arms run one after another by default: each holds every YouTube frame in memory. PARALLEL_ARMS=1 runs the three
  # together (size the pod's memory for 3x one arm's peak RSS, which each arm prints and records in its report).
  ARMS=("pose-ft --epochs 15 --lr 3e-4" "pose-scratch --from-scratch --epochs 30 --lr 1e-3"
        "pose-ft-hands --finetune-hands --epochs 15 --lr 3e-4")
  if [ "${PARALLEL_ARMS:-0}" = 1 ]; then
    export TORCH_THREADS=$(( N / 3 > 0 ? N / 3 : 1 )); PIDS=()
    for A in "${ARMS[@]}"; do
      read NAME FLAGS <<< "$A"
      $T pose $COMMON $FLAGS --out $W/out/$NAME.pt --report $W/out/$NAME.json > $W/out/$NAME.log 2>&1 & PIDS+=($!)
    done
    for P in "${PIDS[@]}"; do wait $P; done
  else
    export TORCH_THREADS=$N
    for A in "${ARMS[@]}"; do
      read NAME FLAGS <<< "$A"
      $T pose $COMMON $FLAGS --out $W/out/$NAME.pt --report $W/out/$NAME.json 2>&1 | tee $W/out/$NAME.log
    done
  fi
  for A in "${ARMS[@]}"; do
    read NAME FLAGS <<< "$A"
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], 'peak RSS', r['peak_rss_mb'], 'MB')" \
      $W/out/$NAME.json $NAME
  done
  $T select --reports $W/out/pose-ft.json $W/out/pose-scratch.json --out $W/out/selection.json
  CHOSEN=$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(['pose-ft','pose-scratch'][s['chosen']])" $W/out/selection.json)
  export TORCH_THREADS=$N
  $T validity --split-file $W/split.json --yt-pairs $W/yt --crops $W/crops --crops-legacy $W/legacy \
    --bundle $W/out/$CHOSEN.pt --bundle-sha256 $(sha256sum $W/out/$CHOSEN.pt | cut -d' ' -f1) \
    --out $W/out/world-regressors-v2-uwb.pt --report $W/out/world-regressors-v2-uwb.json
  store train "lifter v2: pose arms (fine-tuned / from scratch / hands comparison), selection on lifter-val, validity head"
  echo "BUNDLE_SHA=$(sha256sum $W/out/world-regressors-v2-uwb.pt | cut -d' ' -f1)" ;;
gate)
  split_file; xy2; h2s_pairs
  fetch $V2/train/files/world-regressors-v2-uwb.pt "${BUNDLE_SHA:?BUNDLE_SHA from the train phase}" $W/v2.pt
  fetch $OFD_TAR $OFD_SHA $W/ofd.tar && mkdir -p $W/ofd-x && tar -xf $W/ofd.tar -C $W/ofd-x && rm $W/ofd.tar
  fetch $LEGVAL_TAR $LEGVAL_SHA $W/legval.tar && mkdir -p $W/legval-x && tar -xf $W/legval.tar -C $W/legval-x && rm $W/legval.tar
  OFD=$(dirname "$(find $W/ofd-x -maxdepth 3 -type d -name twoview | head -1)")      # the tar's root: val, val-tight, twoview
  LEGVAL=$(dirname "$(find $W/legval-x -name '*.npz' | head -1)")                    # <clip name>.npz arrays
  [ -d $OFD/val ] && [ -d $OFD/val-tight ] && [ -f $OFD/twoview/index.json ] && [ -d "$LEGVAL" ]
  G="python scripts/uwb_lifter_gate.py --split-file $W/split.json --visibility inside --candidate-visibility $CANDIDATES \
    --exclude-videos $DEV --videos 0 --pairs-per-video 12 --bootstrap 2000 --lag-curves --head affine \
    --models-dir /root/work/models --workers $N"
  $G --regressors $W/v2.pt --regressors-sha256 $BUNDLE_SHA --secondary-visibility head --out $W/out/gate-v2.json
  $G --regressors $W/xy2.pt --regressors-sha256 $XY2_SHA --out $W/out/gate-v1-reference.json
  for B in v2:$W/v2.pt:$BUNDLE_SHA xy2:$W/xy2.pt:$XY2_SHA; do
    IFS=: read NAME PT S <<< "$B"
    python scripts/uwb_how2sign_val_medians.py --pairs-dir $W/pairs --legacy-dir $W/legacy-pairs --regressors $PT \
      --regressors-sha256 $S --visibility inside --candidate-visibility $CANDIDATES --out $W/out/how2sign-val-$NAME.json
  done
  python scripts/uwb_outside_frame_accuracy.py --crop-dirs $OFD/val $OFD/val-tight --two-view-dir $OFD/twoview \
    --legacy-dir $LEGVAL --require-legacy --uwb-weights $W/out/gate-v2.json --regressors $W/v2.pt \
    --regressors-sha256 $BUNDLE_SHA --rules $CANDIDATES,head --tight-summary --report $W/out/outside-frame-v2.json
  mkdir -p $W/train && fetch_phase train $W/train
  pairs_index
  python scripts/uwb_gate_verdict.py --store none --gate-version 2 --gate-report $W/out/gate-v2.json \
    --v1-reference $W/out/gate-v1-reference.json --how2sign-val $W/out/how2sign-val-v2.json \
    --how2sign-val-v1 $W/out/how2sign-val-xy2.json --outside-frame $W/out/outside-frame-v2.json \
    --split-file $W/split.json --pairs-index $W/pairs-index.json --training-report $W/train/world-regressors-v2-uwb.json \
    --pose-report $W/train/$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(['pose-ft','pose-scratch'][s['chosen']])" $W/train/selection.json).json \
    --bundle-sha256 $BUNDLE_SHA --out $W/out/gate-verdict-v2-without-downstream.json \
    --markdown $W/out/gate-verdict-v2-without-downstream.md
  store gate "gate v2 (held-out channels, 12 pairs/video) for v2 and the xy2 reference; How2Sign-val medians; outside-frame"
  echo "GATE_SHA=$(sha256sum $W/out/gate-v2.json | cut -d' ' -f1)" ;;
downstream)
  uv pip install -q 'transformers>=4.40,<5' sentencepiece        # the back-translator (T5)
  KPMANIFEST=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2   # as uwb_downstream.sh
  KPSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
  split_file; pairs_index
  fetch $V2/train/files/world-regressors-v2-uwb.pt "${BUNDLE_SHA:?BUNDLE_SHA}" $W/v2.pt
  mkdir -p $W/gate $W/train && fetch_phase gate $W/gate && fetch_phase train $W/train
  need $W/gate/gate-v2.json "${GATE_SHA:?GATE_SHA from the gate phase}"
  python scripts/uwb_downstream_check.py --gate-report $W/gate/gate-v2.json --regressors $W/v2.pt --regressors-sha256 $BUNDLE_SHA \
    --instrument-manifest-sha256 $KPMANIFEST --instrument-sha256 $KPSHA --models-dir /root/work/models --pairs 2000 \
    --seed 0 --bootstrap 2000 --device cuda --out $W/out/downstream-v2.json
  python scripts/uwb_gate_verdict.py --store none --gate-version 2 --gate-report $W/gate/gate-v2.json \
    --v1-reference $W/gate/gate-v1-reference.json --how2sign-val $W/gate/how2sign-val-v2.json \
    --how2sign-val-v1 $W/gate/how2sign-val-xy2.json --outside-frame $W/gate/outside-frame-v2.json \
    --downstream $W/out/downstream-v2.json --split-file $W/split.json --pairs-index $W/pairs-index.json \
    --training-report $W/train/world-regressors-v2-uwb.json \
    --pose-report $W/train/$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(['pose-ft','pose-scratch'][s['chosen']])" $W/train/selection.json).json \
    --bundle-sha256 $BUNDLE_SHA --out $W/out/gate-verdict-v2.json --markdown $W/out/gate-verdict-v2.md
  store downstream "gate v2 downstream row and the full verdict" ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
echo "UWB-V2-$(echo $PHASE | tr a-z A-Z)-DONE"
