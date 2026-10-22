#!/usr/bin/env bash
# post_full_train_retrieval.sh — pod B: waits for the manifest pod A writes, then trains the duration-blind retrieval
# evaluator exactly as post_full_train.sh does. FULL-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
export SPLITS="train val test" MODELS=/root/work/models PY=python DERIVED=acquisition/2026-10-17-v1/derived/full-v1
D=$DERIVED; M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
wait_key() { until python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; do sleep 300; done; }
wait_key "$M"; MSHA=$(sha "$M")
RET=retrieval-hold-full-v1
stage $RET retrieval python scripts/train_text_motion_retrieval.py --manifest-key "$M" --manifest-sha256 "$MSHA" --run $RET \
  --batch-size 128 --steps 30000 --val-every 500 --patience 8 --workers 8
echo FULL-DONE
