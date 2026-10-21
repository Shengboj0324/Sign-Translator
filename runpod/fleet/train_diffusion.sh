#!/usr/bin/env bash
# Continuous text-to-motion diffusion (usage: train_diffusion.sh NAME D_MODEL MOTION_LAYERS): x0-parameterised DiT-style
# denoiser over the canonical channels with the signing-space location term and grammar conditioning, EMA weights,
# fixed-seed validation, best.json and early stopping; then the same evaluation as the token generators. Discrete
# token priors did not generalise on 4k clips (v3 validation never below chance), which a continuous objective avoids.
set -uo pipefail
NAME=$1 DM=$2 ML=$3
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() {
  local run=$1 name=$2; shift 2
  echo "STAGE $name $run $(date)"
  "$@" > "/root/stage-$name-$run.log" 2>&1; local status=$?
  tail -3 "/root/stage-$name-$run.log"
  upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $status = 0 ] || echo "FAILED $name $run (status $status)"
  return $status
}
MSHA=$(sha "$M")
RUN=t2m-diffusion-$NAME-subset-v1
stage $RUN train python scripts/train_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" --run $RUN \
  --objective diffusion --steps 60000 --batch-size 32 --lr 3e-4 --warmup 2000 --d-model $DM --motion-layers $ML \
  --feedforward $((DM * 4)) --eval-every 1000 --eval-batches 16 --checkpoint-every 2000 --patience 6 --workers 8 && \
stage $RUN evaluate python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
  --checkpoint-key "$(best $RUN)" --partition val --guidance 1,2,3 --device cuda --output-key "$D/evaluations/$RUN.json"
echo T2M-DONE
