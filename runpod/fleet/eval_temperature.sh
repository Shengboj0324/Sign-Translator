#!/usr/bin/env bash
# Sampling-temperature evaluation (usage: eval_temperature.sh RUN TEMPERATURE GUIDANCE): the run's best checkpoint on the
# same 200 validation clips at one guidance and temperature, generated states stored for re-scoring; log uploaded to R2.
set -uo pipefail
RUN=$1 T=$2 G=$3
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
MSHA=$(sha "$M"); TAG=$RUN-t$T
echo "STAGE evaluate-temperature $TAG $(date)"
python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" --checkpoint-key "$(best $RUN)" \
  --partition val --guidance "$G" --temperature "$T" --device cuda --save-generated "$D/generated/$TAG" \
  --output-key "$D/evaluations/$TAG.json" > /root/stage-evalT-$TAG.log 2>&1 || echo "FAILED evaluate-temperature $TAG"
tail -1 /root/stage-evalT-$TAG.log | cut -c1-300; upload "training-runs/$RUN/stage-evaluate-t$T.log" "/root/stage-evalT-$TAG.log"
echo EVAL-DONE
