#!/usr/bin/env bash
# Text-motion retrieval evaluator (usage: train_retrieval.sh NAME): contrastive motion/text model on real pairs,
# early-stopped on validation R@3; frozen afterwards to score generators by R-precision. Stage log uploaded to R2.
set -uo pipefail
NAME=$1
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
MSHA=$(sha "$M")
RUN=retrieval-$NAME-subset-v1
echo "STAGE train $RUN $(date)"
python scripts/train_text_motion_retrieval.py --manifest-key "$M" --manifest-sha256 "$MSHA" --run $RUN --batch-size 128 \
  --steps 20000 --val-every 500 --patience 8 --workers 8 > /root/stage-train-$RUN.log 2>&1 || echo "FAILED train $RUN"
tail -3 /root/stage-train-$RUN.log; upload "training-runs/$RUN/stage-train.log" "/root/stage-train-$RUN.log"
echo RETRIEVAL-DONE
