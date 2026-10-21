#!/usr/bin/env bash
# run_sapiens_full.sh I N — Sapiens (no flip) on every How2Sign clip (train, val, test; front and side), rotation I/N.
# Writes under derived/full-v1; the subset's 13,278 keypoint files were copied there first (R2 server-side), and the
# per-clip claims make pods skip finished and in-progress clips.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
export SPLITS="train val test" ROTATE="$1/$2" MODELS=/root/work/models PY=python \
       DERIVED=acquisition/2026-10-17-v1/derived/full-v1 SAPIENS_FLAGS="--no-flip --batch 24 --workers 16"
until bash runpod/pipeline.sh sapiens 0 1; do echo "retrying after failure $(date)"; sleep 30; done
echo SAPIENS-DONE
