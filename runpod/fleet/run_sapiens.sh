#!/usr/bin/env bash
# run_sapiens.sh I N — Sapiens (no flip) on the subset, both views of the train zips, rotation I/N.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
export SPLITS=train SUBSET=runpod/subset-v1.json ROTATE="$1/$2" MODELS=/root/work/models PY=python \
       DERIVED=acquisition/2026-10-17-v1/derived/subset-v1 SAPIENS_FLAGS="--no-flip --batch 24 --workers 16"
until bash runpod/pipeline.sh sapiens 0 1; do echo "retrying after failure $(date)"; sleep 30; done
echo SAPIENS-DONE
