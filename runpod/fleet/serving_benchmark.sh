#!/usr/bin/env bash
# serving_benchmark.sh RELEASE_JSON — plan item 6 on one GPU pod: the release's in-process service latency on 100
# How2Sign validation sentences, without and with the skeleton preview render; reports uploaded beside the release
# (releases/<name>.benchmark-gpu*.json under training-runs/serving/). BENCH-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
REL=$1; NAME=$(basename "$REL" .json)
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2"; }
M=acquisition/2026-10-17-v1/derived/full-v1/how2sign-text-manifest-v1/manifest.json; MSHA=$(sha $M)
nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/benchmark_service.py --release "$REL" --manifest-key $M --manifest-sha256 $MSHA --count 100 --device cuda \
  --output /root/$NAME.benchmark-gpu.json && upload training-runs/serving/$NAME.benchmark-gpu.json /root/$NAME.benchmark-gpu.json
python scripts/benchmark_service.py --release "$REL" --manifest-key $M --manifest-sha256 $MSHA --count 30 --device cuda --render \
  --output /root/$NAME.benchmark-gpu-render.json && upload training-runs/serving/$NAME.benchmark-gpu-render.json /root/$NAME.benchmark-gpu-render.json
echo BENCH-DONE
