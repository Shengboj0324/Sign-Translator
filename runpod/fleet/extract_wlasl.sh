#!/usr/bin/env bash
# extract_wlasl.sh — WLASL v0.3 dictionary-site clips (C-UDA) -> canonical states: every per-host zip under
# wlasl-v0.3/zips/ through extract_video_states.py (ranged reads, write-once states, per-zip report; upright-chest
# prior, since dictionary framings vary). Resumable.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
command -v apt-get > /dev/null && { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 > /dev/null; }
P=acquisition/2026-10-17-v1/wlasl-v0.3/zips
for K in $(python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; print(' '.join(sorted(k for k in R2Store(os.environ).list('$P/') if k.endswith('.zip'))))"); do
  SHA=$(python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get(sys.argv[1]+'.provenance.json'))['sha256'])" $K)
  echo "STAGE extract $(basename $K) $(date)"
  for pass in $(seq 1 8); do
    python scripts/extract_video_states.py --zip-key $K --zip-sha256 $SHA --prefix acquisition/2026-10-17-v1/derived/wlasl-v0.3 \
      --source-id wlasl-v0.3 --models-dir /root/work/models --upright-chest --workers ${WORKERS:-12} 2>&1 | grep -v -E "^W0000|inference_feedback" | tail -2
    code=${PIPESTATUS[0]}; echo "PASS $pass exit $code $(date)"
    [ $code -eq 0 ] && break
    sleep 30
  done
done
echo WLASL-EXTRACT-DONE
