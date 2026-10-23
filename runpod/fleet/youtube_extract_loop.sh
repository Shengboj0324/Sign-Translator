#!/usr/bin/env bash
# youtube_extract_loop.sh — CPU pod: extract the YouTube-ASL videos the owner's home connection stages in the private
# store (scripts/acquire_youtube_asl.py --stage extract; 30 workers sharded by video id), then delete each staged video.
# Passes repeat every 10 min while the home downloader runs; the loop ends when the downloader's finished marker exists
# and nothing is left staged, or after 6 consecutive empty passes (1 h with nothing to do: no idle pod). EXTRACT-DONE.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /root/work/Sign-Translator
DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 >/dev/null 2>&1
uv pip install -q yt-dlp && uv pip install --reinstall -q opencv-contrib-python-headless && python -c "import cv2" || { echo "deps failed"; echo EXTRACT-DONE; exit 1; }
P=acquisition/2026-10-17-v1/youtube-asl-v1
staged() { python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; print(sum(k.endswith('.json') for k in R2Store(os.environ).list('$P/staged/')))"; }
finished() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists('$P/home-download-finished.json') else 1)"; }
empty=0; pass=0
while true; do
  n=$(staged); pass=$((pass + 1)); echo "PASS $pass: $n staged $(date -u +%H:%M)"
  if [ "$n" = 0 ]; then
    finished && break
    empty=$((empty + 1)); [ $empty -ge 6 ] && { echo "nothing staged for an hour"; break; }
    sleep 600; continue
  fi
  empty=0
  for i in $(seq 0 29); do
    python scripts/acquire_youtube_asl.py --stage extract --ids $P/youtube_asl_video_ids.txt --prefix $P \
      --models-dir /root/work/models --shard $i --shards 30 >> /root/yt-x-$i.log 2>&1 &
  done
  wait
  echo "converted so far: $(grep -h '"status": "converted"' /root/yt-x-*.log | wc -l)"
  sleep 60
done
echo EXTRACT-DONE
