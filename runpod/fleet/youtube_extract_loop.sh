#!/usr/bin/env bash
# youtube_extract_loop.sh — CPU pod: extract the YouTube-ASL videos the owner's home connection stages in the private
# store (scripts/acquire_youtube_asl.py --stage extract; 30 workers sharded by video id), then delete each staged video.
# Passes repeat every 10 min while the home downloader runs; the loop ends when the downloader's finished marker exists
# and nothing is left staged, or after 6 consecutive empty passes (1 h with nothing to do: no idle pod). EXTRACT-DONE.
# ARM_RECOVERY=1: the arm-recovery re-extraction (acquire_youtube_asl.py --arm-recovery): staged videos, claims, the
# finished marker and every output live under $P/arm-recovery-v1/; the original corpus keys are never touched.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /root/work/Sign-Translator
DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 >/dev/null 2>&1
uv pip install -q yt-dlp && uv pip install --reinstall -q opencv-contrib-python-headless && python -c "import cv2" || { echo "deps failed"; echo EXTRACT-DONE; exit 1; }
P=acquisition/2026-10-17-v1/youtube-asl-v1
IDS=$P/youtube_asl_video_ids.txt; Q=$P; RECOVERY=()
if [ "${ARM_RECOVERY:-0}" = 1 ]; then IDS=$P/arm-recovery/ids-arm-untracked-v1.txt; Q=$P/arm-recovery-v1; RECOVERY=(--arm-recovery); fi
staged() { python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; print(sum(k.endswith('.json') for k in R2Store(os.environ).list('$Q/staged/')))"; }
finished() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists('$Q/home-download-finished.json') else 1)"; }
# 30 independent, unsharded workers. Each claims one staged video at a time (atomic claims: no video is taken twice;
# a random visiting order per run keeps workers from racing for the same one), converts it and re-lists at once. It
# rests 2 min only when nothing was free. Sharding by id left cores idle while one shard's videos queued behind a
# long video. A worker ends when the home downloader has finished and nothing is staged, or after an hour with
# nothing staged anywhere (no idle pod).
worker() { local i=$1 idle=0
  while true; do
    python scripts/acquire_youtube_asl.py --stage extract --ids $IDS --prefix $P ${RECOVERY[@]+"${RECOVERY[@]}"} \
      --models-dir /root/work/models --max-claims 1 --shuffle-seed $RANDOM$i >> /root/yt-x-$i.log 2>&1
    tail -1 /root/yt-x-$i.log | grep -qE '"done": \{\}|"error"' || { idle=0; continue; }   # a failed video waits
    n=$(staged)
    if [ "$n" = 0 ]; then
      finished && return 0
      idle=$((idle + 1)); [ $idle -ge 30 ] && return 0
    else
      idle=0
    fi
    sleep 120
  done; }
# live claims are refreshed every 20 min: a 5-8 h video must not look abandoned after the 3 h takeover age
( while true; do python scripts/refresh_claims.py --prefix $Q >> /root/claims.log 2>&1; sleep 1200; done ) &
HEARTBEAT=$!
for i in $(seq 0 29); do worker $i & done
while [ "$(jobs -rp | grep -vc "^$HEARTBEAT$")" != 0 ]; do
  sleep 600
  echo "$(date -u +%H:%M) staged $(staged), converted $(grep -h '"status": "converted"' /root/yt-x-*.log | wc -l), errors $(grep -h '"status": "error"' /root/yt-x-*.log | wc -l)"
done
kill $HEARTBEAT 2>/dev/null
echo EXTRACT-DONE
