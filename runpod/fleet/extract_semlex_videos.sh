#!/usr/bin/env bash
# extract_semlex_videos.sh — Sem-Lex val and train videos (.webm, ~70k clips by Deaf signers) -> canonical states with
# our own extraction under the generator convention (scripts/extract_semlex_videos.py): states-video-v1/ and raw
# landmarks landmarks-video-v1/, write-once, one report per shard and pass. One CPU pod runs shard SHARD of SHARDS of
# each split in SPLITS (default "val train"). A pass that ends with exit 2 (store errors) or 1 (an interrupted archive
# stream, or a stopped pool) is run again: it resumes from the stored landmarks. Exit 3 or 4 stops the split for
# investigation. Progress lines every 500 clips. Research use only (CC BY-NC-SA 4.0).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
: "${SHARD:?SHARD is required}" "${SHARDS:?SHARDS is required}"
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$(( C > 4 ? C - 4 : 1 ))}   # one core for the streaming producer, the rest headroom; ~0.8 GB per worker
command -v apt-get > /dev/null && { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 > /dev/null; }
declare -A SHA=([val]=6eca70a5761f2bfeea5d4f58b1ed34431f2d1be39a20947f01a27e4a22516b90
                [train]=fa1890fb79312f50eb1d5f60ef71a3fd52cb57342eb43a54c82bd814d7778547)
for split in ${SPLITS:-val train}; do
  echo "STAGE extract $split shard $SHARD/$SHARDS workers $W $(date -u)"
  for pass in $(seq 1 ${PASSES:-4}); do
    python scripts/extract_semlex_videos.py --split $split --archive-sha256 ${SHA[$split]} --models-dir /root/work/models \
      --shard $SHARD --shards $SHARDS --workers $W 2>&1 | grep --line-buffered -v -E "^W0000|inference_feedback|^I0000|INFO: Created|^objc"
    code=${PIPESTATUS[0]}; echo "PASS $split $pass exit $code $(date -u)"
    [ $code -eq 1 ] || [ $code -eq 2 ] || break
    sleep 60
  done
  [ $code -eq 0 ] || echo "ALERT $split shard $SHARD exit $code: see the report under states-video-v1/reports/"
done
echo SEMLEX-VIDEO-DONE
