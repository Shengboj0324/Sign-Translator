#!/usr/bin/env bash
# extract_rit_citation.sh — round 5 data: the RIT citation-form sign clips of DAI 2 (2 zips, 12,197 annotated sign
# tokens, never extracted before) -> canonical states, like the ASLLVD sign clips (extract_video_states.py, studio
# framing, no upright-chest prior). Ranged reads of the verified zips; states write-once under
# derived/asllrp-dai2-v1/rit-citation-sign-videos. Research use only (ASLLRP Sign Bank Data Terms of Use).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then W=$(( Q / P )); else W=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$(( W < 12 ? W : 12 ))}   # 32 MediaPipe workers ran out of memory (BrokenProcessPool, 2026-10-26 21:07)
command -v apt-get > /dev/null && { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 > /dev/null; }
P=acquisition/2026-10-17-v1/asllrp-dai2-v1/rit-citation
for K in $P/rit-sign-videos__batch_signs_v1_1.zip $P/rit-sign-videos__batch_signs_v1_2.zip; do
  SHA=$(python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get(sys.argv[1]+'.provenance.json'))['sha256'])" $K)
  echo "STAGE extract $(basename $K) workers $W $(date)"
  for pass in $(seq 1 12); do          # each pass resumes; a pass that ends clean (exit 0) means every clip is done
    python scripts/extract_video_states.py --zip-key $K --zip-sha256 $SHA \
      --prefix acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/rit-citation-sign-videos --source-id rit-citation-sign-videos \
      --models-dir /root/work/models --workers $W ${LIMIT:+--limit $LIMIT} 2>&1 | grep -v -E "^W0000|inference_feedback" | tail -3
    code=${PIPESTATUS[0]}; echo "PASS $pass exit $code $(date)"
    [ $code -eq 0 ] && break
    sleep 60
  done
done
echo RIT-EXTRACT-DONE
