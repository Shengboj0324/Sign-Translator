#!/usr/bin/env bash
# extract_asl_citizen.sh SHARD SHARDS — monocular canonical states for ASL Citizen's 83,399 isolated-sign videos
# (MediaPipe landmarkers with ROI crops -> swing IK with the upright-chest prior, since webcam framing cuts the hips; extract_video_states.py), ranged reads of the verified zip, states
# write-once under derived/asl-citizen-v1/states, a per-shard report of every clip (converted or refused). Research use
# under the Microsoft Research Licence Terms; states stay private in R2.
set -uo pipefail
SHARD=$1 SHARDS=$2
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
KEY=acquisition/2026-10-17-v1/asl-citizen-v1/ASL_Citizen.zip
SHA=349c0e33f13b876f19ad4dee72205f4d2eae02937613e275200aa92283268eab
# nproc reports the host's cores (256); the pod's allotment is the cgroup CPU quota.
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1   # cgroup v2
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then W=$(( Q / P )); else W=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$W}
# MediaPipe's tasks runtime loads libEGL/libGLESv2 even on CPU; the pytorch image ships without them.
command -v apt-get > /dev/null && { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 > /dev/null; }
echo "STAGE extract shard $SHARD/$SHARDS workers $W $(date)"
# Exit 2: some clips met transient store/network errors (reported as "error"); the shard resumes, skipping
# converted clips, until none is left (at most four passes).
for pass in 1 2 3 4; do
  python scripts/extract_video_states.py --zip-key $KEY --zip-sha256 $SHA --prefix acquisition/2026-10-17-v1/derived/asl-citizen-v1 \
    --source-id asl_citizen_video_estimate --upright-chest --models-dir /root/work/models --shard $SHARD --shards $SHARDS --workers $W ${LIMIT:+--limit $LIMIT}
  code=$?; echo "PASS $pass exit $code $(date)"
  [ $code -eq 0 ] && break
  [ $pass -eq 4 ] && echo "FAILED extract $SHARD"
  sleep 60
done
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" \
  "acquisition/2026-10-17-v1/derived/asl-citizen-v1/logs/extract-shard-$SHARD-of-$SHARDS.log" /root/extract.log || true
echo EXTRACT-DONE
