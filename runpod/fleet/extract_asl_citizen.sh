#!/usr/bin/env bash
# extract_asl_citizen.sh SHARD SHARDS — monocular canonical states for ASL Citizen's 83,399 isolated-sign videos
# (MediaPipe landmarkers with ROI crops -> swing IK; extract_video_states.py), ranged reads of the verified zip, states
# write-once under derived/asl-citizen-v1/states, a per-shard report of every clip (converted or refused). Research use
# under the Microsoft Research Licence Terms; states stay private in R2.
set -uo pipefail
SHARD=$1 SHARDS=$2
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
KEY=acquisition/2026-10-17-v1/asl-citizen-v1/ASL_Citizen.zip
SHA=349c0e33f13b876f19ad4dee72205f4d2eae02937613e275200aa92283268eab
W=$(( $(nproc) * 3 / 4 ))
echo "STAGE extract shard $SHARD/$SHARDS workers $W $(date)"
python scripts/extract_video_states.py --zip-key $KEY --zip-sha256 $SHA --prefix acquisition/2026-10-17-v1/derived/asl-citizen-v1 \
  --source-id asl_citizen_video_estimate --models-dir /root/work/models --shard $SHARD --shards $SHARDS --workers $W \
  || echo "FAILED extract $SHARD"
echo EXTRACT-DONE
