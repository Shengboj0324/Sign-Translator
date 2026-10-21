#!/usr/bin/env bash
# post_full_tri.sh FIRST COUNT TOTAL — full How2Sign: wait until both views' keypoints are complete, then triangulate
# shards FIRST..FIRST+COUNT-1 of TOTAL in parallel (camera intrinsics reused from the subset's validated calibration:
# How2Sign's two cameras are the same rig). Logs per shard uploaded to R2. RECUT_DONE=K also waits for K completion
# markers of recut_take5.sh, so take-5 sessions triangulate from the re-cut lateral keypoints (audit B80).
set -uo pipefail
FIRST=$1 COUNT=$2 TOTAL=$3
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
cd /root/work/Sign-Translator
export SPLITS="train val test" MODELS=/root/work/models PY=python DERIVED=acquisition/2026-10-17-v1/derived/full-v1
count() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; print(len(R2Store(os.environ).list(sys.argv[1])))" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
last=-1; still=0
while true; do
  f=$(count "$DERIVED/how2sign-sapiens-front-v1/keypoints/"); s=$(count "$DERIVED/how2sign-sapiens-side-v1/keypoints/")
  r=$(count "$DERIVED/how2sign-sapiens-side-recut-v1/done/")
  t=$((f + s)); echo "$(date +%H:%M) front $f side $s recut markers $r/${RECUT_DONE:-0}"
  [ "$t" -eq "$last" ] && still=$((still + 1)) || still=0; last=$t
  if [ "$f" -ge 34900 ] && [ "$s" -ge 35000 ] && [ "$still" -ge 2 ] && [ "$r" -ge "${RECUT_DONE:-0}" ]; then break; fi
  sleep 300
done
echo "STAGE triangulate $FIRST..$((FIRST + COUNT - 1)) of $TOTAL $(date)"
for i in $(seq $FIRST $((FIRST + COUNT - 1))); do
  ( bash runpod/pipeline.sh triangulate $i $TOTAL > /root/tri-$i.log 2>&1 || echo "FAILED triangulate $i" >> /root/tri-$i.log
    upload "$DERIVED/how2sign-two-view-v1/logs/triangulate-$i-of-$TOTAL.log" /root/tri-$i.log ) &
done
wait
grep -h '"sessions"\|FAILED' /root/tri-*.log
echo TRI-DONE
