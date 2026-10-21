#!/usr/bin/env bash
# recut_take5.sh I N GROUP LEDGERS SPLITS... — take-5 lateral keypoints re-cut from the raw recordings (audit B80) for
# sessions I mod N of SPLITS, into full-v1/how2sign-sapiens-side-recut-v1. A session record is written once from the
# frontal keypoints present at that moment, so this waits until the frontal keypoints of SPLITS (counted against the
# released frontal zips of those splits) have stopped growing at >= 99% of their clips. It then runs in passes:
# a pass recovers every session whose raw recordings (both cameras) are acquired and verified, and skips the others
# without writing them; passes repeat every 10 min until a pass has run after the raw acquisition of SPLITS completed
# (LEDGERS ledger files under how2sign-raw/ledgers/). Sapiens without flip, as the full-v1 keypoints. Writes
# done/<GROUP>-I-of-N when finished; post_full_tri.sh waits for these markers.
set -uo pipefail
I=$1 N=$2 GROUP=$3 LEDGERS=$4; shift 4; SPLITS_LIST="$*"
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
export DERIVED=acquisition/2026-10-17-v1/derived/full-v1 MODELS=/root/work/models PY=python
RECUT=$DERIVED/how2sign-sapiens-side-recut-v1
LOG=/root/recut-$GROUP-$I-of-$N.log PASS=/root/recut-$GROUP-$I-of-$N-pass.log   # per shard: two shards may share a pod
count() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; print(len(R2Store(os.environ).list(sys.argv[1])))" "$1"; }
ledgers() { python -c "
import os,sys; from signtranslator.data.text_motion_dataset import R2Store
ks=R2Store(os.environ).list('acquisition/2026-10-17-v1/how2sign-raw/ledgers/')
print(sum(any(f'raw-ledger-{s}-' in k for s in sys.argv[1:]) for k in ks))" $SPLITS_LIST; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
# Frontal keypoints of SPLITS only (the fleet goes on to other splits' frontal clips): names from the released
# frontal zips of these splits.
split_count() { python -c "
import io, os, sys, zipfile
from pathlib import Path
sys.path.insert(0, 'scripts')
from signtranslator.data.text_motion_dataset import R2Store
from extract_video_states import StoreRangeFile
s = R2Store(os.environ); names = set()
for split in sys.argv[1:]:
    z = zipfile.ZipFile(io.BufferedReader(StoreRangeFile(s.s3, s.bucket, f'acquisition/2026-10-17-v1/how2sign-public/{split}_rgb_front_clips.zip'), 1 << 20))
    names |= {Path(n).stem for n in z.namelist() if n.endswith('.mp4')}
done = {Path(k).stem for k in s.list('$DERIVED/how2sign-sapiens-front-v1/keypoints/')}
print(len(names & done), len(names))" $SPLITS_LIST; }
last=-1; still=0
while true; do
  read f total < <(split_count)
  [ "$f" -eq "$last" ] && still=$((still + 1)) || still=0; last=$f
  echo "$(date +%H:%M) frontal keypoints of $SPLITS_LIST: $f/$total (unchanged x$still)"
  ready=0
  # Complete when it has stopped growing for two checks at >= 99% of the split's clips (a few clips are refused).
  [ "$still" -ge 2 ] && [ $(( f * 100 )) -ge $(( total * 99 )) ] && ready=1
  [ $ready = 1 ] && break
  sleep 300
done
failed=0
for pass in $(seq 1 200); do
  l=$(ledgers); complete=0; [ "$l" -ge "$LEDGERS" ] && complete=1     # raw complete before this pass started
  echo "STAGE recut $GROUP $I/$N pass $pass (raw ledgers $l/$LEDGERS) $(date)"
  SPLITS="$SPLITS_LIST" RECUT_FLAGS="--no-flip --batch 24 --offset-workers ${OFFSET_WORKERS:-12}" bash runpod/pipeline.sh recut $I $N > $PASS 2>&1 \
    || { echo "FAILED recut $GROUP $I pass $pass" >> $PASS; failed=$((failed + 1)); }
  cat $PASS >> $LOG
  grep -a '"sessions"' $PASS | tail -1
  upload "$RECUT/logs/recut-$GROUP-$I-of-$N.log" $LOG
  if grep -aq "FAILED recut" $PASS; then [ $failed -ge 5 ] && break; sleep 120; continue; fi
  [ $complete = 1 ] && break
  sleep 600
done
grep -aq "FAILED recut" $PASS && echo "FAILED recut $GROUP $I" \
  || python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], b'done')" "$RECUT/done/$GROUP-$I-of-$N"
echo RECUT-DONE
