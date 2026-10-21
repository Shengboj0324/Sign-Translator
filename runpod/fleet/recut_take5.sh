#!/usr/bin/env bash
# recut_take5.sh I N GROUP LEDGERS SPLITS... — take-5 lateral keypoints re-cut from the raw recordings (audit B80) for
# sessions I mod N of SPLITS, into full-v1/how2sign-sapiens-side-recut-v1. A session record is written once from the
# frontal keypoints present at that moment, so this waits until frontal keypoints have stopped growing (no frontal work
# left for these splits: every pod has moved on), and for val/test until frontal is complete. It then runs in passes:
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
count() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; print(len(R2Store(os.environ).list(sys.argv[1])))" "$1"; }
ledgers() { python -c "
import os,sys; from signtranslator.data.text_motion_dataset import R2Store
ks=R2Store(os.environ).list('acquisition/2026-10-17-v1/how2sign-raw/ledgers/')
print(sum(any(f'raw-ledger-{s}-' in k for s in sys.argv[1:]) for k in ks))" $SPLITS_LIST; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
last=-1; still=0
while true; do
  f=$(count "$DERIVED/how2sign-sapiens-front-v1/keypoints/")
  [ "$f" -eq "$last" ] && still=$((still + 1)) || still=0; last=$f
  echo "$(date +%H:%M) front $f (unchanged x$still)"
  ready=0
  if [ "$still" -ge 2 ]; then
    case " $SPLITS_LIST " in *" val "*|*" test "*) [ "$f" -ge 34900 ] && ready=1 ;; *) [ "$f" -ge 30000 ] && ready=1 ;; esac
  fi
  [ $ready = 1 ] && break
  sleep 300
done
failed=0
for pass in $(seq 1 200); do
  l=$(ledgers); complete=0; [ "$l" -ge "$LEDGERS" ] && complete=1     # raw complete before this pass started
  echo "STAGE recut $GROUP $I/$N pass $pass (raw ledgers $l/$LEDGERS) $(date)"
  SPLITS="$SPLITS_LIST" RECUT_FLAGS="--no-flip --batch 24" bash runpod/pipeline.sh recut $I $N > /root/recut-pass.log 2>&1 \
    || { echo "FAILED recut $GROUP $I pass $pass" >> /root/recut-pass.log; failed=$((failed + 1)); }
  cat /root/recut-pass.log >> /root/recut.log
  grep -a '"sessions"' /root/recut-pass.log | tail -1
  upload "$RECUT/logs/recut-$GROUP-$I-of-$N.log" /root/recut.log
  if grep -aq "FAILED recut" /root/recut-pass.log; then [ $failed -ge 5 ] && break; sleep 120; continue; fi
  [ $complete = 1 ] && break
  sleep 600
done
grep -aq "FAILED recut" /root/recut-pass.log && echo "FAILED recut $GROUP $I" \
  || python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], b'done')" "$RECUT/done/$GROUP-$I-of-$N"
echo RECUT-DONE
