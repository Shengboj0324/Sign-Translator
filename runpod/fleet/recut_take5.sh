#!/usr/bin/env bash
# recut_take5.sh I N GROUP LEDGERS SPLITS... — take-5 lateral keypoints re-cut from the raw recordings (audit B80) for
# sessions I mod N of SPLITS, into full-v1/how2sign-sapiens-side-recut-v1. A session record is written once from the
# frontal keypoints present at that moment, so this waits until (1) the raw acquisition of SPLITS is complete (LEDGERS
# ledger files under how2sign-raw/ledgers/ for those splits) and (2) frontal keypoints have stopped growing (no frontal
# work left for these splits: every pod has moved on), and for val/test until frontal is complete. Sapiens without flip,
# as the full-v1 keypoints. Writes done/<GROUP>-I-of-N when finished; post_full_tri.sh waits for these markers.
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
  l=$(ledgers); f=$(count "$DERIVED/how2sign-sapiens-front-v1/keypoints/")
  [ "$f" -eq "$last" ] && still=$((still + 1)) || still=0; last=$f
  echo "$(date +%H:%M) raw ledgers $l/$LEDGERS front $f (unchanged x$still)"
  ready=0
  if [ "$l" -ge "$LEDGERS" ] && [ "$still" -ge 2 ]; then
    case " $SPLITS_LIST " in *" val "*|*" test "*) [ "$f" -ge 34900 ] && ready=1 ;; *) [ "$f" -ge 30000 ] && ready=1 ;; esac
  fi
  [ $ready = 1 ] && break
  sleep 300
done
echo "STAGE recut $GROUP $I/$N $(date)"
SPLITS="$SPLITS_LIST" RECUT_FLAGS="--no-flip --batch 24" bash runpod/pipeline.sh recut $I $N > /root/recut.log 2>&1 \
  || echo "FAILED recut $GROUP $I" >> /root/recut.log
upload "$RECUT/logs/recut-$GROUP-$I-of-$N.log" /root/recut.log
grep -a '"sessions"\|FAILED' /root/recut.log
grep -aq "FAILED recut" /root/recut.log || python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], b'done')" "$RECUT/done/$GROUP-$I-of-$N"
echo RECUT-DONE
