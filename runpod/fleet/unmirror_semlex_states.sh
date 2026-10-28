#!/usr/bin/env bash
# unmirror_semlex_states.sh — round 14's handedness amendment (docs/NEXT_PHASE_PLAN.md, 2026-10-28 18:01 UTC): only if
# Sem-Lex is judged mirrored. One CPU pod (~32 vCPU):
#   1. re-solve every Sem-Lex val and train video state from its stored raw landmarks flipped horizontally
#      (scripts/unmirror_semlex_states.py; each checked against mirror_state of the stored state, write-once to
#      sem-lex-mirror-v1/states-video-v1-unmirrored/, one report per split and pass). Needs every extraction shard's
#      full-pass report (train shard 2 was still extracting on 2026-10-28): the job refuses until then.
#   2. verification, reported only: the dominance check on the re-solved states (scripts/handedness_check.py
#      --semlex-states-prefix), and v2 + the tokenizer pre-check on the same 300 val clips as the 14:23 UTC pre-check
#      (scripts/tokenizer_domain_read.py --states-prefix).
# A pass that ends with exit 2 (store errors) or 1 (watchdog / stopped pool) is run again: it resumes. Exit 3 stops the
# split for investigation. UNMIRROR-DONE <status> at the end. Research use only (Sem-Lex: CC BY-NC-SA 4.0).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
SL=acquisition/2026-10-17-v1/sem-lex-mirror-v1; UM=$SL/states-video-v1-unmirrored
R10=acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json@ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$(( C > 4 ? C - 4 : 1 ))}
OUT=/root/unmirror-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT
status=0
for split in ${SPLITS:-val train}; do
  echo "STAGE unmirror $split workers $W $(date -u)"
  for pass in $(seq 1 ${PASSES:-4}); do
    python scripts/unmirror_semlex_states.py --split $split --shard ${SHARD:-0} --shards ${SHARDS:-1} --workers $W 2>&1 \
      | grep --line-buffered -v -E "^W0000|^I0000"
    code=${PIPESTATUS[0]}; echo "PASS $split $pass exit $code $(date -u)"
    [ $code -eq 1 ] || [ $code -eq 2 ] || break
    sleep 60
  done
  [ $code -eq 0 ] || { echo "ALERT unmirror $split exit $code: see the report under $UM/reports/"; status=$code; }
done
[ $status = 0 ] || { echo "UNMIRROR-DONE $status"; exit $status; }

echo "STAGE verify-dominance $(date -u)"
python scripts/handedness_check.py --semlex --semlex-states-prefix $UM --manifest $R10 --per-signer 30 --max-units 60 \
  --workers $W --output $OUT/handedness-unmirrored.json > $OUT/verify-dominance.log 2>&1
echo "dominance exit $?"; tail -c 2500 $OUT/verify-dominance.log; echo
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; s=R2Store(os.environ); k=sys.argv[1]; sys.exit('exists: '+k) if s.exists(k) else s.put(k, open(sys.argv[2],'rb').read())" \
  $SL/adapter-dev/2026-10-28-handedness-unmirrored.json $OUT/handedness-unmirrored.json || true

echo "STAGE verify-v2-precheck $(date -u)"
python scripts/tokenizer_domain_read.py --states-prefix $UM --cache-dir $OUT/cache --output $OUT/precheck-unmirrored.json \
  --output-key $SL/adapter-dev/2026-10-28-round14-precheck-tokenizer-unmirrored.json > $OUT/verify-precheck.log 2>&1
echo "precheck exit $?"; tail -c 2500 $OUT/verify-precheck.log; echo
echo "UNMIRROR-DONE 0"
