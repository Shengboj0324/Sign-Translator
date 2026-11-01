#!/bin/bash
# tok_v0.sh — GPU evaluation pod: V0, do real's validity marks carry content (scripts/tok_v0.py)? Report only: nothing trains, and the report decides nothing.
# Prepared, not run: it waits for an independent review.
#
#   bash runpod/fleet/tok_v0.sh            smoke (c02 train clips with the longest among them), then D20
#   SMOKE=1 bash runpod/fleet/tok_v0.sh    the smoke only
#
# One process. On D20, real and blank are read first and must reproduce the pinned stored report (`canary`): a miss stops the job before any other form and
# voids the read. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Outputs (write-once) under $D. A failure uploads the work so far to $D/partial/<UTC>/; a retry on another pod starts over, since the read is short.
# Ends with TOK-V0-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# No output passes through a pipe.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "TOK-V0-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/tok-probes-v0
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/tok-v0}
DEV=${DEVICE:-cuda}
T=scripts/tok_v0.py
S=$W/smoke-$TS
R=$W/d20
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"

finish() {
  local status=$?
  trap - EXIT
  if [ "$status" != 0 ]; then
    python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || echo "ALERT partial upload failed"
  fi
  echo "TOK-V0-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }
mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }   # the version of the stored D20 reads

# ---- smoke: train clips only -------------------------------------------------------------------------------------------
python $T read --out-dir "$S/forms" --smoke-train 8 --smoke-longest 4 $COMMON || { echo "ALERT smoke read failed"; done_ 2; }
python $T canary --out-dir "$S/forms" --void-prefix "$D/void" || { echo "ALERT smoke canary failed"; done_ 2; }
python $T merge --out-dir "$S/forms" --output-file "$S/report.json" || { echo "ALERT smoke merge failed"; done_ 2; }
python $T check --report "$S/report.json" || { echo "ALERT smoke check failed"; done_ 2; }
python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { echo "ALERT smoke upload failed"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------
python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { echo "ALERT preflight refused"; done_ 1; }
python $T read --forms real,blank --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of real and blank failed"; done_ 1; }
python $T canary --out-dir "$R/forms" --void-prefix "$D/void"
canary=$?
[ "$canary" != 3 ] || { echo "ALERT the canary failed: the read is void, do not retry"; done_ 3; }
[ "$canary" = 0 ] || { echo "ALERT the canary did not run"; done_ 1; }
python $T read --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read failed"; done_ 1; }
python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { echo "ALERT merge failed"; done_ 1; }
python $T upload --out-dir "$R" --prefix "$D" || { echo "ALERT upload failed"; done_ 1; }
python $T check --report "$R/report.json" || { echo "ALERT check failed: the report is stored, read it as broken"; done_ 1; }
done_ 0
