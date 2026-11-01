#!/bin/bash
# path_read.sh — GPU evaluation pod: does instrument v1 read real D20 through the monocular lifting path lower than through two-view (scripts/path_read.py)? Report only: real forms only, nothing trains, and the report decides nothing.
#
#   bash runpod/fleet/path_read.sh            the 5-clip smoke (c02 train clips; D20's lookup from index objects), then D20, on this pod
#   SMOKE=1 bash runpod/fleet/path_read.sh    the smoke only
#
# One process at a time. On D20, real_two_view and blank are read first and must reproduce the pinned stored report (`control`). A miss stops the job before any monocular form and voids the read. No automatic retry: this code never
# reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Outputs (write-once) under $D. A failure uploads the work so far to $D/partial/<UTC>/; a retry on another pod starts over.
# Ends with PATH-READ-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# Every step's output passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "PATH-READ-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/round20/path-read
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/path-read}
DEV=${DEVICE:-cuda}
T=scripts/path_read.py
S=$W/smoke-$TS
R=$W/d20
LOG=$W/path_read.log
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"

finish() {
  local status=$?
  trap - EXIT
  if [ "$status" != 0 ]; then
    python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || echo "ALERT partial upload failed"
  fi
  echo "PATH-READ-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

# a command's output to the log and the console; its status, or the log's failure, comes back
step() {
  "$@" 2>&1 | tee -a "$LOG"
  local codes=("${PIPESTATUS[@]}")
  [ "${codes[0]}" = 0 ] || return "${codes[0]}"
  [ "${codes[1]}" = 0 ] || { echo "ALERT the log could not be written (tee ${codes[1]})"; return 1; }
  return 0
}

mkdir -p "$S/forms" "$S/d20" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }   # the version of the stored D20 reads

# ---- smoke: 5 c02 train clips through every form; D20 only as a lookup in index objects, no D20 state, no instrument on D20 ----------
step python $T lookups --out-dir "$S/d20" || { echo "ALERT smoke D20 lookup failed"; done_ 2; }
step python $T lookups --out-dir "$S/forms" --smoke-train 5 || { echo "ALERT smoke lookup failed"; done_ 2; }
step python $T read --out-dir "$S/forms" --smoke-train 5 $COMMON || { echo "ALERT smoke read failed"; done_ 2; }
step python $T control --out-dir "$S/forms" --void-prefix "$D/void" || { echo "ALERT smoke control failed"; done_ 2; }
step python $T merge --out-dir "$S/forms" --output-file "$S/report.json" || { echo "ALERT smoke merge failed"; done_ 2; }
step python $T check --report "$S/report.json" || { echo "ALERT smoke check failed"; done_ 2; }
step python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { echo "ALERT smoke upload failed"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { echo "ALERT preflight refused"; done_ 1; }
step python $T lookups --out-dir "$R/forms" || { echo "ALERT lookup failed"; done_ 1; }
step python $T read --forms real_two_view,blank --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of the control failed"; done_ 1; }
step python $T control --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { echo "ALERT the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { echo "ALERT the control did not run"; done_ 1; }
step python $T read --forms real_monocular,real_monocular_unconformed,round_trip_monocular --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of the monocular forms failed"; done_ 1; }
step python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { echo "ALERT merge failed"; done_ 1; }
step python $T upload --out-dir "$R" --prefix "$D" || { echo "ALERT upload failed"; done_ 1; }
step python $T check --report "$R/report.json" || { echo "ALERT check failed: the report is stored, read it as broken"; done_ 1; }
done_ 0
