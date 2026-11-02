#!/bin/bash
# sensitivity_ladder.sh — GPU evaluation pod: how far each instrument's reading falls when real two-view D20 is damaged in known parts by known amounts (scripts/sensitivity_ladder.py). Report only: real forms only, nothing trains, and the report decides nothing.
#
#   bash runpod/fleet/sensitivity_ladder.sh               the smoke (c02 train clips), then D20 through v1 and the parent, on this pod
#   SMOKE=1 bash runpod/fleet/sensitivity_ladder.sh       the smoke only
#   SET=mono bash runpod/fleet/sensitivity_ladder.sh      later, once training-runs/back-translator-kp2d-uwb-ft-mono-v1/trained.json stands: the same rungs through the mono-tuned instrument, under round20/sensitivity-ladder-mono/
#
# One process. The first step reads the instruments' records; no checkpoint is fetched for that. On D20 the control's forms are read first and must reproduce their stored reads (`control`). A miss stops the job before any rung and voids the read, and `read` itself refuses another form without a passed control. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Outputs (write-once) under $D, the log under $D/logs/. A failure uploads the work so far to $D/partial/<UTC>/; rerunning on the same pod resumes what stands in $WORK. Use the L40S: the control compares hypotheses, and a miss voids the read.
# Ends with SENSITIVITY-LADDER-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "SENSITIVITY-LADDER-DONE 1"; exit 1; }
set -uo pipefail
SET=${SET:-base}
case $SET in base) D_NAME=sensitivity-ladder ;; mono) D_NAME=sensitivity-ladder-mono ;; *) echo "ALERT SET must be base or mono"; echo "SENSITIVITY-LADDER-DONE 1"; exit 1 ;; esac
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/round20/$D_NAME
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/sensitivity-ladder}
DEV=${DEVICE:-cuda}
T=scripts/sensitivity_ladder.py
S=$W/smoke-$TS
R=$W/d20
LOG=$W/ladder.log
COMMON="--set $SET --device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"

# a command's output to the log and the console; its status, or the log's failure, comes back
step() {
  "$@" 2>&1 | tee -a "$LOG"
  local codes=("${PIPESTATUS[@]}")
  [ "${codes[0]}" = 0 ] || return "${codes[0]}"
  [ "${codes[1]}" = 0 ] || { echo "ALERT the log could not be written (tee ${codes[1]})"; return 1; }
  return 0
}
alert() { step echo "ALERT $*" || true; }

finish() {
  local status=$?
  trap - EXIT
  if [ "$status" != 0 ]; then
    python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || alert "partial upload failed"
  fi
  python $T upload-log --file "$LOG" --key "$D/logs/$TS.log" || echo "ALERT log upload failed"
  echo "SENSITIVITY-LADDER-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

step python $T instruments --set "$SET" || { alert "the instruments are not as the header states"; done_ 1; }

# ---- smoke: c02 train clips through every rung; no D20 state ----------------------------------------------------------------------------
step python $T read --forms control --out-dir "$S/forms" --smoke-train 5 --smoke-longest 2 --smoke-never-tracked 1 $COMMON || { alert "smoke read of the control failed"; done_ 2; }
step python $T control --set "$SET" --out-dir "$S/forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
step python $T read --forms rest --out-dir "$S/forms" --smoke-train 5 --smoke-longest 2 --smoke-never-tracked 1 $COMMON || { alert "smoke read failed"; done_ 2; }
step python $T merge --set "$SET" --out-dir "$S/forms" --output-file "$S/report.json" || { alert "smoke merge failed"; done_ 2; }
step python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { alert "smoke upload failed"; done_ 2; }
step python $T check --report "$S/report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
step python $T read --forms control --resume --out-dir "$R/forms" $COMMON || { alert "read of the control failed"; done_ 1; }
step python $T control --set "$SET" --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
step python $T read --forms rest --resume --out-dir "$R/forms" $COMMON || { alert "read of the rungs failed"; done_ 1; }
step python $T merge --set "$SET" --out-dir "$R/forms" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
step python $T upload --out-dir "$R" --prefix "$D" || { alert "upload failed"; done_ 1; }
step python $T check --report "$R/report.json" || { alert "check raised an alert: the report is stored, its label withheld"; done_ 1; }
done_ 0
