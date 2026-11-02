#!/bin/bash
# parent_comp.sh — how much of real signing's clip-specific margin composed motion carries on v1's parent: the eight A5 word draws' top10 on D20 (scripts/parent_comp.py). Report only: nothing trains, and the report decides nothing.
#
#   bash runpod/fleet/parent_comp.sh                                  all eight draws on this pod: the smoke, the exports, the reads, the merge
#   SMOKE=1 bash runpod/fleet/parent_comp.sh                          the smoke only
#   POD=a PODS=a,b DRAWS="1 2 3 4" bash runpod/fleet/parent_comp.sh   two pods (the other POD=b DRAWS="5 6 7 8"); whichever ends last merges
#
# Exports are CPU only, EXPORT_PROCS at a time (default 4), and reused when stored. No instrument sees them until the anchors have reproduced the parts job's stored forms (`control`): a miss stops the job, voids the read and `read` refuses a composition without a passed control. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE.
# Outputs (write-once) under $D: composed/draw<K>/, pods/<POD>/forms/, report.json, logs/. A failure uploads the work so far to $D/partial/<UTC>/; rerunning on the same pod resumes what stands in $WORK. Use the L40S: the control compares hypotheses, and a miss voids the read.
# Ends with PARENT-COMP-DONE <status> (0 this pod's work done, 1 failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "PARENT-COMP-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/round20/parent-comp
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/parent-comp}
DEV=${DEVICE:-cuda}
T=scripts/parent_comp.py
POD=${POD:-a}
PODS=${PODS:-$POD}
DRAWS=${DRAWS:-1 2 3 4 5 6 7 8}
EXPORT_PROCS=${EXPORT_PROCS:-4}
S=$W/smoke-$TS
R=$W/d20
LOG=$W/parent_comp.log
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"
case $POD in *[!a-z0-9]*|"") echo "ALERT POD must be lower-case letters and digits"; echo "PARENT-COMP-DONE 1"; exit 1 ;; esac

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
  python $T upload-log --file "$LOG" --key "$D/logs/$POD-$TS.log" || echo "ALERT log upload failed"
  echo "PARENT-COMP-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

# every draw's export, at most EXPORT_PROCS at a time, one log file each; any failure fails the whole
export_all() {
  local pids=() k running=0 failed=0
  for k in $DRAWS; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python $T export --draw "$k" --work "$W/export" --prefix "$D/composed" > "$W/export-$k-$TS.log" 2>&1 &
    pids+=($!); running=$(( running + 1 ))
    if [ "$running" -ge "$EXPORT_PROCS" ]; then wait "${pids[0]}" || failed=1; pids=("${pids[@]:1}"); running=$(( running - 1 )); fi
  done
  for k in "${pids[@]:-}"; do [ -z "$k" ] || wait "$k" || failed=1; done
  return $failed
}

mkdir -p "$S/forms" "$R/forms" "$W/export" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

step python $T inputs || { alert "the draws' pins do not hold"; done_ 1; }

# ---- smoke: one draw's 5 D20 sentences composed, read against c02 train references; the parent sees no D20 text or real clip -------------
SMOKE_DRAW=${DRAWS%% *}
step python $T export --draw "$SMOKE_DRAW" --clips 5 --work "$S/export" --prefix "$D/smoke/$TS/composed" || { alert "smoke export failed"; done_ 2; }
step python $T read --forms control --out-dir "$S/forms" --smoke-train 5 $COMMON || { alert "smoke read of the anchors failed"; done_ 2; }
step python $T control --out-dir "$S/forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
step python $T fidelity --draw "$SMOKE_DRAW" --clips 5 --composed "$D/smoke/$TS/composed" --out-dir "$S/forms" --device "$DEV" --instrument-cache "$W/instrument" || { alert "smoke fidelity failed"; done_ 2; }
step python $T read --draws "$SMOKE_DRAW" --composed "$D/smoke/$TS/composed" --out-dir "$S/forms" --smoke-train 5 $COMMON || { alert "smoke read failed"; done_ 2; }
step python $T upload --out-dir "$S/forms" --prefix "$D/smoke/$TS/pods/$POD/forms" || { alert "smoke upload failed"; done_ 2; }
step python $T merge --pods "$POD" --prefix "$D/smoke/$TS/pods" --output-file "$S/report.json" || { alert "smoke merge failed"; done_ 2; }
step python $T put-report --file "$S/report.json" --key "$D/smoke/$TS/report.json" || { alert "smoke report upload failed"; done_ 2; }
step python $T check --report "$S/report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
export_all || { alert "an export failed (logs $W/export-*-$TS.log)"; done_ 1; }
step python $T read --forms control --resume --out-dir "$R/forms" $COMMON || { alert "read of the anchors failed"; done_ 1; }
step python $T control --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
for k in $DRAWS; do
  [ -f "$R/forms/fidelity-d$k.json" ] || step python $T fidelity --draw "$k" --composed "$D/composed" --out-dir "$R/forms" --device "$DEV" --instrument-cache "$W/instrument" || { alert "fidelity of draw $k failed"; done_ 1; }
  step python $T read --draws "$k" --composed "$D/composed" --resume --out-dir "$R/forms" $COMMON || { alert "read of draw $k failed"; done_ 1; }
done
step python $T upload --out-dir "$R/forms" --prefix "$D/pods/$POD/forms" || { alert "upload of this pod's forms failed"; done_ 1; }
step python $T ready --pods "$PODS" --prefix "$D/pods"
ready=$?
if [ "$ready" = 5 ]; then step echo "this pod's draws are stored; the merge waits for the others"; done_ 0; fi
[ "$ready" = 0 ] || { alert "ready did not run"; done_ 1; }
step python $T merge --pods "$PODS" --prefix "$D/pods" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
step python $T put-report --file "$R/report.json" --key "$D/report.json" || { alert "the final report was not stored"; done_ 1; }
step python $T check --report "$R/report.json" || { alert "check raised an alert: the report is stored"; done_ 1; }
done_ 0
