#!/bin/bash
# path_parts.sh — GPU evaluation pod: which part of the monocular path loses v1's reading, and is it v1's fit or lost content (scripts/path_parts.py)? Report only: real forms only, nothing trains, and the report decides nothing.
#
#   bash runpod/fleet/path_parts.sh            the smoke (c02 train clips, one stored at another rate), then D20, on this pod
#   SMOKE=1 bash runpod/fleet/path_parts.sh    the smoke only
#
# One process at a time. The first step checks the instruments against their run records; no checkpoint is fetched for that. On D20, real_two_view and blank under v1 are read first and must reproduce the pinned stored report (`control`). A miss stops
# the job before any other form and voids the read, and `read` itself refuses another form without a passed control. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Outputs (write-once) under $D, the log under $D/logs/. A failure uploads the work so far to $D/partial/<UTC>/; rerunning on the same pod resumes what stands in $WORK. Use the L40S: the control compares hypotheses, and a miss voids the read.
# Ends with PATH-PARTS-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "PATH-PARTS-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/round20/path-read-parts
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/path-parts}
DEV=${DEVICE:-cuda}
T=scripts/path_parts.py
S=$W/smoke-$TS
R=$W/d20
LOG=$W/path_parts.log
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"
V1_REST=monocular_values_two_view_marks,mono_body_two_view_hands,two_view_body_mono_hands,mono_arms_only,mono_hand_pose_only,nearest_two_view,nearest_monocular
UNFITTED=uwb_blank,uwb_nearest_two_view,uwb_nearest_monocular

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
  echo "PATH-PARTS-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

step python $T instruments || { alert "the instruments are not as the header states"; done_ 1; }

# ---- smoke: c02 train clips through every form, on both instruments; no D20 state, no instrument on D20 ----------------------------------
step python $T read --out-dir "$S/forms" --smoke-train 5 --smoke-longest 4 --smoke-never-tracked 2 --smoke-other-rate 1 $COMMON || { alert "smoke read failed"; done_ 2; }
step python $T control --out-dir "$S/forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
step python $T merge --out-dir "$S/forms" --output-file "$S/report.json" || { alert "smoke merge failed"; done_ 2; }
step python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { alert "smoke upload failed"; done_ 2; }
step python $T check --report "$S/report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
step python $T read --forms real_two_view,blank --resume --out-dir "$R/forms" $COMMON || { alert "read of the control failed"; done_ 1; }
step python $T control --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
step python $T read --forms "$V1_REST" --resume --out-dir "$R/forms" $COMMON || { alert "read of v1's forms failed"; done_ 1; }
step python $T read --forms "$UNFITTED" --resume --out-dir "$R/forms" $COMMON || { alert "read of the unfitted instrument failed"; done_ 1; }
step python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
step python $T upload --out-dir "$R" --prefix "$D" || { alert "upload failed"; done_ 1; }
step python $T check --report "$R/report.json" || { alert "check failed: the report is stored, read it as broken"; done_ 1; }
done_ 0
