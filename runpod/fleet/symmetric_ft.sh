#!/bin/bash
# symmetric_ft.sh — one GPU pod: v1's parent fine-tuned on How2Sign's monocular states with v1's recipe, then read on D20 through both paths (scripts/symmetric_ft.py). Report only: the instrument gates nothing, and the report decides nothing.
#
#   bash runpod/fleet/symmetric_ft.sh            coverage, the smoke (8 c02 train clips, 10 steps), the shards, training, the read
#   SMOKE=1 bash runpod/fleet/symmetric_ft.sh    coverage and the smoke only
#
# C20's clips, which v1's lists hold, are refused by exact record: see the header. Under 90% of v1's training clips with monocular states the job says so and stops with 4, before any training. Shards are built once; a finished run (DONE) is not re-entered; checkpoints without DONE refuse unless ALLOW_RESUME=1, since a resumed trainer resets its patience, and the report says so.
# On D20, v1's real and blank are read first and must reproduce the pinned report (`control`). A miss stops the job and voids the read. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE.
# Outputs under $D (write-once): keypoints-mono/, checkpoints/, trained.json, read/, read/report.json, logs/. A failure uploads the work so far to $D/partial/<UTC>/. Use the L40S: the control compares hypotheses.
# Ends with SYM-FT-DONE <status> (0 done, 1 failed, 2 the smoke stopped the job, 3 the read is void: do not retry, 4 too few monocular clips). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "SYM-FT-DONE 1"; exit 1; }
set -uo pipefail
D=training-runs/back-translator-kp2d-uwb-ft-mono-v1
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/symmetric-ft}
DEV=${DEVICE:-cuda}
T=scripts/symmetric_ft.py
S=$W/smoke-$TS
R=$W/d20
LOG=$W/symmetric_ft.log
SMOKE_RUN=back-translator-kp2d-uwb-ft-mono-v1/smoke/$TS/run
SMOKE_INDEX=$D/smoke/$TS/keypoints-mono
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"

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
  echo "SYM-FT-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

step python $T recipe || { alert "v1's run record is not the header's recipe"; done_ 1; }
step python $T coverage --out-dir "$W"
covered=$?
[ "$covered" != 4 ] || { alert "too few of v1's training clips have monocular states: this part stops"; done_ 4; }
[ "$covered" = 0 ] || { alert "coverage did not run"; done_ 1; }

# ---- smoke: 8 c02 train clips through the shards, 10 steps of the trainer from the parent, and every form; no D20 state --------------------
step python $T build --smoke --prefix "$SMOKE_INDEX" --workers 4 || { alert "smoke build failed"; done_ 2; }
step python $T train --smoke --run "$SMOKE_RUN" --index-prefix "$SMOKE_INDEX" || { alert "smoke train failed"; done_ 2; }
step python $T trained --smoke --run "$SMOKE_RUN" --index-prefix "$SMOKE_INDEX" || { alert "smoke trained record failed"; done_ 2; }
step python $T val-control --run "$SMOKE_RUN" --index-prefix "$SMOKE_INDEX" --out-dir "$S/forms" --val-clips 2 --batch-size 2 --device "$DEV" --instrument-cache "$W/smoke-instrument" || { alert "smoke selection control failed"; done_ 2; }
step python $T read --forms control --out-dir "$S/forms" --run "$SMOKE_RUN" --smoke-train 5 --smoke-longest 2 --smoke-never-tracked 1 $COMMON || { alert "smoke read of the control failed"; done_ 2; }
step python $T control --out-dir "$S/forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
step python $T read --forms rest --out-dir "$S/forms" --run "$SMOKE_RUN" --smoke-train 5 --smoke-longest 2 --smoke-never-tracked 1 --device "$DEV" --threads "$THREADS" --instrument-cache "$W/smoke-instrument" --nll-group 4 || { alert "smoke read failed"; done_ 2; }
step python $T merge --out-dir "$S/forms" --run "$SMOKE_RUN" --output-file "$S/report.json" || { alert "smoke merge failed"; done_ 2; }
step python $T upload --out-dir "$S" --prefix "$D/smoke/$TS/read" || { alert "smoke upload failed"; done_ 2; }
step python $T check --report "$S/report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the instrument -------------------------------------------------------------------------------------------------------------------
step python $T build --workers "${BUILD_WORKERS:-12}"
built=$?
[ "$built" != 4 ] || { alert "too few of v1's training clips were built: this part stops"; done_ 4; }
[ "$built" = 0 ] || { alert "build failed"; done_ 1; }
step python $T train ${ALLOW_RESUME:+--allow-resume} || { alert "training failed"; done_ 1; }
step python $T trained || { alert "the run is not v1's recipe"; done_ 1; }

# ---- the read: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/read/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
[ -f "$R/forms/val_control.json" ] || step python $T val-control --out-dir "$R/forms" --device "$DEV" --instrument-cache "$W/instrument-mono" || { alert "the selection control did not run"; done_ 1; }
step python $T read --forms control --resume --out-dir "$R/forms" $COMMON || { alert "read of the control failed"; done_ 1; }
step python $T control --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
step python $T read --forms rest --resume --out-dir "$R/forms" --device "$DEV" --threads "$THREADS" --instrument-cache "$W/instrument-mono" --nll-group 4 || { alert "read of the new instrument failed"; done_ 1; }
step python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
step python $T upload --out-dir "$R" --prefix "$D/read" || { alert "upload failed"; done_ 1; }
step python $T check --report "$R/report.json" || { alert "check raised an alert: the report is stored, its label withheld"; done_ 1; }
done_ 0
