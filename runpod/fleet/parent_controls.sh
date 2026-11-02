#!/bin/bash
# parent_controls.sh — why composition reads above real signing on v1's parent, and what the mirror makes of the same forms: real word clips stitched in the same format (oracle), the same generated clips in shuffled order (shuffled, 3 seeds), both together (oracle_shuffled), and the generated clips of the oracle's words (covered), on D20 (scripts/parent_controls.py). Report only: nothing trains, the mirror never moves M1, the reports decide nothing, and none of it shows human comprehension.
#
#   bash runpod/fleet/parent_controls.sh                              pins, both smokes, the compose, the parent's eight draws, the mirror's three, each merged
#   SMOKE=1 bash runpod/fleet/parent_controls.sh                      the smokes only
#   DRAWS="1 2 3 4 5 6" bash runpod/fleet/parent_controls.sh          fewer draws named; each report's draws say so. MIRROR_DRAWS (default "1 2 3") must be among them
#
# Estimated, not measured: about 3.8 h and about $3.25 on one RTX 6000 Ada or L40S (range about $2.70 to $4.30), with 2 reads at a time at a guessed 1.7 times the speed; about 5.3 h and $4.50 without. DRAWS="1 2 3 4 5 6" is about $2.85. The measured cost goes beside this figure.
# The clock rule, blind to every score: before starting each draw past the minimum (parent 4, mirror 2), the job starts no more of that instrument's draws if the elapsed time plus one more draw's measured wall time would pass its limit (PARENT_BUDGET_SECONDS 10800; BUDGET_SECONDS 14400, 4 h, about $3.40, for the mirror), and merges the draws it holds.
# Composing is one process per draw, COMPOSE_PROCS at a time (default 4), each under COMPOSE_TIMEOUT seconds (default 5400), and reused when stored. It loads no instrument, and a rebuilt composition off its stored bytes stops that draw. No instrument sees a composed form until its anchors have passed: the parent's reproduce the parts job's stored forms (`control`) and draw one's composition its stored form (`comp-check`), the mirror's reproduce its report's values and stored forms (`control`). A miss stops the job, voids that instrument's read and `read` refuses. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE.
# Outputs (write-once) under $D: composed/draw<K>/, forms/ and report.json (the parent), forms-mirror/ and report-mirror.json (the mirror), logs/. A failure uploads the work so far to $D/partial/<UTC>/; rerunning on the same pod resumes what stands in $WORK. Use the GPU of the stored read (RTX 6000 Ada) or the L40S: the checks compare hypotheses, and a miss voids the read.
# Ends with PARENT-CONTROLS-DONE <status> (0 this pod's work done, 1 failed, 2 a smoke stopped the job, 3 a read is void: do not retry). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "PARENT-CONTROLS-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/round20/parent-controls
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/parent-controls}
DEV=${DEVICE:-cuda}
T=scripts/parent_controls.py
DRAWS=${DRAWS:-1 2 3 4 5 6 7 8}
MIRROR_DRAWS=${MIRROR_DRAWS:-1 2 3}
COMPOSE_PROCS=${COMPOSE_PROCS:-4}
COMPOSE_TIMEOUT=${COMPOSE_TIMEOUT:-5400}
READ_PROCS=${READ_PROCS:-2}
PARENT_BUDGET_SECONDS=${PARENT_BUDGET_SECONDS:-10800}
BUDGET_SECONDS=${BUDGET_SECONDS:-14400}
STARTED=$(date +%s)
S=$W/smoke-$TS
R=$W/d20
LOG=$W/parent_controls.log
MODEL=$W/model.json
COMMON="--threads $THREADS --device $DEV --instrument-cache $W/instrument --nll-group 4"
FIRST=${DRAWS%% *}

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
  echo "PARENT-CONTROLS-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }

# every draw composed, at most COMPOSE_PROCS at a time, one log file each; any failure fails the whole
compose_all() {
  local pids=() k running=0 failed=0
  for k in $DRAWS; do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout -k 60 "$COMPOSE_TIMEOUT" python $T compose --draw "$k" --work "$W/compose" --prefix "$D/composed" --model-file "$MODEL" --device "$DEV" > "$W/compose-$k-$TS.log" 2>&1 &
    pids+=($!); running=$(( running + 1 ))
    if [ "$running" -ge "$COMPOSE_PROCS" ]; then wait "${pids[0]}" || failed=1; pids=("${pids[@]:1}"); running=$(( running - 1 )); fi
  done
  for k in "${pids[@]:-}"; do [ -z "$k" ] || wait "$k" || failed=1; done
  return $failed
}

# read_draws INSTRUMENT OUT DRAWS MIN LIMIT: each draw's forms, at most READ_PROCS at a time, one log file each, under the clock rule; READ holds the draws started. Any failure fails the whole, naming its draw
read_draws() {
  local instrument=$1 out=$2 list=$3 min=$4 limit=$5
  local pids=() names=() k now finished=0 failed=0 begun
  begun=$(date +%s)
  READ=""
  for k in $list; do
    now=$(date +%s)
    if [ "$finished" -ge "$min" ] && [ $(( now - STARTED + (now - begun) / finished )) -gt "$limit" ]; then
      step echo "the clock rule starts no more $instrument draws after${READ//,/ }" || return 1
      break
    fi
    python $T read --draws "$k" --instrument "$instrument" --composed "$D/composed" --resume --out-dir "$out" $COMMON > "$W/read-$instrument-$k-$TS.log" 2>&1 &
    pids+=($!); names+=("$k")
    READ="$READ,$k"
    if [ "${#pids[@]}" -ge "$READ_PROCS" ]; then
      wait "${pids[0]}" || { alert "read of $instrument draw ${names[0]} failed (log $W/read-$instrument-${names[0]}-$TS.log)"; failed=1; }
      pids=("${pids[@]:1}"); names=("${names[@]:1}"); finished=$(( finished + 1 ))
    fi
  done
  while [ "${#pids[@]}" -gt 0 ]; do
    wait "${pids[0]}" || { alert "read of $instrument draw ${names[0]} failed (log $W/read-$instrument-${names[0]}-$TS.log)"; failed=1; }
    pids=("${pids[@]:1}"); names=("${names[@]:1}")
  done
  READ=${READ#,}
  step cat "$W"/read-"$instrument"-*-"$TS".log || failed=1
  return $failed
}

mkdir -p "$S/forms" "$S/forms-mirror" "$R/forms" "$R/forms-mirror" "$W/compose" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

step python $T inputs --model-file "$MODEL" || { alert "the pins do not hold"; done_ 1; }

# ---- smoke: one draw's 5 D20 sentences composed, read by each instrument against c02 train references; neither sees a D20 text or real clip -------------
step timeout -k 60 "$COMPOSE_TIMEOUT" python $T compose --draw "$FIRST" --clips 5 --work "$S/compose" --prefix "$D/smoke/$TS/composed" --model-file "$MODEL" --device "$DEV" || { alert "smoke compose failed"; done_ 2; }
for instrument in parent mirror; do
  forms=$S/forms; name=forms; report=report
  [ "$instrument" = parent ] || { forms=$S/forms-mirror; name=forms-mirror; report=report-mirror; }
  step python $T read --forms control --instrument "$instrument" --out-dir "$forms" --smoke-train 5 $COMMON || { alert "smoke read of the $instrument anchors failed"; done_ 2; }
  step python $T control --instrument "$instrument" --out-dir "$forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
  step python $T read --draws "$FIRST" --instrument "$instrument" --composed "$D/smoke/$TS/composed" --out-dir "$forms" --smoke-train 5 $COMMON || { alert "smoke read failed"; done_ 2; }
  step python $T upload --out-dir "$forms" --prefix "$D/smoke/$TS/$name" || { alert "smoke upload failed"; done_ 2; }
  step python $T merge --draws "$FIRST" --instrument "$instrument" --composed "$D/smoke/$TS/composed" --prefix "$D/smoke/$TS/$name" --output-file "$S/$report.json" || { alert "smoke merge failed"; done_ 2; }
  step python $T put-report --file "$S/$report.json" --key "$D/smoke/$TS/$report.json" || { alert "smoke report upload failed"; done_ 2; }
  step python $T check --report "$S/$report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
done
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the reads: D20 ------------------------------------------------------------------------------------------------------------------
step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
step python $T preflight --out-dir "$R/mirror" --final-key "$D/report-mirror.json" --void-prefix "$D/void-mirror" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused the mirror"; done_ 1; }
compose_all || { alert "a compose failed (logs $W/compose-*-$TS.log)"; done_ 1; }

# the parent
step python $T read --forms control --instrument parent --resume --out-dir "$R/forms" $COMMON || { alert "read of the anchors failed"; done_ 1; }
step python $T control --instrument parent --out-dir "$R/forms" --void-prefix "$D/void"
control=$?
[ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
step python $T read --forms comp-check --draws "$FIRST" --composed "$D/composed" --resume --out-dir "$R/forms" $COMMON || { alert "read of the stored composition failed"; done_ 1; }
step python $T comp-check --draw "$FIRST" --out-dir "$R/forms" --void-prefix "$D/void"
check=$?
[ "$check" != 3 ] || { alert "the stored composition was not reproduced: the read is void, do not retry"; done_ 3; }
[ "$check" = 0 ] || { alert "the composition check did not run"; done_ 1; }
read_draws parent "$R/forms" "$DRAWS" 4 "$PARENT_BUDGET_SECONDS" || { alert "a read of the parent failed"; done_ 1; }
step python $T upload --out-dir "$R/forms" --prefix "$D/forms" || { alert "upload of the forms failed"; done_ 1; }
step python $T merge --draws "$READ" --instrument parent --composed "$D/composed" --prefix "$D/forms" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
step python $T put-report --file "$R/report.json" --key "$D/report.json" || { alert "the final report was not stored"; done_ 1; }
step python $T check --report "$R/report.json" || { alert "check raised an alert: the report is stored"; done_ 1; }

# the mirror: report only, it never moves M1
step python $T read --forms control --instrument mirror --resume --out-dir "$R/forms-mirror" $COMMON || { alert "read of the mirror's anchors failed"; done_ 1; }
step python $T control --instrument mirror --out-dir "$R/forms-mirror" --void-prefix "$D/void-mirror"
control=$?
[ "$control" != 3 ] || { alert "the mirror's control failed: its read is void, do not retry"; done_ 3; }
[ "$control" = 0 ] || { alert "the mirror's control did not run"; done_ 1; }
read_draws mirror "$R/forms-mirror" "$MIRROR_DRAWS" 2 "$BUDGET_SECONDS" || { alert "a read of the mirror failed"; done_ 1; }
step python $T upload --out-dir "$R/forms-mirror" --prefix "$D/forms-mirror" || { alert "upload of the mirror's forms failed"; done_ 1; }
step python $T merge --draws "$READ" --instrument mirror --composed "$D/composed" --prefix "$D/forms-mirror" --output-file "$R/report-mirror.json" || { alert "the mirror's merge failed"; done_ 1; }
step python $T put-report --file "$R/report-mirror.json" --key "$D/report-mirror.json" || { alert "the mirror's final report was not stored"; done_ 1; }
step python $T check --report "$R/report-mirror.json" || { alert "check raised an alert on the mirror: its report is stored"; done_ 1; }
done_ 0
