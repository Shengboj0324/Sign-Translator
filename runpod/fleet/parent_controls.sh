#!/bin/bash
# parent_controls.sh — controls on composition reading above real signing on v1's parent, read through the parent and the mirror: real word clips stitched in the same format (oracle), the generated clips of the oracle's words (covered), and the same forms in shuffled order (2 seeds each), on D20 (scripts/parent_controls.py). Report only: nothing trains, the mirror never moves M1, the reports decide nothing, and none of it shows human comprehension.
#
#   bash runpod/fleet/parent_controls.sh                              pins, both smokes, the compose, the parent's eight draws, the mirror's three, each merged
#   SMOKE=1 bash runpod/fleet/parent_controls.sh                      the smokes only
#   DRAWS="1 2 3 4 5 6" bash runpod/fleet/parent_controls.sh          fewer draws named; each report's stage says so. MIRROR_DRAWS (default "1 2 3") must be among them
#
# Request an NVIDIA L40S: the mirror's stored forms were read on one, a different type may void its read, and `inputs` refuses any GPU but EXPECT_GPU.
# Estimated, not measured: about 4 h and about $3.40 (range about $2.80 to $4.50), with 2 reads at a time at a guessed 1.7 times the speed; about 5.5 h and $4.70 without. DRAWS="1 2 3 4 5 6" is about $3. The measured cost goes beside this figure.
# The clock rule, blind to every score: once an instrument has started its minimum of draws (parent 4, mirror 2), the job starts no further draw of it if the time since the job began, plus the wall time since its reads began divided by the draws finished, would pass its limit (PARENT_BUDGET_SECONDS 10800; BUDGET_SECONDS 14400, 4 h, about $3.40, for the mirror), and merges the draws it holds. After a read fails no further draw starts.
# Composing is one process per draw, COMPOSE_PROCS at a time (default 4), each under COMPOSE_TIMEOUT seconds (default 5400), and reused when stored. It loads no instrument, and a rebuilt composition off its stored bytes stops that draw. Reads are READ_PROCS at a time (default 2), each draw under READ_TIMEOUT seconds (default 3600). No instrument sees a composed form until its anchors have passed: the parent's reproduce the parts job's stored forms (`control`) and the first draw's composition its stored form (`comp-check`), the mirror's reproduce its report's values and stored forms (`control`). A miss stops the job and voids that read. A parent void or a parent check alert stops the job before the mirror reads.
# Resuming: rerun on the same pod. A stage whose final key holds its local report's bytes is skipped and its check runs again, so a check alert stops the job again; standing forms are not read again; merge and put-report accept identical bytes; the smoke is not repeated for the same code; a stage's draws and stop reason stand from when its reads first finished, while the clock restarts. After a void, this code never reads again: a rerun needs new code and RERUN_CAUSE, and its work and outputs go under runs/<code digest>/.
# Outputs (write-once) under $D: composed/draw<K>/, forms/ and report.json (the parent), forms-mirror/ and report-mirror.json (the mirror), logs/. A failure uploads the work so far to $D/partial/<UTC>/, and the job's log, holding each compose and read log, to logs/.
# Ends with PARENT-CONTROLS-DONE <status> (0 this pod's work done, 1 failed, 2 a smoke stopped the job, 3 a read is void: do not retry this code). The exit status is the same.
# Every step's output and every ALERT passes through tee to $LOG, and `step` returns the failing side's status from PIPESTATUS, so no pipe hides a failure.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "PARENT-CONTROLS-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
BASE=$O/round20/parent-controls
D=$BASE
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/parent-controls}
DEV=${DEVICE:-cuda}
T=scripts/parent_controls.py
DRAWS=${DRAWS:-1 2 3 4 5 6 7 8}
MIRROR_DRAWS=${MIRROR_DRAWS:-1 2 3}
EXPECT_GPU=${EXPECT_GPU:-NVIDIA L40S}
COMPOSE_PROCS=${COMPOSE_PROCS:-4}
COMPOSE_TIMEOUT=${COMPOSE_TIMEOUT:-5400}
READ_PROCS=${READ_PROCS:-2}
READ_TIMEOUT=${READ_TIMEOUT:-3600}
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
  local status=$? kids f
  trap - EXIT TERM INT HUP
  kids=$(jobs -p)
  # shellcheck disable=SC2086
  [ -z "$kids" ] || kill $kids 2>/dev/null
  wait 2>/dev/null
  if [ "$status" != 0 ]; then
    for f in "$W"/compose-*-"$TS".log "$W"/read-*-"$TS".log; do
      [ ! -f "$f" ] || cat "$f" >> "$LOG"
    done
    python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || alert "partial upload failed"
  fi
  python $T upload-log --file "$LOG" --key "$BASE/logs/$TS.log" || echo "ALERT log upload failed"
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
  step cat "$W"/compose-*-"$TS".log || failed=1
  return $failed
}

# read_draws INSTRUMENT OUT DRAWS MIN LIMIT: each draw's forms, at most READ_PROCS at a time, one log file each, under the clock rule. READ, STOP and ELAPSED hold what it did. A failure lets running reads end and starts no further draw
read_draws() {
  local instrument=$1 out=$2 list=$3 min=$4 limit=$5
  local pids=() names=() k now begun started=0 finished=0 failed=0
  begun=$(date +%s)
  READ=""
  STOP=all_planned
  for k in $list; do
    now=$(date +%s)
    if [ "$started" -ge "$min" ] && [ "$finished" -gt 0 ] && [ $(( now - STARTED + (now - begun) / finished )) -gt "$limit" ]; then
      STOP=clock_rule
      step echo "the clock rule starts no more $instrument draws after${READ//,/ }" || failed=1
      break
    fi
    timeout -k 60 "$READ_TIMEOUT" python $T read --draws "$k" --instrument "$instrument" --composed "$D/composed" --resume --out-dir "$out" $COMMON > "$W/read-$instrument-$k-$TS.log" 2>&1 &
    pids+=($!); names+=("$k")
    READ="$READ,$k"
    started=$(( started + 1 ))
    if [ "${#pids[@]}" -ge "$READ_PROCS" ]; then
      wait "${pids[0]}" || { alert "read of $instrument draw ${names[0]} failed (log $W/read-$instrument-${names[0]}-$TS.log)"; failed=1; }
      pids=("${pids[@]:1}"); names=("${names[@]:1}"); finished=$(( finished + 1 ))
    fi
    [ "$failed" = 0 ] || break
  done
  while [ "${#pids[@]}" -gt 0 ]; do
    wait "${pids[0]}" || { alert "read of $instrument draw ${names[0]} failed (log $W/read-$instrument-${names[0]}-$TS.log)"; failed=1; }
    pids=("${pids[@]:1}"); names=("${names[@]:1}")
  done
  READ=${READ#,}
  ELAPSED=$(( $(date +%s) - STARTED ))
  step cat "$W"/read-"$instrument"-*-"$TS".log || failed=1
  return $failed
}

# reads_of INSTRUMENT OUT DRAWS MIN LIMIT: read_draws once, its result kept in a file: a rerun takes that file's draws, stop reason and seconds, so its merge writes the same bytes
reads_of() {
  local record="$R/stage-$1.txt"
  if [ -f "$record" ]; then
    read -r READ STOP ELAPSED < "$record"
    step echo "$1: the reads stand: draws $READ, $STOP" || return 1
    return 0
  fi
  read_draws "$@" || return 1
  echo "$READ $STOP $ELAPSED" > "$record" || return 1
}

for k in $MIRROR_DRAWS; do
  case " $DRAWS " in *" $k "*) ;; *) echo "ALERT MIRROR_DRAWS holds draw $k, which DRAWS does not"; done_ 1 ;; esac
done
mkdir -p "$W/compose" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { alert "transformers install failed"; done_ 1; }   # the version of the stored D20 reads

DIGEST=$(python $T digest) || { alert "this code has no digest"; done_ 1; }
case $DIGEST in *[!0-9a-f]*|"") alert "this code's digest reads $DIGEST"; done_ 1 ;; esac
if [ -n "${RERUN_CAUSE:-}" ]; then      # new code writes beside what the voided code left; the voids it answers stay in the common prefix
  D=$BASE/runs/$DIGEST
  R=$W/runs/$DIGEST/d20
fi
RM=$R/mirror/forms
mkdir -p "$S/forms" "$S/mirror/forms" "$R/forms" "$RM" || done_ 1

step python $T inputs --model-file "$MODEL" --expect-gpu "${EXPECT_GPU// /_}" || { alert "the pins do not hold"; done_ 1; }

# ---- smoke: one draw's 5 D20 sentences composed, read by each instrument against c02 train references; neither sees a D20 text or real clip -------------
SMOKED=$W/smoke-$DIGEST.done
if [ ! -f "$SMOKED" ]; then
  step timeout -k 60 "$COMPOSE_TIMEOUT" python $T compose --draw "$FIRST" --clips 5 --work "$S/compose" --prefix "$D/smoke/$TS/composed" --model-file "$MODEL" --device "$DEV" || { alert "smoke compose failed"; done_ 2; }
  for instrument in parent mirror; do
    forms=$S/forms; name=forms; report=report; beside=""
    [ "$instrument" = parent ] || { forms=$S/mirror/forms; name=forms-mirror; report=report-mirror; beside="--parent-forms $D/smoke/$TS/forms"; }
    step python $T read --forms control --instrument "$instrument" --out-dir "$forms" --smoke-train 5 $COMMON || { alert "smoke read of the $instrument anchors failed"; done_ 2; }
    step python $T control --instrument "$instrument" --out-dir "$forms" --void-prefix "$D/void" || { alert "smoke control failed"; done_ 2; }
    step python $T read --draws "$FIRST" --instrument "$instrument" --composed "$D/smoke/$TS/composed" --out-dir "$forms" --smoke-train 5 $COMMON || { alert "smoke read failed"; done_ 2; }
    step python $T upload --out-dir "$forms" --prefix "$D/smoke/$TS/$name" || { alert "smoke upload failed"; done_ 2; }
    step python $T merge --draws "$FIRST" --instrument "$instrument" --composed "$D/smoke/$TS/composed" --prefix "$D/smoke/$TS/$name" --output-file "$S/$report.json" $beside || { alert "smoke merge failed"; done_ 2; }
    step python $T put-report --file "$S/$report.json" --key "$D/smoke/$TS/$report.json" || { alert "smoke report upload failed"; done_ 2; }
    step python $T check --report "$S/$report.json" || { alert "smoke check failed: its report is stored"; done_ 2; }
  done
  touch "$SMOKED" || done_ 1
fi
if [ "${SMOKE:-0}" = 1 ]; then
  step echo "SMOKE done"
  done_ 0
fi

# ---- the reads: D20 ------------------------------------------------------------------------------------------------------------------
stage_state() {
  step python $T stage-state --key "$1" --file "$2"
}
stage_state "$D/report.json" "$R/report.json"
parent_state=$?
[ "$parent_state" = 0 ] || [ "$parent_state" = 5 ] || { alert "the parent's final report is not this run's"; done_ 1; }
stage_state "$D/report-mirror.json" "$R/report-mirror.json"
mirror_state=$?
[ "$mirror_state" = 0 ] || [ "$mirror_state" = 5 ] || { alert "the mirror's final report is not this run's"; done_ 1; }

if [ "$parent_state" = 5 ]; then
  step python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$BASE/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused"; done_ 1; }
fi
if [ "$mirror_state" = 5 ]; then
  step python $T preflight --out-dir "$R/mirror" --final-key "$D/report-mirror.json" --void-prefix "$BASE/void-mirror" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { alert "preflight refused the mirror"; done_ 1; }
fi
if [ "$parent_state" = 5 ] || [ "$mirror_state" = 5 ]; then
  compose_all || { alert "a compose failed (logs $W/compose-*-$TS.log)"; done_ 1; }
fi

# the parent
if [ "$parent_state" = 5 ]; then
  step python $T read --forms control --instrument parent --resume --out-dir "$R/forms" $COMMON || { alert "read of the anchors failed"; done_ 1; }
  step python $T control --instrument parent --out-dir "$R/forms" --void-prefix "$BASE/void"
  control=$?
  [ "$control" != 3 ] || { alert "the control failed: the read is void, do not retry this code; a rerun needs new code and RERUN_CAUSE"; done_ 3; }
  [ "$control" = 0 ] || { alert "the control did not run"; done_ 1; }
  step python $T read --forms comp-check --draws "$FIRST" --composed "$D/composed" --resume --out-dir "$R/forms" $COMMON || { alert "read of the stored composition failed"; done_ 1; }
  step python $T comp-check --draw "$FIRST" --out-dir "$R/forms" --void-prefix "$BASE/void"
  check=$?
  [ "$check" != 3 ] || { alert "the stored composition was not reproduced: the read is void, do not retry this code; a rerun needs new code and RERUN_CAUSE"; done_ 3; }
  [ "$check" = 0 ] || { alert "the composition check did not run"; done_ 1; }
fi
reads_of parent "$R/forms" "$DRAWS" 4 "$PARENT_BUDGET_SECONDS" || { alert "a read of the parent failed"; done_ 1; }
PARENT_READ=$READ
if [ "$parent_state" = 5 ]; then
  step python $T upload --out-dir "$R/forms" --prefix "$D/forms" || { alert "upload of the forms failed"; done_ 1; }
  step python $T merge --draws "$READ" --instrument parent --planned-draws "${DRAWS// /,}" --stop-reason "$STOP" --elapsed-seconds "$ELAPSED" --composed "$D/composed" --prefix "$D/forms" --output-file "$R/report.json" || { alert "merge failed"; done_ 1; }
  step python $T put-report --file "$R/report.json" --key "$D/report.json" || { alert "the final report was not stored"; done_ 1; }
fi
step python $T check --report "$R/report.json" || { alert "check raised an alert: the report is stored, and the mirror does not read"; done_ 1; }

# the mirror: report only, it never moves M1; only draws the parent read, so the two stand side by side
if [ "$mirror_state" = 5 ]; then
  step python $T read --forms control --instrument mirror --resume --out-dir "$RM" $COMMON || { alert "read of the mirror's anchors failed"; done_ 1; }
  step python $T control --instrument mirror --out-dir "$RM" --void-prefix "$BASE/void-mirror"
  control=$?
  [ "$control" != 3 ] || { alert "the mirror's control failed: its read is void, do not retry this code; a rerun needs new code and RERUN_CAUSE"; done_ 3; }
  [ "$control" = 0 ] || { alert "the mirror's control did not run"; done_ 1; }
fi
SHARED=""
for k in $MIRROR_DRAWS; do
  case ",$PARENT_READ," in *",$k,"*) SHARED="$SHARED $k" ;; esac
done
SHARED=${SHARED# }
[ -n "$SHARED" ] || { alert "the parent read none of MIRROR_DRAWS"; done_ 1; }
reads_of mirror "$RM" "$SHARED" 2 "$BUDGET_SECONDS" || { alert "a read of the mirror failed"; done_ 1; }
if [ "$mirror_state" = 5 ]; then
  step python $T upload --out-dir "$RM" --prefix "$D/forms-mirror" || { alert "upload of the mirror's forms failed"; done_ 1; }
  step python $T merge --draws "$READ" --instrument mirror --planned-draws "${MIRROR_DRAWS// /,}" --stop-reason "$STOP" --elapsed-seconds "$ELAPSED" --composed "$D/composed" --prefix "$D/forms-mirror" --parent-forms "$D/forms" --output-file "$R/report-mirror.json" || { alert "the mirror's merge failed"; done_ 1; }
  step python $T put-report --file "$R/report-mirror.json" --key "$D/report-mirror.json" || { alert "the mirror's final report was not stored"; done_ 1; }
fi
step python $T check --report "$R/report-mirror.json" || { alert "check raised an alert on the mirror: its report is stored"; done_ 1; }
done_ 0
