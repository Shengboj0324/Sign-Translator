#!/bin/bash
# tok_probes.sh — GPU evaluation pod: tokenizer probes B1, B2, B4, B5, B6 and the seam ablation (reports/post_s2_plan.md,
# sections 4 and 5; scripts/tok_probes.py). Report only: nothing trains, nothing is declared, the report decides nothing.
# Prepared, not run: it waits for an independent review.
#
#   bash runpod/fleet/tok_probes.sh            smoke (c02 train sentences with the longest clips, train words, a smoke fit), then D20
#   SMOKE=1 bash runpod/fleet/tok_probes.sh    the smoke only
#   PROCS=4                                    processes reading forms side by side
#
# The smoke stops the job when a decoder that returns real does not read real on real clips (`check`). On D20, real and blank are read
# first and must reproduce the pinned stored report (`canary`): a miss stops the job before any other form and voids the read. No automatic
# retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Outputs (write-once): $D/report.json, forms, fit report and codebooks; smokes under $D/smoke/<UTC>/.
# A failure uploads the work so far to $D/partial/<UTC>/, on exit and after each finished stage. A retry restores the latest such run, if it
# has this code and environment (`restore`), and reads only what is missing. A kill -9 keeps only the last finished stage.
# Ends with TOK-PROBES-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit
# status is the same.
# No output passes through a pipe; each background read's status is waited on one by one.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "TOK-PROBES-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/tok-probes
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/tok-probes}
DEV=${DEVICE:-cuda}
PROCS=${PROCS:-4}
T=scripts/tok_probes.py
S=$W/smoke-$TS
R=$W/d20
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"

# the work so far, keys already stored left alone; a failed upload never hides the job's own status
sync_partial() {
  python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || echo "ALERT partial upload failed"
}
# on any exit: stop what still runs, keep the work when the job failed, then the one DONE line, last
finish() {
  local status=$?
  trap - EXIT
  local p
  for p in $(jobs -p); do kill "$p" 2>/dev/null; done
  [ "$status" = 0 ] || sync_partial
  echo "TOK-PROBES-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }
mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers>=4.40,<5' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }

# real and blank alone (the first process writes the slim instrument; a declared read checks them), then the rest side by side,
# leaving out whatever is already stored
read_all() {
  local dir=$1; shift
  if [ ! -f "$dir/forms/form-real.json" ] || [ ! -f "$dir/forms/form-blank.json" ]; then
    python $T read --forms real,blank --out-dir "$dir/forms" --fit-dir "$dir/fit" $COMMON "$@" || return 1
  fi
  python $T canary --out-dir "$dir/forms" --void-prefix "$D/void"
  local canary=$?
  [ "$canary" != 3 ] || { echo "ALERT the canary failed: the read is void, do not retry"; done_ 3; }
  [ "$canary" = 0 ] || return 1
  local lists
  lists=$(python $T groups --processes "$PROCS" --skip real,blank --stored-in "$dir/forms") || return 1
  local pids=() n=0 group
  while IFS= read -r group; do
    [ -n "$group" ] || continue
    python $T read --forms "$group" --out-dir "$dir/forms" --fit-dir "$dir/fit" $COMMON "$@" > "$dir/read-$n.log" 2>&1 &
    pids+=("$!")
    n=$((n + 1))
  done <<< "$lists"
  [ "$n" -gt 0 ] || return 0                     # a retry with every form stored
  local rc=0 pid k=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "ALERT read-$k failed"
      tail -n 20 "$dir/read-$k.log"
      rc=1
    fi
    sync_partial
    k=$((k + 1))
  done
  return $rc
}

# ---- smoke: train clips only, the longest among them --------------------------------------------------------------------
python $T fit --out-dir "$S/fit" --smoke-train 16 --iterations 2 --device "$DEV" --threads "$THREADS" \
  || { echo "ALERT smoke fit failed"; done_ 2; }
read_all "$S" --smoke-train 8 --smoke-longest 4 || { echo "ALERT smoke read failed"; done_ 2; }
python $T words --out-dir "$S/forms" --fit-dir "$S/fit" --smoke-train 4 --device "$DEV" --threads "$THREADS" \
  || { echo "ALERT smoke words failed"; done_ 2; }
python $T merge --out-dir "$S/forms" --output-file "$S/report.json" || { echo "ALERT smoke merge failed"; done_ 2; }
python $T check --report "$S/report.json" || { echo "ALERT smoke check failed"; done_ 2; }
python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { echo "ALERT smoke upload failed"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  echo "SMOKE done"
  done_ 0
fi

# ---- the read: fit on c02 train, forms on D20, B6 on the gate halves ----------------------------------------------------
python $T restore --prefix "$D/partial" --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} \
  || { echo "ALERT restore refused"; done_ 1; }
if [ ! -f "$R/fit/fit.json" ]; then
  python $T fit --out-dir "$R/fit" --device "$DEV" --threads "$THREADS" || { echo "ALERT fit failed"; done_ 1; }
  sync_partial
fi
read_all "$R" || { echo "ALERT read failed"; done_ 1; }
python $T words --out-dir "$R/forms" --fit-dir "$R/fit" --resume --device "$DEV" --threads "$THREADS" || { echo "ALERT words failed"; done_ 1; }
python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { echo "ALERT merge failed"; done_ 1; }
python $T upload --out-dir "$R" --prefix "$D" || { echo "ALERT upload failed"; done_ 1; }
python $T check --report "$R/report.json" || { echo "ALERT check failed: the report is stored, read it as broken"; done_ 1; }
done_ 0
