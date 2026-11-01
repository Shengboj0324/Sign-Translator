#!/bin/bash
# tok_v1.sh — GPU evaluation pod: V1, does neutral fill limit generated signing (scripts/tok_v1.py)? Report only: nothing trains, and the report decides nothing. Prepared, not run: it waits for an independent review.
#
#   bash runpod/fleet/tok_v1.sh            smoke (c02 train clips; a composed export of 5 D20 sentences, read by no instrument), then D20
#   SMOKE=1 bash runpod/fleet/tok_v1.sh    the smoke only
#
# One process. On D20, real, blank, identity, round_trip and v0_nearest_fill are read first: real and blank must reproduce the pinned stored report (`canary`), the others their stored reads (`path-check`). A miss stops the
# job before any other form and voids the read. No automatic retry: this code never reads again, and a rerun needs new code and RERUN_CAUSE, which the report records.
# Composed states were never stored, so a9@predicted and a3-gen top 10 are composed again by `round15_compose_eval --export-only`, write-once under $D/composed/.
# Outputs (write-once) under $D. A failure uploads the work so far to $D/partial/<UTC>/; a retry on another pod starts over, except the stored exports.
# Ends with TOK-V1-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job, 3 the read is void: do not retry). The exit status is the same.
# No output passes through a pipe.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "TOK-V1-DONE 1"; exit 1; }
set -uo pipefail
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/tok-probes-v1
A=$O/round20/phase-a
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/tok-v1}
DEV=${DEVICE:-cuda}
T=scripts/tok_v1.py
S=$W/smoke-$TS
R=$W/d20
COMMON="--device $DEV --threads $THREADS --instrument-cache $W/instrument --nll-group 4"
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21
CACHE9=$A/cache-w14b-nogains-r10/index.json; CACHE9SHA=507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b
CACHE3=$O/round20/comp-a/cache-a3-top30-nogains/index.json; CACHE3SHA=2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c
TOKENS=reports/comp_a3_tokens.json; TOKENSSHA=c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776
DURATION=-0.083,0.3691,18289

finish() {
  local status=$?
  trap - EXIT
  if [ "$status" != 0 ]; then
    python $T upload --partial --out-dir "$W" --prefix "$D/partial/$TS" || echo "ALERT partial upload failed"
  fi
  echo "TOK-V1-DONE $status"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM INT HUP
done_() { exit "$1"; }
mkdir -p "$S/forms" "$R/forms" || done_ 1
uv pip install -q 'transformers==4.57.6' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }   # the version of the stored D20 reads

# composed export of one arm, write-once; nothing is read by an instrument: export PREFIX ARMNAME CLIPS ARGS...
export_arm() {
  local prefix=$1 arm=$2 clips=$3; shift 3
  python $T stored --key "$prefix/index.json" && return 0
  local list="$W/export-$arm-$clips.json" sha
  if [ "$clips" = 500 ]; then sha=$(python $T export-list --out "$list") || return 1; else sha=$(python $T export-list --out "$list" --first "$clips") || return 1; fi
  python scripts/round15_compose_eval.py run --mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips "$clips" --selection stage1 --timing sign-fitted \
    --expect-model-sha256 $MODEL --device "$DEV" --variants core_crossfade --word-cache-key $CACHE9 --word-cache-sha256 $CACHE9SHA --word-cache-n 16 --recognizer-run sign-recognizer-v2 \
    --predicted-duration --expect-duration-model=$DURATION --export-prefix "$prefix" --export-arm "$arm" --export-sentences-file "$list" --export-sentences-sha256 "$sha" --export-only \
    --export-forms "core_crossfade@predicted" "$@" --output-file "$W/export-$arm-$clips-out.json"
}
A9ARGS="--core-margin 0.1"
A3ARGS="--core-margin 0 --vocabulary-extra-file $TOKENS --vocabulary-extra-sha256 $TOKENSSHA --vocabulary-extra-top 10 --word-cache-extra-key $CACHE3 --word-cache-extra-sha256 $CACHE3SHA"

# ---- smoke: train clips for every read; 5 D20 sentences composed, read by no instrument ------------------------------------------------
python $T read --out-dir "$S/forms" --smoke-train 8 --smoke-longest 4 --smoke-never-tracked 2 $COMMON || { echo "ALERT smoke read failed"; done_ 2; }
python $T neutral --out-dir "$S/forms" --smoke-train 8 || { echo "ALERT smoke neutral failed"; done_ 2; }
python $T canary --out-dir "$S/forms" --void-prefix "$D/void" || { echo "ALERT smoke canary failed"; done_ 2; }
python $T path-check --out-dir "$S/forms" --void-prefix "$D/void" || { echo "ALERT smoke path check failed"; done_ 2; }
python $T composed-check --out-dir "$S/forms" || { echo "ALERT smoke composed check failed"; done_ 2; }
python $T merge --out-dir "$S/forms" --output-file "$S/report.json" || { echo "ALERT smoke merge failed"; done_ 2; }
python $T check --report "$S/report.json" || { echo "ALERT smoke check failed"; done_ 2; }
export_arm "$D/smoke/$TS/composed/a9" a9-smoke 5 $A9ARGS || { echo "ALERT smoke export a9 failed"; done_ 2; }
export_arm "$D/smoke/$TS/composed/a3" a3-smoke 5 $A3ARGS || { echo "ALERT smoke export a3 failed"; done_ 2; }
python $T check-export --key "$D/smoke/$TS/composed/a9/index.json" --arm a9 || { echo "ALERT smoke export a9 settings differ"; done_ 2; }
python $T check-export --key "$D/smoke/$TS/composed/a3/index.json" --arm a3 || { echo "ALERT smoke export a3 settings differ"; done_ 2; }
python $T upload --out-dir "$S" --prefix "$D/smoke/$TS" || { echo "ALERT smoke upload failed"; done_ 2; }
if [ "${SMOKE:-0}" = 1 ]; then
  echo "SMOKE done"
  done_ 0
fi

# ---- the read: D20 ------------------------------------------------------------------------------------------------------
python $T preflight --out-dir "$R" --final-key "$D/report.json" --void-prefix "$D/void" ${RERUN_CAUSE:+--rerun-cause "$RERUN_CAUSE"} || { echo "ALERT preflight refused"; done_ 1; }
python $T read --forms real,blank,identity,round_trip,v0_nearest_fill --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of the anchors failed"; done_ 1; }
python $T canary --out-dir "$R/forms" --void-prefix "$D/void"
canary=$?
[ "$canary" != 3 ] || { echo "ALERT the canary failed: the read is void, do not retry"; done_ 3; }
[ "$canary" = 0 ] || { echo "ALERT the canary did not run"; done_ 1; }
python $T path-check --out-dir "$R/forms" --void-prefix "$D/void"
path_check=$?
[ "$path_check" != 3 ] || { echo "ALERT the path check failed: the read is void, do not retry"; done_ 3; }
[ "$path_check" = 0 ] || { echo "ALERT the path check did not run"; done_ 1; }
python $T read --forms rt_nearest,no_quant_nearest,rt_post_fill --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of the cap failed"; done_ 1; }
export_arm "$D/composed/a9" a9 500 $A9ARGS || { echo "ALERT export a9 failed"; done_ 1; }
export_arm "$D/composed/a3-top10" a3-top10 500 $A3ARGS || { echo "ALERT export a3 failed"; done_ 1; }
python $T check-export --key "$D/composed/a9/index.json" --arm a9 || { echo "ALERT export a9 settings differ"; done_ 1; }
python $T check-export --key "$D/composed/a3-top10/index.json" --arm a3 || { echo "ALERT export a3 settings differ"; done_ 1; }
python $T neutral --out-dir "$R/forms" || { echo "ALERT neutral failed"; done_ 1; }
python $T read --forms a9p,a9p_replaced,a9p_control --resume --out-dir "$R/forms" $COMMON || { echo "ALERT read of the sensitivity failed"; done_ 1; }
python $T composed-check --out-dir "$R/forms" || { echo "ALERT composed check failed"; done_ 1; }
python $T merge --out-dir "$R/forms" --output-file "$R/report.json" || { echo "ALERT merge failed"; done_ 1; }
python $T upload --out-dir "$R" --prefix "$D" || { echo "ALERT upload failed"; done_ 1; }
python $T check --report "$R/report.json" || { echo "ALERT check failed: the report is stored, read it as broken"; done_ 1; }
done_ 0
