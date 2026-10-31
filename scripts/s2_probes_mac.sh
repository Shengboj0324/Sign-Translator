#!/bin/bash
# s2_probes_mac.sh — stage S2's probes on the Mac, within its limits (at most 6 processes, 1 BLAS thread, nice, each phase
# under 30 minutes, disk under 5 GB). Bars: docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design received". Probes run after
# review. Each phase reads R2 only and writes under $WORK, write-once; nothing goes to R2. Source .env.r2.local first.
#
#   WORK=~/s2-probes bash scripts/s2_probes_mac.sh PHASE [--smoke N]
#
# Times are extrapolated from 8-clip train smokes and timed steps (2026-10-31), not measured on D20; peak memory is per process.
#   p1a  c02 codes (8,000), stop and D20 codes, flan-T5 tokens; word control's   about 18 min, 2.7 GB, disk 0.8 GB
#   p1b  fit the head (6,000 steps at most) and the control's head               about 15 min, 1.6 GB
#   p1c  gain, ranks, samples, the control's read, AR priors on the same clips   about 10 min, 3.6 GB
#   p1v  P1's verdict                                                            seconds
#   p2   11 forms in 5 processes, after `real` alone                             about 15 min, 10 GB then 4.3 GB each, disk 1.2 GB
#   p3   lengths, then 3 forms                                                   about 6 min, as p2
#   p4   6 forms in 5 processes                                                  about 9 min, as p2
# A pod does better for: B0 (runpod/fleet/s2_b0.sh, GPU only: about 25 CPU-s per clip and form, about 14 h here); P1's one
# confirmation of a fail (all c02 clips, about 1 GPU-h, `--device cuda --threads 8 --scale pod`); PMI ranks (`--candidates 100`).
set -euo pipefail
PHASE=${1:?phase}
SMOKE=()
FIT=8000; CONTROL=3000; STEPS=(--steps 6000); CONTROL_STEPS=(--steps 3000)
if [ "${2:-}" = "--smoke" ]; then
  SMOKE=(--smoke-train "$3"); FIT=64; CONTROL=64; STEPS=(--steps 20 --every 10); CONTROL_STEPS=(--steps 20 --every 10)
fi
: "${WORK:?set WORK}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="nice ${PYTHON:-python}"
P1=scripts/s2_text_info_probe.py
CACHE=$WORK/instrument
FLANT5=flant5=training-runs/tokens-rvq-d0b-prior-r4-flant5/checkpoints/step-00023000.pt
W14B=w14b=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt@0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
pids=()
bg() { "$@" & pids+=($!); }
join() { for p in "${pids[@]}"; do wait "$p"; done; pids=(); }

# P2, P3 and P4: the first form alone writes the slim instrument; then at most 5 processes, one list of forms each.
forms() {
  local script=$1 dir=$2 first=$3; shift 3
  $PY "$script" read --out-dir "$dir" --forms "$first" --instrument-cache "$CACHE" "${SMOKE[@]+"${SMOKE[@]}"}"
  for group in "$@"; do
    bg $PY "$script" read --out-dir "$dir" --forms "$group" --instrument-cache "$CACHE" "${SMOKE[@]+"${SMOKE[@]}"}"
  done
  join
  $PY "$script" verdict --out-dir "$dir" --output-file "$dir/verdict.json"
}

case "$PHASE" in
  p1a)
    $PY $P1 cache --work "$WORK/p1" --fit-clips $FIT "${SMOKE[@]+"${SMOKE[@]}"}"
    bg $PY $P1 embed --work "$WORK/p1"
    bg $PY $P1 control-cache --work "$WORK/p1" --control-fit-clips $CONTROL
    join ;;
  p1b)
    bg $PY $P1 fit --work "$WORK/p1" "${STEPS[@]}"
    bg $PY $P1 control-fit --work "$WORK/p1" "${CONTROL_STEPS[@]}"
    join ;;
  p1c)
    bg $PY $P1 read --work "$WORK/p1"
    bg $PY $P1 rank --work "$WORK/p1"
    bg $PY $P1 sample --work "$WORK/p1" "${SMOKE[@]+"${SMOKE[@]}"}"
    bg $PY $P1 control-read --work "$WORK/p1"
    if [ ${#SMOKE[@]} -gt 0 ]; then
      bg $PY scripts/s1_text_kl.py "${SMOKE[@]}" --device cpu --batch 4 --checkpoint "$FLANT5" --checkpoint "$W14B" --output-file "$WORK/p1/s1.json"
    else
      bg $PY scripts/s1_text_kl.py --sentence-set d20 --device cpu --batch 4 --checkpoint "$FLANT5" --checkpoint "$W14B" --output-file "$WORK/p1/s1.json"
    fi
    join ;;
  p1v)
    $PY $P1 verdict --work "$WORK/p1" --s1-report "$WORK/p1/s1.json" --output-file "$WORK/p1/verdict.json" --scale mac ;;
  p2)
    forms scripts/s2_noise_curve.py "$WORK/p2" real blank,clean iid10,iid25 iid50,block25 block50,jitter key12,key6 ;;
  p3)
    $PY scripts/s2_length_probe.py lengths --out-dir "$WORK/p3" "${SMOKE[@]+"${SMOKE[@]}"}"
    forms scripts/s2_length_probe.py "$WORK/p3" real blank predicted ;;
  p4)
    forms scripts/s2_amplitude_probe.py "$WORK/p4" real blank amp0.8 amp0.9 amp1.1 amp1.27 ;;
  *) echo "unknown phase $PHASE" >&2; exit 2 ;;
esac
