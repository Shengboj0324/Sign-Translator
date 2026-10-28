#!/usr/bin/env bash
# f12_chain.sh PHASE [PART] — the declared evaluation chain on the leader prior F12 (tokens-rvq-d0b-prior-r12-f12) at
# its step 60000 (docs/NEXT_PHASE_PLAN.md: round 12's selection, "the winner continues to 60000 steps, then the full
# declared chain"). The chain and its settings are round 3e's (runpod/fleet/round3c_eval.sh with R3_HANDS='1.0 1.05 1.1'
# R3_FACES='0.9 1.0' R3_NPARTS=3 R3_ROTATE=1; "Round 3e design", 2026-10-26 07:21 UTC), unchanged except:
#   - the generator: F12's step-60000 checkpoint by name (not best.json, which tracks teacher-forced validation loss:
#     round 11 amendment 3), on the manifest F12 trained on (round10-mixed-v1; its How2Sign train/val/test/
#     heldout_official records are identical to round3c-mixed-v1's, checked 2026-10-28);
#   - R5's words pinned to the pilots' 150 (round 13 amendment 1): strict_prior_review.py --r5-vocab-mix-key
#     round10-mixed-v1, and it refuses to run unless the list equals the F12 pilot report's words;
#   - test-set protection (to be declared as an amendment before the run): nothing that reads the test or
#     heldout_official partitions runs unless the exit gate, R1, R2, R2b, R3, R4, R5 and R7 on the validation states
#     all passed (scripts/r6_guard.py, checked again inside strict_prior_review.py --r6-from-validation-key). Otherwise
#     the chain stops before reading any test data and records "R6 not run: validation rules failed (test partition
#     left untouched)". So the cross-signer baseline runs twice: validation bands first (no --test-limit; the gate's
#     envelopes never used the test signer), the test envelope only after the guard;
#   - reports under round12/chain-f12-step60000/, stage logs under its logs/.
# Phases (each its own pod, as round 3e):
#   sweep a|b|c (GPU) hands/body {1.0,1.05,1.1} x face {0.9,1.0} x signer groups c01,c03,c05 (rotated per setting),
#              guidance {1,1.5,2}, 300 validation clips each, 3 at a time; a failed evaluation is rerun once alone.
#   grid       (CPU) the post-processing grid (--round3c): head smoothing {0.1,0.2,0.3} x damping {1,0.8,0.6} x face
#              calibration, head translation smoothed with the head-rotation window, R7 and R4 slices inside.
#   review     (GPU) validation cross-signer baseline -> multi-signer exit gate -> strict review R1-R5
#              (--validation-only) -> R7 on the validation states -> TEST-SET GUARD -> [only if it allows: test
#              cross-signer baseline -> strict review R6 -> R7 on the test states] -> verdict.
# F12_SMOKE=1: toy sizes into a -smoke prefix (plumbing; the guard always blocks a --quick review, so a smoke run never
# reads test data). F12_DRY=1: print every stage's command without running anything (F12_DRY_GUARD=pass|block picks the
# guard's outcome); used by tests/test_f12_chain_script.py. F12_STEP: another checkpoint step (default 60000).
set -uo pipefail
PHASE=${1:?phase}; PART=${2:-}
DRY=${F12_DRY:-0}
if [ "$DRY" = 1 ]; then
  cd "$(dirname "$0")/../.."; PY=${PY:-python3} LOGDIR=${LOGDIR:-/tmp} DEVICE=cuda
else
  source /root/r2env.sh; export PATH=/root/venv/bin:$PATH; cd /root/work/Sign-Translator; PY=python LOGDIR=/root DEVICE=cuda
  uv pip install -q 'transformers>=4.40,<5' sentencepiece   # R2 / R2b translators (as round3b.sh and round3c_eval.sh)
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8} PY LOGDIR DEVICE DRY
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round10-mixed-v1/manifest.json; MSHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json
PILOT=$ACQ/youtube-asl-v1/round12/pilot-f12-step10000.json   # the pinned R5 words (round 13 amendment 1)
RUN=tokens-rvq-d0b-prior-r12-f12 STEP=${F12_STEP:-60000}
GEN=training-runs/$RUN/checkpoints/step-$(printf %08d "$STEP").pt
D=$ACQ/youtube-asl-v1/round12/chain-f12-step$STEP SIGNER_GROUPS=c01,c03,c05   # never GROUPS (reserved)
if [ "${F12_SMOKE:-0}" = 1 ]; then
  D=$D-smoke LIMIT=12 TESTLIM=12 QUICK=--quick HANDS="1.1" FACES="1.0" SMOOTH="0.1" DAMP="1,0.6"
else
  LIMIT=300 TESTLIM=92 QUICK= HANDS="1.0 1.05 1.1" FACES="0.9 1.0" SMOOTH="0.1,0.2,0.3" DAMP="1,0.8,0.6"   # round 3e
fi
NPARTS=3 ROTATE=1 EV=$D/evaluations
export MIX MSHA D RUN GEN LIMIT EV

if [ "$DRY" = 1 ]; then
  sha() { [ "$1" = "$MIX" ] && echo $MSHA || echo "sha256-of:$1"; }
  best() { echo "training-runs/$1/checkpoints/best-of-$1.pt"; }
  upload() { :; }
  exists() { case "$1" in "$GEN") return 0 ;; */evaluations/*) [ "${F12_DRY_STORED:-0}" = 1 ] ;; *) return 1 ;; esac; }
  stage() { local name=$1; shift; echo "STAGE $name: $*"
    case "$name" in guard) [ "${F12_DRY_GUARD:-block}" = pass ] && return 0 || return 3 ;; esac; return 0; }
else
  sha() { $PY -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
  best() { $PY -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
  upload() { $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
  exists() { $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
  stage() { local name=$1; shift; echo "STAGE $name $RUN $(date)"; "$@" > "$LOGDIR/stage-$name.log" 2>&1; local s=$?
    tail -3 "$LOGDIR/stage-$name.log"; upload "$D/logs/stage-$name.log" "$LOGDIR/stage-$name.log"
    [ $s = 0 ] || echo "FAILED $name (status $s)"; return $s; }
fi
export -f sha best upload exists stage

[ "$(sha $MIX)" = $MSHA ] || { echo "round10-mixed-v1 is not the declared manifest"; echo "$PHASE-DONE"; exit 1; }
exists "$GEN" || { echo "F12 checkpoint $GEN is not in R2 yet"; echo "$PHASE-DONE"; exit 1; }
FSHA=$(sha $FULL); RKEY=$(best retrieval-hold-full-v1); export FSHA RKEY
echo "generator $GEN sha256 $(sha "$GEN")"

settings() { for h in $HANDS; do for f in $FACES; do echo "$h $f"; done; done; }
sweep_one() { local h=$1 f=$2 g=$3; local TAG=$RUN-sweep-h$h-f$f; local hand=${h%b*} body=${h#*b}
  exists $EV/$TAG-$g.json && { echo "stored: $TAG-$g"; return 0; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 stage eval-h$h-f$f-$g \
    $PY scripts/evaluate_text_to_motion.py --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" \
    --partition val --corpus how2sign-two-view-text-v1 --signer-component $g --limit $LIMIT --guidance 1,1.5,2 --fps 24 \
    --device $DEVICE --batch 32 --part-temperature body=$body,left_hand=$hand,right_hand=$hand,face=$f --retrieval-key "$RKEY" \
    --retrieval-manifest-sha256 $FSHA --save-generated $D/generated/$TAG-$g --output-key $EV/$TAG-$g.json; }
export -f sweep_one

case $PHASE in
sweep)
  # round 3e's 18 evaluations in its fixed order, signer groups rotated per setting, dealt to parts a, b, c
  jobs=$(settings | awk -v r="$ROTATE" '{n=split("c01 c03 c05", g, " "); for (i = 0; i < n; i++)
      print $1, $2, g[(r == 1 ? (i + NR - 1) % n : i) + 1]}' | awk -v p="$PART" \
    -v n="$NPARTS" 'p == "" || (NR - 1) % n == index("abcdefgh", p) - 1')
  echo "$jobs" | xargs -P 3 -L 1 bash -c 'sweep_one "$@"' _
  while read h f g; do      # (a here-string, not a pipe: the exit leaves the script, not a subshell)
    exists $EV/$RUN-sweep-h$h-f$f-$g.json || OMP_NUM_THREADS=8 sweep_one $h $f $g \
      || { echo "evaluation h$h-f$f-$g failed twice"; echo EVAL-DONE; exit 1; }
  done <<< "$jobs"
  echo EVAL-DONE ;;
grid)
  for h in $HANDS; do for f in $FACES; do for g in c01 c03 c05; do
    exists $EV/$RUN-sweep-h$h-f$f-$g.json || { echo "sweep result h$h-f$f-$g missing"; echo GRID-DONE; exit 1; }
  done; done; done
  grid_one() { local TAG=$RUN-sweep-h$1-f$2
    OMP_NUM_THREADS=2 stage grid-h$1-f$2 $PY scripts/score_postprocessing_grid.py --manifest-key $MIX \
      --manifest-sha256 $MSHA --run $TAG --groups $SIGNER_GROUPS --pool-group pool --round3c \
      --generated-template "$D/generated/{run}-{group}" --evaluations $EV \
      --head-smoothing $SMOOTH --head-damping $DAMP; }
  export -f grid_one; export SIGNER_GROUPS SMOOTH DAMP
  settings | xargs -P 6 -L 1 bash -c 'grid_one "$@"' _
  echo GRID-DONE ;;
review)
  # 1. the validation signers' real cross-signer envelopes (no --test-limit: no test clip is read)
  stage cross-signer-baseline-validation $PY scripts/cross_signer_baseline.py --manifest-key $MIX --manifest-sha256 $MSHA \
    --groups $SIGNER_GROUPS --limit $LIMIT --output-key $EV/cross-signer-baseline-validation.json \
    || { echo REVIEW-DONE; exit 1; }
  RUNS=""
  for hf in $(settings | tr ' ' '-'); do h=${hf%-*} f=${hf#*-}; TAG=$RUN-sweep-h$h-f$f
    for s in ${SMOOTH//,/ }; do for a in ${DAMP//,/ }; do
      V=$TAG-hs$s$([ $a = 1 ] || echo -hd$a); RUNS="$RUNS $V $V-fc"
    done; done; done
  if [ "$DRY" != 1 ]; then   # every grid result present (one listing; round 3e's 432 single checks took ~20 min)
    $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store
have=set(R2Store(os.environ).list(sys.argv[1])); miss=[k for k in sys.argv[2:] if k not in have]
print('grid results present: %d of %d' % (len(sys.argv) - 2 - len(miss), len(sys.argv) - 2)); [print('grid result missing:', k) for k in miss[:10]]
sys.exit(1 if miss else 0)" "$EV/" $(for v in $RUNS; do for g in ${SIGNER_GROUPS//,/ } pool; do echo $EV/$v-$g.json; done; done) \
      || { echo REVIEW-DONE; exit 1; }
  fi
  # 2. the exit gate (multi-signer, every check against each group's real envelope)
  stage gate $PY scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $SIGNER_GROUPS,pool \
    --envelope-key $EV/cross-signer-baseline-validation.json --output-key $EV/$RUN-gate.json \
    || { echo REVIEW-DONE; exit 1; }
  if [ "$DRY" = 1 ]; then
    read SEL_RUN SEL_G HS HD <<< "$RUN-sweep-h1.1-f1.0 1 0.1 1"
  else
    read SEL_RUN SEL_G HS HD < <($PY -c "import os,re,json; from signtranslator.data.text_motion_dataset import R2Store
g=json.loads(R2Store(os.environ).get('$EV/$RUN-gate.json'))['selected']; r=g['run']
hs=float(re.search(r'-hs([0-9.]+)', r).group(1)); hd=re.search(r'-hd([0-9.]+)', r)
print(re.sub(r'-hs.*', '', r), g['setting'].split()[-1], hs, hd.group(1) if hd else 1)")
  fi
  TS=$($PY -c "print(max(float('$HS'), 0.1))")
  echo "selected $SEL_RUN guidance $SEL_G head smoothing $HS damping $HD; head-translation smoothing $TS"
  BKEY=$(best back-translator-full-v1) KP2D=$(best back-translator-kp2d-uwb-ft-v1)
  REVIEW=(scripts/strict_prior_review.py --derived $D --gate-key $EV/$RUN-gate.json --manifest-key $MIX
    --manifest-sha256 $MSHA --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY"
    --retrieval-manifest-sha256 $FSHA --translator-manifest-sha256 $FSHA --groups $SIGNER_GROUPS
    --heldout-corpus how2sign-two-view-text-v1 --second-translator-key "$KP2D" --word-seconds 0 --translation-smoothing $TS
    --r5-vocab-mix-key $MIX --r5-vocab-mix-sha256 $MSHA --r5-expect-words-key $PILOT --device $DEVICE $QUICK)
  # 3. strict review, validation rules R1-R5 only
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stage strict-review-validation $PY "${REVIEW[@]}" --validation-only \
    --output-key $EV/$RUN-strict-review-validation.json || { echo REVIEW-DONE; exit 1; }
  # 4. R7 on the selected candidate's validation states
  PREFIXES=""; for g in ${SIGNER_GROUPS//,/ }; do PREFIXES="$PREFIXES --generated-prefix $D/generated/$SEL_RUN-$g/guidance-$SEL_G"; done
  stage r7-validation $PY scripts/smoothness_check.py $PREFIXES --manifest-key $MIX --manifest-sha256 $MSHA \
    --head-smoothing $HS --head-damping $HD --translation-smoothing $TS --output-key $EV/$RUN-r7-validation.json \
    || { echo "R7 on the validation states failed to run: R6 not run, no verdict"; echo REVIEW-DONE; exit 1; }
  # 5. the test-set guard: exit 0 = every validation rule passed, 3 = blocked, anything else = the guard failed
  stage guard $PY scripts/r6_guard.py decide --gate-key $EV/$RUN-gate.json \
    --review-key $EV/$RUN-strict-review-validation.json --r7-validation-key $EV/$RUN-r7-validation.json \
    --output-key $EV/$RUN-test-set-guard.json
  G=$?
  VERDICT=(scripts/r6_guard.py verdict --gate-key $EV/$RUN-gate.json --r7-validation-key $EV/$RUN-r7-validation.json
    --guard-key $EV/$RUN-test-set-guard.json --output-key $EV/$RUN-verdict.json)
  if [ $G = 0 ]; then
    echo "TEST-SET GUARD: every validation rule passed; R6 runs"
    # 6. the test signer's envelope (reads the test partition's real clips)
    stage cross-signer-baseline $PY scripts/cross_signer_baseline.py --manifest-key $MIX --manifest-sha256 $MSHA \
      --groups $SIGNER_GROUPS --limit $LIMIT --test-limit $TESTLIM --output-key $EV/cross-signer-baseline.json \
      || { echo REVIEW-DONE; exit 1; }
    if [ "$DRY" != 1 ]; then   # its validation part must be the one the gate used
      $PY -c "import os,json; from signtranslator.data.text_motion_dataset import R2Store
s=R2Store(os.environ); a=json.loads(s.get('$EV/cross-signer-baseline-validation.json')); b=json.loads(s.get('$EV/cross-signer-baseline.json'))
k=('groups','cross','pooled_others','within'); raise SystemExit(0 if all(a[x]==b[x] for x in k) else 'validation envelopes differ')" \
        || { echo "the validation envelopes changed between the two baseline runs: R6 not run"; echo REVIEW-DONE; exit 1; }
    fi
    # 7. R6 (test and heldout_official), merged into the validation report; the review re-checks the guard itself
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stage strict-review-r6 $PY "${REVIEW[@]}" \
      --r6-from-validation-key $EV/$RUN-strict-review-validation.json --r7-validation-key $EV/$RUN-r7-validation.json \
      --envelope-key $EV/cross-signer-baseline.json --heldout-save-generated $D/review-generated \
      --output-key $EV/$RUN-strict-review.json || { echo REVIEW-DONE; exit 1; }
    # 8. R7 on R6's test states (already post-processed)
    stage r7-test $PY scripts/smoothness_check.py --generated-prefix $D/review-generated/test/guidance-$SEL_G \
      --manifest-key $MIX --manifest-sha256 $MSHA --output-key $EV/$RUN-r7-test.json \
      || { echo "R7 on the test states failed to run: no verdict"; echo REVIEW-DONE; exit 1; }
    stage verdict $PY "${VERDICT[@]}" --review-key $EV/$RUN-strict-review.json --r7-test-key $EV/$RUN-r7-test.json
  elif [ $G = 3 ]; then
    echo "R6 not run: validation rules failed (test partition left untouched)"
    stage verdict $PY "${VERDICT[@]}" --review-key $EV/$RUN-strict-review-validation.json
  else
    echo "TEST-SET GUARD FAILED TO RUN (status $G): R6 not run, no verdict"; echo REVIEW-DONE; exit 1
  fi
  echo REVIEW-DONE ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
