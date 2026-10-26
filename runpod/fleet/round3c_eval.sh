#!/usr/bin/env bash
# round3c_eval.sh PHASE [PART] — round 3c's evaluation (docs/NEXT_PHASE_PLAN.md, "Round 3c design" 2026-10-25 00:05 PDT
# and its amendment 00:20 PDT, both before any round-3c result), split so each pod only runs what it needs:
#   sweep a|b  (GPU) half of hands {1.2,1.25,1.3} x face {0.8,1.0} x 3 validation signers, guidance {1,1.5,2}, 300 clips,
#              3 at a time; a failed evaluation is rerun once alone. EVAL-DONE.
#   grid       (CPU) the post-processing grid with --round3c: head smoothing {0.1,0.2,0.3} x damping {1,0.8,0.6} x face
#              calibration, head translation smoothed with the head-rotation window, R7 in every group and R4's length
#              slices on the pooled group. GRID-DONE.
#   review     (GPU) real cross-signer envelopes, the envelope gate (now also gating R4 slices and R7), the strict review
#              R1-R6 with head-translation smoothing max(head smoothing, 0.1 s), then R7 (smoothness_check.py) on the
#              selected candidate's validation states and on R6's test states. Every rule must pass. REVIEW-DONE.
# R3C_SMOKE=1: the toy smoke prior at toy sizes into -smoke keys (plumbing only). R3C_LOCAL=1: run from this checkout
# with the project venv (for the Mac smoke run) instead of the pod layout.
set -uo pipefail
PHASE=${1:?phase}; PART=${2:-}
if [ "${R3C_LOCAL:-0}" = 1 ]; then
  cd "$(dirname "$0")/../.."; set -a; source ./.env.r2.local; set +a; PY=${PY:-$HOME/.venvs/sign-translator/bin/python}
  LOGDIR=${LOGDIR:-/tmp/r3c-eval-logs}; mkdir -p "$LOGDIR"; DEVICE=${DEVICE:-cpu}
else
  source /root/r2env.sh; export PATH=/root/venv/bin:$PATH; cd /root/work/Sign-Translator; PY=python LOGDIR=/root DEVICE=cuda
  uv pip install -q 'transformers>=4.40,<5' sentencepiece   # R2 / R2b translators (as round3b.sh)
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8} PY LOGDIR DEVICE
sha() { $PY -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { $PY -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
exists() { $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "$LOGDIR/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "$LOGDIR/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "$LOGDIR/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
export -f sha best upload exists stage
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3c-mixed-v1/manifest.json; MSHA=004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json
# round 3d reuses this chain unchanged under its own names (R3_RUN, R3_D; docs "Round 3d design")
RUN=${R3_RUN:-tokens-rvq-d0b-prior-r3c} D=${R3_D:-$ACQ/youtube-asl-v1/round3c} SIGNER_GROUPS=c01,c03,c05   # never GROUPS (reserved)
if [ "${R3C_SMOKE:-0}" = 1 ]; then
  RUN=$RUN-smoke D=$D-smoke LIMIT=12 TESTLIM=12 QUICK=--quick HANDS="1.25" FACES="1.0" SMOOTH="0.1" DAMP="1,0.6"
else
  LIMIT=300 TESTLIM=92 QUICK= HANDS="1.2 1.25 1.3" FACES="${R3_FACES:-0.8 1.0}" SMOOTH="0.1,0.2,0.3" DAMP="1,0.8,0.6"
fi
# round 3d amendment 4: R3_FACES (face temperatures {1.0, 1.1, 1.2}); R3_NPARTS sweep pods (part a, b, c, ...)
export MIX MSHA D RUN LIMIT
[ "$(sha $MIX)" = $MSHA ] || { echo "round-3c mix is not the declared one"; echo "$PHASE-DONE"; exit 1; }
FSHA=$(sha $FULL); GEN=$(best $RUN); RKEY=$(best retrieval-hold-full-v1); export FSHA GEN RKEY

settings() { for h in $HANDS; do for f in $FACES; do echo "$h $f"; done; done; }
sweep_one() { local h=$1 f=$2 g=$3; local TAG=$RUN-sweep-h$h-f$f
  exists $D/evaluations/$TAG-$g.json && { echo "stored: $TAG-$g"; return 0; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 stage $RUN eval-h$h-f$f-$g \
    $PY scripts/evaluate_text_to_motion.py --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" \
    --partition val --corpus how2sign-two-view-text-v1 --signer-component $g --limit $LIMIT --guidance 1,1.5,2 --fps 24 \
    --device $DEVICE --batch 32 --part-temperature body=$h,left_hand=$h,right_hand=$h,face=$f --retrieval-key "$RKEY" \
    --retrieval-manifest-sha256 $FSHA --save-generated $D/generated/$TAG-$g --output-key $D/evaluations/$TAG-$g.json; }
export -f sweep_one

case $PHASE in
sweep)
  # the 18 evaluations in a fixed order, alternately to part a and b
  jobs=$(settings | while read h f; do for g in c01 c03 c05; do echo "$h $f $g"; done; done | awk -v p="$PART" \
    -v n="${R3_NPARTS:-2}" 'p == "" || (NR - 1) % n == index("abcdefgh", p) - 1')
  echo "$jobs" | xargs -P 3 -L 1 bash -c 'sweep_one "$@"' _
  echo "$jobs" | while read h f g; do
    exists $D/evaluations/$RUN-sweep-h$h-f$f-$g.json || OMP_NUM_THREADS=8 sweep_one $h $f $g \
      || { echo "evaluation h$h-f$f-$g failed twice"; echo EVAL-DONE; exit 1; }
  done
  echo EVAL-DONE ;;
grid)
  for h in $HANDS; do for f in $FACES; do for g in c01 c03 c05; do
    exists $D/evaluations/$RUN-sweep-h$h-f$f-$g.json || { echo "sweep result h$h-f$f-$g missing"; echo GRID-DONE; exit 1; }
  done; done; done
  grid_one() { local TAG=$RUN-sweep-h$1-f$2
    OMP_NUM_THREADS=2 stage $RUN grid-h$1-f$2 $PY scripts/score_postprocessing_grid.py --manifest-key $MIX \
      --manifest-sha256 $MSHA --run $TAG --groups $SIGNER_GROUPS --pool-group pool --round3c \
      --generated-template "$D/generated/{run}-{group}" --evaluations $D/evaluations \
      --head-smoothing $SMOOTH --head-damping $DAMP; }
  export -f grid_one; export SIGNER_GROUPS SMOOTH DAMP
  settings | xargs -P 6 -L 1 bash -c 'grid_one "$@"' _
  echo GRID-DONE ;;
review)
  stage $RUN cross-signer-baseline $PY scripts/cross_signer_baseline.py --manifest-key $MIX --manifest-sha256 $MSHA \
    --groups $SIGNER_GROUPS --limit $LIMIT --test-limit $TESTLIM --output-key $D/evaluations/cross-signer-baseline.json \
    || { echo REVIEW-DONE; exit 1; }
  RUNS=""
  for hf in $(settings | tr ' ' '-'); do h=${hf%-*} f=${hf#*-}; TAG=$RUN-sweep-h$h-f$f
    for s in ${SMOOTH//,/ }; do for a in ${DAMP//,/ }; do
      V=$TAG-hs$s$([ $a = 1 ] || echo -hd$a); RUNS="$RUNS $V $V-fc"
    done; done; done
  for v in $RUNS; do for g in ${SIGNER_GROUPS//,/ } pool; do
    exists $D/evaluations/$v-$g.json || { echo "grid result $v-$g missing"; echo REVIEW-DONE; exit 1; }
  done; done
  stage $RUN gate $PY scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $SIGNER_GROUPS,pool \
    --envelope-key $D/evaluations/cross-signer-baseline.json --output-key $D/evaluations/$RUN-gate.json \
    || { echo REVIEW-DONE; exit 1; }
  read SEL_RUN SEL_G HS HD < <($PY -c "import os,re,json; from signtranslator.data.text_motion_dataset import R2Store
g=json.loads(R2Store(os.environ).get('$D/evaluations/$RUN-gate.json'))['selected']; r=g['run']
hs=float(re.search(r'-hs([0-9.]+)', r).group(1)); hd=re.search(r'-hd([0-9.]+)', r)
print(re.sub(r'-hs.*', '', r), g['setting'].split()[-1], hs, hd.group(1) if hd else 1)")
  TS=$($PY -c "print(max(float('$HS'), 0.1))")
  echo "selected $SEL_RUN guidance $SEL_G head smoothing $HS damping $HD; head-translation smoothing $TS"
  BKEY=$(best back-translator-full-v1) KP2D=$(best back-translator-kp2d-uwb-ft-v1)
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stage $RUN strict-review $PY scripts/strict_prior_review.py \
    --derived $D --gate-key $D/evaluations/$RUN-gate.json --manifest-key $MIX --manifest-sha256 $MSHA \
    --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY" --retrieval-manifest-sha256 $FSHA \
    --translator-manifest-sha256 $FSHA --groups $SIGNER_GROUPS --heldout-corpus how2sign-two-view-text-v1 \
    --second-translator-key "$KP2D" --word-seconds 0 --translation-smoothing $TS \
    --heldout-save-generated $D/review-generated --envelope-key $D/evaluations/cross-signer-baseline.json \
    --output-key $D/evaluations/$RUN-strict-review.json --device $DEVICE $QUICK
  PREFIXES=""; for g in ${SIGNER_GROUPS//,/ }; do PREFIXES="$PREFIXES --generated-prefix $D/generated/$SEL_RUN-$g/guidance-$SEL_G"; done
  stage $RUN r7-validation $PY scripts/smoothness_check.py $PREFIXES --manifest-key $MIX --manifest-sha256 $MSHA \
    --head-smoothing $HS --head-damping $HD --translation-smoothing $TS --output-key $D/evaluations/$RUN-r7-validation.json
  stage $RUN r7-test $PY scripts/smoothness_check.py --generated-prefix $D/review-generated/test/guidance-$SEL_G \
    --manifest-key $MIX --manifest-sha256 $MSHA --output-key $D/evaluations/$RUN-r7-test.json   # already post-processed
  $PY -c "import os,json; from signtranslator.data.text_motion_dataset import R2Store
s=R2Store(os.environ); D='$D'; R='$RUN'
review=json.loads(s.get(f'{D}/evaluations/{R}-strict-review.json'))
r7=[json.loads(s.get(f'{D}/evaluations/{R}-r7-{p}.json'))['pass'] if s.exists(f'{D}/evaluations/{R}-r7-{p}.json') else False for p in ('validation','test')]
rules={k: v.get('pass') for k, v in review['rules'].items()}; rules['R7_smoothness']=all(r7)
verdict=dict(rules=rules, review_pass=review.get('pass'), r7_validation=r7[0], r7_test=r7[1], **{'pass': bool(review.get('pass')) and all(r7)})
s.put(f'{D}/evaluations/{R}-verdict.json', json.dumps(verdict, indent=2).encode()); print(json.dumps(verdict))"
  echo REVIEW-DONE ;;
*) echo "unknown phase $PHASE"; exit 2 ;;
esac
