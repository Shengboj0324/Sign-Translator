#!/usr/bin/env bash
# post_gate_sampling_sweep.sh — plan item 4 (tuning) on one GPU pod, after the full-corpus prior is trained: sweep the
# sampling settings of tokens-rvq-d0b-prior-loci-full-v1 on the validation partition (test untouched) and select with
# the same declared exit gate (scripts/exit_gate.py), so the sweep cannot change the rule it is judged by.
#   - per-part temperatures: hands {1.0, 1.2, 1.4} x face {0.8, 1.0} (body follows the hands). The face part carries the
#     head-rotation codes, whose over-activity (1.9-3.3x too fast) was the subset round's carried problem, so a cooler
#     face is part of the grid;
#   - guidance {1, 1.5, 2} inside each evaluation.
# Every evaluation is saved under $D/evaluations/$PRIOR-sweep-<setting>.json with its generated states, scored by
# retrieval R-precision (blank control inside) and back-translation, and the gate runs over the core chain's own
# evaluations plus every sweep setting. SWEEP-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/full-v1; M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
PRIOR=tokens-rvq-d0b-prior-loci-full-v1 RET=retrieval-hold-full-v1 BT=back-translator-full-v1
MSHA=$(sha "$M"); GEN=$(best $PRIOR); RKEY=$(best $RET); BKEY=$(best $BT)
RUNS="$PRIOR-pt $PRIOR-pt-hs0.1 $PRIOR-pt-hs0.2 $PRIOR-pt-hs0.3"
for hands in 1.0 1.2 1.4; do
  for face in 0.8 1.0; do
    TAG=$PRIOR-sweep-h$hands-f$face
    stage $PRIOR sweep-h$hands-f$face python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
      --checkpoint-key "$GEN" --partition val --limit 600 --guidance 1,1.5,2 --device cuda \
      --part-temperature body=$hands,left_hand=$hands,right_hand=$hands,face=$face --retrieval-key "$RKEY" \
      --save-generated "$D/generated/$TAG" --output-key "$D/evaluations/$TAG.json" && RUNS="$RUNS $TAG"
  done
done
GENERATED=""
for r in $RUNS; do [[ $r == *sweep* ]] && GENERATED="$GENERATED $D/generated/$r"; done
stage $PRIOR sweep-back-translation python scripts/score_back_translation.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
  --translator-key "$BKEY" --generated $GENERATED --output-key "$D/evaluations/$PRIOR-sweep-bleu.json"
stage $PRIOR sweep-gate python scripts/exit_gate.py --derived "$D" --runs $RUNS --output-key "$D/evaluations/$PRIOR-sweep-gate.json"
echo SWEEP-DONE
