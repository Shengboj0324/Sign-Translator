#!/usr/bin/env bash
# Re-evaluate generators with the frozen retrieval evaluator (usage: eval_round.sh RETRIEVAL_RUN RUN [RUN ...]): each
# run's best.json checkpoint on the same 200 validation clips, guidance 1,2,3, R-precision of generated and real motion,
# generated states stored under generated/<run>/ for later scoring and for the human reviewers. Logs uploaded to R2.
set -uo pipefail
RET=$1; shift
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
MSHA=$(sha "$M"); RKEY=$(best "$RET"); echo "retrieval $RKEY"
for run in "$@"; do
  echo "STAGE evaluate2 $run $(date)"
  if [ "$run" = tokens-prior-subset-v1 ]; then CK=training-runs/tokens-prior-subset-v1/checkpoints/step-00008000.pt; else CK=$(best "$run"); fi
  python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" --checkpoint-key "$CK" \
    --partition val --guidance 1,2,3 --device cuda --retrieval-key "$RKEY" --save-generated "$D/generated/$run" \
    --output-key "$D/evaluations-r2/$run.json" > /root/stage-eval2-$run.log 2>&1 || echo "FAILED evaluate2 $run"
  tail -1 /root/stage-eval2-$run.log | cut -c1-300; upload "training-runs/$run/stage-evaluate2.log" "/root/stage-eval2-$run.log"
done
echo EVAL-DONE
