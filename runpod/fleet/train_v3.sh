#!/usr/bin/env bash
# Tokens v3 candidate (usage: train_v3.sh NAME STAGES CODEBOOK): a generation-oriented tokenizer at 7.5 code steps/s
# (--down-layers 2, the rate class of published RVQ text-to-motion), then its error and learnability diagnosis, the
# plain prior and its evaluation. v2/v2b reconstruct well but their codes are near uniform and change almost every
# step (bigram rate 422 / 1,070 nats/s against v1's 8), which a prior cannot learn from 4k clips. Every stage's
# output is uploaded to R2 (training-runs/<run>/stage-<name>.log), pass or fail.
set -uo pipefail
NAME=$1 Q=$2 C=$3
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() {   # stage RUN NAME COMMAND...
  local run=$1 name=$2; shift 2
  echo "STAGE $name $run $(date)"
  "$@" > "/root/stage-$name-$run.log" 2>&1; local status=$?
  tail -3 "/root/stage-$name-$run.log"
  upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $status = 0 ] || echo "FAILED $name $run (status $status)"
  return $status
}
MSHA=$(sha "$M")
COMMON="--architecture rvq --manifest-key $M --manifest-sha256 $MSHA --workers 8 --val-every 500 --val-batches 16 --checkpoint-every 1000 --patience 4"
TOKRUN=tokens-$NAME-subset-v1 PRIORRUN=tokens-$NAME-prior-subset-v1
stage $TOKRUN tokenizer python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage tokenizer --run $TOKRUN \
  --steps 15000 --hidden 512 --code-dim 128 --codebook $C --stages $Q --down-layers 2 --base-blocks 4 || { echo V3-DONE; exit 1; }
TOK=$(best $TOKRUN); TSHA=$(sha "$TOK"); echo "best tokenizer $TOK"
stage $TOKRUN diagnose python scripts/diagnose_tokenizer.py --manifest-key "$M" --manifest-sha256 "$MSHA" --tokenizer-key "$TOK" \
  --output-key "$D/diagnostics/$TOKRUN.json"
stage $PRIORRUN prior python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage prior --run $PRIORRUN \
  --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" --steps 12000 --d-model 384 --feedforward 1536 --token-layers 6 \
  --depth-layers 2 && \
stage $PRIORRUN evaluate python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
  --checkpoint-key "$(best $PRIORRUN)" --partition val --guidance 1,2,3 --device cuda --output-key "$D/evaluations/$PRIORRUN.json"
echo V3-DONE
