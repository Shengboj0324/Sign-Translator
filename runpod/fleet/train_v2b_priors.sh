#!/usr/bin/env bash
# Tokens v2b priors (relaunch of 2026-10-21): the plain and locus-conditioned priors on the best v2b tokenizer,
# each evaluated from its best.json. Batch 64 as v2's priors: at 128 both priors failed on the first steps of the
# first chain (most likely GPU memory: full frame rate gives up to 600 steps x 16 codes per clip). Every stage's
# output is uploaded to R2 (training-runs/<run>/stage-<name>.log) when it ends, pass or fail, so a failure stays
# readable after the pod stops.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() {   # stage RUN NAME COMMAND...: run, keep the output, upload it, return the command's status
  local run=$1 name=$2; shift 2
  echo "STAGE $name $run $(date)"
  "$@" > "/root/stage-$name-$run.log" 2>&1; local status=$?
  tail -5 "/root/stage-$name-$run.log"
  upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $status = 0 ] || echo "FAILED $name $run (status $status)"
  return $status
}
MSHA=$(sha "$M")
TOK=$(best tokens-rvq-d0b-subset-v1); TSHA=$(sha "$TOK"); echo "tokenizer $TOK"
COMMON="--architecture rvq --manifest-key $M --manifest-sha256 $MSHA --workers 8 --val-every 500 --val-batches 16 --checkpoint-every 1000 --patience 4"
for run in tokens-rvq-d0b-prior-subset-v1-r2 tokens-rvq-d0b-prior-loci-subset-v1-r2; do
  extra=""; [ "$run" = tokens-rvq-d0b-prior-loci-subset-v1-r2 ] && extra="--locus-conditioning"
  stage $run prior python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage prior --run $run \
    --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" --steps 12000 --d-model 384 --feedforward 1536 --token-layers 6 \
    --depth-layers 2 $extra || continue
  stage $run evaluate python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
    --checkpoint-key "$(best $run)" --partition val --guidance 1,2,3 --device cuda --output-key "$D/evaluations/$run.json"
done
echo PRIORS-DONE
