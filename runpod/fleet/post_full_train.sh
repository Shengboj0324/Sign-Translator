#!/usr/bin/env bash
# post_full_train.sh — full How2Sign, after triangulation: manifest v2, the selected generator family retrained (v2b
# tokenizer: full frame rate, base blocks; locus-conditioned RVQ prior at batch 64), the duration-blind retrieval
# evaluator, then validation evaluation at the selected sampling settings with R-precision and the duration-only control,
# and the exit gate (head rotation included). Every stage's log goes to R2; the test partition is not touched here.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
export SPLITS="train val test" MODELS=/root/work/models PY=python DERIVED=acquisition/2026-10-17-v1/derived/full-v1
D=$DERIVED; M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
stage full-manifest manifest bash runpod/pipeline.sh manifest || { echo FULL-DONE; exit 1; }
MSHA=$(sha "$M")
COMMON="--architecture rvq --manifest-key $M --manifest-sha256 $MSHA --workers 8 --val-every 1000 --val-batches 32 --checkpoint-every 2000 --patience 5"
TOKRUN=tokens-rvq-d0b-full-v1 PRIOR=tokens-rvq-d0b-prior-loci-full-v1 RET=retrieval-hold-full-v1
stage $TOKRUN tokenizer python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage tokenizer --run $TOKRUN \
  --steps 40000 --hidden 512 --code-dim 128 --codebook 512 --stages 4 --down-layers 0 --base-blocks 4 || { echo FULL-DONE; exit 1; }
TOK=$(best $TOKRUN); TSHA=$(sha "$TOK")
stage $TOKRUN diagnose python scripts/diagnose_tokenizer.py --manifest-key "$M" --manifest-sha256 "$MSHA" --tokenizer-key "$TOK" \
  --limit 300 --output-key "$D/diagnostics/$TOKRUN.json"
stage $RET retrieval python scripts/train_text_motion_retrieval.py --manifest-key "$M" --manifest-sha256 "$MSHA" --run $RET \
  --batch-size 128 --steps 30000 --val-every 500 --patience 8 --workers 8
stage $PRIOR prior python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage prior --run $PRIOR \
  --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" --steps 40000 --d-model 384 --feedforward 1536 --token-layers 6 \
  --depth-layers 2 --locus-conditioning || { echo FULL-DONE; exit 1; }
stage $PRIOR evaluate python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" \
  --checkpoint-key "$(best $PRIOR)" --partition val --limit 600 --guidance 1,2 \
  --part-temperature body=1.2,left_hand=1.2,right_hand=1.2,face=1.0 --device cuda --retrieval-key "$(best $RET)" \
  --save-generated "$D/generated/$PRIOR-pt" --output-key "$D/evaluations/$PRIOR-pt.json"
stage $PRIOR gate python scripts/exit_gate.py --derived "$D" --runs $PRIOR-pt --output-key "$D/evaluations/$PRIOR-pt-gate.json"
echo FULL-DONE
