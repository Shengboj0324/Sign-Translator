#!/usr/bin/env bash
# Tokens v2 on the subset manifest: tokenizer -> prior -> locus prior, each selected by validation (best.json),
# then evaluation and the tokenizer error decomposition. Writes V2-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
MSHA=$(sha "$M")
COMMON="--architecture rvq --manifest-key $M --manifest-sha256 $MSHA --batch-size 64 --workers 8 --val-every 500 --val-batches 16 --checkpoint-every 1000"
echo "STAGE v2-tokenizer $(date)"
python scripts/train_motion_tokens.py $COMMON --stage tokenizer --run tokens-rvq-subset-v1 --steps ${TOK_STEPS:-15000} \
  --hidden 384 --code-dim 128 --codebook 512 --stages 4 --down-layers 1 || { echo "FAILED v2-tokenizer"; exit 1; }
TOK=$(best tokens-rvq-subset-v1); TSHA=$(sha "$TOK"); echo "best tokenizer $TOK"
echo "STAGE v2-diagnose $(date)"
python scripts/diagnose_tokenizer.py --manifest-key "$M" --manifest-sha256 "$MSHA" --tokenizer-key "$TOK" \
  --output-key "$D/diagnostics/tokens-rvq-subset-v1.json" || echo "FAILED v2-diagnose"
for run in tokens-rvq-prior-subset-v1 tokens-rvq-prior-loci-subset-v1; do
  extra=""; [ "$run" = tokens-rvq-prior-loci-subset-v1 ] && extra="--locus-conditioning"
  echo "STAGE v2-prior $run $(date)"
  python scripts/train_motion_tokens.py $COMMON --stage prior --run $run --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" \
    --steps ${PRIOR_STEPS:-8000} --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 $extra \
    || { echo "FAILED v2-prior $run"; continue; }
  echo "STAGE v2-evaluate $run $(date)"
  python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" --checkpoint-key "$(best $run)" \
    --partition val --guidance 1,2,3 --device cuda --output-key "$D/evaluations/$run.json" || echo "FAILED v2-evaluate $run"
done
echo V2-DONE
