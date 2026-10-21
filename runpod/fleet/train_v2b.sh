#!/usr/bin/env bash
# Tokens v2b (full frame rate, base residual blocks): tokenizer with early stopping (resumes its run), error
# decomposition, then the plain and locus-conditioned priors with early stopping, each evaluated from best.json.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
D=acquisition/2026-10-17-v1/derived/subset-v1
M=$D/how2sign-text-manifest-v1/manifest.json
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
MSHA=$(sha "$M")
COMMON="--architecture rvq --manifest-key $M --manifest-sha256 $MSHA --workers 8 --val-every 500 --val-batches 16 --checkpoint-every 1000 --patience 4"
echo "STAGE v2b-tokenizer $(date)"
python scripts/train_motion_tokens.py $COMMON --batch-size 64 --stage tokenizer --run tokens-rvq-d0b-subset-v1 --steps 15000 \
  --hidden 512 --code-dim 128 --codebook 512 --stages 4 --down-layers 0 --base-blocks 4 || { echo "FAILED v2b-tokenizer"; exit 1; }
TOK=$(best tokens-rvq-d0b-subset-v1); TSHA=$(sha "$TOK"); echo "best tokenizer $TOK"
echo "STAGE v2b-diagnose $(date)"
python scripts/diagnose_tokenizer.py --manifest-key "$M" --manifest-sha256 "$MSHA" --tokenizer-key "$TOK" \
  --output-key "$D/diagnostics/tokens-rvq-d0b-subset-v1.json" || echo "FAILED v2b-diagnose"
for run in tokens-rvq-d0b-prior-subset-v1 tokens-rvq-d0b-prior-loci-subset-v1; do
  extra=""; [ "$run" = tokens-rvq-d0b-prior-loci-subset-v1 ] && extra="--locus-conditioning"
  echo "STAGE v2b-prior $run $(date)"
  python scripts/train_motion_tokens.py $COMMON --batch-size 128 --stage prior --run $run --tokenizer-key "$TOK" \
    --tokenizer-sha256 "$TSHA" --steps 12000 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 $extra \
    || { echo "FAILED v2b-prior $run"; continue; }
  echo "STAGE v2b-evaluate $run $(date)"
  python scripts/evaluate_text_to_motion.py --manifest-key "$M" --manifest-sha256 "$MSHA" --checkpoint-key "$(best $run)" \
    --partition val --guidance 1,2,3 --device cuda --output-key "$D/evaluations/$run.json" || echo "FAILED v2b-evaluate $run"
done
echo V2B-DONE
