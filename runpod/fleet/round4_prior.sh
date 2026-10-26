#!/usr/bin/env bash
# round4_prior.sh — a round-4 pilot prior (docs/NEXT_PHASE_PLAN.md "Round 4 pilot", declared 2026-10-26 15:08 UTC):
# round 3d's prior recipe (60000 steps, ASLLVD sampled at weight 3, locus conditioning) on round 3d's tokenizer (env
# TOKKEY), with a frozen pretrained text encoder (env TEXT_MODEL, Hugging Face name) instead of the byte embedding.
# R3D_SMOKE=1: 300 steps into a -smoke run. PRIOR-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece   # the pretrained text encoder
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3c-mixed-v1/manifest.json; MSHA=$(sha $MIX)
[ "$MSHA" = 004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd ] || { echo "mix SHA mismatch"; exit 1; }
RUN=${PRUN:-tokens-rvq-d0b-prior-r3d}   # PRUN: a speculative prior on a fallback candidate (amendment 3)
if [ "${R3D_SMOKE:-0}" = 1 ]; then RUN=$RUN-smoke PSTEPS=300 PEVERY=100; else PSTEPS=60000 PEVERY=1000; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; exit 1; }
TSHA=$(sha "$TOKKEY"); echo "tokenizer $TOKKEY $TSHA"
echo "STAGE prior $RUN $(date)"
python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOKKEY" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --corpus-weight asllvd-text-v1=3 --text-model "$TEXT_MODEL" --val-every $PEVERY --val-batches 32 \
  --checkpoint-every $((PEVERY * 2)) --patience 5 --workers 10 --cache-bytes 107374182400 > /root/stage-prior-$RUN.log 2>&1
s=$?; tail -3 /root/stage-prior-$RUN.log; upload "training-runs/$RUN/stage-prior.log" "/root/stage-prior-$RUN.log"
[ $s = 0 ] || echo "FAILED prior $RUN (status $s)"
echo PRIOR-DONE
exit $s
