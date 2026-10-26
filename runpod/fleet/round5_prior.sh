#!/usr/bin/env bash
# round5_prior.sh — a round-5 pilot prior (docs/NEXT_PHASE_PLAN.md "Round 5", declared 2026-10-26 20:55 UTC, amendment 1):
# round 4's recipe (60000 steps, ASLLVD and RIT citation signs at weight 3, locus conditioning) on round 3d's tokenizer (env
# TOKKEY) and mix round5-mixed-v1 (env MIX, MSHA); TEXT_MODEL (Hugging Face name) or empty for the byte encoder.
# R3D_SMOKE=1: 300 steps into a -smoke run. PRIOR-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
[ -n "${TEXT_MODEL:-}" ] && uv pip install -q 'transformers>=4.40,<5' sentencepiece   # the pretrained text encoder
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
ACQ=acquisition/2026-10-17-v1
TMSHA=004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd   # round3c-mixed-v1: the tokenizer's own manifest
[ "$(sha $MIX)" = "$MSHA" ] || { echo "mix SHA mismatch"; exit 1; }
RUN=$PRUN
if [ "${R3D_SMOKE:-0}" = 1 ]; then RUN=$RUN-smoke PSTEPS=300 PEVERY=100; else PSTEPS=60000 PEVERY=1000; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; exit 1; }
TSHA=$(sha "$TOKKEY"); echo "tokenizer $TOKKEY $TSHA"
echo "STAGE prior $RUN $(date)"
python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOKKEY" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $TMSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --corpus-weight asllvd-text-v1=3 --corpus-weight rit-citation-text-v1=3 ${TEXT_MODEL:+--text-model "$TEXT_MODEL"} --val-every $PEVERY --val-batches 32 \
  --checkpoint-every $((PEVERY * 2)) --patience 5 --workers 10 --cache-bytes 107374182400 > /root/stage-prior-$RUN.log 2>&1
s=$?; tail -3 /root/stage-prior-$RUN.log; upload "training-runs/$RUN/stage-prior.log" "/root/stage-prior-$RUN.log"
[ $s = 0 ] || echo "FAILED prior $RUN (status $s)"
echo PRIOR-DONE
exit $s
