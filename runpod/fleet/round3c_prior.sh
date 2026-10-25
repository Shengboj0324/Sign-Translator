#!/usr/bin/env bash
# round3c_prior.sh — generator round 3c, stage B only (the prior), one GPU pod; design declared 2026-10-25 00:05 PDT
# (docs/NEXT_PHASE_PLAN.md, "Round 3c design"), before any round-3c result. The evaluation stages (sweep on two GPU
# pods, grid on CPU pods, baseline, gate with R4 slices and R7, strict review R1-R7) run as separate jobs after this
# one, so the GPU pod is never held for CPU work.
#   - mix round3c-mixed-v1: round 3b's mix without ASL Citizen's motion (its single-word targets tokenize to static
#     hands: the measured cause of R5's 0/111); ASLLVD up-weighted 3x (--corpus-weight);
#   - tokenizer: round 3a's, unchanged; it passed round 3b's domain check (YouTube r3b2, ASLLVD) and recognisability
#     round trip on the same corpora, all of which are in this mix;
#   - prior from scratch, round 3b's architecture and schedule, <= 60,000 steps, early stop.
# R3C_SMOKE=1: 300 steps into <run>-smoke (plumbing only, never a verdict). PRIOR-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3c-mixed-v1/manifest.json; MSHA=$(sha $MIX)
[ "$MSHA" = 004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd ] \
  || { echo "round-3c mix SHA $MSHA is not the declared one"; echo PRIOR-DONE; exit 1; }
R3A_SHA=dc576f4aff4bef38b933b3646670d98c3160e2474c80fb0719d91631ce00d92c     # round 3a's mix (its tokenizer's manifest)
RUN=tokens-rvq-d0b-prior-r3c
if [ "${R3C_SMOKE:-0}" = 1 ]; then RUN=$RUN-smoke PSTEPS=300 PEVERY=100; else PSTEPS=60000 PEVERY=1000; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; echo PRIOR-DONE; exit 1; }
TOK=$(best tokens-rvq-d0b-r3a); TSHA=$(sha "$TOK")
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $R3A_SHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --corpus-weight asllvd-text-v1=3 --val-every $PEVERY --val-batches 32 \
  --checkpoint-every $((PEVERY * 2)) --patience 5 --workers 10 --cache-bytes 107374182400 || { echo PRIOR-DONE; exit 1; }
echo PRIOR-DONE
