#!/usr/bin/env bash
# round3_prior.sh — generator round 3, stage B, one GPU pod, after round3_tokenizer.sh. Designed from task No.1's strict
# review failures (docs/NEXT_PHASE_PLAN.md, "Task No.1 verdict"): selection across several unseen signers, a pooled
# face calibration, single-sign training data, a tokenizer trained on the mix.
#   1. prior from scratch on the round-3 mix with the round-3 tokenizer (full prior's architecture and schedule);
#   2. sampling sweep hands {1.0,1.2,1.4} x face {0.8,1.0,1.2} x guidance {1,1.5,2}, evaluated separately on each
#      validation signer (c01 signers 1/2, c03 signer 3, c05 signer 11; How2Sign, 300 clips each, batched generation);
#   3. post-processing grid per setting: head smoothing {0,0.1,0.2,0.3} x {none, face calibration pooled over signers};
#   4. multi-signer exit gate (declared checks, every signer must pass);
#   5. strict review (R1-R6, rules unchanged) on the selection; R6 = How2Sign test signer 9, once.
# R3-DONE at the end.
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
export -f sha best upload stage
MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round3a-mixed-v1/manifest.json; MSHA=$(sha $MIX)
FULL=acquisition/2026-10-17-v1/derived/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha $FULL)
D=acquisition/2026-10-17-v1/youtube-asl-v1/round3a
TOKRUN=tokens-rvq-d0b-r3a RUN=tokens-rvq-d0b-prior-r3a SIGNER_GROUPS=c01,c03,c05
TOK=$(best $TOKRUN); TSHA=$(sha "$TOK")
python scripts/prefetch_cache.py $MIX
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" \
  --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps 40000 --lr 3e-4 --warmup 1000 --batch-size 64 \
  --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 --locus-conditioning \
  --val-every 1000 --val-batches 32 --checkpoint-every 2000 --patience 5 --workers 10 --cache-bytes 107374182400 \
  || { echo R3-DONE; exit 1; }
GEN=$(best $RUN)
RKEY=$(best retrieval-hold-full-v1) BKEY=$(best back-translator-full-v1)
export MIX MSHA FSHA D RUN GEN RKEY
# 2. sweep: 27 (setting, signer) evaluations, four at a time on the GPU (batched generation)
sweep_one() { local h=$1 f=$2 g=$3; local TAG=$RUN-sweep-h$h-f$f
  OMP_NUM_THREADS=2 stage $RUN eval-h$h-f$f-$g python scripts/evaluate_text_to_motion.py --manifest-key $MIX \
    --manifest-sha256 $MSHA --checkpoint-key "$GEN" --partition val --corpus how2sign-two-view-text-v1 \
    --signer-component $g --limit 300 --guidance 1,1.5,2 --fps 24 --device cuda --batch 32 \
    --part-temperature body=$h,left_hand=$h,right_hand=$h,face=$f --retrieval-key "$RKEY" --retrieval-manifest-sha256 $FSHA \
    --save-generated $D/generated/$TAG-$g --output-key $D/evaluations/$TAG-$g.json; }
export -f sweep_one
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do for g in c01 c03 c05; do echo "$h $f $g"; done; done; done \
  | xargs -P 4 -L 1 bash -c 'sweep_one "$@"' _
# 3. post-processing grid (CPU), all nine settings in parallel
grid_one() { local TAG=$RUN-sweep-h$1-f$2
  OMP_NUM_THREADS=1 stage $RUN grid-h$1-f$2 python scripts/score_postprocessing_grid.py --manifest-key $MIX --manifest-sha256 $MSHA \
    --run $TAG --groups c01,c03,c05 --generated-template "$D/generated/{run}-{group}" --evaluations $D/evaluations \
    --head-smoothing 0,0.1,0.2,0.3; }
export -f grid_one
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do echo "$h $f"; done; done | xargs -P 9 -L 1 bash -c 'grid_one "$@"' _
RUNS=""
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do TAG=$RUN-sweep-h$h-f$f; RUNS="$RUNS $TAG"
  for s in 0 0.1 0.2 0.3; do RUNS="$RUNS $TAG-hs$s $TAG-hs$s-fc"; done; done; done
stage $RUN gate python scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $SIGNER_GROUPS --output-key $D/evaluations/$RUN-gate.json
stage $RUN strict-review python scripts/strict_prior_review.py --derived $D --gate-key $D/evaluations/$RUN-gate.json \
  --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY" \
  --retrieval-manifest-sha256 $FSHA --translator-manifest-sha256 $FSHA --groups $SIGNER_GROUPS \
  --heldout-corpus how2sign-two-view-text-v1 --output-key $D/evaluations/$RUN-strict-review.json --device cuda
echo R3-DONE
