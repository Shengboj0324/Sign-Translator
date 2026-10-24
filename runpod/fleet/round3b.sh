#!/usr/bin/env bash
# round3b.sh — generator round 3b, one GPU pod; design declared 2026-10-23 23:48 PDT (docs/NEXT_PHASE_PLAN.md,
# "Round 3b design"), gate 23:07 PDT ("Round 3b gate"), both before any round-3b result.
#   A. tokenizer: the round-3a tokenizer is kept if the domain check (YouTube r3b, ASLLVD) and the recognisability round
#      trip (ASLLVD val) pass; otherwise a tokenizer is trained on the round-3b mix and must pass both;
#   B. prior from scratch on the round-3b mix (round 3a's architecture and schedule, <= 60,000 steps, early stop);
#   C. sweep hands {1.1,1.2,1.3} x face {0.8,1.0,1.2} x guidance {1,1.5,2}, per validation signer (300 clips), at most
#      3 at a time with expandable CUDA segments; any failed evaluation is rerun once before the grid;
#   D. post-processing grid: head smoothing x damping {1,0.8,0.6} x pooled face calibration, per signer and pooled;
#   E. real cross-signer baseline (the envelopes, from real clips only), the envelope gate, the strict review
#      (R1-R6 with R2b; R5 at the generator's predicted durations; R6 within the test signer's envelope).
# R3B-DONE at the end. The How2Sign test partition is read only by R6 and by the baseline's real test clips.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece   # R2b's 2D back-translator is T5-based
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
export -f sha best upload exists stage
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3b-mixed-v1/manifest.json; MSHA=$(sha $MIX)
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha $FULL)
R3A_SHA=dc576f4aff4bef38b933b3646670d98c3160e2474c80fb0719d91631ce00d92c     # round 3a's mix (its tokenizer's manifest)
D=$ACQ/youtube-asl-v1/round3b
RUN=tokens-rvq-d0b-prior-r3b SIGNER_GROUPS=c01,c03,c05      # never GROUPS: a reserved bash variable
GLOSS_MAP=$ACQ/derived/signbank-v1/asl-citizen-gloss-map.json
# R3B_SMOKE=1: every stage end to end at toy sizes into <run>-smoke / <D>-smoke (plumbing only, never a verdict;
# its R6 dry run reads 12 test clips with a 300-step toy prior, written to -dryrun keys and used for no decision)
if [ "${R3B_SMOKE:-0}" = 1 ]; then
  RUN=$RUN-smoke D=$D-smoke PSTEPS=300 PEVERY=100 LIMIT=12 TLIMIT=24 TESTLIM=12 QUICK=--quick
  HANDS="1.2" FACES="1.0"
else
  PSTEPS=60000 PEVERY=1000 LIMIT=300 TLIMIT=300 TESTLIM=92 QUICK=
  HANDS="1.1 1.2 1.3" FACES="0.8 1.0 1.2"
fi
export MIX MSHA FSHA D RUN LIMIT
python scripts/prefetch_cache.py $MIX

# A. tokenizer acceptance (declared): domain check and recognisability round trip
tok_ok() { local run=$1 tag=$2 key=$3
  stage $run domain-check-$tag python scripts/tokenizer_domain_check.py --manifest-key $MIX --manifest-sha256 $MSHA \
    --conform-to-corpus how2sign-two-view-text-v1 --corpus youtube-asl-v1 --corpus asllvd-text-v1 --limit $TLIMIT \
    --tokenizer-key "$key" --output-key $D/tokenizer-domain-check-$tag.json \
  && stage $run recognisability-$tag python scripts/tokenizer_recognisability_check.py --manifest-key $MIX \
    --manifest-sha256 $MSHA --corpus asllvd-text-v1 --gloss-map-key $GLOSS_MAP --tokenizer-key "$key" --limit $TLIMIT \
    --device cuda --output-key $D/tokenizer-recognisability-$tag.json; }
TOK=$(best tokens-rvq-d0b-r3a) TOK_MANIFEST=$R3A_SHA
if ! tok_ok $RUN r3a "$TOK"; then
  [ "${R3B_SMOKE:-0}" = 1 ] && { echo "smoke: tokenizer checks failed"; echo R3B-DONE; exit 1; }
  echo "round-3a tokenizer failed the round-3b acceptance: training one on the round-3b mix"
  stage tokens-rvq-d0b-r3b tokenizer python scripts/train_motion_tokens.py --architecture rvq --stage tokenizer \
    --run tokens-rvq-d0b-r3b --manifest-key $MIX --manifest-sha256 $MSHA --fps 24 \
    --conform-to-corpus how2sign-two-view-text-v1 --hidden 512 --code-dim 128 --codebook 512 --stages 4 \
    --down-layers 0 --base-blocks 4 --steps 40000 --batch-size 64 --lr 3e-4 --warmup 1000 --val-every 1000 \
    --val-batches 32 --checkpoint-every 2000 --patience 5 --workers 10 --cache-bytes 107374182400 \
    || { echo R3B-DONE; exit 1; }
  TOK=$(best tokens-rvq-d0b-r3b) TOK_MANIFEST=$MSHA
  tok_ok tokens-rvq-d0b-r3b r3b "$TOK" || { echo "the round-3b tokenizer failed its acceptance"; echo R3B-DONE; exit 1; }
fi
TSHA=$(sha "$TOK")

# B. prior from scratch
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $TOK_MANIFEST --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --val-every $PEVERY --val-batches 32 --checkpoint-every $((PEVERY * 2)) --patience 5 \
  --workers 10 --cache-bytes 107374182400 || { echo R3B-DONE; exit 1; }
GEN=$(best $RUN) RKEY=$(best retrieval-hold-full-v1) BKEY=$(best back-translator-full-v1)
KP2D=$(best back-translator-kp2d-uwb-ft-v1)
export GEN RKEY

# C. sweep, 3 evaluations at a time; a failed one is rerun alone
sweep_one() { local h=$1 f=$2 g=$3; local TAG=$RUN-sweep-h$h-f$f
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 stage $RUN eval-h$h-f$f-$g \
    python scripts/evaluate_text_to_motion.py --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" \
    --partition val --corpus how2sign-two-view-text-v1 --signer-component $g --limit $LIMIT --guidance 1,1.5,2 --fps 24 \
    --device cuda --batch 32 --part-temperature body=$h,left_hand=$h,right_hand=$h,face=$f --retrieval-key "$RKEY" \
    --retrieval-manifest-sha256 $FSHA --save-generated $D/generated/$TAG-$g --output-key $D/evaluations/$TAG-$g.json; }
export -f sweep_one
for h in $HANDS; do for f in $FACES; do for g in c01 c03 c05; do echo "$h $f $g"; done; done; done \
  | xargs -P 3 -L 1 bash -c 'sweep_one "$@"' _
for h in $HANDS; do for f in $FACES; do for g in c01 c03 c05; do
  exists $D/evaluations/$RUN-sweep-h$h-f$f-$g.json || OMP_NUM_THREADS=8 sweep_one $h $f $g \
    || { echo "evaluation h$h-f$f-$g failed twice"; echo R3B-DONE; exit 1; }
done; done; done

# D. post-processing grid (CPU), per signer and pooled (grid and gate refuse empty input)
grid_one() { local TAG=$RUN-sweep-h$1-f$2
  OMP_NUM_THREADS=1 stage $RUN grid-h$1-f$2 python scripts/score_postprocessing_grid.py --manifest-key $MIX \
    --manifest-sha256 $MSHA --run $TAG --groups c01,c03,c05 --pool-group pool \
    --generated-template "$D/generated/{run}-{group}" --evaluations $D/evaluations \
    --head-smoothing 0,0.1,0.2,0.3 --head-damping 1,0.8,0.6; }
export -f grid_one
for h in $HANDS; do for f in $FACES; do echo "$h $f"; done; done | xargs -P 9 -L 1 bash -c 'grid_one "$@"' _

# E. envelopes (real clips only), envelope gate, strict review
stage $RUN cross-signer-baseline python scripts/cross_signer_baseline.py --manifest-key $MIX --manifest-sha256 $MSHA \
  --groups $SIGNER_GROUPS --limit $LIMIT --test-limit $TESTLIM --output-key $D/evaluations/cross-signer-baseline.json \
  || { echo R3B-DONE; exit 1; }
RUNS=""
for h in $HANDS; do for f in $FACES; do TAG=$RUN-sweep-h$h-f$f
  for s in 0 0.1 0.2 0.3; do for a in 1 0.8 0.6; do
    V=$TAG-hs$s$([ $a = 1 ] || echo -hd$a); RUNS="$RUNS $V $V-fc"; done; done; done; done
stage $RUN gate python scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $SIGNER_GROUPS,pool \
  --envelope-key $D/evaluations/cross-signer-baseline.json --output-key $D/evaluations/$RUN-gate.json \
  || { echo R3B-DONE; exit 1; }
stage $RUN strict-review python scripts/strict_prior_review.py --derived $D --gate-key $D/evaluations/$RUN-gate.json \
  --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY" \
  --retrieval-manifest-sha256 $FSHA --translator-manifest-sha256 $FSHA --groups $SIGNER_GROUPS \
  --heldout-corpus how2sign-two-view-text-v1 --second-translator-key "$KP2D" --word-seconds 0 \
  --envelope-key $D/evaluations/cross-signer-baseline.json --output-key $D/evaluations/$RUN-strict-review.json \
  --device cuda $QUICK
echo R3B-DONE
