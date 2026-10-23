#!/usr/bin/env bash
# round3_supplement.sh — round 3, declared 2026-10-23 11:50 PDT before any round-3 result: head-excursion damping
# (postprocess.damp_rotation_channel, factors 0.8 and 0.6) joins the post-processing grid, against the full prior's
# R6 failure on an unseen signer (nod rate 19.8x, head variation 2.3x). Runs after round3_prior.sh on its stored
# generations: grid of the damped variants per validation signer, the multi-signer gate over every candidate (stage B's
# plus these), and the strict review on that union selection, which is round 3's verdict (stage B's own strict review
# on the undamped grid is reported as preliminary). SUPP-DONE at the end.
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
RUN=tokens-rvq-d0b-prior-r3a GROUPS=c01,c03,c05
GEN=$(best $RUN) RKEY=$(best retrieval-hold-full-v1) BKEY=$(best back-translator-full-v1)
export MIX MSHA D RUN
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
# 0. repair stage B: a sweep evaluation that failed there (4 concurrent evaluations ran a 44 GB GPU out of memory at
#    h1.2-f1.2-c01) is rerun here with the same arguments, one at a time, and a setting whose undamped grid is
#    incomplete is re-scored, so the union gate sees every declared candidate on every signer.
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do TAG=$RUN-sweep-h$h-f$f; redo=0
  for g in c01 c03 c05; do
    exists $D/evaluations/$TAG-$g.json && continue
    redo=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 stage $RUN eval-h$h-f$f-$g \
      python scripts/evaluate_text_to_motion.py --manifest-key $MIX \
      --manifest-sha256 $MSHA --checkpoint-key "$GEN" --partition val --corpus how2sign-two-view-text-v1 \
      --signer-component $g --limit 300 --guidance 1,1.5,2 --fps 24 --device cuda --batch 32 \
      --part-temperature body=$h,left_hand=$h,right_hand=$h,face=$f --retrieval-key "$RKEY" --retrieval-manifest-sha256 $FSHA \
      --save-generated $D/generated/$TAG-$g --output-key $D/evaluations/$TAG-$g.json
  done
  for s in 0 0.1 0.2 0.3; do for g in c01 c03 c05; do
    exists $D/evaluations/$TAG-hs$s-fc-$g.json || redo=1; done; done
  if [ $redo = 1 ]; then
    OMP_NUM_THREADS=8 stage $RUN grid-h$h-f$f python scripts/score_postprocessing_grid.py --manifest-key $MIX \
      --manifest-sha256 $MSHA --run $TAG --groups c01,c03,c05 --generated-template "$D/generated/{run}-{group}" \
      --evaluations $D/evaluations --head-smoothing 0,0.1,0.2,0.3
  fi
done; done
grid_one() { local TAG=$RUN-sweep-h$1-f$2
  OMP_NUM_THREADS=1 stage $RUN grid-hd-h$1-f$2 python scripts/score_postprocessing_grid.py --manifest-key $MIX --manifest-sha256 $MSHA \
    --run $TAG --groups c01,c03,c05 --generated-template "$D/generated/{run}-{group}" --evaluations $D/evaluations \
    --head-smoothing 0,0.1,0.2,0.3 --head-damping 0.8,0.6; }
export -f grid_one
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do echo "$h $f"; done; done | xargs -P 9 -L 1 bash -c 'grid_one "$@"' _
RUNS=""
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do TAG=$RUN-sweep-h$h-f$f; RUNS="$RUNS $TAG"
  for s in 0 0.1 0.2 0.3; do RUNS="$RUNS $TAG-hs$s $TAG-hs$s-fc"
    for a in 0.8 0.6; do RUNS="$RUNS $TAG-hs$s-hd$a $TAG-hs$s-hd$a-fc"; done; done; done; done
stage $RUN gate-union python scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $GROUPS \
  --output-key $D/evaluations/$RUN-gate-union.json
stage $RUN strict-review-union python scripts/strict_prior_review.py --derived $D --gate-key $D/evaluations/$RUN-gate-union.json \
  --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY" \
  --retrieval-manifest-sha256 $FSHA --translator-manifest-sha256 $FSHA --groups $GROUPS \
  --heldout-corpus how2sign-two-view-text-v1 --output-key $D/evaluations/$RUN-strict-review-union.json --device cuda
echo SUPP-DONE
