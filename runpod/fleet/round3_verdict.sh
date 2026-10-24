#!/usr/bin/env bash
# round3_verdict.sh — round 3 verdict only: the union gate and the union strict review of round3_supplement.sh, rerun
# (postprocess.damp_rotation_channel, factors 0.8 and 0.6) joins the post-processing grid, against the full prior's
# R6 failure on an unseen signer (nod rate 19.8x, head variation 2.3x). Runs after round3_prior.sh on its stored
# generations: grid of the damped variants per validation signer, the multi-signer gate over every candidate (stage B's
# plus these), and the strict review on that union selection, which is round 3's verdict (stage B's own strict review
# on the undamped grid is reported as preliminary). SUPP-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece   # R2b's 2D back-translator is T5-based
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
RUN=tokens-rvq-d0b-prior-r3a SIGNER_GROUPS=c01,c03,c05
GEN=$(best $RUN) RKEY=$(best retrieval-hold-full-v1) BKEY=$(best back-translator-full-v1)
KP2D=$(best back-translator-kp2d-uwb-ft-v1)    # R2b: the accepted 2D back-translator (declared 2026-10-23 17:35 PDT)
# (the supplement's union gate read an evaluation a moment before the last grid's writes landed; every candidate's
# reports exist now). Same runs, groups, instruments and output keys as the supplement. VERDICT-DONE at the end.
RUNS=""
for h in 1.0 1.2 1.4; do for f in 0.8 1.0 1.2; do TAG=$RUN-sweep-h$h-f$f; RUNS="$RUNS $TAG"
  for s in 0 0.1 0.2 0.3; do RUNS="$RUNS $TAG-hs$s $TAG-hs$s-fc"
    for a in 0.8 0.6; do RUNS="$RUNS $TAG-hs$s-hd$a $TAG-hs$s-hd$a-fc"; done; done; done; done
stage $RUN gate-union python scripts/multi_signer_gate.py --derived $D --runs $RUNS --groups $SIGNER_GROUPS \
  --output-key $D/evaluations/$RUN-gate-union.json || { echo VERDICT-DONE; exit 1; }
stage $RUN strict-review-union python scripts/strict_prior_review.py --derived $D --gate-key $D/evaluations/$RUN-gate-union.json \
  --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" --retrieval-key "$RKEY" --translator-key "$BKEY" \
  --retrieval-manifest-sha256 $FSHA --translator-manifest-sha256 $FSHA --groups $SIGNER_GROUPS \
  --heldout-corpus how2sign-two-view-text-v1 --second-translator-key "$KP2D" \
  --output-key $D/evaluations/$RUN-strict-review-union.json --device cuda
echo VERDICT-DONE
