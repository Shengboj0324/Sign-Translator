#!/usr/bin/env bash
# round3b_sweep_helper.sh — a second GPU pod for round 3b's sweep (runpod/fleet/round3b.sh, unchanged): it evaluates
# the helper's share with exactly the chain's arguments and output keys, 3 at a time, skipping any already stored:
# hands 1.3 x every face (9) and hands 1.2 x face 1.2 (3). The main pod's chain stops its own copies of these while
# its xargs sweep runs (round3b_helper_guard.sh) and finds them stored at its rerun step, which re-evaluates anything
# still missing, so no evaluation is lost if this pod fails. HELPER-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name-helper.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
export -f sha best upload exists stage
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3b-mixed-v2/manifest.json; MSHA=$(sha $MIX)
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha $FULL)
D=$ACQ/youtube-asl-v1/round3b RUN=tokens-rvq-d0b-prior-r3b LIMIT=300
GEN=$(best $RUN) RKEY=$(best retrieval-hold-full-v1)
export MIX MSHA FSHA D RUN LIMIT GEN RKEY
sweep_one() { local h=$1 f=$2 g=$3; local TAG=$RUN-sweep-h$h-f$f       # round3b.sh's sweep_one, verbatim arguments
  exists $D/evaluations/$TAG-$g.json && { echo "stored: $TAG-$g"; return 0; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 stage $RUN eval-h$h-f$f-$g \
    python scripts/evaluate_text_to_motion.py --manifest-key $MIX --manifest-sha256 $MSHA --checkpoint-key "$GEN" \
    --partition val --corpus how2sign-two-view-text-v1 --signer-component $g --limit $LIMIT --guidance 1,1.5,2 --fps 24 \
    --device cuda --batch 32 --part-temperature body=$h,left_hand=$h,right_hand=$h,face=$f --retrieval-key "$RKEY" \
    --retrieval-manifest-sha256 $FSHA --save-generated $D/generated/$TAG-$g --output-key $D/evaluations/$TAG-$g.json; }
export -f sweep_one
{ for f in 0.8 1.0 1.2; do for g in c01 c03 c05; do echo "1.3 $f $g"; done; done
  for g in c01 c03 c05; do echo "1.2 1.2 $g"; done; } | xargs -P 3 -L 1 bash -c 'sweep_one "$@"' _
echo HELPER-DONE
