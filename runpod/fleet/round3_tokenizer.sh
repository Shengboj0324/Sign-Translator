#!/usr/bin/env bash
# round3_tokenizer.sh — generator round 3, stage A, one GPU pod: the motion tokenizer trained on the round-3 mix
# (How2Sign multi-signer train, YouTube-ASL, ASL Citizen one-word signs, ASLLRP plans; all conformed to How2Sign's
# layouts at 24 fps). The frozen full-corpus tokenizer encodes ASLLRP face/blink 1.5-2.3x worse than How2Sign
# (tokenizer domain check, 2026-10-22), so round 3 does not reuse it.
#   1. domain check of the full-corpus tokenizer on YouTube-ASL, ASL Citizen and ASLLRP (recorded: the "before");
#   2. tokenizer training on the mix (same architecture as tokens-rvq-d0b-full-v1; 40k steps, early stop);
#   3. the same domain check with the new tokenizer: its acceptance (every channel <= 1.5x How2Sign val, every corpus).
# TOK-DONE at the end.
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
MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round3a-mixed-v1/manifest.json; MSHA=$(sha $MIX)
[ "$MSHA" = dc576f4aff4bef38b933b3646670d98c3160e2474c80fb0719d91631ce00d92c ] || { echo "mix manifest changed"; echo TOK-DONE; exit 1; }
D=acquisition/2026-10-17-v1/youtube-asl-v1/round3a
RUN=tokens-rvq-d0b-r3a
CHECK="--manifest-key $MIX --manifest-sha256 $MSHA --conform-to-corpus how2sign-two-view-text-v1 --corpus youtube-asl-v1 --corpus asl-citizen-text-v1 --corpus asllrp-dai2-plan-v2 --limit 120"
python scripts/prefetch_cache.py $MIX
stage $RUN domain-check-before python scripts/tokenizer_domain_check.py $CHECK --tokenizer-key "$(best tokens-rvq-d0b-full-v1)" \
  --output-key $D/tokenizer-domain-check-full-v1.json
stage $RUN tokenizer python scripts/train_motion_tokens.py --architecture rvq --stage tokenizer --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 \
  --hidden 512 --code-dim 128 --codebook 512 --stages 4 --down-layers 0 --base-blocks 4 \
  --steps 40000 --batch-size 64 --lr 3e-4 --warmup 1000 --val-every 1000 --val-batches 32 --checkpoint-every 2000 \
  --patience 5 --workers 10 --cache-bytes 107374182400 || { echo TOK-DONE; exit 1; }
stage $RUN domain-check-after python scripts/tokenizer_domain_check.py $CHECK --tokenizer-key "$(best $RUN)" \
  --output-key $D/tokenizer-domain-check-r3a.json
echo "tokenizer acceptance status $?"
echo TOK-DONE
