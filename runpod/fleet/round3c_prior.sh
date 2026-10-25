#!/usr/bin/env bash
# round3c_prior.sh — generator round 3c, stages A (tokenizer) and B (prior), one GPU pod. Design declared 2026-10-25
# 00:05 PDT; tokenizer stage added 00:20 PDT (docs/NEXT_PHASE_PLAN.md, "Round 3c amendment"), both before any
# round-3c result. The evaluation stages run as separate jobs afterwards, so the GPU pod is never held for CPU work.
#   A. three tokenizer candidates trained in parallel on the round-3c mix (round 3a's recipe; 6 or 8 residual stages,
#      or a 1024-entry codebook), each judged by the domain check, the recognisability round trip and the fidelity
#      check (tokenizer_fidelity_check.py: speed and variation ratios in [0.85, 1.15], R2b round trip above blank);
#      the passing candidate with the fewest codes per step is used; if none passes, the round stops (TOKENIZER-FAILED);
#   B. prior from scratch on round3c-mixed-v1 (no ASL Citizen motion, ASLLVD x3), round 3b's architecture and
#      schedule, <= 60,000 steps, early stop.
# R3C_SMOKE=1: toy sizes into -smoke runs (plumbing only, never a verdict). PRIOR-DONE at the end.
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
RUN=tokens-rvq-d0b-prior-r3c D=$ACQ/youtube-asl-v1/round3c
if [ "${R3C_SMOKE:-0}" = 1 ]; then RUN=$RUN-smoke D=$D-smoke PSTEPS=300 PEVERY=100 TSTEPS=200 TEVERY=100 TLIMIT=24 FLIMIT=24 SFX=-smoke
else PSTEPS=60000 PEVERY=1000 TSTEPS=40000 TEVERY=1000 TLIMIT=300 FLIMIT=300 SFX=; fi
export MIX MSHA D TLIMIT FLIMIT TSTEPS TEVERY SFX
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; echo PRIOR-DONE; exit 1; }
GLOSS_MAP=$ACQ/derived/signbank-v1/asl-citizen-gloss-map.json
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
export GLOSS_MAP KP2D KPSHA
# A. tokenizer candidates: name stages codebook (codes per step = 4 parts x stages)
tok_one() { local name=$1 stages=$2 book=$3; local run=tokens-rvq-d0b-r3c-$name$SFX
  stage $run tokenizer python scripts/train_motion_tokens.py --architecture rvq --stage tokenizer --run $run \
    --manifest-key $MIX --manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --hidden 512 \
    --code-dim 128 --codebook $book --stages $stages --down-layers 0 --base-blocks 4 --steps $TSTEPS --batch-size 64 \
    --lr 3e-4 --warmup 1000 --val-every $TEVERY --val-batches 32 --checkpoint-every $((TEVERY * 2)) --patience 5 \
    --workers 6 --cache-bytes 107374182400 || return 1
  local key rc verdict=PASS; key=$(best $run)
  # each check exits 0 (pass), 4 (a failing verdict: the candidate is rejected) or anything else (an error)
  for check in domain recognisability fidelity; do
    case $check in
      domain) stage $run domain-check python scripts/tokenizer_domain_check.py --manifest-key $MIX --manifest-sha256 $MSHA \
        --conform-to-corpus how2sign-two-view-text-v1 --corpus youtube-asl-v1 --corpus asllvd-text-v1 --limit $TLIMIT \
        --tokenizer-key "$key" --output-key $D/tokenizer-domain-check-$name.json ;;
      recognisability) stage $run recognisability python scripts/tokenizer_recognisability_check.py --manifest-key $MIX \
        --manifest-sha256 $MSHA --corpus asllvd-text-v1 --gloss-map-key $GLOSS_MAP --tokenizer-key "$key" --limit $TLIMIT \
        --device cuda --output-key $D/tokenizer-recognisability-$name.json ;;
      fidelity) stage $run fidelity python scripts/tokenizer_fidelity_check.py --manifest-key $MIX --manifest-sha256 $MSHA \
        --tokenizer-key "$key" --translator-key "$KP2D" --translator-manifest-sha256 $KPSHA --limit $FLIMIT --device cuda \
        --output-key $D/tokenizer-fidelity-$name.json ;;
    esac
    rc=$?
    [ $rc = 0 ] && continue
    [ $rc = 4 ] && { verdict="REJECTED ($check)"; continue; }
    echo "TOKENIZER-ERROR $name ($check, status $rc)"; return 1
  done
  [ "$verdict" = PASS ] && echo "TOKENIZER-PASS $name $key" || echo "TOKENIZER-$verdict $name"; }
export -f sha best upload stage
export -f tok_one
printf "s6 6 512\ns8 8 512\nc1024 4 1024\n" | xargs -P 3 -L 1 bash -c 'tok_one "$@"' _ > /root/tokenizers.out 2>&1
cat /root/tokenizers.out
CHOSEN=""
for name in c1024 s6 s8; do                      # fewest codes per step first: 16, 24, 32
  line=$(grep "^TOKENIZER-PASS $name " /root/tokenizers.out) && { CHOSEN=$name TOK=$(echo "$line" | cut -d" " -f3); break; }
done
if [ -z "$CHOSEN" ]; then
  if [ "${R3C_SMOKE:-0}" = 1 ]; then      # smoke: toy tokenizers never pass; take any trained one to test the prior plumbing
    TOK=$(best tokens-rvq-d0b-r3c-c1024$SFX) CHOSEN=c1024-smoke
  else echo "TOKENIZER-FAILED: no candidate passed its acceptance"; echo PRIOR-DONE; exit 1; fi
fi
echo "tokenizer chosen: $CHOSEN $TOK"
TSHA=$(sha "$TOK") R3A_SHA=$MSHA
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $R3A_SHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --corpus-weight asllvd-text-v1=3 --val-every $PEVERY --val-batches 32 \
  --checkpoint-every $((PEVERY * 2)) --patience 5 --workers 10 --cache-bytes 107374182400 || { echo PRIOR-DONE; exit 1; }
echo PRIOR-DONE
