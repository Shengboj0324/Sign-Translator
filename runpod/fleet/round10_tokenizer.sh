#!/usr/bin/env bash
# round10_tokenizer.sh — round 10 coarse tokenizers (declared 2026-10-27 18:10 UTC): round 3d's tokenizer stage and checks with DOWN and STAGES set by the candidate.
# design", declared 2026-10-25 08:20 PDT before any round-3d result): train, then the domain, recognisability and
# fidelity checks (fidelity with reliable-joint speeds and R2b on 1000 clips), each writing its JSON report. No prior
# here and no waiting: the prior is launched separately once every candidate has its verdict (precedence: the passing
# candidate with the highest R2b round-trip BLEU). Env: CAND (name), TFLAGS (extra trainer flags), TSTEPS, TEVERY.
# R3D_SMOKE=1: toy sizes into -smoke runs. TOK10-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece   # the fidelity check's T5 translator (as round3b.sh)
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3c-mixed-v1/manifest.json; MSHA=$(sha $MIX)
[ "$MSHA" = 004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd ] || { echo "mix SHA mismatch"; echo TOK10-DONE; exit 1; }
D=$ACQ/youtube-asl-v1/round10
JOINTS=$ACQ/youtube-asl-v1/round3d/joint-reliability-how2sign-train.json
if [ "${R3D_SMOKE:-0}" = 1 ]; then D=$D-smoke SFX=-smoke STEPS=200 EVERY=100 TLIMIT=24 FLIMIT=24
else SFX= STEPS=${TSTEPS:-100000} EVERY=${TEVERY:-1000} TLIMIT=300 FLIMIT=1000; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; echo TOK10-DONE; exit 1; }
GLOSS_MAP=$ACQ/derived/signbank-v1/asl-citizen-gloss-map.json
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
name=$CAND run=tokens-rvq-r10-$CAND$SFX
# CHECKS_ONLY=1: a run that stopped before its checks (d4: the spike guard's divergence stop, 2026-10-25 11:59 PDT)
# is judged on its best checkpoint, as every early-stopped candidate is
[ "${CHECKS_ONLY:-0}" = 1 ] || stage $run tokenizer python scripts/train_motion_tokens.py --architecture rvq --stage tokenizer --run $run \
  --manifest-key $MIX --manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --hidden 512 \
  --code-dim 128 --codebook 512 --stages ${STAGES:?} --down-layers ${DOWN:?} --base-blocks 4 --steps $STEPS --batch-size 64 \
  --lr 3e-4 --warmup 1000 --val-every $EVERY --val-batches 32 --checkpoint-every $((EVERY * 2)) --patience 10 \
  --codebook-follows-lr --workers 10 --cache-bytes 107374182400 ${TFLAGS:-} || { echo TOK10-DONE; exit 1; }
key=$(best $run); verdict=PASS
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
      --speed-joints-key $JOINTS --output-key $D/tokenizer-fidelity-$name.json ;;
  esac
  rc=$?
  [ $rc = 0 ] && continue
  [ $rc = 4 ] && { verdict="REJECTED ($check)"; continue; }
  echo "TOKENIZER-ERROR $name ($check, status $rc)"; echo TOK10-DONE; exit 1
done
[ "$verdict" = PASS ] && echo "TOKENIZER-PASS $name $key" || echo "TOKENIZER-$verdict $name"
[ "${R3D_SMOKE:-0}" = 1 ] && echo "smoke: $name plumbing done"
echo TOK10-DONE
