#!/usr/bin/env bash
# round3c_tokenizer_v.sh — round 3c's fourth tokenizer candidate, s8v, on its own GPU pod (amendment declared 2026-10-25
# 02:45 PDT, before any round-3c tokenizer verdict; docs/NEXT_PHASE_PLAN.md "Round 3c amendment 2").
#   s8v: the s8 recipe (8 residual stages) plus per-channel velocity matching (--velocity-weight 1.0 on body, both hands,
#   head rotation and head translation), judged by the same three acceptance checks as s6/s8/c1024.
# Precedence (declared): if any of s6, s8, c1024 passes, round3c_prior.sh uses it and this pod trains no prior. Only if
# all three fail and s8v passes does this pod train the round-3c prior on s8v (the same prior recipe).
# R3C_SMOKE=1: toy sizes into -smoke runs, no prior verdict. TOKV-DONE at the end.
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
[ "$MSHA" = 004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd ] || { echo "mix SHA mismatch"; echo TOKV-DONE; exit 1; }
D=$ACQ/youtube-asl-v1/round3c RUN=tokens-rvq-d0b-prior-r3c
if [ "${R3C_SMOKE:-0}" = 1 ]; then D=$D-smoke RUN=$RUN-smoke TSTEPS=200 TEVERY=100 TLIMIT=24 FLIMIT=24 SFX=-smoke PSTEPS=300 PEVERY=100
else TSTEPS=40000 TEVERY=1000 TLIMIT=300 FLIMIT=300 SFX= PSTEPS=60000 PEVERY=1000; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing or not matching the manifest"; echo TOKV-DONE; exit 1; }
GLOSS_MAP=$ACQ/derived/signbank-v1/asl-citizen-gloss-map.json
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
name=s8v run=tokens-rvq-d0b-r3c-s8v$SFX
stage $run tokenizer python scripts/train_motion_tokens.py --architecture rvq --stage tokenizer --run $run \
  --manifest-key $MIX --manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --hidden 512 \
  --code-dim 128 --codebook 512 --stages 8 --down-layers 0 --base-blocks 4 --steps $TSTEPS --batch-size 64 \
  --lr 3e-4 --warmup 1000 --val-every $TEVERY --val-batches 32 --checkpoint-every $((TEVERY * 2)) --patience 5 \
  --velocity-weight body=1 --velocity-weight left_hand=1 --velocity-weight right_hand=1 \
  --velocity-weight head_rotation=1 --velocity-weight head_translation=1 \
  --workers 10 --cache-bytes 107374182400 || { echo TOKV-DONE; exit 1; }
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
      --output-key $D/tokenizer-fidelity-$name.json ;;
  esac
  rc=$?
  [ $rc = 0 ] && continue
  [ $rc = 4 ] && { verdict="REJECTED ($check)"; continue; }
  echo "TOKENIZER-ERROR $name ($check, status $rc)"; echo TOKV-DONE; exit 1
done
[ "$verdict" = PASS ] && echo "TOKENIZER-PASS $name $key" || echo "TOKENIZER-$verdict $name"
[ "${R3C_SMOKE:-0}" = 1 ] && { echo "smoke: s8v plumbing done"; echo TOKV-DONE; exit 0; }
# precedence: wait for the original three candidates' verdicts (every check writes its JSON report)
original() { python - "$D" <<'EOF'
import os, sys, json
from signtranslator.data.text_motion_dataset import R2Store
s, D = R2Store(os.environ), sys.argv[1]
state = {}
for n in ("c1024", "s6", "s8"):
    keys = [f"{D}/tokenizer-{c}-{n}.json" for c in ("domain-check", "recognisability", "fidelity")]
    found = [json.loads(s.get(k))["pass"] for k in keys if s.exists(k)]
    state[n] = "fail" if False in found else ("pass" if len(found) == 3 else "pending")
print(" ".join(f"{n}={v}" for n, v in state.items()))
EOF
}
for i in $(seq 1 480); do st=$(original); echo "$st" | grep -q pending || break; sleep 60; done
echo "original candidates: $st"
if echo "$st" | grep -q "=pass"; then echo "an original candidate passed: it is used; s8v trains no prior"; echo TOKV-DONE; exit 0; fi
echo "$st" | grep -q pending && { echo "original candidates still pending after 8 h: no prior from s8v"; echo TOKV-DONE; exit 1; }
[ "$verdict" = PASS ] || { echo "TOKENIZER-FAILED: no round-3c candidate passed its acceptance"; echo TOKV-DONE; exit 1; }
echo "tokenizer chosen: s8v $key"
TSHA=$(sha "$key")
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --stage prior --run $RUN \
  --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$key" --tokenizer-sha256 "$TSHA" \
  --tokenizer-manifest-sha256 $MSHA --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --steps $PSTEPS \
  --lr 3e-4 --warmup 1000 --batch-size 64 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --corpus-weight asllvd-text-v1=3 --val-every $PEVERY --val-batches 32 \
  --checkpoint-every $((PEVERY * 2)) --patience 5 --workers 10 --cache-bytes 107374182400
echo PRIOR-DONE
echo TOKV-DONE
