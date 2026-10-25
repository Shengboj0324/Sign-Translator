#!/usr/bin/env bash
# round3d_gains.sh — round 3d amendment 3 (docs/NEXT_PHASE_PLAN.md): amendment 5's fallback for one qualifying
# candidate (env CAND, KEY = its checkpoint): fit per-channel gains on 300 How2Sign training clips, then rerun the
# recognisability check and the fidelity check (1000 clips, reliable-joint speeds) with the gains. Each writes its
# report; the orchestrator accepts the gains only if both pass. GAINS-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
ACQ=acquisition/2026-10-17-v1
MIX=$ACQ/youtube-asl-v1/round3c-mixed-v1/manifest.json MSHA=004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd
D=$ACQ/youtube-asl-v1/round3d GAINS=$D/tokenizer-gains-$CAND.json
# already done for this checkpoint (run early for one candidate): nothing to redo
if python - "$D" "$CAND" "$KEY" <<'PY'
import os, sys, json
from signtranslator.data.text_motion_dataset import R2Store
s, (D, n, key) = R2Store(os.environ), sys.argv[1:]
keys = [f"{D}/tokenizer-{c}-{n}-gained.json" for c in ("recognisability", "fidelity")]
sys.exit(0 if all(s.exists(k) and json.loads(s.get(k))["tokenizer_key"] == key for k in keys) else 1)
PY
then echo "gained reports for $CAND $KEY already exist"; echo GAINS-DONE; exit 0; fi
python scripts/prefetch_cache.py $MIX || { echo "states missing"; echo GAINS-DONE; exit 1; }
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
echo "STAGE fit $(date)"
python scripts/fit_tokenizer_gains.py --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$KEY" --limit 300 \
  --output-key $GAINS || { echo "FAILED fit"; echo GAINS-DONE; exit 1; }
echo "STAGE recognisability-gained $(date)"
python scripts/tokenizer_recognisability_check.py --manifest-key $MIX --manifest-sha256 $MSHA --corpus asllvd-text-v1 \
  --gloss-map-key $ACQ/derived/signbank-v1/asl-citizen-gloss-map.json --tokenizer-key "$KEY" --limit 300 --device cuda \
  --gains-key $GAINS --output-key $D/tokenizer-recognisability-$CAND-gained.json; echo "recognisability-gained status $?"
echo "STAGE fidelity-gained $(date)"
python scripts/tokenizer_fidelity_check.py --manifest-key $MIX --manifest-sha256 $MSHA --tokenizer-key "$KEY" \
  --translator-key "$KP2D" --translator-manifest-sha256 $KPSHA --limit 1000 --device cuda \
  --speed-joints-key $D/joint-reliability-how2sign-train.json --gains-key $GAINS \
  --output-key $D/tokenizer-fidelity-$CAND-gained.json; echo "fidelity-gained status $?"
echo GAINS-DONE
