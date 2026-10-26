#!/usr/bin/env bash
# round5_recognizer.sh — round 5 instrument: sign-recognizer-v3 = v2's recipe exactly (same arguments) on the
# right-dominant canonical ASL Citizen (CMAN@CSHA) plus the ASLLVD train signers, then v2's validation unchanged.
# The two extra round-5 checks (canonical val top-5 >= 0.70, ASLLVD val top-1 >= 0.58) are read from validation.json.
set -euo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
A=acquisition/2026-10-17-v1/derived
LVD=$A/asllrp-dai2-v1/asllvd-sign-manifest-v1/manifest.json LSHA=86df1dfd00df235c7df2b0feaa021e618d9bfc591e13810153683475d9961842
RUN=${RUN:-sign-recognizer-v3}
echo "STAGE prefetch $(date)"
python scripts/prefetch_cache.py $CMAN $LVD
echo "STAGE train $RUN $(date)"
python scripts/train_sign_recognizer.py --manifest-key $CMAN --manifest-sha256 $CSHA --extra-manifest-key $LVD \
  --extra-manifest-sha256 $LSHA --fps 24 --augment --run $RUN --workers 12 --cache-bytes 12884901888 ${STEPS:+--steps $STEPS}
echo "STAGE validate $RUN $(date)"
python scripts/validate_sign_recognizer.py --run $RUN --citizen-manifest-key $CMAN --citizen-manifest-sha256 $CSHA \
  --asllvd-manifest-key $LVD --asllvd-manifest-sha256 $LSHA --fps 24 --cross-partitions val \
  --continuous-manifest-key $A/asllrp-dai2-v1/gloss-manifest-v1/manifest.json \
  --continuous-manifest-sha256 f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944 \
  --output-key training-runs/$RUN/validation.json
echo "ROUND5-RECOGNIZER-DONE $(date)"
