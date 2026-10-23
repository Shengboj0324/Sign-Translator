#!/usr/bin/env bash
# uwb_keypoint_shards.sh — CPU pod, after acquire_uwb_keypoints.sh: the 10 UWB zips -> 48-point layout shards
# (scripts/build_uwb_keypoint_shards.py; one worker per zip, ranged reads from R2), with How2Sign and ASLLRP val/test
# sentences excluded. SHARDS-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
ACQ=acquisition/2026-10-17-v1/derived
FULL=$ACQ/full-v1/how2sign-text-manifest-v1/manifest.json
PLAN=$ACQ/asllrp-dai2-v1/plan-manifest-v2/manifest.json
python scripts/build_uwb_keypoint_shards.py --workers 10 --eval-manifest "$FULL@$(sha $FULL)" --eval-manifest "$PLAN@$(sha $PLAN)"
echo "SHARDS-DONE $?"
