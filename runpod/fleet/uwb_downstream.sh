#!/usr/bin/env bash
# uwb_downstream.sh — GPU pod (CPU workers for the lifting): the UWB lifter gate's downstream row, then the row-by-row
# verdict (plan "UWB lifter: declared selection and gate", 2026-10-30 03:34 UTC, amendment 1 03:59, owner 04:02,
# selection inside 04:20; review of d2591fa):
#   (B) back-translator-kp2d-uwb-ft-v1 reading lifted vs our states vs blank on the gate's pairs, pooled (<= 2000 pairs,
#       one per video first, seed 0) and on every UWB dev-partition pair (uwb_downstream_check.py);
#   uwb_gate_verdict.py over the gate report, the precomputed How2Sign-val medians (A: uwb_how2sign_val_medians.py, run
#   locally; reports/uwb-how2sign-val-medians-v1.json, uploaded and SHA-pinned here), (B) and the selection report.
# Usage: runpod/fleet/uwb_downstream.sh <gate report key> <How2Sign-val medians key>
#   e.g. .../lifter-v1/gate-rerun/files/gate-xy2-v1.json  .../lifter-v1/gate-rerun/how2sign-val-medians-v1.json
# Outputs under lifter-v1/gate-rerun/ (each script refuses an existing key). UWB-DOWNSTREAM-DONE at the end.
set -euo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece        # the back-translator (T5)
GATE=${1:?usage: uwb_downstream.sh <gate report key> <How2Sign-val medians key>}
H2S=${2:?usage: uwb_downstream.sh <gate report key> <How2Sign-val medians key>}
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
K=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1
BUNDLE=$K/files/world-regressors-xy2-v1.pt; SHA=02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde
SEL=$K/selection/outside-frame-xy2-v1.json; SELSHA=3e26180a5f6510f68adecb5bbb3d977aa03d469f29930dac38919d8533590555
H2SSHA=751e4abc1bf17bc358081e0d096f9ee05b3f6ffb27d279afa8144a91f90426fa      # reports/uwb-how2sign-val-medians-v1.json
KPMANIFEST=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2   # as round17_pt.sh:36 (round3b-mixed-v2)
KPSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8   # streamed sha256 (2026-10-30) of training-runs/back-translator-kp2d-uwb-ft-v1/checkpoints/step-00009000.pt
OUT=$K/gate-rerun
GSHA=$(sha "$GATE"); echo "GATE REPORT $GATE $GSHA"
[ "$(sha $SEL)" = "$SELSHA" ] && [ "$(sha "$H2S")" = "$H2SSHA" ] || { echo "pin mismatch"; exit 1; }
W=/root/dstream; rm -rf $W && mkdir -p $W/out
python - "$W" "$BUNDLE" "$SHA" <<'PY'
import os, sys, hashlib
from signtranslator.data.text_motion_dataset import R2Store
W, key, sha = sys.argv[1:4]
b = R2Store(os.environ).get(key)
assert hashlib.sha256(b).hexdigest() == sha, key
open(f"{W}/xy2.pt", "wb").write(b)                          # every lifting worker loads this local copy
print("bundle verified", flush=True)
PY
python scripts/uwb_downstream_check.py --gate-report "$GATE@$GSHA" --regressors $W/xy2.pt --regressors-sha256 $SHA \
  --instrument-manifest-sha256 $KPMANIFEST --instrument-sha256 $KPSHA \
  --models-dir /root/work/models --pairs 2000 --seed 0 --bootstrap 2000 --device cuda \
  --out $W/out/downstream-v1.json --output-key $OUT/downstream-v1.json
python scripts/uwb_gate_verdict.py --gate-report "$GATE@$GSHA" --how2sign-val "$H2S@$H2SSHA" \
  --downstream $W/out/downstream-v1.json --outside-frame "$SEL@$SELSHA" --out $W/out/gate-verdict-v1.json \
  --markdown $W/out/gate-verdict-v1.md --output-key $OUT/gate-verdict-v1.json --markdown-key $OUT/gate-verdict-v1.md
echo UWB-DOWNSTREAM-DONE
