#!/usr/bin/env bash
# uwb_downstream.sh — GPU pod (CPU workers for the lifting): the two UWB lifter gate rows uwb_lifter_gate.py does not
# compute (plan "UWB lifter: declared selection and gate", 2026-10-30 03:34 UTC, amendment 1 03:59, owner 04:02,
# selection inside 04:20), then the row-by-row verdict:
#   (A) the xy2 bundle's How2Sign-val median per gate group, by the gate's code path (uwb_how2sign_val_medians.py, on the
#       SHA-pinned v2 world pairs, validation signers 1 and 2; signer 9 never);
#   (B) the downstream row: back-translator-kp2d-uwb-ft-v1 reading lifted vs our states vs blank on the gate's pairs
#       (uwb_downstream_check.py, <= 2000 pairs, one per video first, seed 0);
#   uwb_gate_verdict.py over the gate report, (A), (B) and the visibility selection report.
# Usage: runpod/fleet/uwb_downstream.sh <gate report key>   (e.g. .../lifter-v1/gate-rerun/files/gate-xy2-v1.json)
# Outputs (never overwritten: each script refuses an existing key) under lifter-v1/gate-rerun/. UWB-DOWNSTREAM-DONE.
set -euo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece        # the back-translator (T5)
GATE=${1:?usage: uwb_downstream.sh <gate report key>}
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
K=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1
BUNDLE=$K/files/world-regressors-xy2-v1.pt; SHA=02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde
SEL=$K/selection/outside-frame-xy2-v1.json; SELSHA=3e26180a5f6510f68adecb5bbb3d977aa03d469f29930dac38919d8533590555
OUT=$K/gate-rerun
GSHA=$(sha "$GATE"); echo "GATE REPORT $GATE $GSHA"
[ "$(sha $BUNDLE)" = "$SHA" ] && [ "$(sha $SEL)" = "$SELSHA" ] || { echo "pin mismatch"; exit 1; }
W=/root/dstream; rm -rf $W && mkdir -p $W/out
python - "$W" <<'PY'
import os, sys, hashlib, io, tarfile
from signtranslator.data.text_motion_dataset import R2Store
s, W = R2Store(os.environ), sys.argv[1]
for name, sha in (("pairs.tar", "6c4d3ecc18e66f2bbc885aacda9a86499d7f5ce48618fa2438540ca958526758"),
                  ("legacy-pairs.tar", "7042e0a8bcd7452d13743fcac5f6b37be1854fafb06c417ab54fa8aa24c65fa0")):
    b = s.get(f"scratch/uwb-lifter-dev/world-pairs-v2/{name}")
    assert hashlib.sha256(b).hexdigest() == sha, name
    with tarfile.open(fileobj=io.BytesIO(b)) as t:          # our own pinned archive; signer 9 is never extracted
        t.extractall(W, members=[m for m in t.getmembers() if m.isfile() and "-9-rgb_front" not in m.name],
                     **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
b = s.get("acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1/files/world-regressors-xy2-v1.pt")
assert hashlib.sha256(b).hexdigest() == "02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde"
open(f"{W}/xy2.pt", "wb").write(b)
print("inputs verified", flush=True)
PY
python scripts/uwb_how2sign_val_medians.py --pairs-dir $W/pairs --legacy-dir $W/legacy-pairs \
  --regressors $W/xy2.pt --regressors-sha256 $SHA --visibility inside --candidate-visibility learned,inside,margin:0.1,margin:0.25 \
  --val-signers h2s-1,h2s-2 --out $W/out/how2sign-val-medians-v1.json --output-key $OUT/how2sign-val-medians-v1.json
python scripts/uwb_downstream_check.py --gate-report "$GATE@$GSHA" --regressors $BUNDLE --regressors-sha256 $SHA \
  --models-dir /root/work/models --pairs 2000 --seed 0 --bootstrap 2000 --device cuda \
  --out $W/out/downstream-v1.json --output-key $OUT/downstream-v1.json
python scripts/uwb_gate_verdict.py --gate-report "$GATE@$GSHA" --how2sign-val $W/out/how2sign-val-medians-v1.json \
  --downstream $W/out/downstream-v1.json --outside-frame "$SEL@$SELSHA" --out $W/out/gate-verdict-v1.json \
  --markdown $W/out/gate-verdict-v1.md --output-key $OUT/gate-verdict-v1.json
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" \
  $OUT/gate-verdict-v1.md $W/out/gate-verdict-v1.md
echo UWB-DOWNSTREAM-DONE
