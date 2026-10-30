#!/usr/bin/env bash
# uwb_train_regressors.sh — CPU pod: train the z-free world regressors for the UWB lifter (plan "UWB lifter: declared
# selection and gate", 2026-10-30 03:34 UTC): xy2 (primary, framing-invariant) and xy with framing augmentation
# (comparison only), on the SHA-pinned v2 pairs; stores both bundles and their training reports under
# acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1/ with an INDEX.json. UWB-REGRESSORS-DONE at the end.
set -euo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4
cd /root/work/Sign-Translator
W=/root/uwb; rm -rf $W && mkdir -p $W/out
python - "$W" <<'PY'
import os, sys, hashlib, tarfile
from signtranslator.data.text_motion_dataset import R2Store
s, W = R2Store(os.environ), sys.argv[1]
for name, sha in (("pairs.tar", "6c4d3ecc18e66f2bbc885aacda9a86499d7f5ce48618fa2438540ca958526758"),
                  ("legacy-pairs.tar", "7042e0a8bcd7452d13743fcac5f6b37be1854fafb06c417ab54fa8aa24c65fa0")):
    b = s.get(f"scratch/uwb-lifter-dev/world-pairs-v2/{name}")
    assert hashlib.sha256(b).hexdigest() == sha, name
    open(f"{W}/{name}", "wb").write(b); tarfile.open(f"{W}/{name}").extractall(W)
b = s.get("acquisition/2026-10-17-v1/sem-lex-mirror-v1/adapter-dev/world-regressors-v2/world-regressors-v2.pt")
assert hashlib.sha256(b).hexdigest() == "0953cd2cb2b48506808dff2d7199061eac3218ad555ea35327ff34edac3acab1"
open(f"{W}/v2.pt", "wb").write(b)
print("inputs verified", flush=True)
PY
T=scripts/train_world_regressors.py
python $T --pairs $W/pairs --legacy-dir $W/legacy-pairs --features xy2 --epochs 30 --seed 0 --val-signers h2s-1,h2s-2 \
  --baseline-bundle $W/v2.pt --out $W/out/world-regressors-xy2-v1.pt --report $W/out/world-regressors-xy2-v1-training.json
python $T --pairs $W/pairs --legacy-dir $W/legacy-pairs --features xy --augment-framing 3 --epochs 30 --seed 0 \
  --val-signers h2s-1,h2s-2 --baseline-bundle $W/v2.pt --out $W/out/world-regressors-xy-aug-v1.pt \
  --report $W/out/world-regressors-xy-aug-v1-training.json
[ "$(ls $W/out | wc -l)" -eq 4 ]
python scripts/store_local_files.py --dir $W/out --prefix acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1 \
  --source "trained on the v2 world pairs (How2Sign; validation signers 1, 2; signer 9 never), seed 0, 30 epochs" \
  --licence "derived from How2Sign (CC BY-NC 4.0) pairs; project-internal" --note "plan 2026-10-30 03:34 UTC declaration"
echo UWB-REGRESSORS-DONE
