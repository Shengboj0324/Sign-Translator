#!/usr/bin/env bash
# Pod setup without git: code from an uploaded archive, torch 2.13 to match the repo, models from R2.
set -euo pipefail
source /root/r2env.sh
W=/root/work; mkdir -p $W/models; cd $W
rm -rf Sign-Translator && mkdir Sign-Translator && tar xzf /root/code.tgz -C Sign-Translator
cd Sign-Translator
# Python 3.12 venv (numpy 2.5 needs it), torch 2.13 for CUDA 13 to match the repo's pins.
pip install -q uv
[ -x /root/venv/bin/python ] || uv venv -q -p 3.12 /root/venv
export VIRTUAL_ENV=/root/venv PATH=/root/venv/bin:$PATH
python -c "import torch" 2>/dev/null || uv pip install -q "torch==2.13.*" --index-url https://download.pytorch.org/whl/cu130
uv pip install -q -r runpod/requirements-runpod.txt
uv pip uninstall -q opencv-contrib-python opencv-python || true
uv pip install -q "opencv-contrib-python-headless==5.0.0.93" || uv pip install -q opencv-contrib-python-headless
uv pip install -q -e . --no-deps
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
WORK=$W python - <<'PY'
import os, hashlib
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.video_landmarks import MODEL_SHA256, verify_models
store, out = R2Store(os.environ), Path(os.environ["WORK"]) / "models"
for name in MODEL_SHA256:
    path = out / name
    if not path.exists():
        path.write_bytes(store.get(f"models/mediapipe-tasks-0.10.21/{name}"))
verify_models(out)
sapiens = out / "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
if not sapiens.exists():
    sapiens.write_bytes(store.get("acquisition/2026-10-17-v1/models-sapiens/" + sapiens.name))
d = hashlib.sha256()
with sapiens.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""): d.update(chunk)
assert d.hexdigest() == "d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3", "sapiens hash"
print("models verified")
PY
echo SETUP-DONE
