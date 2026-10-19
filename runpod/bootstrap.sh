#!/usr/bin/env bash
# Prepare a Runpod pod: clone the code, install dependencies, fetch and verify models.
# Requires Runpod secrets exported as R2_ENDPOINT_URL, R2_BUCKET, AWS_ACCESS_KEY_ID,
# AWS_SECRET_ACCESS_KEY. Nothing here prints them.
set -euo pipefail
WORK=${WORK:-/workspace}
REPO=${REPO:-https://github.com/Shengboj0324/Sign-Translator.git}
REF=${REF:-main}
for v in R2_ENDPOINT_URL R2_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  [[ -n "${!v:-}" ]] || { echo "missing secret $v" >&2; exit 1; }
done
cd "$WORK"
[[ -d Sign-Translator ]] || git clone --depth 1 --branch "$REF" "$REPO"
cd Sign-Translator
python -c "import torch; assert torch.__version__.startswith('2.13'), torch.__version__; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install --quiet -r runpod/requirements-runpod.txt
# MediaPipe pulls the GUI build of OpenCV, which needs libGL; server images lack it.
pip uninstall -y -q opencv-contrib-python || true
pip install --quiet "opencv-contrib-python-headless==5.0.0.93"
pip install --quiet -e . --no-deps
mkdir -p "$WORK/models"
python - <<'PY'
import os
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.video_landmarks import MODEL_SHA256, verify_models
store, out = R2Store(os.environ), Path(os.environ.get("WORK", "/workspace")) / "models"
for name in MODEL_SHA256:
    path = out / name
    if not path.exists():
        path.write_bytes(store.get(f"models/mediapipe-tasks-0.10.21/{name}"))
verify_models(out)
import hashlib
sapiens = out / "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
if not sapiens.exists():
    sapiens.write_bytes(store.get("acquisition/2026-10-17-v1/models-sapiens/" + sapiens.name))
digest = hashlib.sha256()
with sapiens.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 24), b""):
        digest.update(chunk)
if digest.hexdigest() != "d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3":
    sapiens.unlink()
    raise SystemExit("Sapiens checkpoint hash mismatch; removed")
print("models verified (MediaPipe and Sapiens)")
PY
python -m pytest -q -x -p no:cacheprovider tests/test_generation_text_to_motion.py tests/test_train_text_to_motion_script.py tests/test_de_video_landmarks.py tests/test_de_two_view_geometry.py tests/test_de_two_view_states.py tests/test_de_sapiens_keypoints.py \
  tests/test_generation_diffusion.py tests/test_generation_motion_tokens.py tests/test_train_motion_tokens_script.py \
  tests/test_triangulate_how2sign_sessions_script.py tests/test_calibrate_how2sign_cameras_script.py
echo "bootstrap complete"
