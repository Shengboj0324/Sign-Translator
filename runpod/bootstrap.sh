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
print("models verified")
PY
python -m pytest -q -x -p no:cacheprovider tests/test_generation_text_to_motion.py tests/test_train_text_to_motion_script.py tests/test_de_video_landmarks.py
echo "bootstrap complete"
