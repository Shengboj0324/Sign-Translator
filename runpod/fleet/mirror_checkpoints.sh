#!/usr/bin/env bash
# Copy checkpoints from R2 to the network volume (usage: mirror_checkpoints.sh KEY [KEY ...]) on a pod with the volume
# mounted at /workspace. Each object lands at /workspace/sign-translator/checkpoints/<key>, is re-hashed from disk, and
# must equal the SHA-256 of the R2 bytes; a manifest of (key, bytes, sha256) is written beside them. Only the
# sign-translator/ subdirectory is touched (the volume also holds another project's files).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
OUT=/workspace/sign-translator/checkpoints
mkdir -p "$OUT"
python - "$OUT" "$@" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
out, keys = Path(sys.argv[1]), sys.argv[2:]
store, rows = R2Store(os.environ), []
for key in keys:
    data = store.get(key)
    digest = hashlib.sha256(data).hexdigest()
    path = out / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"mismatch after write: {key}")
    rows.append(dict(key=key, bytes=len(data), sha256=digest))
    print("mirrored", key, len(data), digest[:12], flush=True)
manifest = out / "MANIFEST.json"
old = json.loads(manifest.read_text()) if manifest.exists() else []
merged = {r["key"]: r for r in old + rows}
manifest.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["key"]), indent=2))
PY
echo MIRROR-DONE
