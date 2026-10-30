#!/bin/bash
# round18_pilot.sh — GPU pod: R18's pilot and verdict (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test",
# declared 2026-10-30 13:03 UTC). The pilot is R16C's command (job_r16c-pilot.sh: 1,000 How2Sign val clips,
# instrument v1, recogniser v3d, word list round14/r5-word-split.json, the same seeds) with the checkpoint changed, and
# with the two keys that cannot stay R16C's:
#   - --mix-key/--mix-sha256 name round18-mixed-v1, because load_generator refuses a checkpoint trained on another
#     manifest than --mix-sha256; the How2Sign records are byte-identical to round14b-mixed-v1's and the builder
#     refuses a mix whose 1,000-clip pilot sample differs, so the clips and references are R16C's;
#   - --output-key is round18/pilot-r18.json (R16C's key would overwrite R16C's stored pilot).
# Then the paired comparisons exactly as round 16 ran them (compare_pilot_lexicon.py; --per-subset against R16C) and
# the declared rule (scripts/round18_verdict.py). Controls are SHA-checked first.
#   MANIFEST_SHA=<round18-mixed-v1 SHA> bash runpod/fleet/round18_pilot.sh
#   SMOKE=1 MANIFEST_SHA=.. [SMOKE_CHECKPOINT=<key>] bash runpod/fleet/round18_pilot.sh   (8 clips into
#     round18/smoke/, no comparison; SMOKE_CHECKPOINT defaults to R18's step-300 smoke checkpoint)
# Exit 0 PASS, 4 FAIL, 2 not evaluable, 1 error; ends with R18-PILOT-DONE <status>.
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
O=acquisition/2026-10-17-v1/youtube-asl-v1
MSHA=${MANIFEST_SHA:-}
done_() { echo "R18-PILOT-DONE $1"; exit $1; }
[[ $MSHA =~ ^[0-9a-f]{64}$ ]] || { echo "ALERT MANIFEST_SHA (64 hex) is required"; done_ 2; }
CKPT=training-runs/tokens-rvq-d0b-prior-r18-uwb/checkpoints/step-00010000.pt
OUTKEY=$O/round18/pilot-r18.json
CLIPS=1000
if [ "${SMOKE:-0}" = 1 ]; then
  CKPT=${SMOKE_CHECKPOINT:-training-runs/tokens-rvq-d0b-prior-r18-uwb-smoke/checkpoints/step-00000300.pt}
  OUTKEY=$O/round18/smoke/pilot-smoke-$(date -u +%Y%m%dT%H%M%SZ).json
  CLIPS=8
fi
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
# the stored controls are the ones round 16 compared (SHA-256 read from R2 on 2026-10-30)
python - <<'PY' || { echo "ALERT a control pilot is not the stored one"; done_ 2; }
import hashlib, os, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
O = "acquisition/2026-10-17-v1/youtube-asl-v1"
for key, sha in ((f"{O}/round16/pilot-r16c.json", "b90c537a15535d4cbd05818f2a7d66d9392f577c37cdca35021228caf97a7f06"),
                 (f"{O}/v3read/pilot-w14b-step30000-v3d.json", "d90daf723c0bd2e22648b887550a03a224b60d34b84e3bea6aa612e8148a590b")):
    got = hashlib.sha256(s.get(key)).hexdigest()
    if got != sha:
        sys.exit(f"{key}: {got}, expected {sha}")
PY
uv pip install -q 'transformers>=4.40,<5' sentencepiece
if exists $OUTKEY; then
  echo "stored: $OUTKEY (the pilot is not rerun)"
else
  python scripts/pilot_content_check.py --mix-key $O/round18-mixed-v1/manifest.json --mix-sha256 $MSHA --vocab-mix-key acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json --vocab-mix-sha256 ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a --word-split-key acquisition/2026-10-17-v1/youtube-asl-v1/round14/r5-word-split.json --word-split-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a --word-list-key acquisition/2026-10-17-v1/youtube-asl-v1/round14/r5-word-split.json --word-list-sha256 676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a --recognizer-run sign-recognizer-v3d --clips $CLIPS --checkpoint-key $CKPT --output-key $OUTKEY \
    || { echo "ALERT the pilot failed"; done_ 1; }
fi
[ "${SMOKE:-0}" = 1 ] && { echo "SMOKE pilot written to $OUTKEY"; done_ 0; }
# comparisons and verdict: run into local files (each status checked), the verdict reads those files, and every
# output key is written once (an existing compare or verdict key refuses the run before anything is computed)
set -o pipefail
D=/root/r18-verdict; rm -rf $D && mkdir -p $D
for k in compare-r18-vs-r16c.json compare-r18-vs-w14b.json verdict-r18.json; do
  exists $O/round18/$k && { echo "ALERT $O/round18/$k exists: refusing to overwrite"; done_ 2; }
done
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; open(sys.argv[2],'wb').write(R2Store(os.environ).get(sys.argv[1]))" \
  $OUTKEY $D/pilot-r18.json || { echo "ALERT the pilot report could not be read"; done_ 1; }
python scripts/compare_pilot_lexicon.py --arm-key $OUTKEY --control-key $O/round16/pilot-r16c.json --per-subset \
  2> $D/vs-r16c.err | tail -1 > $D/vs-r16c.json || { echo "ALERT compare vs R16C failed"; tail -3 $D/vs-r16c.err; done_ 1; }
python scripts/compare_pilot_lexicon.py --arm-key $OUTKEY --control-key $O/v3read/pilot-w14b-step30000-v3d.json \
  2> $D/vs-w14b.err | tail -1 > $D/vs-w14b.json || { echo "ALERT compare vs W14b failed"; tail -3 $D/vs-w14b.err; done_ 1; }
sed 's/^/R18-VS-R16C /' $D/vs-r16c.json; sed 's/^/R18-VS-W14B /' $D/vs-w14b.json
python scripts/round18_verdict.py --arm-pilot $D/pilot-r18.json --vs-r16c $D/vs-r16c.json --vs-w14b $D/vs-w14b.json \
  --arm-key $OUTKEY --checkpoint-key $CKPT --out $D/verdict-r18.json
s=$?
[ $s = 0 ] || [ $s = 4 ] || [ $s = 2 ] || { echo "ALERT the verdict failed (status $s)"; done_ 1; }
python - $O/round18 $D <<'PY' || { echo "ALERT the comparisons or the verdict were not stored"; done_ 1; }
import os, sys
from signtranslator.data.text_motion_dataset import R2Store
s, prefix, d = R2Store(os.environ), sys.argv[1], sys.argv[2]
for local, key in (("vs-r16c.json", "compare-r18-vs-r16c.json"), ("vs-w14b.json", "compare-r18-vs-w14b.json"),
                   ("verdict-r18.json", "verdict-r18.json")):
    if s.exists(f"{prefix}/{key}"):
        sys.exit(f"{prefix}/{key} exists")
    s.put(f"{prefix}/{key}", open(f"{d}/{local}", "rb").read())
PY
done_ $s
