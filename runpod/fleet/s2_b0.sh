#!/bin/bash
# s2_b0.sh — GPU evaluation pod: stage S2's B0 (docs/NEXT_PHASE_PLAN.md, "Road to 3.8: S2 design received"). W14b end-to-end
# on D20, gains and no gains, reference and predicted lengths, with PMI ranks. Evaluation only: nothing is trained.
# Prepared, not run: it waits for the probes' review.
#
#   bash runpod/fleet/s2_b0.sh            smoke (8 c02 train clips, no D20 read) first, then the D20 read
#   SMOKE=1 bash runpod/fleet/s2_b0.sh    the smoke only
#
# Outputs (write-once): $D/b0-w14b-d20-v1.json and $D/lengths-d20-v1.json under the round's s2/ prefix.
# Ends with S2-B0-DONE <status> (0 read done, 1 the read failed, 2 the smoke stopped the job).
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /root/work/Sign-Translator
set -uo pipefail
done_() { echo "S2-B0-DONE $1"; exit $1; }
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/s2
TS=$(date -u +%Y%m%dT%H%M%SZ)
TSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8     # v1 checkpoint
PILOT=$O/round14b/pilot-w14b-step30000.json
W=/root/s2-b0
mkdir -p $W/smoke $W/d20
uv pip install -q 'transformers>=4.40,<5' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }

python scripts/s2_length_probe.py lengths --smoke-train 8 --out-dir $W/smoke || { echo "ALERT smoke lengths failed"; done_ 2; }
python scripts/s0_ceilings.py --smoke-train 8 --forms real,blank --device cuda --translator-sha256 $TSHA --nll-group 4 \
  --candidates 8 --b0-pilot $PILOT --b0-lengths $W/smoke/lengths.json --output-key $D/smoke/b0-smoke-$TS.json \
  || { echo "ALERT B0 smoke failed"; done_ 2; }
[ "${SMOKE:-0}" = 1 ] && { echo "SMOKE done"; done_ 0; }

python scripts/s2_length_probe.py lengths --out-dir $W/d20 || { echo "ALERT lengths failed"; done_ 1; }
python scripts/s0_ceilings.py --sentence-set d20 --forms real,blank --device cuda --translator-sha256 $TSHA --nll-group 4 \
  --candidates 100 --b0-pilot $PILOT --b0-lengths $W/d20/lengths.json --output-key $D/b0-w14b-d20-v1.json \
  || { echo "ALERT B0 failed"; done_ 1; }
python - "$W/d20/lengths.json" "$D/lengths-d20-v1.json" <<'PY' || done_ 1
import os, pathlib, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
if s.exists(sys.argv[2]):
    sys.exit(f"{sys.argv[2]} exists (write-once)")
s.put(sys.argv[2], pathlib.Path(sys.argv[1]).read_bytes())
PY
done_ 0
