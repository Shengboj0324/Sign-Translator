#!/usr/bin/env bash
# uwb_gate.sh — CPU pod: the declared UWB lifter gate (plan "UWB lifter: declared selection and gate", 2026-10-30
# 03:34 UTC, amendment 1 03:59, owner-declared 04:02; visibility selection 04:20 -> inside) with the xy2 bundle, then
# the drift run that writes the fps table. Outputs under .../youtube-asl-uwb-keypoints-v1/lifter-v1/gate/. UWB-GATE-DONE.
set -euo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
K=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1
SHA=02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde
N=$(( $(nproc) - 2 )); [ $N -gt 30 ] && N=30
O=/root/gate; rm -rf $O && mkdir -p $O
python scripts/uwb_lifter_gate.py --regressors $K/files/world-regressors-xy2-v1.pt --regressors-sha256 $SHA \
  --visibility inside --candidate-visibility learned,inside,margin:0.1,margin:0.25 \
  --exclude-videos reports/uwb_gate_smoke_videos.txt --videos 0 --pairs-per-video 5 --head affine \
  --workers $N --out $O/gate-xy2-v1.json
python scripts/uwb_lifter_gate.py --regressors $K/files/world-regressors-xy2-v1.pt --regressors-sha256 $SHA \
  --visibility inside --candidate-visibility "" --exclude-videos reports/uwb_gate_smoke_videos.txt --videos 0 \
  --pairs-per-video 12 --workers $N --out $O/gate-drift.json --fps-table-out $O/uwb-fps-table-v3.json
[ -s $O/gate-xy2-v1.json ] && [ -s $O/uwb-fps-table-v3.json ]
python scripts/store_local_files.py --dir $O --prefix $K/gate --source "uwb_lifter_gate.py, xy2 bundle $SHA, visibility inside" \
  --licence "derived (UWB CC BY 4.0; our YouTube-ASL states research-only)" --note "declared gate 2026-10-30"
echo UWB-GATE-DONE
