#!/usr/bin/env bash
# uwb_drift.sh — CPU pod: the UWB lifter gate's drift run alone, for the drift-checked fps table (12 pairs/video), with
# --bootstrap 0: its intervals are not part of any gate (the gate report, 5 pairs/video, carries them) and recomputing
# 2,000-resample intervals over 2.4x the pairs would take hours. Stores the fps table under lifter-v1/gate/. UWB-DRIFT-DONE.
set -euo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
K=acquisition/2026-10-17-v1/youtube-asl-uwb-keypoints-v1/lifter-v1
SHA=02a90e4ff333202d556bee62b12216ffca85761e45dafb011144fcbc1bd82dde
N=$(python -c "import os
cpus = len(os.sched_getaffinity(0)) - 2
try: mem = int(open('/sys/fs/cgroup/memory.max').read()) / 2**30
except (OSError, ValueError): mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
print(max(1, min(30, cpus, int(mem // 2))))")   # not nproc (it honours OMP_NUM_THREADS=1: gave --workers -1); 2 GB per worker
O=/root/drift; rm -rf $O && mkdir -p $O
python scripts/uwb_lifter_gate.py --regressors $K/files/world-regressors-xy2-v1.pt --regressors-sha256 $SHA \
  --visibility inside --candidate-visibility "" --exclude-videos reports/uwb_gate_smoke_videos.txt --videos 0 \
  --models-dir /root/work/models --pairs-per-video 12 --bootstrap 0 --workers $N --out $O/gate-drift.json \
  --fps-table-out $O/uwb-fps-table-v3.json
[ -s $O/uwb-fps-table-v3.json ]
python scripts/store_local_files.py --dir $O --prefix $K/gate-drift --source "uwb_lifter_gate.py drift run, 12 pairs/video, bootstrap 0" \
  --licence "derived (UWB CC BY 4.0; our YouTube-ASL states research-only)" --note "drift-checked fps table for conversion"
echo UWB-DRIFT-DONE
