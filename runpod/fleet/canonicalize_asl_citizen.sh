#!/usr/bin/env bash
# canonicalize_asl_citizen.sh — round 5 data: every ASL Citizen clip right-dominant (mirrored where the signing hand is
# stored on the left) and trimmed to its active signing (scripts/canonicalize_asl_citizen.py), then the canonical
# manifest. Shards resume from their reports; research use only, everything private in R2.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then W=$(( Q / P )); else W=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$(( W * 2 ))}       # I/O-bound: two workers per core
SHARDS=64
echo "STAGE canonicalize workers $W $(date)"
for s in $(seq ${FROM:-0} ${TO:-63}); do
  for pass in 1 2 3; do
    python scripts/canonicalize_asl_citizen.py --shard $s --shards $SHARDS --workers $W && break
    echo "shard $s pass $pass exit $?"; sleep 30
  done
done
[ -n "${ASSEMBLE:-1}" ] && python scripts/canonicalize_asl_citizen.py --assemble
echo CANONICALIZE-DONE
