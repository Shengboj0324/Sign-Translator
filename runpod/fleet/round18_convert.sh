#!/usr/bin/env bash
# round18_convert.sh — CPU pod: round 18's UWB conversion (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test",
# declared 2026-10-30 13:03 UTC; runs only if UWB gate v2 passes). Every UWB clip of an estimated-fps, stream-matched
# video is lifted with the gate-v2 bundle, visibility `head`, fps table v3, the converter's face and head defaults
# (blendshapes, no head rotation: the settings the gate-v2 downstream row read) and `generator` flags.
#
#   BUNDLE_KEY=.. BUNDLE_SHA=.. round18_convert.sh SHARD SHARDS     one shard (resumable: finished units are skipped)
#   round18_convert.sh merge                                         after every shard: index.json, refused unless complete
#   BUNDLE_SHA=.. [DRY_RUN=1] round18_convert.sh manifest            after merge: build_round18_manifest.py ->
#                       round18-mixed-v1 (+ lifter-v2/youtube-asl-uwb-lifted-v2/manifest.json); prints MANIFEST_SHA=;
#                       exit 5 with ROUND18-NO-MEANINGFUL-GAIN when fewer than 20,000 lifted records remain (nothing written)
#   BUNDLE_KEY=.. BUNDLE_SHA=.. DRY_RUN=1 round18_convert.sh 0 1     count the units and check every pinned input only
#
# BUNDLE_KEY / BUNDLE_SHA: the lifter-v2 chain's bundle (uwb_lifter_v2.sh train: lifter-v2/train/files/
# world-regressors-v2-uwb.pt, printed BUNDLE_SHA=...); refused when unset. Output: lifter-v2/converted-v1/ (states/,
# reports/, shards/, index.json). A shard whose summary says it finished is refused (exit 3), never rerun over.
# Workers: sched_getaffinity and cgroup memory (2 GB each), never nproc. Ends with ROUND18-CONVERT-DONE <status>.
set -uo pipefail
MODE=${1:?usage: round18_convert.sh SHARD SHARDS | merge | manifest}
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
ACQ=acquisition/2026-10-17-v1
U=$ACQ/youtube-asl-uwb-keypoints-v1
OUT=${R18_OUT:-$U/lifter-v2/converted-v1}
# pinned inputs (SHA-256 read from R2 on 2026-10-30)
FPS=$U/lifter-v1/gate-rerun/files/uwb-fps-table-v3.json@b3d2a80ce33d266bc309b2a948d5c98ea35689e6f55c2f494c19586f6540dfaa
STREAM=$U/lifter-v1/gate-rerun/files/gate-drift.json@dd130f78de24282fe12a5595ea0103a99ba7d9df5f137fc82bc728191122e95d
OVERLAP=scratch/uwb-lifter-dev/uwb-overlap-table-v2.json@c617a2af0e696591f8b47bc8f1e02fda44a84d77dc2aad3d20167aeed0cfd6d3
EVAL=(
  --eval-manifest $ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json@0529f229428f6b245af2c290ea13dbf638d7900d63b2148cb8f529ae031d38f1
  --eval-manifest $ACQ/derived/asllrp-dai2-v1/plan-manifest-v2/manifest.json@b01877253ea45b62aeaec313e2b9ffb075c0363af9d9e33ff5e4fe8c1902044a
  # round14b-mixed-v1: its How2Sign val holds 1,460 sentences (the pilot's population) that full-v1 lists as train
  --eval-manifest $ACQ/youtube-asl-v1/round14b-mixed-v1/manifest.json@710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
)
done_() { echo "ROUND18-CONVERT-DONE $1"; exit $1; }

if [ "$MODE" = manifest ]; then           # after merge: round18-mixed-v1 (write-once), prints MANIFEST_SHA=...
  [[ ${BUNDLE_SHA:-} =~ ^[0-9a-f]{64}$ ]] || { echo "ALERT BUNDLE_SHA (64 hex) is required"; done_ 2; }
  python scripts/build_round18_manifest.py --converted-prefix $OUT "${EVAL[@]}" \
    --lifted-key $U/lifter-v2/youtube-asl-uwb-lifted-v2/manifest.json \
    --manifest-key $ACQ/youtube-asl-v1/round18-mixed-v1/manifest.json \
    --expect-bundle-sha $BUNDLE_SHA --expect-visibility head --expect-fps-sources estimated \
    --expect-fps-table-sha ${FPS#*@} --expect-stream-check-sha ${STREAM#*@} --expect-overlap-sha ${OVERLAP#*@} \
    ${DRY_RUN:+--dry-run} 2>&1 | tee /root/r18-manifest.log
  s=${PIPESTATUS[0]}   # 5: ROUND18-NO-MEANINGFUL-GAIN (< 20,000 lifted records; amendment 2: round 18 is not trained)
  python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" \
    "$OUT/logs/manifest-$(date -u +%Y%m%dT%H%M%SZ).log" /root/r18-manifest.log || true
  done_ $s
fi
if [ "$MODE" = merge ]; then
  python scripts/convert_uwb_clips.py --out-prefix $OUT --merge-index | tee /root/r18-merge.json || done_ 1
  python - "$OUT" <<'PY' || done_ 1
import hashlib, json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
b = R2Store(os.environ).get(f"{sys.argv[1]}/index.json")
i = json.loads(b)
print(f"units {i['units_finished']}/{i['units_total']} complete {i['complete']} counts {json.dumps(i['counts'])}")
print(f"checks {json.dumps(i['checks'])}")
print(f"INDEX_SHA={hashlib.sha256(b).hexdigest()}")
sys.exit(0 if i["complete"] else "the index is not complete: rerun the unfinished shards, then merge again")
PY
  done_ 0
fi

SHARD=$MODE; SHARDS=${2:?usage: round18_convert.sh SHARD SHARDS}
[[ $SHARD =~ ^[0-9]+$ && $SHARDS =~ ^[0-9]+$ && $SHARD -lt $SHARDS ]] || { echo "ALERT bad shard $SHARD of $SHARDS"; done_ 2; }
[ -n "${BUNDLE_KEY:-}" ] && [[ ${BUNDLE_SHA:-} =~ ^[0-9a-f]{64}$ ]] || { echo "ALERT BUNDLE_KEY and BUNDLE_SHA (64 hex) are required"; done_ 2; }
# a finished shard is never rerun over: its summary lists only present/written units
python - "$OUT/shards/shard-$(printf %03d $SHARD)-of-$(printf %03d $SHARDS).json" <<'PY'
import json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
if s.exists(sys.argv[1]):
    summary = json.loads(s.get(sys.argv[1]))
    statuses, run = summary.get("statuses", {}), summary.get("run", {})
    if not any(run.values()) and set(statuses) <= {"present", "written"} and sum(statuses.values()) == summary["units"]:
        sys.exit(3)
PY
case $? in 0) ;; 3) echo "ALERT shard $SHARD of $SHARDS already finished ($OUT/shards): refusing to rerun"; done_ 3 ;; *) done_ 1 ;; esac
N=$(python -c "import os
cpus = len(os.sched_getaffinity(0))
try:                                             # the cgroup v2 CPU quota (cpu.max: '<quota> <period>' or 'max ...')
    quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
    cpus = min(cpus, int(quota) // int(period)) if quota != 'max' else cpus
except (OSError, ValueError):
    try:                                         # cgroup v1
        quota = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        if quota > 0: cpus = min(cpus, quota // int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read()))
    except (OSError, ValueError): pass
try: mem = int(open('/sys/fs/cgroup/memory.max').read()) / 2**30
except (OSError, ValueError): mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
print(max(1, min(30, cpus - 2, int(mem // 2))))")   # not nproc (it honours OMP_NUM_THREADS=1); 2 GB per worker
[ -s /root/work/models/face_landmarker.task ] || { echo "ALERT /root/work/models/face_landmarker.task missing (blendshapes)"; done_ 2; }
echo "round 18 conversion shard $SHARD/$SHARDS, bundle $BUNDLE_KEY ($BUNDLE_SHA), workers $N, out $OUT $(date -u +%FT%TZ)"
python scripts/convert_uwb_clips.py --out-prefix $OUT --regressors "$BUNDLE_KEY" --regressors-sha256 "$BUNDLE_SHA" \
  --world learned --visibility head --face blendshapes --head none --flags generator --models-dir /root/work/models \
  --fps-table $FPS --fps-sources estimated --stream-check $STREAM --overlap-table $OVERLAP "${EVAL[@]}" \
  --shard $SHARD --shards $SHARDS --workers $N ${DRY_RUN:+--dry-run} > /root/r18-convert-$SHARD.log 2>&1
s=$?
head -1 /root/r18-convert-$SHARD.log; tail -2 /root/r18-convert-$SHARD.log
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" \
  "$OUT/logs/convert-shard-$SHARD-of-$SHARDS-$(date -u +%Y%m%dT%H%M%SZ).log" /root/r18-convert-$SHARD.log || true
done_ $s
