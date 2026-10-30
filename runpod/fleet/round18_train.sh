#!/bin/bash
# round18_train.sh — GPU pod: arm R18 (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test", declared 2026-10-30
# 13:03 UTC): W14b step 30000 + 10,000 steps on round18-mixed-v1. The command is R16C's (job_r16c-train.sh, round 16)
# with only the manifest (MIX, MSHA) and the run name (PRUN) changed: the smoke (R3D_SMOKE=1, 300 steps into
# <run>-smoke), then the full run (PATIENCE=0). round5_prior.sh checks the mix SHA and prefetches every state against
# its SHA before each stage; this job also prefetches and checks once before the smoke, so a missing or altered state
# fails before any GPU step.
#   MANIFEST_SHA=<build_round18_manifest.py's MANIFEST_SHA> bash runpod/fleet/round18_train.sh
# Ends with R18-TRAIN-DONE <status>.
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json
MSHA=${MANIFEST_SHA:-}
PRUN=tokens-rvq-d0b-prior-r18-uwb
done_() { echo "R18-TRAIN-DONE $1"; exit $1; }
[[ $MSHA =~ ^[0-9a-f]{64}$ ]] || { echo "ALERT MANIFEST_SHA (64 hex) is required"; done_ 2; }
# the mix must be round14b-mixed-v1 (710a3941...) extended by youtube-asl-uwb-lifted-v2 and nothing else, and the run new
python - "$MIX" "$MSHA" "$PRUN" <<'PY' || done_ 2
import hashlib, json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
mix, sha, run = sys.argv[1:4]
s = R2Store(os.environ)
b = s.get(mix)
if hashlib.sha256(b).hexdigest() != sha:
    sys.exit(f"{mix}: SHA-256 {hashlib.sha256(b).hexdigest()} is not MANIFEST_SHA {sha}")
m = json.loads(b)
inputs = m.get("inputs") or {}
if m.get("corpus_id") != "round18-mixed-v1" or set(inputs) != {"round14b-mixed-v1", "youtube-asl-uwb-lifted-v2"} \
        or inputs["round14b-mixed-v1"] != "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d":
    sys.exit(f"{mix}: not round14b-mixed-v1 + youtube-asl-uwb-lifted-v2 ({m.get('corpus_id')}, {inputs})")
if s.list(f"training-runs/{run}/checkpoints/"):
    sys.exit(f"training-runs/{run} already has checkpoints: a rerun needs a new run name")
lifted = sum(r.get("corpus") == "youtube-asl-uwb-lifted-v2" for r in m["records"])
print(f"mix {mix} sha {sha}: {len(m['records'])} records, {lifted} lifted")
PY
python scripts/prefetch_cache.py $MIX || { echo "ALERT states missing or not matching the manifest"; done_ 1; }
R3D_SMOKE=1 PRUN=tokens-rvq-d0b-prior-r18-uwb TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh && PATIENCE=0 PRUN=tokens-rvq-d0b-prior-r18-uwb TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh
done_ $?
