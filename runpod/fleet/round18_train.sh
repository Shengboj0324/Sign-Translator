#!/bin/bash
# round18_train.sh — GPU pod: arm R18 (docs/NEXT_PHASE_PLAN.md "Round 18: the UWB 4x-data test", declared 2026-10-30
# 13:03 UTC): W14b step 30000 + 10,000 steps on round18-mixed-v1. The command is R16C's (job_r16c-train.sh, round 16)
# with only the manifest (MIX, MSHA) and the run name (PRUN) changed: the smoke (R3D_SMOKE=1, 300 steps into
# <run>-smoke), then the full run (PATIENCE=0). round5_prior.sh checks the mix SHA and prefetches every state against
# its SHA before each stage; this job also prefetches and checks once before the smoke, so a missing or altered state
# fails before any GPU step. Between the smoke and the full run the smoke's recorded recipe is checked against R16C's
# (same_recipe_as_r16c: the only change to R16C's command line).
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
# between the smoke and the full run: R18's smoke run.json must be R16C's smoke run.json but for the run name and the
# manifest (the same model config SHA; the trainer gained round 17's opt-in plan flags since R16C, which must sit at
# their defaults). R16C trained from a bundle of main 7876a0d (holds 52965cc); R18 trains on the current main.
same_recipe_as_r16c() {
  python - <<'PY' || { echo "ALERT R18's smoke recipe differs from R16C's"; return 1; }
import json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
a = json.loads(s.get("training-runs/tokens-rvq-d0b-prior-r16c-control-smoke/run.json"))
b = json.loads(s.get("training-runs/tokens-rvq-d0b-prior-r18-uwb-smoke/run.json"))
allowed = {"run", "manifest_key", "manifest_sha256"}
new_defaults = dict(plan_tokenizer_key=None, plan_tokenizer_sha256=None, plan_tokenizer_manifest_sha256=None,
                    plan_dropout=0., plan_corruption=0., plan_corruption_max=.5, plan_corruption_source="prior",
                    plan_corruption_prior_key=None, plan_corruption_prior_sha256=None, plan_memory=True,
                    init_text_key=None, init_text_sha256=None)
problems = [f"args.{k}: {a['args'][k]!r} -> {b['args'].get(k, '<absent>')!r}" for k in a["args"]
            if k not in allowed and a["args"][k] != b["args"].get(k, object())]
problems += [f"new args.{k} = {b['args'][k]!r}" for k in set(b["args"]) - set(a["args"])
             if k not in new_defaults or b["args"][k] != new_defaults[k]]
if a["config_sha256"] != b["config_sha256"]:
    problems.append(f"config_sha256 {a['config_sha256']} -> {b['config_sha256']}")
if b["manifest_key"].rsplit("/", 2)[-2] != "round18-mixed-v1" or a["kind"] != b["kind"]:
    problems.append(f"manifest {b['manifest_key']} / kind {b['kind']}")
print(f"R18 smoke recipe vs R16C smoke: config_sha256 {b['config_sha256']}; train records {a['train_records']} -> "
      f"{b['train_records']}; {len(problems)} differences")
sys.exit("; ".join(problems) if problems else None)
PY
}
R3D_SMOKE=1 PRUN=tokens-rvq-d0b-prior-r18-uwb TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh && same_recipe_as_r16c && PATIENCE=0 PRUN=tokens-rvq-d0b-prior-r18-uwb TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh
done_ $?
