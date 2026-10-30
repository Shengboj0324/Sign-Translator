#!/bin/bash
# round19_train.sh — GPU pod: arm R19 (docs/NEXT_PHASE_PLAN.md "Round 19: refresh our own YouTube data", declared
# 2026-10-30 14:16 UTC): W14b step 30000 + 10,000 steps on round19-mixed-v1. round18_train.sh with only the manifest and
# the run name changed: R16C's command (job_r16c-train.sh) with MIX/MSHA and PRUN changed, the smoke (R3D_SMOKE=1, 300
# steps into <run>-smoke), the recipe check against R16C's smoke run.json, then the full run (PATIENCE=0). Every state
# is fetched and SHA-checked by prefetch_cache.py before the smoke (and again by round5_prior.sh before each stage):
# about 76k YouTube states are new to any training run (the duplicate-overwrite rule).
#   MANIFEST_SHA=<build_round19_manifest.py's MANIFEST_SHA> bash runpod/fleet/round19_train.sh
# Ends with R19-TRAIN-DONE <status>.
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8
cd /root/work/Sign-Translator
MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round19-mixed-v1/manifest.json
MSHA=${MANIFEST_SHA:-}
PRUN=tokens-rvq-d0b-prior-r19-refresh
done_() { echo "R19-TRAIN-DONE $1"; exit $1; }
[[ $MSHA =~ ^[0-9a-f]{64}$ ]] || { echo "ALERT MANIFEST_SHA (64 hex) is required"; done_ 2; }
# the mix must be round14b-mixed-v1 (710a3941...) with its youtube-asl-v1 replaced (build_round19_manifest.py), and the run new
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
if m.get("corpus_id") != "round19-mixed-v1" or set(inputs) != {"round14b-mixed-v1", "youtube-asl-v1-r19"} \
        or inputs["round14b-mixed-v1"] != "710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d" \
        or (m.get("replaces") or {}).get("corpus") != "youtube-asl-v1":
    sys.exit(f"{mix}: not round14b-mixed-v1 with youtube-asl-v1 replaced ({m.get('corpus_id')}, {inputs})")
if s.list(f"training-runs/{run}/checkpoints/"):
    sys.exit(f"training-runs/{run} already has checkpoints: a rerun needs a new run name")
youtube = sum(r.get("corpus") == "youtube-asl-v1" for r in m["records"])
print(f"mix {mix} sha {sha}: {len(m['records'])} records, {youtube} youtube-asl-v1")
PY
python scripts/prefetch_cache.py $MIX || { echo "ALERT states missing or not matching the manifest"; done_ 1; }
# between the smoke and the full run: R19's smoke run.json must be R16C's smoke run.json but for the run name and the
# manifest (the same model config SHA; the trainer gained round 17's opt-in plan flags since R16C, which must sit at
# their defaults). R16C trained from a bundle of main 7876a0d (holds 52965cc); R19 trains on the current main.
same_recipe_as_r16c() {
  python - <<'PY' || { echo "ALERT R19's smoke recipe differs from R16C's"; return 1; }
import json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
a = json.loads(s.get("training-runs/tokens-rvq-d0b-prior-r16c-control-smoke/run.json"))
b = json.loads(s.get("training-runs/tokens-rvq-d0b-prior-r19-refresh-smoke/run.json"))
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
if b["manifest_key"].rsplit("/", 2)[-2] != "round19-mixed-v1" or a["kind"] != b["kind"]:
    problems.append(f"manifest {b['manifest_key']} / kind {b['kind']}")
print(f"R19 smoke recipe vs R16C smoke: config_sha256 {b['config_sha256']}; train records {a['train_records']} -> "
      f"{b['train_records']}; {len(problems)} differences")
sys.exit("; ".join(problems) if problems else None)
PY
}
R3D_SMOKE=1 PRUN=tokens-rvq-d0b-prior-r19-refresh TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round19-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh && same_recipe_as_r16c && PATIENCE=0 PRUN=tokens-rvq-d0b-prior-r19-refresh TEXT_MODEL= RIT_CORPUS= PLR=1e-4 PWARMUP=200 PSTEPS_FULL=10000 MIX=acquisition/2026-10-17-v1/youtube-asl-v1/round19-mixed-v1/manifest.json MSHA=$MSHA TOKKEY=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt EXTRA_WEIGHTS='--corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight asllvd-text-v1=3 --corpus-weight semlex-text-v2=2 --history-dropout 0.5 --part-valid-loss --history-stages 1 --init-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt --init-sha256 0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563 ' bash runpod/fleet/round5_prior.sh
done_ $?
