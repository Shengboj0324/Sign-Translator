#!/bin/bash
# s0s1_probes.sh — GPU evaluation pod: Road to 3.8 part 2, stages S0 (v1 parts) and S1 (docs/NEXT_PHASE_PLAN.md,
# declared 2026-10-30 19:32 UTC). Evaluation only: nothing is trained. Branch probes-s0s1; report reports/s0s1_dev.md.
#
#   bash runpod/fleet/s0s1_probes.sh            smoke (20 clips) first, then the full reads
#   SMOKE=1 bash runpod/fleet/s0s1_probes.sh    the smoke only
#   PHASES="s0 regen s1 words" (default all)     which full phases run (each skips if its output key exists)
#
# Smoke (always first; keys timestamped under s0s1/smoke/): s0_ceilings.py on 20 clips with every form, the stored
# reports and the W14b regeneration (one whole block of 32), s1_text_kl.py on 20 clips, and the 64-way word retrieval
# of W14b on 8 words. The smoke must show the instrument reproducing the stored real hypotheses (identical_to_computed
# = 1 on eval1000/w14b-step30000) and v1's face-blindness (face_mean / face_absent identical_to_real = 1); otherwise
# the job stops before the full reads (ALERT). W14b's regeneration reproduction rate is printed (a rate below 1 is an
# ALERT but not a stop: the panel then reads the regenerated outputs, as reported).
#
# Full (write-once keys under acquisition/2026-10-17-v1/youtube-asl-v1/s0s1/):
#   s0     s0-ceilings-v1.json      every form on the 1,000 + text metrics of every stored 1,000-clip report
#   regen  s0-regenerated-v1.json   real, blank + W14b, R16, R16C, R19, R19w regenerated; orderings merged with s0
#   s1     s1-text-kl-v1.json       W14b, byte prior at 24k, flan-T5 at 23k, MiniLM at 24k (1,000 val clips)
#   words  s1-word-retrieval-<label>.json   word_retrieval_check.py (64 ASLLVD words, round6-mixed-v1, as round 7)
# Ends with S0S1-DONE <status> (0 all phases ok, 1 a phase failed, 2 the smoke stopped the job).
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /root/work/Sign-Translator
set -uo pipefail
done_() { echo "S0S1-DONE $1"; exit $1; }
O=acquisition/2026-10-17-v1/youtube-asl-v1
D=$O/s0s1
TS=$(date -u +%Y%m%dT%H%M%SZ)
PHASES=${PHASES:-s0 regen s1 words}
TSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8     # v1 checkpoint (step 9000), read 2026-10-30
R6MIX=$O/round6-mixed-v1/manifest.json; R6SHA=3c8024e37d59883bed3f5c69717c8e14d3a06d29778cb615af83c2c5afe55a89
uv pip install -q 'transformers>=4.40,<5' sentencepiece || { echo "ALERT transformers install failed"; done_ 1; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
STORED=""
for k in round15/eval1000/w14b-step30000 round15/eval1000/f12-step60000 round15/eval1000/oracle-stage2 \
         round15/eval1000/oracle-refined-r1 round15/eval1000/oracle-knobs-all round15/eval1000/oracle-knobs-jerk \
         round15/eval1000/oracle-knobs-space round15/eval1000/oracle-knobs-tight round14b/pilot-w14b-step10000 \
         round14b/pilot-w14b-step30000 round16/pilot-r16 round16/pilot-r16c round19/pilot-r19 round19/pilot-r19w \
         v3read/pilot-w14b-step30000-v3d v3read/pilot-f12-step60000-v3d v3read/pilot-control-c1000-step20000-v3d \
         v3read/pilot-w14-step10000-v3d; do
  STORED="$STORED --stored $O/$k.json"
done
REGEN=""
for k in round14b/pilot-w14b-step30000 round16/pilot-r16 round16/pilot-r16c round19/pilot-r19 round19/pilot-r19w; do
  REGEN="$REGEN --regenerate $O/$k.json"
done
COMMON="--device cuda --translator-sha256 $TSHA --nll-group 4"

# ---- smoke ----------------------------------------------------------------------------------------------------------
S0SMOKE=$D/smoke/s0-smoke-$TS.json; S1SMOKE=$D/smoke/s1-smoke-$TS.json
python scripts/s0_ceilings.py --clips 20 $COMMON $STORED --regenerate $O/round14b/pilot-w14b-step30000.json \
  --output-key $S0SMOKE || { echo "ALERT s0 smoke failed"; done_ 2; }
python scripts/s1_text_kl.py --clips 20 --device cuda --output-key $S1SMOKE || { echo "ALERT s1 smoke failed"; done_ 2; }
python scripts/word_retrieval_check.py --checkpoint-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt \
  --mix-key $R6MIX --mix-sha256 $R6SHA --corpus asllvd-text-v1 --clips 8 --device cuda \
  --output-key $D/smoke/words-smoke-$TS.json || { echo "ALERT word retrieval smoke failed"; done_ 2; }
python - "$S0SMOKE" "$S1SMOKE" <<'PY' || done_ 2
import json, os, sys
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
r = json.loads(s.get(sys.argv[1]))
key = "acquisition/2026-10-17-v1/youtube-asl-v1/round15/eval1000/w14b-step30000.json"
same = r["stored"][key]["forms"]["real"]["identical_to_computed"]
face = [r["forms"][f]["identical_to_real"] for f in ("face_mean", "face_absent")]
regen = [v["reproduction"] for k, v in r["forms"].items() if k.startswith("regen:")]
print("SMOKE real reproduces stored:", same, "| v1 face-blind:", face, "| W14b regeneration:",
      [(g["identical"], g["stored_bleu"], g["regenerated_bleu"]) for g in regen])
print("SMOKE forms:", {k: (v["bleu"]["bleu"], v["delta_nll"]["mean"], v["retrieval"]["nll"]["top1"],
                          v["rho_faith"]["dtw"]["mean"]) for k, v in r["forms"].items()})
k1 = json.loads(s.get(sys.argv[2]))
print("SMOKE s1:", {k: (v["kl"], v["gain"]) for k, v in k1["priors"].items()})
if regen and regen[0]["identical"] < 1:
    print("ALERT W14b regeneration does not reproduce every stored hypothesis (the panel reads the regenerated outputs)")
if same != 1 or face != [1, 1] or len(k1["priors"]) != 4:
    sys.exit("ALERT smoke checks failed: the instrument does not reproduce, v1 is not face-blind, or a prior is missing")
PY
[ "${SMOKE:-0}" = 1 ] && { echo "SMOKE done"; done_ 0; }

# ---- full -----------------------------------------------------------------------------------------------------------
rc=0
run() {   # run KEY CMD...: skip when KEY exists (write-once), else run and record failure
  local key=$1; shift
  if exists "$key"; then echo "stored: $key (not rerun)"; return 0; fi
  echo "$(date -u +%H:%M) start $key"
  "$@" || { echo "ALERT failed: $key"; rc=1; return 1; }
  echo "$(date -u +%H:%M) done $key"
}
for phase in $PHASES; do
  case $phase in
    s0)    run $D/s0-ceilings-v1.json python scripts/s0_ceilings.py --clips 1000 $COMMON $STORED \
             --output-key $D/s0-ceilings-v1.json ;;
    regen) if exists $D/s0-ceilings-v1.json; then MERGE="--merge-from $D/s0-ceilings-v1.json"; else MERGE=""; fi
           run $D/s0-regenerated-v1.json python scripts/s0_ceilings.py --clips 1000 $COMMON --forms real,blank $REGEN \
             $MERGE --output-key $D/s0-regenerated-v1.json ;;
    s1)    run $D/s1-text-kl-v1.json python scripts/s1_text_kl.py --clips 1000 --device cuda \
             --output-key $D/s1-text-kl-v1.json ;;
    words) for spec in w14b=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt \
                       byte24k=training-runs/tokens-rvq-d0b-prior-r3d-d2g/checkpoints/step-00024000.pt \
                       flant5=training-runs/tokens-rvq-d0b-prior-r4-flant5/checkpoints/step-00023000.pt \
                       minilm=training-runs/tokens-rvq-d0b-prior-r4-minilm/checkpoints/step-00024000.pt; do
             label=${spec%%=*}; ck=${spec#*=}
             run $D/s1-word-retrieval-$label.json python scripts/word_retrieval_check.py --checkpoint-key $ck \
               --mix-key $R6MIX --mix-sha256 $R6SHA --corpus asllvd-text-v1 --clips 64 --device cuda \
               --output-key $D/s1-word-retrieval-$label.json
           done ;;
    *) echo "ALERT unknown phase $phase"; rc=1 ;;
  esac
done
python - <<'PY'
import os
from signtranslator.data.text_motion_dataset import R2Store
s = R2Store(os.environ)
D = "acquisition/2026-10-17-v1/youtube-asl-v1/s0s1"
keys = [f"{D}/s0-ceilings-v1.json", f"{D}/s0-regenerated-v1.json", f"{D}/s1-text-kl-v1.json"] + \
       [f"{D}/s1-word-retrieval-{l}.json" for l in ("w14b", "byte24k", "flant5", "minilm")]
print("OUTPUTS", {k.rsplit("/", 1)[-1]: s.exists(k) for k in keys})
PY
done_ $rc
