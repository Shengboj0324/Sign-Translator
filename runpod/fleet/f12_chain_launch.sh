#!/bin/bash
# f12_chain_launch.sh — Mac-side launcher for runpod/fleet/f12_chain.sh (round 3e's launcher, after_prior_3e.sh, with
# F12's names): waits for F12's step-60000 checkpoint in R2 and for three free pod slots (at most 5 pods), then three
# sweep GPU pods, the CPU grid pod and the review GPU pod, each through ~/.sign-translator-ops/run_job_pod.sh (which
# deletes each pod once its log is verified in R2). The pods run the code bundle of ~/src/Sign-Translator HEAD
# (make_bundle.sh), so this branch must be on main first. F12_SMOKE=1 runs the -smoke chain (one sweep pod, toy sizes).
# Log: ~/.sign-translator-ops/after_f12_chain.log (run it as: nohup bash runpod/fleet/f12_chain_launch.sh >> ... &).
cd ~/src/Sign-Translator; set -a; source ./.env.r2.local; set +a; cd ~/.sign-translator-ops
PY=~/.venvs/sign-translator/bin/python
STEP=${F12_STEP:-60000}; SMOKE=${F12_SMOKE:-0}
D=acquisition/2026-10-17-v1/youtube-asl-v1/round12/chain-f12-step$STEP$([ "$SMOKE" = 1 ] && echo -smoke)
RUN=tokens-rvq-d0b-prior-r12-f12 GEN=training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-$(printf %08d $STEP).pt
r2py() { $PY -c "import os,sys,json; sys.path.insert(0, os.path.expanduser('~/src/Sign-Translator')); from signtranslator.data.text_motion_dataset import R2Store; s=R2Store(os.environ); $1" "${@:2}"; }
r2_exists() { r2py "sys.exit(0 if s.exists(sys.argv[1]) else 1)" "$1"; }
r2_count() { r2py "print(sum(1 for k in s.list(sys.argv[1]) if k.endswith('.json')))" "$1"; }
running() { python3 -c "
from rp import gql
r=gql('{ myself { pods { desiredStatus } } }'); m=r.get('myself') or r.get('data',{}).get('myself')
print(sum(1 for p in m['pods'] if p['desiredStatus']=='RUNNING'))" 2>/dev/null || echo 9; }
DCS_GPU="SECURE:US-IL-1 SECURE:US-TX-4 SECURE:EU-NL-1 SECURE:EUR-IS-2 SECURE:OC-AU-1 SECURE:US-TX-3 COMMUNITY:"
E="F12_STEP=$STEP F12_SMOKE=$SMOKE"
r2_exists $D/evaluations/$RUN-verdict.json && { echo "a verdict already exists under $D: not relaunching over it"; exit 1; }
until r2_exists $GEN; do sleep 120; done
echo "F12 $GEN in R2 $(date -u +%H:%M) UTC"
NSWEEP=$([ "$SMOKE" = 1 ] && echo 1 || echo 3)
until [ "$(running)" -le $((5 - NSWEEP)) ]; do sleep 120; done
bash make_bundle.sh
echo "launching the F12 chain sweep ($NSWEEP pods) $(date -u +%H:%M) UTC"
if [ "$SMOKE" = 1 ]; then
  BAD_DCS="US-NC-1 US-MO-1" DCS="$DCS_GPU" bash run_job_pod.sh f12-chain-smoke-sweep "$E bash runpod/fleet/f12_chain.sh sweep" \
    >> job_f12_chain_smoke_sweep.log 2>&1
  WANT=3 WANT_GRID=19   # 1 setting x 3 groups; grid 2 dampings x 2 (fc) x 4 groups
else
  for part in a b c; do
    BAD_DCS="US-NC-1 US-MO-1" DCS="$DCS_GPU" bash run_job_pod.sh f12-chain-sweep-$part "$E bash runpod/fleet/f12_chain.sh sweep $part" \
      >> job_f12_chain_sweep_$part.log 2>&1 &
  done
  wait
  WANT=18 WANT_GRID=450
fi
n=$(r2_count $D/evaluations/$RUN-sweep-); echo "sweep results stored: $n of $WANT $(date -u +%H:%M)"
[ "$n" -ge $WANT ] || { echo "SWEEP INCOMPLETE: not launching the grid"; exit 1; }
CPU=32 DISK=100 bash run_job_pod.sh f12-chain-grid$([ "$SMOKE" = 1 ] && echo -smoke) "$E OMP_NUM_THREADS=2 bash runpod/fleet/f12_chain.sh grid" \
  >> job_f12_chain_grid.log 2>&1
n=$(r2_count $D/evaluations/$RUN-sweep-h); echo "evaluation files after the grid: $n ($WANT_GRID expected) $(date -u +%H:%M)"
[ "$n" -ge $WANT_GRID ] || { echo "GRID INCOMPLETE: not launching the review"; exit 1; }
BAD_DCS="US-NC-1 US-MO-1" DCS="$DCS_GPU" bash run_job_pod.sh f12-chain-review$([ "$SMOKE" = 1 ] && echo -smoke) \
  "$E bash runpod/fleet/f12_chain.sh review" >> job_f12_chain_review.log 2>&1
if r2_exists $D/evaluations/$RUN-verdict.json; then
  echo "F12 CHAIN VERDICT WRITTEN $(date -u +%H:%M)"
  r2py "v=json.loads(s.get(sys.argv[1])); print(json.dumps(dict(rules=v['rules'], guard=v['test_set_guard']['message'], **{'pass': v['pass']})))" \
    $D/evaluations/$RUN-verdict.json
else
  echo "REVIEW ENDED WITHOUT A VERDICT $(date -u +%H:%M)"
fi
