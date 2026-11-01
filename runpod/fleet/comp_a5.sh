#!/usr/bin/env bash
# comp_a5.sh — post-S2 A5 on D20: A3 generated top10 against margin0, paired within each of five fresh word draws (scripts/comp_a5_read.py). Evaluation only: NOTHING TRAINS.
# PREPARED, NOT LAUNCHED. NOTHING IS DECLARED BY THIS FILE: a full run stops unless COMP_A_DECLARED passes scripts/comp_a5_declared.py, a declaration apart from A1 to A4's.
#
#   COMP_PHASE=<phase> DRAW=<1..5> [SMOKE=1] [SMOKE_STAMP=<stamp>] [COMP_A_DECLARED=<plan commit>] [hashes] bash runpod/fleet/comp_a5.sh
#
# D20 only, pinned; no option for another set. Reports are --report-only and write-once under $OUT. SMOKE=1 reads at most 5 real D20 clips under $OUT/smoke/<stamp>/ at 4 samples per
# word. Nothing decides on a smoke, and a smoke has no summary.
# Phases, on one pod per draw (GPU for a full run; runpod/fleet/comp_a5_chain.sh runs a draw in order):
#   cache    The draw's r10 entries: 16 samples, best by v3d, no gains, on CACHE_PROCS processes (a failed shard stops the others).
#   extra    The draw's seven new entries of the top-10 arm, the same recipe.
#   margin0  a9's list at margin 0 and predicted timing, on this draw's words.
#   top10    margin0's list plus the top-10 arm. MARGIN0_SHA pins its comparator.
#   summary  All five draws (no DRAW; every report pinned by M1_SHA..M5_SHA, T1_SHA..T5_SHA): mean, between-draw SD, two-way bootstrap, the rule.
# Ends with one line: COMP-A5-DONE <0|1> phase=<phase> draw=<draw> smoke=<0|1>. Any other way out prints ALERT and COMP-A5-DONE 1.
set -uo pipefail
PHASE=${COMP_PHASE:-none}; K=${DRAW:-}; ENDED=""; SMOKING=0
done_line() { ENDED=1; echo "COMP-A5-DONE $1 phase=$PHASE draw=${K:-none} smoke=$SMOKING"; }
die() { echo "ALERT $*"; done_line 1; exit 1; }
unexpected() { code=$?; if [ -z "$ENDED" ]; then echo "ALERT comp_a5 stopped unexpectedly, exit $code"; done_line 1; exit 1; fi; }
trap unexpected EXIT
source /root/r2env.sh
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(/root/venv/bin/python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
T=${THREADS:-$C}
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
uv pip install -q 'transformers>=4.40,<5' sentencepiece || die "uv pip install failed"
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }
said() { local hash; hash=$(sha "$2"); [ -n "$hash" ] || die "$1: no hash of $2"; echo "COMP-A5 $1 $2 $hash"; }

[ -z "${SET:-}" ] || [ "$SET" = d20 ] || die "SET: this read is D20 only; C20, the 1,000 and test are refused"
case ${SMOKE:-0} in 0) ;; 1) SMOKING=1 ;; *) die "SMOKE must be 0 or 1" ;; esac
case $PHASE in cache|extra|margin0|top10) case $K in 1|2|3|4|5) ;; *) die "DRAW must be 1, 2, 3, 4 or 5: the rule names five, and no other" ;; esac ;;
  summary) [ -z "$K" ] || die "summary reads all five draws: no DRAW" ;;
  *) die "unknown COMP_PHASE $PHASE: cache extra margin0 top10 summary" ;; esac
if [ $SMOKING = 0 ]; then
  [ -n "${COMP_A_DECLARED:-}" ] || die "a full run needs COMP_A_DECLARED=<plan commit>: nothing is declared yet"
  python scripts/comp_a5_declared.py "$COMP_A_DECLARED" || die "COMP_A_DECLARED $COMP_A_DECLARED does not declare A5 for this code"
fi
O=acquisition/2026-10-17-v1/youtube-asl-v1
A=$O/round20/phase-a
OUT=$O/round20/comp-a5
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
R14B=$O/round14b-mixed-v1/manifest.json; R14BSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
BT=training-runs/back-translator-kp2d-uwb-ft-v1; BTSHA=3db49bb084ed499bceffbbd93b83d6f62059da0d6727369c9472b029668b0dba
BTCK=$BT/checkpoints/step-00009000.pt; BTCKSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
V2=training-runs/sign-recognizer-v2; V2SHA=8989a03b8af3137659f375f1f7c64507a7341061e77b19a5f677cfa230661497
V2CK=$V2/checkpoints/step-00027000.pt; V2CKSHA=a126bf42c0fa4ce8a00e168efbe785668a9ee98600d1146e0d0503c087ba82fb
V3D=training-runs/sign-recognizer-v3d; V3DSHA=3b50a8168e9365984a1f5eb42dd1525466e700bae4036c9bb485a23704cd7ba0
V3DCK=$V3D/checkpoints/step-00012000.pt; V3DCKSHA=0c6c10e232037e869f3983ed6df16df44b3e0a092cbfa34d40ca744cbf00a4e2
TOKENS=reports/comp_a3_tokens.json; TOKENSSHA=c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776
A2FILE=reports/comp_a2_left_arm.json; A2SHA=1dab85c88ba9380cdba89fe4b830d0987cfbeb29d0dec638636af5034e84644a
DURATION=-0.083,0.3691,18289
JOBFILE=runpod/fleet/comp_a5.sh
for p in "$R10 $R10SHA" "$R14B $R14BSHA" "$D20 $D20SHA" "$BT/best.json $BTSHA" "$BTCK $BTCKSHA" "$V2/best.json $V2SHA" "$V2CK $V2CKSHA" "$V3D/best.json $V3DSHA" "$V3DCK $V3DCKSHA"; do
  set -- $p; pin "$1" "$2"
done
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
CLIPS=500; SAMPLES=16; CHOOSE=16; N=16
if [ $SMOKING = 1 ]; then
  [ "$PHASE" != summary ] || die "summary reads all of D20: no smoke"
  OUT=$OUT/smoke/${SMOKE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}; CLIPS=5; SAMPLES=4; CHOOSE=4; N=4
else
  [ "$DEV" = cuda ] || die "a full run needs a GPU (DEV=$DEV)"
fi
DECL=""; [ -z "${COMP_A_DECLARED:-}" ] || DECL="--declared-commit $COMP_A_DECLARED"
CACHEARGS="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS --checkpoint-key $W14B --generator-mix-sha256 $R14BSHA \
  --samples $SAMPLES --choose $CHOOSE --recognizer-run sign-recognizer-v3d --device $DEV --no-decoder-gains --comp-a-job-file $JOBFILE"
BASE="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS \
  --selection stage1 --timing sign-fitted --expect-model-sha256 $MODEL --likelihood --device $DEV --comp-a-job-file $JOBFILE"
PREDICT="--core-margin 0 --predicted-duration --expect-duration-model=$DURATION"
EXTRA="--vocabulary-extra-file $TOKENS --vocabulary-extra-sha256 $TOKENSSHA --vocabulary-extra-top 10"

# one write-once run and its report-only merge: arm NAME ARGS...
arm() {
  local name=$1; shift
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json
  if exists $merged; then echo "stored: $merged (not rerun)"; return 0; fi
  exists $part || python scripts/round15_compose_eval.py run $BASE "$@" --output-key $part || die "run $name"
  python scripts/round15_compose_eval.py merge --report-only --shard-keys $part --output-key $merged || die "merge $name"
  said $name $merged
}
words9() {
  local idx=$OUT/draw$K-cache/index.json idxsha
  exists $idx || die "$PHASE: the draw's cache $idx is not stored"
  idxsha=$(sha $idx); [ -n "$idxsha" ] || die "$PHASE: no hash of $idx"
  WORDS9="--variants core_crossfade --word-cache-key $idx --word-cache-sha256 $idxsha --word-cache-n $N --recognizer-run sign-recognizer-v2"
}

case $PHASE in
  cache)
    PREFIX=$OUT/draw$K-cache
    if exists $PREFIX/index.json; then echo "stored: $PREFIX/index.json"
    else
      pin $W14B $W14BSHA
      P=${CACHE_PROCS:-4}; pids=()
      for s in $(seq 0 $(( P - 1 ))); do
        exists $PREFIX/shard-$s-of-$P.json && continue
        OMP_NUM_THREADS=$(( T / P > 0 ? T / P : 1 )) python scripts/round20_word_cache.py run --prefix $PREFIX $CACHEARGS --shards $P --shard $s --draw $K > /root/r20-a5-cache-$K-$s.log 2>&1 & pids+=($!)
      done
      failed=0; ended=" "
      while :; do
        running=0
        for q in "${pids[@]:-}"; do
          [ -n "$q" ] || continue
          case "$ended" in *" $q "*) continue ;; esac
          if kill -0 $q 2>/dev/null; then running=1; continue; fi
          wait $q || failed=1; ended="$ended$q "
        done
        if [ $failed = 1 ]; then for q in "${pids[@]}"; do kill $q 2>/dev/null; done; break; fi
        [ $running = 0 ] && break
        sleep 10
      done
      wait
      [ $failed = 0 ] || die "a cache shard failed; the others were stopped (logs /root/r20-a5-cache-$K-*.log)"
      python scripts/round20_word_cache.py merge --prefix $PREFIX --shards $P || die "cache merge"
      said draw$K-cache $PREFIX/index.json
    fi ;;
  extra)
    PREFIX=$OUT/draw$K-extra
    if exists $PREFIX/index.json; then echo "stored: $PREFIX/index.json"
    else
      pin $W14B $W14BSHA
      exists $PREFIX/shard-0-of-1.json || python scripts/round20_word_cache.py run --prefix $PREFIX $CACHEARGS --draw $K $EXTRA || die "extra run"
      python scripts/round20_word_cache.py merge --prefix $PREFIX || die "extra merge"
      said draw$K-extra $PREFIX/index.json
    fi ;;
  margin0)
    words9
    arm draw$K-margin0 $WORDS9 $PREDICT --comp-a-read ;;
  top10)
    words9
    X=$OUT/draw$K-extra/index.json
    exists $X || die "top10: the draw's new entries $X are not stored"
    XSHA=$(sha $X); [ -n "$XSHA" ] || die "top10: no hash of $X"
    if [ $SMOKING = 0 ]; then
      exists $OUT/draw$K-margin0.json || die "top10: its comparator draw$K-margin0 comes first"
      [ -n "${MARGIN0_SHA:-}" ] || die "top10: give MARGIN0_SHA, the hash of its comparator"
      pin $OUT/draw$K-margin0.json "$MARGIN0_SHA"
    fi
    arm draw$K-top10 $WORDS9 $PREDICT $EXTRA --word-cache-extra-key $X --word-cache-extra-sha256 $XSHA ;;
  summary)
    KEY=$OUT/a5-summary.json
    if exists $KEY; then echo "stored: $KEY"
    else
      ARGS=""
      for d in 1 2 3 4 5; do
        m=$OUT/draw$d-margin0.json; t=$OUT/draw$d-top10.json
        exists $m && exists $t || die "incomplete: draw $d is not stored; fewer than five finished draws is no smaller test"
        ms=M${d}_SHA; ts=T${d}_SHA
        [ -n "${!ms:-}" ] && [ -n "${!ts:-}" ] || die "summary: give $ms and $ts"
        pin $m "${!ms}"; pin $t "${!ts}"
        ARGS="$ARGS --draw $d=$m,$t"
      done
      python scripts/comp_a5_read.py $ARGS --a9-sides-file $A2FILE --a9-sides-sha256 $A2SHA $DECL --output-key $KEY || die "summary"
      said a5-summary $KEY
    fi ;;
esac
done_line 0
