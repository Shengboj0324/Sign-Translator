#!/usr/bin/env bash
# comp_a6.sh — post-S2 A6 on D20: a narrower left-arm hold and a better length, each against A3 generated top10, paired within word draws (scripts/comp_a6_read.py). Evaluation only: NOTHING TRAINS.
# PREPARED, NOT LAUNCHED. NOTHING IS DECLARED BY THIS FILE: a full run stops unless COMP_A_DECLARED passes scripts/comp_a6_declared.py, a declaration apart from A1 to A5's.
#
#   COMP_PHASE=<phase> WORDS=<a3|1..8> [ATTEMPT=<n, default 1>] [SMOKE=1] [SMOKE_STAMP=<stamp>] [COMP_A_DECLARED=<plan commit>] [hashes] bash runpod/fleet/comp_a6.sh
#
# D20 only, pinned; no option for another set. Reports are --report-only and write-once under $OUT. No word is generated: every arm reads stored caches. SMOKE=1 reads at most 5 real D20 clips under $OUT/smoke/<stamp>/.
# Nothing decides on a smoke, and a smoke has no summary.
# WORDS: a3 is A3's stored words, a screen that decides nothing. 1 to 8 is A5's stored draw, pinned by CACHE_SHA and EXTRA_SHA from its COMP-A5 lines, and a full run waits for A5's summary.
# Attempts: a draw with all arms stored is final, so an attempt is refused once another has its last arm. An arm that started and did not finish stays unfinished: the next attempt reruns all arms on the same words.
# Phases, on one pod per draw (runpod/fleet/comp_a6_chain.sh runs them in order):
#   base     top10 at today's length: the comparator of the others.
#   hold-x   entries with at least 25% two-handed records stay anchors; the arm is held between anchors.
#   hold-xr  the same anchors; the held arm goes to the rest pose and back.
#   length   top10 at the length fitted signer-balanced on c02 + c04.
#   summary  WORDS=a3 or WORDS=draws: each draw's one attempt with all arms, pinned by B<k>_SHA, HX<k>_SHA, HXR<k>_SHA, L<k>_SHA; the rule on the draws, nothing on a3.
# A merge's output, which holds the arm's numbers, goes to a file on the pod, not this log: no per-draw number is read until the summary.
# Ends with one line: COMP-A6-DONE <0|1> phase=<phase> words=<words> smoke=<0|1>. Any other way out prints ALERT and COMP-A6-DONE 1.
set -uo pipefail
PHASE=${COMP_PHASE:-none}; K=${WORDS:-}; A=${ATTEMPT:-1}; ENDED=""; SMOKING=0
done_line() { ENDED=1; echo "COMP-A6-DONE $1 phase=$PHASE words=${K:-none} smoke=$SMOKING"; }
die() { echo "ALERT $*"; done_line 1; exit 1; }
unexpected() { code=$?; if [ -z "$ENDED" ]; then echo "ALERT comp_a6 stopped unexpectedly, exit $code"; done_line 1; exit 1; fi; }
trap unexpected EXIT
source /root/r2env.sh
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(/root/venv/bin/python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
T=${THREADS:-$C}
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
uv pip install -q 'transformers==4.57.6' 'tokenizers==0.22.2' sentencepiece || die "uv pip install failed"
python scripts/comp_a6_guard.py env || die "this pod's python, torch, transformers or tokenizers are not the pinned ones"
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }
said() { local hash; hash=$(sha "$2"); [ -n "$hash" ] || die "$1: no hash of $2"; echo "COMP-A6 $1 $2 $hash"; }
case $A in 1|2|3|4|5) ;; *) die "ATTEMPT must be 1 to 5" ;; esac

[ -z "${SET:-}" ] || [ "$SET" = d20 ] || die "SET: this read is D20 only; C20, the 1,000 and test are refused"
case ${SMOKE:-0} in 0) ;; 1) SMOKING=1 ;; *) die "SMOKE must be 0 or 1" ;; esac
case $PHASE in
  base|hold-x|hold-xr|length) case $K in a3|1|2|3|4|5|6|7|8) ;; *) die "WORDS must be a3 or 1 to 8" ;; esac ;;
  summary) case $K in a3|draws) ;; *) die "summary: WORDS=a3 or WORDS=draws" ;; esac ;;
  *) die "unknown COMP_PHASE $PHASE: base hold-x hold-xr length summary" ;; esac
if [ $SMOKING = 0 ]; then
  [ -n "${COMP_A_DECLARED:-}" ] || die "a full run needs COMP_A_DECLARED=<plan commit>: nothing is declared yet"
  python scripts/comp_a6_declared.py "$COMP_A_DECLARED" || die "COMP_A_DECLARED $COMP_A_DECLARED does not declare A6 for this code"
fi
O=acquisition/2026-10-17-v1/youtube-asl-v1
OUT=$O/round20/comp-a6
A5OUT=$O/round20/comp-a5
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
A3CACHE=$O/round20/phase-a/cache-w14b-nogains-r10/index.json; A3CACHESHA=507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b
A3EXTRA=$O/round20/comp-a/cache-a3-top30-nogains/index.json; A3EXTRASHA=2d614f416abf2d7558c36b7ba117d6042aac81be31373f31878654dab2dfff5c
BT=training-runs/back-translator-kp2d-uwb-ft-v1; BTSHA=3db49bb084ed499bceffbbd93b83d6f62059da0d6727369c9472b029668b0dba
BTCK=$BT/checkpoints/step-00009000.pt; BTCKSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
V2=training-runs/sign-recognizer-v2; V2SHA=8989a03b8af3137659f375f1f7c64507a7341061e77b19a5f677cfa230661497
V2CK=$V2/checkpoints/step-00027000.pt; V2CKSHA=a126bf42c0fa4ce8a00e168efbe785668a9ee98600d1146e0d0503c087ba82fb
TOKENS=reports/comp_a3_tokens.json; TOKENSSHA=c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776
HOLDSTATS=reports/comp_a_hold_stats.json; HOLDSHA=1b7c24a81aa75c58c523096595b40df537fcd4a49094b88b1eb19686c17440d7
REST=reports/comp_a6_rest_pose.json; RESTSHA=f75c2a29870ed3b414a76fdf676713971dad5dba9e8dd9aa4a647f0bfdedd3de
DURATION=-0.083,0.3691,18289
BALANCED=-0.3577,0.3445,18289
JOBFILE=runpod/fleet/comp_a6.sh,runpod/fleet/comp_a6_chain.sh
for p in "$R10 $R10SHA" "$D20 $D20SHA" "$BT/best.json $BTSHA" "$BTCK $BTCKSHA" "$V2/best.json $V2SHA" "$V2CK $V2CKSHA"; do
  set -- $p; pin "$1" "$2"
done
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
CLIPS=500
if [ $SMOKING = 1 ]; then
  [ "$PHASE" != summary ] || die "summary reads all of D20: no smoke"
  OUT=$OUT/smoke/${SMOKE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}; CLIPS=5
else
  [ "$DEV" = cuda ] || die "a full run needs a GPU (DEV=$DEV)"
fi
DECL=""; [ -z "${COMP_A_DECLARED:-}" ] || DECL="--declared-commit $COMP_A_DECLARED"
BASE="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS \
  --selection stage1 --timing sign-fitted --expect-model-sha256 $MODEL --likelihood --device $DEV --comp-a-job-file $JOBFILE"
TODAY="--core-margin 0 --predicted-duration --expect-duration-model=$DURATION"
FITTED="--core-margin 0 --predicted-duration --duration-balanced --expect-duration-model=$BALANCED"
EXTRA="--vocabulary-extra-file $TOKENS --vocabulary-extra-sha256 $TOKENSSHA --vocabulary-extra-top 10"
HOLDX="--hold-left-file $HOLDSTATS --hold-left-sha256 $HOLDSHA --hold-left-exclude 0.25"

stored_a5() { local j; for j in 1 2 3 4 5; do if exists $A5OUT/draw$K-$1/attempt$j/index.json; then echo $A5OUT/draw$K-$1/attempt$j/index.json; return 0; fi; done; return 0; }
# this words' stored caches, pinned: A3's by this file, a draw's by the ledger's COMP-A5 lines
words() {
  if [ "$K" = a3 ]; then
    IDX=$A3CACHE; IDXSHA=$A3CACHESHA; X=$A3EXTRA; XSHA=$A3EXTRASHA
  else
    IDX=$(stored_a5 cache); X=$(stored_a5 extra)
    [ -n "$IDX" ] && [ -n "$X" ] || die "draw $K: A5's caches are not stored"
    [ -n "${CACHE_SHA:-}" ] && [ -n "${EXTRA_SHA:-}" ] || die "draw $K: give CACHE_SHA and EXTRA_SHA from A5's COMP-A5 lines"
    IDXSHA=$CACHE_SHA; XSHA=$EXTRA_SHA
  fi
  pin $IDX $IDXSHA; pin $X $XSHA
  WORDARGS="--variants core_crossfade --word-cache-key $IDX --word-cache-sha256 $IDXSHA --word-cache-n 16 --recognizer-run sign-recognizer-v2 --word-cache-extra-key $X --word-cache-extra-sha256 $XSHA $EXTRA"
}
LAST=length
# a draw with all arms stored is final: no other attempt runs
refuse_after_final() { local j; for j in 1 2 3 4 5; do [ "$j" != "$A" ] || continue; exists $OUT/$K-attempt$j-$LAST.json && die "$K attempt $j has all its arms: a finished draw is final"; done; return 0; }
# one write-once arm under this attempt and its report-only merge: arm ARM ARGS...
arm() {
  local name=$K-attempt$A-$1; shift
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json started=$OUT/$name.started
  if exists $merged; then echo "stored: $merged (not rerun)"; said $name $merged; return 0; fi
  exists $started && die "$name started and did not finish: rerun every arm under ATTEMPT=$(( A + 1 ))"
  python scripts/comp_a6_guard.py started $started || die "$name: cannot mark its start"
  python scripts/round15_compose_eval.py run $BASE $WORDARGS "$@" --output-key $part || die "run $name"
  python scripts/round15_compose_eval.py merge --report-only --shard-keys $part --output-key $merged > /root/r20-a6-$name-merge.log 2>&1 || die "merge $name (log /root/r20-a6-$name-merge.log)"
  said $name $merged
}
# an arm's comparator: this attempt's base, stored, pinned, from this pod
comparator() {
  B0=$OUT/$K-attempt$A-base.json
  if [ $SMOKING = 0 ]; then
    exists $B0 || die "$PHASE: its comparator $B0 comes first"
    [ -n "${BASE_SHA:-}" ] || die "$PHASE: give BASE_SHA, the hash of its comparator"
    pin $B0 "$BASE_SHA"
  fi
  if exists $B0; then
    exists $OUT/$K-attempt$A-$PHASE.json || python scripts/comp_a6_guard.py same-pod $B0 || die "$PHASE: base came from other code or another environment; rerun every arm under ATTEMPT=$(( A + 1 ))"
  fi
}

if [ "$PHASE" != summary ]; then
  if [ "$K" != a3 ] && [ $SMOKING = 0 ]; then python scripts/comp_a6_guard.py a5-done || die "draw $K: A6's draws wait for A5's summary"; fi
  refuse_after_final
fi
case $PHASE in
  base)
    words
    arm base $TODAY ;;
  hold-x)
    words; comparator
    arm hold-x $TODAY $HOLDX ;;
  hold-xr)
    words; comparator
    arm hold-xr $TODAY $HOLDX --hold-rest-file $REST --hold-rest-sha256 $RESTSHA ;;
  length)
    words; comparator
    arm length $FITTED ;;
  summary)
    KEY=$OUT/a6-summary-$K.json
    if exists $KEY; then echo "stored: $KEY"; said a6-summary-$K $KEY
    else
      ARGS=""; LIST="1 2 3 4 5 6 7 8"; [ "$K" = draws ] || LIST=a3
      for d in $LIST; do
        finished=""; log=""
        for j in 1 2 3 4 5; do
          has=""
          exists $OUT/$d-attempt$j-base.json || continue       # every other arm needs its base
          for a in base hold-x hold-xr length; do [ $a = base ] || exists $OUT/$d-attempt$j-$a.json || continue; has="$has$a+"; done
          [ -z "$has" ] || log="$log$j:$has;"
          if [ "$has" = "base+hold-x+hold-xr+length+" ]; then [ -z "$finished" ] || die "$d has two finished attempts, $finished and $j: a finished draw is final"; finished=$j; fi
        done
        [ -n "$finished" ] || die "incomplete: $d has no attempt with all its arms ($log); fewer than eight finished draws is no smaller test"
        echo "COMP-A6-ATTEMPTS words=$d attempts=$log used=$finished"
        keys=""
        for pair in "B base" "HX hold-x" "HXR hold-xr" "L length"; do
          set -- $pair
          var=$1$d"_SHA"; key=$OUT/$d-attempt$finished-$2.json
          [ -n "${!var:-}" ] || die "summary: give $var"
          pin $key "${!var}"
          keys="$keys,$key"
        done
        ARGS="$ARGS --draw $d=${keys#,} --attempts $d=$log"
      done
      python scripts/comp_a6_read.py $ARGS $DECL --output-key $KEY || die "summary"
      said a6-summary-$K $KEY
    fi ;;
esac
done_line 0
