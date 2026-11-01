#!/usr/bin/env bash
# comp_a6.sh — post-S2 A6 on D20: a 2x2 of hold-x and the fitted length on A3 generated top10, paired within A5's stored word draws (scripts/comp_a6_read.py). Evaluation only: NOTHING TRAINS.
# THE LAUNCH MUST WAIT: A5's record is code digest 08b9a150, and this branch moves the digest of everything it merges into. Don't merge it into A5's code. A6's full runs wait for A5's summary, and run only if it reads confirmed or direction only.
# PREPARED, NOT LAUNCHED. NOTHING IS DECLARED BY THIS FILE: a full run stops unless COMP_A_DECLARED passes scripts/comp_a6_declared.py, a declaration apart from A1 to A5's.
#
#   COMP_PHASE=<phase> WORDS=<a3|1..8|draws> [ATTEMPT=<n, default 1>] [SMOKE=1] [SMOKE_STAMP=<stamp>] [COMP_A_DECLARED=<plan commit>] [hashes] bash runpod/fleet/comp_a6.sh
#
# D20 only, pinned; no option for another set. Reports are --report-only and write-once under $OUT. No word is generated: every arm reads stored caches. SMOKE=1 reads at most 5 real D20 clips under $OUT/smoke/<stamp>/.
# Nothing decides on a smoke, a smoke has no summary, and a smoke reads A3's words alone: never A5's draws before their summary.
# WORDS: a3 is A3's stored words, a screen with no rule of its own that adds hold-xr, report only, and gates the hold arms. 1 to 8 is A5's stored draw: A5T_SHA pins that draw's top10 report, and its settings name the word caches and their hashes.
# OUT below names this declaration's store. Any reader bug, or any code change after the screen or a draw, means re-declaring under a new OUT named here, with what was stored disclosed; no key is scoped by digest automatically.
# Threads are pinned at 8 (THREADS may only say 8): the match with A5's top10 is read on the same GPU and pinned thread count.
# HOLD_ARMS, below, is read from the committed span report at declaration. off: top10 sits inside the real-signer span on all six readings, so the hold has no wrist purpose: base and length alone, no screen, nothing spent on hold, and the hold rule reads "not run, wrist inside the real-signer span".
# on: the hold gate. WORDS=1..8 and WORDS=draws need the screen's stored summary, pinned by SCREEN_SHA. hold-x and hold-x-length run on the draws only if its point estimates of hold-x minus base sit above the hold floors on likelihood and spotting. Otherwise
# the draws run base and length, the length rule still reads, and the hold rule reads "not run, screen failed".
# Attempts: a draw with all arms stored is final, so an attempt is refused once another has its last arm. An arm that started and did not finish stays unfinished: the next attempt reruns all arms on the same words.
# Phases, on one pod per draw (runpod/fleet/comp_a6_chain.sh runs them in order):
#   base           top10 at today's length: hold-x's and length's comparator. A draw's must match A5's top10 for that draw and the screen's A3's stored top10: on the same GPU, at least 98%, or the draw stops.
#   hold-x         entries with at least 25% two-handed records and the list's new entries stay anchors; the arm is held between anchors. Read beside the rule.
#   length         top10 at the length fitted signer-balanced on c02 + c04: the length rule's arm.
#   hold-x-length  both: the hold rule's arm, against length (LENGTH_SHA).
#   hold-xr        WORDS=a3 only, report only: hold-x's anchors, and the held arm goes toward the rest pose and back on the longest ramps.
#   arms           SMOKE=1, WORDS=a3: prints COMP-A6-ARMS on or off for the chain.
#   gate           WORDS=1..8: prints COMP-A6-GATE open or closed for the chain.
#   smoke-read     SMOKE=1, WORDS=a3: the five smoke reports through the reader's every check and summary. It prints no number and decides nothing.
#   summary        WORDS=a3 or WORDS=draws: each draw's one attempt with all arms, pinned by B<k>_SHA, HX<k>_SHA, L<k>_SHA, HXL<k>_SHA (and HXRa3_SHA); the rules on the draws, nothing on a3. A5T<k>_SHA pins A5's top10 for the match; at least one draw must be on the same GPU.
# A merge's output, which holds the arm's numbers, goes to a file on the pod, not this log: no per-draw number is read until the summary.
# Ends with one line: COMP-A6-DONE <0|1> phase=<phase> words=<words> smoke=<0|1>. Any other way out prints ALERT and COMP-A6-DONE 1.
set -uo pipefail
HOLD_ARMS=off
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
T=8
[ "${THREADS:-8}" = 8 ] || die "THREADS: this job is pinned at 8"
[ -z "$C" ] || [ "$C" -ge 8 ] || die "this pod has $C CPUs: the job is pinned at 8 threads"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
uv pip install -q 'transformers==4.57.6' 'tokenizers==0.22.2' sentencepiece || die "uv pip install failed"
python scripts/comp_a6_guard.py env || die "this pod's python, torch, transformers or tokenizers are not the pinned ones"
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
# 0 stored, 3 absent; anything else is a store error (Python exits 1 on one), and stops the job
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 3)" "$1"; local rc=$?; [ $rc = 0 ] && return 0; [ $rc = 3 ] && return 1; die "store error on $1"; }
next_attempt() { if [ "$A" -lt 5 ]; then echo "rerun every arm under ATTEMPT=$(( A + 1 ))"; else echo "attempt 5 was the last: this draw needs a decision"; fi; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }
said() { local hash; hash=$(sha "$2"); [ -n "$hash" ] || die "$1: no hash of $2"; echo "COMP-A6 $1 $2 $hash"; }
case $A in 1|2|3|4|5) ;; *) die "ATTEMPT must be 1 to 5" ;; esac

[ -z "${SET:-}" ] || [ "$SET" = d20 ] || die "SET: this read is D20 only; C20, the 1,000 and test are refused"
case ${SMOKE:-0} in 0) ;; 1) SMOKING=1 ;; *) die "SMOKE must be 0 or 1" ;; esac
case $HOLD_ARMS in on|off) ;; *) die "HOLD_ARMS must be on or off" ;; esac
if [ "$HOLD_ARMS" = off ]; then
  case $PHASE in hold-x|hold-x-length|hold-xr) die "$PHASE: not run, wrist inside the real-signer span" ;; esac
  [ $SMOKING = 1 ] || [ "$K" != a3 ] || die "WORDS=a3: HOLD_ARMS=off, so no hold screen runs and nothing is spent on hold"
fi
case $PHASE in
  base|hold-x|length|hold-x-length) case $K in a3|1|2|3|4|5|6|7|8) ;; *) die "WORDS must be a3 or 1 to 8" ;; esac ;;
  hold-xr) [ "$K" = a3 ] || die "hold-xr: WORDS=a3 only, report only; it is not on the draws" ;;
  gate) case $K in 1|2|3|4|5|6|7|8) ;; *) die "gate: WORDS must be 1 to 8" ;; esac ;;
  arms) [ $SMOKING = 1 ] && [ "$K" = a3 ] || die "arms: SMOKE=1 and WORDS=a3" ;;
  smoke-read) [ $SMOKING = 1 ] && [ "$K" = a3 ] || die "smoke-read: SMOKE=1 and WORDS=a3" ;;
  summary) case $K in a3|draws) ;; *) die "summary: WORDS=a3 or WORDS=draws" ;; esac ;;
  *) die "unknown COMP_PHASE $PHASE: base hold-x length hold-x-length hold-xr gate arms smoke-read summary" ;; esac
[ $SMOKING = 0 ] || [ "$K" = a3 ] || die "a smoke reads WORDS=a3 only: never A5's draws before their summary"
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
A3TOP=$O/round20/comp-a/a3-gen-top10.json; A3TOPSHA=cc47eeb474db85573d92197e65d03165e1c41e94d5f67cce1de2638a4f67ec12
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

# A5's top10 of this draw with this hash, whatever its attempt, in FOUND (no subshell: a store error must stop the job)
stored_a5_top10() { local j key; FOUND=""; for j in 1 2 3 4 5; do key=$A5OUT/draw$1-attempt$j-top10.json; if exists $key && [ "$(sha $key)" = "$2" ]; then FOUND=$key; return 0; fi; done; return 0; }
# this words' stored caches, pinned: A3's by this file, a draw's by the ledger's COMP-A5 lines
words() {
  if [ "$K" = a3 ]; then
    IDX=$A3CACHE; IDXSHA=$A3CACHESHA; X=$A3EXTRA; XSHA=$A3EXTRASHA
  else
    [ -n "${A5T_SHA:-}" ] || die "draw $K: give A5T_SHA, the hash of A5's top10 of this draw"
    stored_a5_top10 $K "$A5T_SHA"; TOP=$FOUND
    [ -n "$TOP" ] || die "draw $K: no stored top10 of A5's has A5T_SHA"
    LINE=$(python scripts/comp_a6_guard.py a5-caches $TOP) || die "draw $K: cannot read the word caches from $TOP"
    set -- $LINE
    [ $# = 4 ] || die "draw $K: $TOP names no word caches"
    IDX=$1; IDXSHA=$2; X=$3; XSHA=$4
  fi
  pin $IDX $IDXSHA; pin $X $XSHA
  WORDARGS="--variants core_crossfade --word-cache-key $IDX --word-cache-sha256 $IDXSHA --word-cache-n 16 --recognizer-run sign-recognizer-v2 --word-cache-extra-key $X --word-cache-extra-sha256 $XSHA $EXTRA"
}
# the hold gate, from the screen's stored summary: open on the screen itself and on a smoke, which read A3's words
GATE=open; SCREENKEY=$OUT/a6-summary-a3.json; HOLD_WHY="not run, screen failed"
hold_gate() {
  if [ "$HOLD_ARMS" = off ]; then GATE=closed; HOLD_WHY="not run, wrist inside the real-signer span"; return 0; fi
  [ -n "${SCREEN_SHA:-}" ] || die "$K: give SCREEN_SHA, the hash of the screen's summary: it gates the hold arms"
  exists $SCREENKEY || die "$K: the screen's summary is not stored: the screen comes first"
  pin $SCREENKEY "$SCREEN_SHA"
  python scripts/comp_a6_guard.py hold-gate $SCREENKEY "${COMP_A_DECLARED:-}"; local rc=$?
  case $rc in 0) GATE=open ;; 3) GATE=closed ;; *) die "$K: the hold gate cannot read the screen" ;; esac
}
# a draw with all arms stored is final: no other attempt runs
STOPPED=$OUT/$K-match-stopped
# the instrument check: base against its top10. Below 98% on the same GPU the draw stops, and a marker refuses its later attempts until it is declared under a new OUT. A crash is a crash, never a difference
match_top10() {
  python scripts/comp_a6_guard.py a5-match $OUT/$K-attempt$A-base.json $1; local rc=$?
  case $rc in
    0) ;;
    4) python scripts/comp_a6_guard.py started $STOPPED || die "$K: cannot mark the stop"; die "$K: base differs from its top10 on the same GPU, below 98%: this draw stops until it is declared under a new OUT" ;;
    *) die "$K: the match guard crashed, exit $rc: this says nothing about the match" ;;
  esac
}
refuse_after_final() { local j LAST=hold-x-length; [ "$GATE" = open ] || LAST=length; exists $STOPPED && die "$K stopped on the match with its top10: refused until it is declared under a new OUT"; for j in 1 2 3 4 5; do [ "$j" != "$A" ] || continue; exists $OUT/$K-attempt$j-$LAST.json && die "$K attempt $j has all its arms: a finished draw is final"; done; return 0; }
# one write-once arm under this attempt and its report-only merge: arm ARM ARGS...
arm() {
  local name=$K-attempt$A-$1; shift
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json started=$OUT/$name.started
  if exists $merged; then echo "stored: $merged (not rerun)"; said $name $merged; return 0; fi
  exists $started && die "$name started and did not finish: $(next_attempt)"
  python scripts/comp_a6_guard.py started $started || die "$name: cannot mark its start"
  python scripts/round15_compose_eval.py run $BASE $WORDARGS "$@" --output-key $part || die "run $name"
  python scripts/round15_compose_eval.py merge --report-only --shard-keys $part --output-key $merged > /root/r20-a6-$name-merge.log 2>&1 || die "merge $name (log /root/r20-a6-$name-merge.log)"
  said $name $merged
}
# an arm's comparator: this attempt's, stored, pinned, from this pod: comparator NAME HASH
comparator() {
  local B0=$OUT/$K-attempt$A-$1.json
  if [ $SMOKING = 0 ]; then
    exists $B0 || die "$PHASE: its comparator $B0 comes first"
    [ -n "$2" ] || die "$PHASE: give the hash of its comparator $1"
    pin $B0 "$2"
  fi
  if exists $B0; then
    exists $OUT/$K-attempt$A-$PHASE.json || python scripts/comp_a6_guard.py same-pod $B0 || die "$PHASE: $1 came from other code or another environment; $(next_attempt)"
  fi
}

# a not-confirmed A5 spends nothing and reads no hold-x: every full phase, the screen's and the summary's among them
if [ $SMOKING = 0 ]; then python scripts/comp_a6_guard.py a5-done || die "A6's full runs wait for A5's summary, and run only if it reads confirmed or direction only"; fi
if [ "$PHASE" != summary ] && [ "$PHASE" != smoke-read ] && [ "$PHASE" != arms ]; then
  if [ "$K" != a3 ] && [ $SMOKING = 0 ]; then hold_gate; fi
  refuse_after_final
fi
case $PHASE in
  base)
    TOP=""
    words
    if [ "$K" = a3 ] && [ $SMOKING = 0 ]; then pin $A3TOP $A3TOPSHA; TOP=$A3TOP; fi
    arm base $TODAY
    [ -z "$TOP" ] || match_top10 $TOP ;;
  smoke-read)
    SMOKEARMS="base hold-x length hold-x-length hold-xr"; [ "$HOLD_ARMS" = on ] || SMOKEARMS="base length"
    keys=""; for a in $SMOKEARMS; do keys="$keys,$OUT/a3-attempt$A-$a.json"; done
    python scripts/comp_a6_read.py --smoke --draw a3=${keys#,} --output-key $OUT/a3-smoke-read.json > /root/r20-a6-smoke-read.log 2>&1 || die "smoke-read (log /root/r20-a6-smoke-read.log)"
    echo "A6 smoke read passed: five reports through every check and summary; no number printed" ;;
  arms)
    echo "COMP-A6-ARMS $HOLD_ARMS" ;;
  gate)
    echo "COMP-A6-GATE $GATE"
    [ "$GATE" = open ] || echo "COMP-A6-HOLD $HOLD_WHY" ;;
  hold-x)
    [ "$GATE" = open ] || die "hold-x: $HOLD_WHY"
    words; comparator base "${BASE_SHA:-}"
    arm hold-x $TODAY $HOLDX ;;
  length)
    words; comparator base "${BASE_SHA:-}"
    arm length $FITTED ;;
  hold-x-length)
    [ "$GATE" = open ] || die "hold-x-length: $HOLD_WHY"
    words; comparator length "${LENGTH_SHA:-}"
    arm hold-x-length $FITTED $HOLDX ;;
  hold-xr)
    words; comparator base "${BASE_SHA:-}"
    arm hold-xr $TODAY $HOLDX --hold-rest-file $REST --hold-rest-sha256 $RESTSHA ;;
  summary)
    KEY=$OUT/a6-summary-$K.json
    if exists $KEY; then echo "stored: $KEY"; said a6-summary-$K $KEY
    else
      ARGS=""; LIST="1 2 3 4 5 6 7 8"; ALLARMS="base hold-x length hold-x-length"; SHAS="B:base HX:hold-x L:length HXL:hold-x-length"
      if [ "$K" = a3 ]; then LIST=a3; ALLARMS="$ALLARMS hold-xr"; SHAS="$SHAS HXR:hold-xr"
      else
        hold_gate
        [ "$HOLD_ARMS" = off ] || ARGS="--screen-key $SCREENKEY --screen-sha256 $SCREEN_SHA"
        if [ "$GATE" = closed ]; then ALLARMS="base length"; SHAS="B:base L:length"; echo "COMP-A6-HOLD $HOLD_WHY"; fi
      fi
      WANT=""; for a in $ALLARMS; do WANT="$WANT$a+"; done
      for d in $LIST; do
        finished=""; log=""
        for j in 1 2 3 4 5; do
          has=""
          exists $OUT/$d-attempt$j-base.json || continue       # every other arm needs its base
          for a in $ALLARMS; do [ $a = base ] || exists $OUT/$d-attempt$j-$a.json || continue; has="$has$a+"; done
          [ -z "$has" ] || log="$log$j:$has;"
          if [ "$has" = "$WANT" ]; then [ -z "$finished" ] || die "$d has two finished attempts, $finished and $j: a finished draw is final"; finished=$j; fi
        done
        [ -n "$finished" ] || die "incomplete: $d has no attempt with all its arms ($log); fewer than eight finished draws is no smaller test"
        echo "COMP-A6-ATTEMPTS words=$d attempts=$log used=$finished"
        keys=""
        for pair in $SHAS; do
          var=${pair%%:*}$d"_SHA"; key=$OUT/$d-attempt$finished-${pair#*:}.json
          [ -n "${!var:-}" ] || die "summary: give $var"
          pin $key "${!var}"
          keys="$keys,$key"
        done
        ARGS="$ARGS --draw $d=${keys#,} --attempts $d=$log"
        if [ "$K" = draws ]; then      # the match: base against A5's top10 on the same words, each draw's count and rate printed
          var=A5T$d"_SHA"; [ -n "${!var:-}" ] || die "summary: give $var, A5's top10 of draw $d"
          stored_a5_top10 $d "${!var}"
          [ -n "$FOUND" ] || die "summary: no stored top10 of A5's draw $d has $var"
          ARGS="$ARGS --a5-top10 $d=$FOUND"
        else
          pin $A3TOP $A3TOPSHA
          ARGS="$ARGS --a5-top10 a3=$A3TOP"
        fi
      done
      python scripts/comp_a6_read.py $ARGS $DECL --output-key $KEY || die "summary"
      said a6-summary-$K $KEY
    fi ;;
esac
done_line 0
