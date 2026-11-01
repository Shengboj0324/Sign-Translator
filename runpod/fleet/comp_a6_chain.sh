#!/usr/bin/env bash
# comp_a6_chain.sh WORDS COMP_A_DECLARED [ATTEMPT]: one word draw of A6 in order on one pod, its smoke first (runpod/fleet/comp_a6.sh), so a draw's arms share code, packages and GPU. WORDS is a3 or A5's draw 1 to 8; a draw takes CACHE_SHA and
# EXTRA_SHA, A5T_SHA for A5's top10 of that draw, and SCREEN_SHA for the screen's summary, from the environment. If the screen closes the hold gate, the draw runs base and length alone and says so. The smoke always reads WORDS=a3, never A5's draws. A comparator's hash comes from its own COMP-A6 line, printed for stored arms too. Any failure stops the chain. Whatever ends it, it prints ALERT for a failure and one
# COMP-A6-CHAIN-DONE <0|1> last. After a failure, rerun this with the next ATTEMPT: every arm runs again on the same words. Evaluation only. PREPARED, NOT LAUNCHED: the launch waits for A5's summary and for this code's own declaration.
set -uo pipefail
ENDED=""
finish() { code=$?; if [ -z "$ENDED" ]; then echo "ALERT chain stopped unexpectedly, exit $code"; echo "COMP-A6-CHAIN-DONE 1"; fi; }
trap finish EXIT
stop() { echo "ALERT chain: $*"; ENDED=1; echo "COMP-A6-CHAIN-DONE 1"; exit 1; }
cd /root/work/Sign-Translator || stop "no code at /root/work/Sign-Translator"
W=${1:?a3 or a draw, 1 to 8}; DECLARED=${2:?the plan commit}; ATTEMPT=${3:-1}; LOG=/root/comp_a6_chain.log
: > "$LOG"
run() { env "$@" bash runpod/fleet/comp_a6.sh 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || stop "$*"; }
hash_of() { local h; h=$(grep -E "^COMP-A6 $1 " "$LOG" | tail -n 1 | awk '{print $4}'); if [ -z "$h" ]; then echo "ALERT chain: no hash for $1" >&2; return 1; fi; echo "$h"; }
STAMP=$W-attempt$ATTEMPT-$(date -u +%Y%m%dT%H%M%SZ)
for P in base hold-x length hold-x-length hold-xr smoke-read; do run COMP_PHASE=$P WORDS=a3 SMOKE=1 SMOKE_STAMP="$STAMP" ATTEMPT=1; done
: > "$LOG"
one() { run COMP_PHASE=$1 WORDS=$W ATTEMPT="$ATTEMPT" COMP_A_DECLARED="$DECLARED" "${@:2}"; }
GATE=open
if [ "$W" != a3 ]; then
  one gate
  GATE=$(grep -E '^COMP-A6-GATE ' "$LOG" | tail -n 1 | awk '{print $2}')
  case $GATE in open|closed) ;; *) stop "no hold gate for $W" ;; esac
fi
one base
B=$(hash_of "$W-attempt$ATTEMPT-base") || stop "no hash for base of $W attempt $ATTEMPT"
HOLDS=""
if [ "$GATE" = open ]; then
  one hold-x BASE_SHA="$B"
  HX=$(hash_of "$W-attempt$ATTEMPT-hold-x") || stop "no hash for hold-x of $W attempt $ATTEMPT"
  HOLDS=" HX${W}_SHA=$HX"
fi
one length BASE_SHA="$B"
L=$(hash_of "$W-attempt$ATTEMPT-length") || stop "no hash for length of $W attempt $ATTEMPT"
EXTRA=""
if [ "$W" = a3 ]; then
  one hold-xr BASE_SHA="$B"
  HXR=$(hash_of "$W-attempt$ATTEMPT-hold-xr") || stop "no hash for hold-xr of $W attempt $ATTEMPT"
  EXTRA=" HXR${W}_SHA=$HXR"
fi
if [ "$GATE" = open ]; then
  one hold-x-length LENGTH_SHA="$L"
  HXL=$(hash_of "$W-attempt$ATTEMPT-hold-x-length") || stop "no hash for hold-x-length of $W attempt $ATTEMPT"
  HOLDS="$HOLDS HXL${W}_SHA=$HXL"
else
  echo "COMP-A6-HOLD not run, screen failed"
fi
echo "COMP-A6-HASHES words=$W attempt=$ATTEMPT B${W}_SHA=$B L${W}_SHA=$L$HOLDS$EXTRA gate=$GATE"
ENDED=1
echo "COMP-A6-CHAIN-DONE 0"
