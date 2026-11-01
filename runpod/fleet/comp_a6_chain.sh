#!/usr/bin/env bash
# comp_a6_chain.sh WORDS COMP_A_DECLARED [ATTEMPT]: one word draw of A6 in order on one pod, its smoke first (runpod/fleet/comp_a6.sh), so a draw's arms share code, packages and GPU. WORDS is a3 or A5's draw 1 to 8; a draw takes CACHE_SHA and
# EXTRA_SHA from the environment. An arm's comparator hash comes from base's own COMP-A6 line, printed for stored arms too. Any failure stops the chain. Whatever ends it, it prints ALERT for a failure and one COMP-A6-CHAIN-DONE <0|1> last. After a
# failure, rerun this with the next ATTEMPT: every arm runs again on the same words. Evaluation only. PREPARED, NOT LAUNCHED.
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
for P in base hold-x hold-xr length; do run COMP_PHASE=$P WORDS=$W SMOKE=1 SMOKE_STAMP="$STAMP" ATTEMPT=1; done
: > "$LOG"
run COMP_PHASE=base WORDS=$W ATTEMPT="$ATTEMPT" COMP_A_DECLARED="$DECLARED"
B=$(hash_of "$W-attempt$ATTEMPT-base") || stop "no hash for base of $W attempt $ATTEMPT"
for P in hold-x hold-xr length; do run COMP_PHASE=$P WORDS=$W ATTEMPT="$ATTEMPT" COMP_A_DECLARED="$DECLARED" BASE_SHA="$B"; done
HX=$(hash_of "$W-attempt$ATTEMPT-hold-x") || stop "no hash for hold-x of $W attempt $ATTEMPT"
HXR=$(hash_of "$W-attempt$ATTEMPT-hold-xr") || stop "no hash for hold-xr of $W attempt $ATTEMPT"
L=$(hash_of "$W-attempt$ATTEMPT-length") || stop "no hash for length of $W attempt $ATTEMPT"
echo "COMP-A6-HASHES words=$W attempt=$ATTEMPT B${W}_SHA=$B HX${W}_SHA=$HX HXR${W}_SHA=$HXR L${W}_SHA=$L"
ENDED=1
echo "COMP-A6-CHAIN-DONE 0"
