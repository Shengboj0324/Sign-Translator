#!/usr/bin/env bash
# comp_a5_chain.sh DRAW COMP_A_DECLARED [ATTEMPT]: one draw of A5 in order on one pod, its smoke first (runpod/fleet/comp_a5.sh), so a draw's two arms share code, packages and GPU. A comparator's
# hash comes from its own COMP-A5 line, printed for stored arms too. Any failure stops the chain. Whatever ends it, it prints ALERT for a failure and one COMP-A5-CHAIN-DONE <0|1> last. After a
# failure, rerun this with the next ATTEMPT: both arms run again on the same seeds. Evaluation only. PREPARED, NOT LAUNCHED.
set -uo pipefail
ENDED=""
finish() { code=$?; if [ -z "$ENDED" ]; then echo "ALERT chain stopped unexpectedly, exit $code"; echo "COMP-A5-CHAIN-DONE 1"; fi; }
trap finish EXIT
stop() { echo "ALERT chain: $*"; ENDED=1; echo "COMP-A5-CHAIN-DONE 1"; exit 1; }
cd /root/work/Sign-Translator || stop "no code at /root/work/Sign-Translator"
D=${1:?a draw, 1 to 8}; DECLARED=${2:?the plan commit}; ATTEMPT=${3:-1}; LOG=/root/comp_a5_chain.log
: > "$LOG"
run() { env "$@" bash runpod/fleet/comp_a5.sh 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || stop "$*"; }
hash_of() { local h; h=$(grep -E "^COMP-A5 $1 " "$LOG" | tail -n 1 | awk '{print $4}'); if [ -z "$h" ]; then echo "ALERT chain: no hash for $1" >&2; return 1; fi; echo "$h"; }
STAMP=draw$D-attempt$ATTEMPT-$(date -u +%Y%m%dT%H%M%SZ)
for P in cache extra margin0 top10; do run COMP_PHASE=$P DRAW=$D SMOKE=1 SMOKE_STAMP="$STAMP" CACHE_PROCS=1 ATTEMPT=1; done
: > "$LOG"
for P in cache extra margin0; do run COMP_PHASE=$P DRAW=$D ATTEMPT="$ATTEMPT" COMP_A_DECLARED="$DECLARED"; done
M=$(hash_of "draw$D-attempt$ATTEMPT-margin0") || stop "no hash for margin0 of draw $D attempt $ATTEMPT"
run COMP_PHASE=top10 DRAW=$D ATTEMPT="$ATTEMPT" COMP_A_DECLARED="$DECLARED" MARGIN0_SHA="$M"
T=$(hash_of "draw$D-attempt$ATTEMPT-top10") || stop "no hash for top10 of draw $D attempt $ATTEMPT"
echo "COMP-A5-HASHES draw=$D attempt=$ATTEMPT M${D}_SHA=$M T${D}_SHA=$T"
ENDED=1
echo "COMP-A5-CHAIN-DONE 0"
