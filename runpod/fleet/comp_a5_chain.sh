#!/usr/bin/env bash
# comp_a5_chain.sh DRAW COMP_A_DECLARED: one draw of A5 in order on one pod, its smoke first (runpod/fleet/comp_a5.sh), so a draw's two arms share code, packages and GPU. A comparator's
# hash comes from its own COMP-A5 line. Any failure stops the chain. Ends with COMP-A5-CHAIN-DONE <0|1>. Evaluation only. PREPARED, NOT LAUNCHED.
set -uo pipefail
cd /root/work/Sign-Translator
D=${1:?a draw, 1 to 5}; DECLARED=${2:?the plan commit}; LOG=/root/comp_a5_chain.log
stop() { echo "ALERT chain: $*"; echo "COMP-A5-CHAIN-DONE 1"; exit 1; }
run() { env "$@" bash runpod/fleet/comp_a5.sh 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || stop "$*"; }
hash_of() { local h; h=$(grep -E "^COMP-A5 $1 " "$LOG" | tail -n 1 | awk '{print $4}'); [ -n "$h" ] || stop "no hash for $1"; echo "$h"; }
STAMP=draw$D-$(date -u +%Y%m%dT%H%M%SZ)
for P in cache extra margin0 top10; do run COMP_PHASE=$P DRAW=$D SMOKE=1 SMOKE_STAMP="$STAMP" CACHE_PROCS=1; done
: > "$LOG"
for P in cache extra margin0; do run COMP_PHASE=$P DRAW=$D COMP_A_DECLARED="$DECLARED"; done
M=$(hash_of draw$D-margin0) || exit 1
run COMP_PHASE=top10 DRAW=$D COMP_A_DECLARED="$DECLARED" MARGIN0_SHA="$M"
echo "COMP-A5-CHAIN-DONE 0"
