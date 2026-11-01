#!/usr/bin/env bash
# comp_a_chain.sh a1|a3|a4 COMP_A_DECLARED: one phase's arms in order on one pod, smokes first (runpod/fleet/comp_a.sh). A comparator's hash comes from
# its own COMP-A line. Any failure stops the chain. Ends with COMP-CHAIN-DONE <0|1>. Evaluation only.
set -uo pipefail
cd /root/work/Sign-Translator
WHICH=${1:?a1, a3 or a4}; DECLARED=${2:?the plan commit}; LOG=/root/comp_chain.log
stop() { echo "ALERT chain: $*"; echo "COMP-CHAIN-DONE 1"; exit 1; }
run() { env "$@" bash runpod/fleet/comp_a.sh 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || stop "$*"; }
line() { grep -E "^$1 $2 " "$LOG" | tail -n 1; }
hash_of() { local h; h=$(line COMP-A "$1" | awk '{print $4}'); [ -n "$h" ] || stop "no hash for $1"; echo "$h"; }
full() { run COMP_A_DECLARED="$DECLARED" "$@"; }
case $WHICH in
  a1)
    run COMP_PHASE=a1 ARM=margin0 SMOKE=1; run COMP_PHASE=a1 ARM=hold SMOKE=1
    : > "$LOG"
    full COMP_PHASE=a1 ARM=margin0
    M=$(hash_of a1-margin0) || exit 1
    for A in hold left0.5 both0.7; do full COMP_PHASE=a1 ARM=$A MARGIN0_SHA="$M"; done ;;
  a3)
    run COMP_PHASE=a3-oracle ARM=base SMOKE=1; run COMP_PHASE=a3-oracle ARM=top10 SMOKE=1
    : > "$LOG"
    full COMP_PHASE=a3-oracle ARM=base
    B=$(hash_of a3-oracle-base) || exit 1
    for A in top10 top30; do full COMP_PHASE=a3-oracle ARM=$A BASE_SHA="$B"; done ;;
  a4)
    run COMP_PHASE=a4-index SMOKE=1
    K=$(line COMP-A a4-clip-choice | awk '{print $3}'); [ -n "$K" ] || stop "no smoke index key"
    run COMP_PHASE=a4 ARM=sample0 SMOKE=1 CHOICE_KEY="$K"
    : > "$LOG"
    full COMP_PHASE=a4-index
    full COMP_PHASE=a4 ARM=sample0
    S=$(hash_of a4-clipchoice-sample0) || exit 1
    for A in n16 draw1 draw2 draw3 draw4; do full COMP_PHASE=a4 ARM=$A SAMPLE0_SHA="$S"; done
    full COMP_PHASE=a4-summary SAMPLE0_SHA="$S" ;;
  *) stop "a1, a3 or a4" ;;
esac
echo "COMP-CHAIN-DONE 0"
