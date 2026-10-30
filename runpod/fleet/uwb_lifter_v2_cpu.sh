#!/usr/bin/env bash
# uwb_lifter_v2_cpu.sh — CPU pod: the lifter-v2 CPU phases back to back (split -> pairs -> crops -> train -> gate),
# each phase's printed SHA passed to the next. The split must reproduce the composition published in amendment 2
# (SHA 27c35505...), else the chain stops before any pair is built. The GPU downstream phase runs separately.
# UWB-V2-CPU-CHAIN-DONE at the end.
set -uo pipefail
PUBLISHED_SPLIT=27c355054920aa58ad095add17d75f5e938b1d4f3f7c8c68d7aa7da841fb8716
run() {   # run PHASE KEY: the phase's output must end with its DONE marker; prints the value of KEY=...
  bash runpod/fleet/uwb_lifter_v2.sh "$1" 2>&1 | tee /root/chain-$1.log >&2
  grep -q "UWB-V2-$(echo $1 | tr a-z A-Z)-DONE" /root/chain-$1.log || { echo "CHAIN-STOP: phase $1 did not finish" >&2; return 1; }
  grep "^$2=" /root/chain-$1.log | tail -1 | cut -d= -f2
}
SPLIT_SHA=$(run split SPLIT_SHA) || exit 1; export SPLIT_SHA
[ "$SPLIT_SHA" = "$PUBLISHED_SPLIT" ] || { echo "CHAIN-STOP: split $SPLIT_SHA is not the published $PUBLISHED_SPLIT"; exit 1; }
PAIRS_INDEX_SHA=$(run pairs PAIRS_INDEX_SHA) || exit 1; export PAIRS_INDEX_SHA
CROPS_SHA=$(run crops CROPS_SHA) || exit 1; export CROPS_SHA
BUNDLE_SHA=$(run train BUNDLE_SHA) || exit 1; export BUNDLE_SHA
GATE_SHA=$(run gate GATE_SHA) || exit 1; export GATE_SHA
echo "CHAIN-SHAS SPLIT_SHA=$SPLIT_SHA PAIRS_INDEX_SHA=$PAIRS_INDEX_SHA CROPS_SHA=$CROPS_SHA BUNDLE_SHA=$BUNDLE_SHA GATE_SHA=$GATE_SHA"
echo UWB-V2-CPU-CHAIN-DONE
