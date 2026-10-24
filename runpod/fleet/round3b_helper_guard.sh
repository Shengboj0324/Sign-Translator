#!/usr/bin/env bash
# round3b_helper_guard.sh — on round 3b's main pod while round3b_sweep_helper.sh runs elsewhere: stops the chain's own
# evaluations of the helper's share, but only those started by the chain's xargs sweep (their parent chain contains
# xargs), never one started by the chain's rerun step, which only runs an evaluation that is still missing. A stopped
# evaluation is reported FAILED by the sweep and found stored at the rerun step. Ends when the sweep's xargs has gone.
HELPER='sweep-h1\.3-f|sweep-h1\.2-f1\.2-'
seen_xargs=0
while true; do
  if pgrep -f "^xargs -P 3 -L 1 bash -c sweep_one" > /dev/null; then seen_xargs=1
  elif [ $seen_xargs = 1 ]; then echo "guard: sweep finished $(date -u +%H:%M)"; exit 0; fi
  for pid in $(pgrep -f "[e]valuate_text_to_motion.py"); do
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    echo "$cmd" | grep -qE -- "--output-key [^ ]*($HELPER)" || continue
    p=$pid; under_xargs=0
    for _ in 1 2 3 4 5 6; do p=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -z "$p" ] || [ "$p" = 1 ] && break
      ps -o args= -p $p | grep -q "^xargs" && { under_xargs=1; break; }; done
    if [ $under_xargs = 1 ]; then kill $pid && echo "guard: stopped the chain's copy $(echo "$cmd" | grep -oE 'sweep-h[^ /]*json') $(date -u +%H:%M)"; fi
  done
  sleep 15
done
