#!/usr/bin/env bash
# round3c_prior_all.sh — round 3c's prior on one GPU pod: the 300-step smoke run first (round3c_prior.sh), the real
# prior only if it passed. R3C-PRIOR-ALL-DONE at the end.
cd /root/work/Sign-Translator
R3C_SMOKE=1 bash runpod/fleet/round3c_prior.sh > /root/smoke.out 2>&1
cat /root/smoke.out
if grep -q "^FAILED" /root/smoke.out || ! grep -q "^PRIOR-DONE" /root/smoke.out || grep -q "is not the declared" /root/smoke.out; then
  echo "smoke run failed: the real prior is not started"; echo R3C-PRIOR-ALL-DONE; exit 1
fi
echo "smoke run passed: starting the round-3c prior"
bash runpod/fleet/round3c_prior.sh
echo R3C-PRIOR-ALL-DONE
