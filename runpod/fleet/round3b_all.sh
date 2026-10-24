#!/usr/bin/env bash
# round3b_all.sh — round 3b on one GPU pod: the smoke run first (round3b.sh at toy sizes); the real round only if
# every smoke stage passed. R3B-ALL-DONE at the end.
cd /root/work/Sign-Translator
R3B_SMOKE=1 bash runpod/fleet/round3b.sh > /root/smoke.out 2>&1
cat /root/smoke.out
if grep -q "^FAILED" /root/smoke.out || ! grep -q "^R3B-DONE" /root/smoke.out; then
  echo "smoke run failed: the real round is not started"; echo R3B-ALL-DONE; exit 1
fi
echo "smoke run passed: starting round 3b"
bash runpod/fleet/round3b.sh
echo R3B-ALL-DONE
