#!/usr/bin/env bash
# round3d_tokenizer_all.sh — the candidate's smoke run first (toy sizes), the real candidate only if the plumbing passed.
cd /root/work/Sign-Translator
R3D_SMOKE=1 bash runpod/fleet/round3d_tokenizer.sh > /root/smoke.out 2>&1
cat /root/smoke.out
if grep -E "^FAILED .*\(status [0-9]+\)" /root/smoke.out | grep -vq "(status 4)" || grep -q "^TOKENIZER-ERROR" /root/smoke.out \
   || ! grep -q "smoke: $CAND plumbing done" /root/smoke.out; then
  echo "smoke run failed: $CAND not started"; echo TOKD-ALL-DONE; exit 1
fi
echo "smoke run passed: starting $CAND"
bash runpod/fleet/round3d_tokenizer.sh
echo TOKD-ALL-DONE
