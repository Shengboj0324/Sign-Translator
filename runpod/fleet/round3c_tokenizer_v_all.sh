#!/usr/bin/env bash
# round3c_tokenizer_v_all.sh — s8v's smoke run first (toy sizes), the real candidate only if the plumbing passed.
cd /root/work/Sign-Translator
R3C_SMOKE=1 bash runpod/fleet/round3c_tokenizer_v.sh > /root/smoke.out 2>&1
cat /root/smoke.out
if grep -E "^FAILED .*\(status [0-9]+\)" /root/smoke.out | grep -vq "(status 4)" || grep -q "^TOKENIZER-ERROR" /root/smoke.out \
   || ! grep -q "smoke: s8v plumbing done" /root/smoke.out; then
  echo "smoke run failed: s8v not started"; echo TOKV-ALL-DONE; exit 1
fi
echo "smoke run passed: starting s8v"
bash runpod/fleet/round3c_tokenizer_v.sh
echo TOKV-ALL-DONE
