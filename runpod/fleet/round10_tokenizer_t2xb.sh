#!/usr/bin/env bash
# round10_tokenizer_t2xb.sh — round-10 fallback for t2x (prepared 2026-10-27 20:01 UTC, velocity matching added 20:20 UTC; launched only by a declared amendment
# if t2x fails its gates): the same tokenizer at half the learning rate with a longer warm-up and gradient-spike skipping
# (round 3d amendment 1's d4 rule: an update above 8x the running median gradient norm is skipped).
export CAND=t2xb DOWN=1 STAGES=4 TSTEPS=60000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --velocity-weight body=0.3 --velocity-weight left_hand=0.3 --velocity-weight right_hand=0.3 --corpus-weight how2sign-two-view-text-v1=2 --lr 1.5e-4 --warmup 2000 --skip-grad-spikes 8"
exec bash /root/work/Sign-Translator/runpod/fleet/round10_tokenizer_all.sh
