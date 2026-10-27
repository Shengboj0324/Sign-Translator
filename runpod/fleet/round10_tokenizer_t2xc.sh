#!/usr/bin/env bash
# round10_tokenizer_t2xc.sh — round 10 amendment 2 (declared 2026-10-27 21:31 UTC): t2xb plus --velocity-cap 4 (t2xb diverged at
# step 8900 when head-translation velocity reached 1.9e7; terms above 4 are log-compressed). Same gates.
export CAND=t2xc DOWN=1 STAGES=4 TSTEPS=60000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --velocity-weight body=0.3 --velocity-weight left_hand=0.3 --velocity-weight right_hand=0.3 --corpus-weight how2sign-two-view-text-v1=2 --lr 1.5e-4 --warmup 2000 --skip-grad-spikes 8 --velocity-cap 4"
exec bash /root/work/Sign-Translator/runpod/fleet/round10_tokenizer_all.sh
