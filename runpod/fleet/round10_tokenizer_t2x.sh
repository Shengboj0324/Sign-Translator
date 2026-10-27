#!/usr/bin/env bash
# round10_tokenizer_t2x.sh — round-10 candidate t2x (declared 2026-10-27 18:10 UTC): d2's recipe from scratch with 2x temporal
# downsampling and 4 residual stages per part (16 codes per 12 Hz step: 192 codes/s against d2's 768), against the copycat shortcut.
export CAND=t2x DOWN=1 STAGES=4 TSTEPS=60000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --corpus-weight how2sign-two-view-text-v1=2"
exec bash /root/work/Sign-Translator/runpod/fleet/round10_tokenizer_all.sh
