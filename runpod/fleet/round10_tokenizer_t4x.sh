#!/usr/bin/env bash
# round10_tokenizer_t4x.sh — round-10 candidate t4x (declared 2026-10-27 18:09 UTC): d2's recipe from scratch with 4x temporal
# downsampling and 4 residual stages per part (16 codes per 6 Hz step: 96 codes/s against d2's 768), against the copycat shortcut.
export CAND=t4x DOWN=2 STAGES=4 TSTEPS=60000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --corpus-weight how2sign-two-view-text-v1=2"
exec bash /root/work/Sign-Translator/runpod/fleet/round10_tokenizer_all.sh
