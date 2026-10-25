#!/usr/bin/env bash
# round3d_tokenizer_d3.sh — round-3d candidate d3 (declared 2026-10-25 08:20 PDT): from scratch, the s8 recipe over 100000 steps; velocity weight 0.3 on head
# rotation and head translation, codebooks following the learning rate, patience 10.
export CAND=d3 TSTEPS=100000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3"
exec bash /root/work/Sign-Translator/runpod/fleet/round3d_tokenizer_all.sh
