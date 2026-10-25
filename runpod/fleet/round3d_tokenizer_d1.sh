#!/usr/bin/env bash
# round3d_tokenizer_d1.sh — round-3d candidate d1 (declared 2026-10-25 08:20 PDT): s8 at step 34000 (EMA) warm-started: lr 1.5e-4 cosine over 60000 steps, warmup 500; velocity weight 0.3 on head
# rotation and head translation, codebooks following the learning rate, patience 10.
export CAND=d1 TSTEPS=60000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --init-key training-runs/tokens-rvq-d0b-r3c-s8/checkpoints/step-00034000.pt --init-sha256 36f2c84ec96d958d2d1adb3feb15413fca1ed07bac3cf3e3e5b9db2b6b0606e1 --lr 1.5e-4 --warmup 500"
exec bash /root/work/Sign-Translator/runpod/fleet/round3d_tokenizer_all.sh
