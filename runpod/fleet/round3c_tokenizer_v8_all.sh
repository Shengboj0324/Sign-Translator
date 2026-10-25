#!/usr/bin/env bash
# round3c_tokenizer_v8_all.sh — candidate s8v8 (amendment 8, declared 2026-10-25 07:00 PDT before any s8v8/s8v9 result):
# s8v6 at step 19000 (EMA weights) fine-tuned with quantiser dropout 0 (was 0.5), the same velocity weights, lr 1e-4,
# up to 10000 steps with validation every 500 (patience 5); judged by the same three checks; after c1024 s6 s8 s8v s8v3 s8v5 s8v6 s8v7 in precedence.
export CAND=s8v8 VW="body=0.05 left_hand=0.05 right_hand=0.05 head_rotation=0.3 head_translation=0.3" PREV="c1024 s6 s8 s8v s8v3 s8v5 s8v6 s8v7"
export FT_STEPS=10000 FT_EVERY=500
export TEXTRA="--quantizer-dropout 0 --init-key training-runs/tokens-rvq-d0b-r3c-s8v6/checkpoints/step-00019000.pt --init-sha256 b52fa16b53176bcfbf54c9c52213994124e2c20a93594fe0462e7f8850bdac83 --lr 1e-4 --warmup 200"
exec bash /root/work/Sign-Translator/runpod/fleet/round3c_tokenizer_v_all.sh
