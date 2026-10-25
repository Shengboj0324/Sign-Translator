#!/usr/bin/env bash
# round3d_tokenizer_d4.sh — round-3d candidate d4 (round 3d amendment 1, declared 2026-10-25 09:35 PDT before any
# round-3d verdict): d3's recipe (from scratch, 100000 steps at lr 3e-4, head velocity 0.3, codebooks following the
# learning rate) plus --skip-grad-spikes 8: an update whose gradient norm exceeds 8x the running median is skipped.
export CAND=d4 TSTEPS=100000 TFLAGS="--velocity-weight head_rotation=0.3 --velocity-weight head_translation=0.3 --skip-grad-spikes 8"
exec bash /root/work/Sign-Translator/runpod/fleet/round3d_tokenizer_all.sh
