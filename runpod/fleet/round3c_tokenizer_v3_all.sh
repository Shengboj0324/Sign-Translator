#!/usr/bin/env bash
# round3c_tokenizer_v3_all.sh — candidate s8v3 (amendment 3): velocity weight 3 on body and both hands, 1 on the head;
# it trains the prior only if c1024, s6, s8 and s8v all fail and it passes.
export CAND=s8v3 VW="body=3 left_hand=3 right_hand=3 head_rotation=1 head_translation=1" PREV="c1024 s6 s8 s8v"
exec bash /root/work/Sign-Translator/runpod/fleet/round3c_tokenizer_v_all.sh
