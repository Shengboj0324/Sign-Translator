#!/usr/bin/env bash
# round3c_tokenizer_v6_all.sh — candidate s8v6 (amendment 6): 8 stages, stable velocity matching at 0.3 on head rotation
# and head translation (near-still channels, where per-frame loss cannot see jitter) and 0.05 on body and both hands;
# after c1024, s6, s8, s8v, s8v3 and s8v5 in precedence.
export CAND=s8v6 VW="body=0.05 left_hand=0.05 right_hand=0.05 head_rotation=0.3 head_translation=0.3" PREV="c1024 s6 s8 s8v s8v3 s8v5"
exec bash /root/work/Sign-Translator/runpod/fleet/round3c_tokenizer_v_all.sh
