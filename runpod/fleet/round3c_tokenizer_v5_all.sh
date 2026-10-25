#!/usr/bin/env bash
# round3c_tokenizer_v5_all.sh — candidate s8v5 (amendment 4): 8 stages, the stable velocity matching (running energy
# scale, 418fb9e) at weight 0.05 on body, both hands, head rotation and head translation, so the per-frame loss keeps
# the poses in place; after c1024, s6, s8, s8v and s8v3 in precedence.
export CAND=s8v5 VW="body=0.05 left_hand=0.05 right_hand=0.05 head_rotation=0.05 head_translation=0.05" PREV="c1024 s6 s8 s8v s8v3"
exec bash /root/work/Sign-Translator/runpod/fleet/round3c_tokenizer_v_all.sh
