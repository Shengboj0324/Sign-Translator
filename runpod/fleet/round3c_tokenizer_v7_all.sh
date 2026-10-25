#!/usr/bin/env bash
# round3c_tokenizer_v7_all.sh — candidate s8v7 (amendment 7): stable velocity matching at 1.0 on head rotation and head
# translation (s8v's head weight, which put head speed in band) and 0.05 on body and both hands (where s8v's weight 1 cost
# pose accuracy); after c1024, s6, s8, s8v, s8v3, s8v5 and s8v6 in precedence.
export CAND=s8v7 VW="body=0.05 left_hand=0.05 right_hand=0.05 head_rotation=1 head_translation=1" PREV="c1024 s6 s8 s8v s8v3 s8v5 s8v6"
exec bash /root/work/Sign-Translator/runpod/fleet/round3c_tokenizer_v_all.sh
