#!/usr/bin/env bash
# youtube_generator_round.sh — plan item 7 Y5 (generator), one GPU pod, after enough YouTube-ASL is acquired and the
# YouTube-pretrained instruments exist (youtube_pretrain_instruments.sh). The prior learns from How2Sign + YouTube-ASL
# (English captions) + ASLLRP plan lines, all conformed to How2Sign's layouts at 24 fps.
#   1. tokenizer domain check (scripts/tokenizer_domain_check.py, rule fixed: every channel's reconstruction error on
#      YouTube-ASL <= 1.5x How2Sign's). Pass: the full-corpus tokenizer stays and the full prior is warm-started.
#      Fail: a tokenizer is trained on the mix first and the prior trained from scratch on it (a full round).
#   2. evaluation on How2Sign val (same 600 clips, batched generation) against the full prior's own evaluation,
#      through the YouTube-pretrained instruments when present, and on ASLLRP val (plan lines);
#   3. the exit gate over both, then the strict review (same six rules) on the new selection.
# ROUND-DONE at the end. Test partitions are touched only by the strict review's R6.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
ACQ=acquisition/2026-10-17-v1; TAG=${YT_TAG:-v1}
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha "$FULL")
PLAN=$ACQ/derived/asllrp-dai2-v1/plan-manifest-v2/manifest.json; PSHA=$(sha "$PLAN")
YT=$ACQ/youtube-asl-v1/manifest-$TAG/manifest.json; YSHA=$(sha "$YT")
MIX=$ACQ/youtube-asl-v1/how2sign-youtube-plan-mixed-$TAG/manifest.json
exists "$MIX" || stage yt-round-$TAG mixed-manifest python scripts/build_mixed_text_manifest.py --input "$FULL@$FSHA" \
  --input "$YT@$YSHA" --input "$PLAN@$PSHA" --corpus-id how2sign-youtube-plan-mixed-$TAG --manifest-key "$MIX" \
  || { echo ROUND-DONE; exit 1; }
MSHA=$(sha "$MIX")
python scripts/prefetch_cache.py "$MIX"
D=$ACQ/youtube-asl-v1/generator-round-$TAG
TOKRUN=tokens-rvq-d0b-full-v1 PRIOR=tokens-rvq-d0b-prior-loci-full-v1 RUN=tokens-rvq-d0b-prior-ytmix-$TAG
TOK=$(best $TOKRUN); TSHA=$(sha "$TOK")
COMMON="--architecture rvq --manifest-key $MIX --manifest-sha256 $MSHA --workers 8 --val-every 1000 --val-batches 32 \
  --checkpoint-every 2000 --patience 5 --batch-size 64 --fps 24 --conform-to-corpus how2sign-two-view-text-v1 --cache-bytes 107374182400"
PRIOR_ARGS="--d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 --locus-conditioning"
if stage $RUN tokenizer-domain-check python scripts/tokenizer_domain_check.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" \
     --tokenizer-key "$TOK" --conform-to-corpus how2sign-two-view-text-v1 --corpus youtube-asl-$TAG \
     --output-key "$D/tokenizer-domain-check.json"; then
  INIT=$(best $PRIOR); ISHA=$(sha "$INIT")
  stage $RUN prior python scripts/train_motion_tokens.py $COMMON --stage prior --run $RUN --tokenizer-key "$TOK" \
    --tokenizer-sha256 "$TSHA" --tokenizer-manifest-sha256 "$FSHA" --init-key "$INIT" --init-sha256 "$ISHA" \
    --lr 1e-4 --warmup 500 --steps 40000 $PRIOR_ARGS || { echo ROUND-DONE; exit 1; }
else
  TOKRUN=tokens-rvq-d0b-ytmix-$TAG
  stage $TOKRUN tokenizer python scripts/train_motion_tokens.py $COMMON --stage tokenizer --run $TOKRUN --steps 40000 \
    --hidden 512 --code-dim 128 --codebook 512 --stages 4 --down-layers 0 --base-blocks 4 || { echo ROUND-DONE; exit 1; }
  TOK=$(best $TOKRUN); TSHA=$(sha "$TOK")
  stage $RUN prior python scripts/train_motion_tokens.py $COMMON --stage prior --run $RUN --tokenizer-key "$TOK" \
    --tokenizer-sha256 "$TSHA" --steps 40000 $PRIOR_ARGS || { echo ROUND-DONE; exit 1; }
fi
GEN=$(best $RUN)
RET=$(exists training-runs/retrieval-ytmix-$TAG/best.json && best retrieval-ytmix-$TAG || best retrieval-hold-full-v1)
BT=$(exists training-runs/back-translator-t5-ytft-$TAG/best.json && best back-translator-t5-ytft-$TAG || best back-translator-full-v1)
RMAN=$(python -c "import os,sys,io,torch; from signtranslator.data.text_motion_dataset import R2Store; print(torch.load(io.BytesIO(R2Store(os.environ).get(sys.argv[1])),map_location='cpu',weights_only=False)['manifest_sha256'])" "$RET")
BMAN=$(python -c "import os,sys,io,torch; from signtranslator.data.text_motion_dataset import R2Store; print(torch.load(io.BytesIO(R2Store(os.environ).get(sys.argv[1])),map_location='cpu',weights_only=False)['manifest_sha256'])" "$BT")
EVAL="--manifest-key $MIX --manifest-sha256 $MSHA --partition val --guidance 1,2 --fps 24 --device cuda --batch 32 \
  --part-temperature body=1.2,left_hand=1.2,right_hand=1.2,face=1.0 --retrieval-key $RET --retrieval-manifest-sha256 $RMAN"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
( stage $RUN eval-how2sign python scripts/evaluate_text_to_motion.py $EVAL --checkpoint-key "$GEN" \
    --corpus how2sign-two-view-text-v1 --limit 600 --save-generated "$D/generated/$RUN-h2s" --output-key "$D/evaluations/$RUN-h2s.json" ) &
( stage $RUN eval-asllrp python scripts/evaluate_text_to_motion.py $EVAL --checkpoint-key "$GEN" \
    --corpus asllrp-dai2-plan-v2 --limit 412 --save-generated "$D/generated/$RUN-asllrp" --output-key "$D/evaluations/$RUN-asllrp.json" ) &
wait
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
stage $RUN back-translation python scripts/score_back_translation.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" \
  --translator-key "$BT" --translator-manifest-sha256 "$BMAN" --fps 24 \
  --generated "$D/generated/$RUN-h2s" "$D/generated/$RUN-asllrp" --output-key "$D/evaluations/$RUN-bleu.json"
for h in 0.1 0.2; do
  stage $RUN smooth-$h python scripts/score_saved_generated.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" --fps 24 \
    --generated "$D/generated/$RUN-h2s" --head-smoothing $h --output-key "$D/evaluations/$RUN-h2s-hs$h.json"
  stage $RUN fc-$h python scripts/score_saved_generated.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" --fps 24 \
    --generated "$D/generated/$RUN-h2s" --head-smoothing $h --face-calibrate --output-key "$D/evaluations/$RUN-h2s-hs$h-fc.json"
done
stage $RUN gate python scripts/exit_gate.py --derived "$D" --runs $RUN-h2s $RUN-h2s-hs0.1 $RUN-h2s-hs0.2 $RUN-h2s-hs0.1-fc \
  $RUN-h2s-hs0.2-fc --output-key "$D/evaluations/$RUN-gate.json"
echo ROUND-DONE
