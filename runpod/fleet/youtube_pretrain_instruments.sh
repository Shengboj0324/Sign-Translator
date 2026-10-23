#!/usr/bin/env bash
# youtube_pretrain_instruments.sh — plan item 7, stage Y5 (instruments first), on one GPU pod, once enough YouTube-ASL has
# been acquired. Builds the filtered YouTube-ASL manifest (evaluation sentences of How2Sign and ASLLRP removed), mixes
# it with How2Sign (How2Sign val stays the only validation), then:
#   1. back-translator v2 (motion encoder + T5) pretrained on How2Sign + YouTube-ASL at 24 fps,
#   2. fine-tuned from that on How2Sign alone,
#   3. retrieval evaluator trained on the mixed manifest (byte text encoder, augmented),
# each selected on How2Sign val. Instruments are private evaluation tools; their numbers are compared with
# back-translator-t5-full-v1 (How2Sign only) and retrieval-hold-full-v1. YT-PRETRAIN-DONE at the end.
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
ACQ=acquisition/2026-10-17-v1
FULL=$ACQ/derived/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha "$FULL")
PLAN=$ACQ/derived/asllrp-dai2-v1/plan-manifest-v2/manifest.json; PSHA=$(sha "$PLAN")
TAG=${YT_TAG:-v1}
YT=$ACQ/youtube-asl-v1/manifest-$TAG/manifest.json
MIX=$ACQ/youtube-asl-v1/how2sign-youtube-mixed-$TAG/manifest.json
exists "$YT" || stage youtube-asl-$TAG yt-manifest python scripts/build_youtube_asl_manifest.py --prefix $ACQ/youtube-asl-v1 \
  --corpus-id youtube-asl-$TAG --manifest-key "$YT" --eval-manifest "$FULL@$FSHA" --eval-manifest "$PLAN@$PSHA" \
  --licence "YouTube-ASL video IDs CC BY 4.0 (Google Research); videos owned by their creators under YouTube Terms; research use, private store, no redistribution" \
  || { echo YT-PRETRAIN-DONE; exit 1; }
YSHA=$(sha "$YT")
exists "$MIX" || stage youtube-asl-$TAG mixed-manifest python scripts/build_mixed_text_manifest.py --input "$FULL@$FSHA" \
  --input "$YT@$YSHA" --corpus-id how2sign-youtube-mixed-$TAG --manifest-key "$MIX" || { echo YT-PRETRAIN-DONE; exit 1; }
MSHA=$(sha "$MIX")
python scripts/prefetch_cache.py "$MIX"
PRE=back-translator-t5-ytpre-$TAG FT=back-translator-t5-ytft-$TAG
stage $PRE bt-pretrain python scripts/train_motion_to_text_t5.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" --run $PRE \
  --fps 24 --batch-size 32 --steps 60000 --val-every 2000 --val-clips 512 --patience 6 --workers 8 --cache-bytes 64424509440 \
  || { echo YT-PRETRAIN-DONE; exit 1; }
INIT=$(best $PRE); ISHA=$(sha "$INIT")
stage $FT bt-finetune python scripts/train_motion_to_text_t5.py --manifest-key "$FULL" --manifest-sha256 "$FSHA" --run $FT \
  --fps 24 --batch-size 32 --steps 20000 --lr 1e-4 --decoder-lr 2e-5 --warmup 500 --val-every 1000 --val-clips 512 \
  --patience 6 --workers 8 --init-key "$INIT" --init-sha256 "$ISHA"
RET=retrieval-ytmix-$TAG
stage $RET retrieval python scripts/train_text_motion_retrieval.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" --run $RET \
  --batch-size 128 --steps 40000 --val-every 500 --patience 8 --workers 8 --augment --fps 24 --dropout 0.3 --cache-bytes 64424509440
echo YT-PRETRAIN-DONE
