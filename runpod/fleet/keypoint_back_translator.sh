#!/usr/bin/env bash
# keypoint_back_translator.sh — plan item 7 fallback, one GPU pod: the 2D "second-opinion" back-translator.
#   1. pretrain on the UWB YouTube-ASL keypoints (353k clips), selected by zero-shot How2Sign-FK val BLEU, UWB dev logged;
#   2. fine-tune on How2Sign through FK (the judged domain);
#   3. ablation: the same model trained on How2Sign FK only (does the YouTube pretraining help?);
#   4. instrument acceptance (scripts/validate_keypoint_back_translator.py; rule declared in the plan) of the
#      better of 2 and 3 by val BLEU, with per-sentence agreement against back-translator v2 (T5);
#   5. only if it passes: second-opinion back-translation of the selected prior's core evaluation (generated vs real
#      vs blank vs shuffled).
# KP-DONE at the end. Test partitions are not read.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
bleu() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['val_bleu'])" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
done_mark() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put('training-runs/'+sys.argv[1]+'/DONE', b'done')" "$1"; }
# a training stage that completed is never re-entered (a resumed trainer would reset its patience and train on)
train_once() { local run=$1 name=$2; shift 2; exists "training-runs/$run/DONE" && { echo "SKIP $name $run (done)"; return 0; }
  stage $run $name "$@" && done_mark $run; }
PHASE=${KP_PHASE:-all}   # h2s: smoke + How2Sign-only model + its acceptance, before the UWB shards exist
ACQ=acquisition/2026-10-17-v1/derived
UWB=$ACQ/youtube-asl-uwb-keypoints-v1/shards-v1/index.json
H2S=$ACQ/full-v1/how2sign-fk-keypoints-v1/index.json
FULL=$ACQ/full-v1/how2sign-text-manifest-v1/manifest.json; FSHA=$(sha $FULL)
OUT=$ACQ/full-v1/keypoint-back-translator-v1
PRE=back-translator-kp2d-uwb-v2 FT=back-translator-kp2d-uwb-ft-v1 ONLY=back-translator-kp2d-h2s-v1
COMMON="--val-index $H2S --workers 10 --batch-size 64"
# smoke run first: every script of the chain on a 300-step model, so a bug surfaces in minutes, not after pretraining
SMOKE=back-translator-kp2d-smoke-v1
train_once $SMOKE smoke-train python scripts/train_keypoints_to_text_t5.py --train-index $H2S $COMMON --run $SMOKE --steps 300 \
  --val-every 300 --val-clips 64 --checkpoint-every 300 || { echo KP-DONE; exit 1; }
stage $SMOKE smoke-accept python scripts/validate_keypoint_back_translator.py --manifest-key $FULL --manifest-sha256 $FSHA \
  --translator-key "$(best $SMOKE)" --limit 32 --beam 1 --output-key "$OUT/smoke-acceptance.json"
[ -n "$(python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; print(R2Store(os.environ).exists('$OUT/smoke-acceptance.json') or '')")" ] \
  || { echo "smoke acceptance wrote no report"; echo KP-DONE; exit 1; }
if [ "$PHASE" = h2s ]; then
  train_once $ONLY h2s-only python scripts/train_keypoints_to_text_t5.py --train-index $H2S $COMMON --run $ONLY --steps 40000 \
    --val-every 1000 --checkpoint-every 2000 --patience 6 || { echo KP-DONE; exit 1; }
  stage $ONLY accept python scripts/validate_keypoint_back_translator.py --manifest-key $FULL --manifest-sha256 $FSHA \
    --translator-key "$(best $ONLY)" --reference-key "$(best back-translator-t5-full-v1)" --output-key "$OUT/acceptance-$ONLY.json"
  echo KP-DONE; exit 0
fi
# pretraining converges on the pretraining corpus's own held-out split (UWB dev); How2Sign zero-shot BLEU is logged but
# is noisy across the video -> FK domain gap (0.88 at 2k, 0.69 at 4k in the first attempt), so it does not stop it
train_once $PRE pretrain python scripts/train_keypoints_to_text_t5.py --train-index $UWB --monitor-index $UWB $COMMON --run $PRE \
  --select-on monitor --steps 100000 --val-every 4000 --checkpoint-every 4000 --patience 5 || { echo KP-DONE; exit 1; }
INIT=$(best $PRE)
train_once $FT finetune python scripts/train_keypoints_to_text_t5.py --train-index $H2S $COMMON --run $FT --init-key "$INIT" \
  --init-sha256 "$(sha $INIT)" --steps 30000 --lr 1e-4 --decoder-lr 3e-5 --warmup 500 --val-every 1000 --checkpoint-every 2000 \
  --patience 6 || { echo KP-DONE; exit 1; }
train_once $ONLY h2s-only python scripts/train_keypoints_to_text_t5.py --train-index $H2S $COMMON --run $ONLY --steps 40000 \
  --val-every 1000 --checkpoint-every 2000 --patience 6
SEL=$FT
if [ -n "$(best $ONLY 2>/dev/null)" ] && python -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$(bleu $ONLY)" "$(bleu $FT)"; then SEL=$ONLY; fi
echo "selected $SEL: val BLEU $(bleu $SEL) (fine-tuned $(bleu $FT), How2Sign-only $(bleu $ONLY 2>/dev/null))"
KP=$(best $SEL); REF=$(best back-translator-t5-full-v1)
if stage $SEL accept python scripts/validate_keypoint_back_translator.py --manifest-key $FULL --manifest-sha256 $FSHA \
     --translator-key "$KP" --reference-key "$REF" --output-key "$OUT/acceptance-$SEL.json"; then
  stage $SEL second-opinion python scripts/score_back_translation.py --manifest-key $FULL --manifest-sha256 $FSHA \
    --translator-key "$KP" --fps 24 --generated $ACQ/full-v1/generated/tokens-rvq-d0b-prior-loci-full-v1-pt \
    --output-key "$OUT/second-opinion-prior-loci-full-v1.json"
else
  echo "the 2D back-translator failed its acceptance rule: not used as an instrument"
fi
echo KP-DONE
