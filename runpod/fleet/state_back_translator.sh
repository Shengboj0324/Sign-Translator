#!/usr/bin/env bash
# state_back_translator.sh — instrument v2, the state-to-text back-translator (run back-translator-state-t5-v1, kind
# state_to_text_t5; plan "Instrument v2: design and declared gates", declared 2026-10-30 09:10 UTC). PREPARED, NOT
# LAUNCHED: the plan requires an independent review of branch instrument-v2 before any pod trains.
# Commands as in reports/instrument_v2_dev.md. Three phases (BT_PHASE), each ending with STATE-BT-<PHASE>-DONE:
#   data  (CPU pod, BT_SHARD/BT_SHARDS): the ASL STEM Wiki and 2M-Flores manifests (write-once), then shard
#         BT_SHARD of the d2 round-trip copies of every training record (build_recon_states.py); BT_SHARD=merge
#         writes the index;
#   train (GPU pod): smoke run (300 steps, selection on 64 clips) and a smoke acceptance (32 clips), then the full run;
#   eval  (GPU pod): the W14b composition re-run twice (v1, v2; 4 shards each, merged), the arms re-read
#         (reread_arms.py), the acceptance (validate_state_back_translator.py: G1-G6).
set -uo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
PHASE=${BT_PHASE:?set BT_PHASE to data, train or eval}
O=acquisition/2026-10-17-v1/youtube-asl-v1; A=acquisition/2026-10-30-v1; OUT=$O/instrument-v2
MIX=$O/round14b-mixed-v1/manifest.json; MSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
TOK=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt; TSHA=021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de
F12=training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt
STEM=$A/asl-stem-wiki/asl-stem-wiki-v1/manifest.json; FLORES=$A/2m-flores-asl/2m-flores-asl-v1/manifest.json
RECON=$OUT/recon-d2-v1
RUN=back-translator-state-t5-v1; SMOKE=back-translator-state-t5-smoke-v1
H2S=how2sign-two-view-text-v1; YT=youtube-asl-v1
[ "$(sha $MIX)" = "$MSHA" ] && [ "$(sha $R10)" = "$R10SHA" ] && [ "$(sha $TOK)" = "$TSHA" ] || { echo "pin mismatch"; exit 1; }

if [ "$PHASE" = data ]; then
  set -e
  exists $STEM || python scripts/build_video_corpus_manifest.py --prefix $A/asl-stem-wiki/extract-v1 \
    --corpus-id asl-stem-wiki-v1 --text-field "sentence text" --holdout-count 4 --manifest-key $STEM \
    --licence "Microsoft Research License (ASL STEM Wiki): non-commercial research, no distribution; instruments only (owner, 2026-10-30 01:12 UTC)"
  exists $FLORES || python scripts/build_video_corpus_manifest.py --prefix $A/2m-flores-asl/extract-v2 \
    --corpus-id 2m-flores-asl-v1 --text-field sentence --crossdomain --manifest-key $FLORES --licence "CC BY-SA 4.0"
  SSHA=$(sha $STEM); echo "STEM $STEM $SSHA"; echo "FLORES $FLORES $(sha $FLORES)"
  if [ "${BT_SHARD:-0}" = merge ]; then
    python scripts/build_recon_states.py --prefix $RECON --merge --shards ${BT_SHARDS:-4}
  else
    python scripts/build_recon_states.py --manifest $MIX@$MSHA --manifest $STEM@$SSHA --corpus $H2S --corpus $YT \
      --corpus asl-stem-wiki-v1 --partition train --prefix $RECON --prior-checkpoint-key $F12 \
      --shards ${BT_SHARDS:-4} --shard ${BT_SHARD:-0} --workers ${BT_WORKERS:-28} --device cpu
  fi
  echo STATE-BT-DATA-DONE; exit 0
fi

SSHA=$(sha $STEM); FSHA=$(sha $FLORES); RISHA=$(sha $RECON/index.json)
CORPORA="--corpus $MIX@$MSHA:${H2S_WEIGHT:-4}:$H2S --corpus $MIX@$MSHA:1:$YT --corpus $STEM@$SSHA:1 --corpus $FLORES@$FSHA:1"
TRAIN="$CORPORA --template $R10@$R10SHA:$H2S --recon-prefix $RECON --recon-index-sha256 $RISHA --recon-p 0.4 \
  --validity-dropout 0.3 --monitor-clips ${MONITOR_CLIPS:-1000} --workers 10 --batch-size 64"

if [ "$PHASE" = train ]; then
  set -e
  # smoke first: every script of the chain on a 300-step model, so a bug surfaces in minutes
  exists training-runs/$SMOKE/DONE || { python scripts/train_state_to_text_t5.py $TRAIN --run $SMOKE --steps 300 \
    --val-every 300 --select-clips 64 --report-clips 32 --checkpoint-every 300; \
    python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put('training-runs/$SMOKE/DONE', b'done')"; }
  SK=$(best $SMOKE)
  set +e      # a smoke model fails the gates (exit 4) or cannot complete them (exit 2); only a crash stops the chain
  python scripts/validate_state_back_translator.py --translator-key $SK --translator-sha256 $(sha $SK) --limit 32 \
    --beam 1 --flores-manifest $FLORES@$FSHA --flores-clips 32 --output-key $OUT/smoke-acceptance.json
  rc=$?; set -e; [ $rc = 0 ] || [ $rc = 4 ] || [ $rc = 2 ] || { echo "smoke acceptance crashed ($rc)"; exit 1; }
  exists $OUT/smoke-acceptance.json || { echo "smoke acceptance wrote no report"; exit 1; }
  exists training-runs/$RUN/DONE || { python scripts/train_state_to_text_t5.py $TRAIN --run $RUN --steps 60000 \
    --val-every 2000 --checkpoint-every 4000 --patience 6; \
    python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put('training-runs/$RUN/DONE', b'done')"; }
  echo "INSTRUMENT-V2 $(best $RUN) $(sha $(best $RUN))"
  echo STATE-BT-TRAIN-DONE; exit 0
fi

if [ "$PHASE" = eval ]; then
  V2=$(best $RUN); V2SHA=$(sha $V2); echo "INSTRUMENT-V2 $V2 $V2SHA"
  # the W14b composition (round 15, 1,000 clips), exactly the declared run's command, read by v1 and by v2
  E=$O/round15/eval1000
  COMP="--clips 1000 --shards 4 --selection stage1 --timing sign-fitted --core-margin 0.1 --variants core_crossfade \
    --expect-model-sha256 42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21 \
    --stage1-report-key $O/round15/compositional-f12-step60000.json --device cuda --word-workers 4 \
    --checkpoint-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt \
    --generator-mix-key $MIX --generator-mix-sha256 $MSHA"
  for T in v1 v2; do
    EXTRA=""; [ $T = v2 ] && EXTRA="--translator-run $RUN"
    for S in 0 1 2 3; do
      exists $OUT/composition-$T-shard$S.json || python scripts/round15_compose_eval.py run $COMP $EXTRA --shard $S \
        --output-key $OUT/composition-$T-shard$S.json || exit 1
    done
    exists $OUT/composition-$T.json || python scripts/round15_compose_eval.py merge \
      --shard-keys $OUT/composition-$T-shard0.json,$OUT/composition-$T-shard1.json,$OUT/composition-$T-shard2.json,$OUT/composition-$T-shard3.json \
      --output-key $OUT/composition-$T.json || exit 1
  done
  python scripts/reread_arms.py --translator-key $V2 --translator-sha256 $V2SHA --composition-v1-rerun $OUT/composition-v1.json \
    --composition-v2 $OUT/composition-v2.json --output-key $OUT/reread-arms.json || exit 1
  python scripts/validate_state_back_translator.py --translator-key $V2 --translator-sha256 $V2SHA \
    --reread-key $OUT/reread-arms.json --flores-manifest $FLORES@$FSHA --output-key $OUT/acceptance.json
  echo "ACCEPTANCE exit $? (0 pass, 4 a gate failed, 2 incomplete)"
  echo STATE-BT-EVAL-DONE; exit 0
fi
echo "unknown BT_PHASE $PHASE"; exit 1
