#!/usr/bin/env bash
# state_back_translator.sh — instrument v2, the state-to-text back-translator (run back-translator-state-t5-v1, kind
# state_to_text_t5; plan "Instrument v2: design and declared gates", 09:10 UTC, and its amendment 1, 09:56 UTC).
# PREPARED, NOT LAUNCHED (independent review first). Commands as in reports/instrument_v2_dev.md. Phases (BT_PHASE),
# in order, each ending with STATE-BT-<PHASE>-DONE:
#   manifests (CPU, once, after the ASL STEM Wiki extraction completes): the STEM and 2M-Flores manifests, write-once;
#             prints their SHA-256s, to be pinned below (STEM_SHA, FLORES_SHA) and committed before any later phase;
#   data      (CPU pods, BT_SHARD of BT_SHARDS): shard BT_SHARD of the d2 round-trip copies of every training record;
#   index     (CPU, once): merges the round-trip shards into the index and prints its SHA-256 (pin RECON_INDEX_SHA);
#   train     (GPU): smoke run (300 steps, selection on 64 clips, acceptance on 32 clips), then the full run;
#   eval      (GPU, the pilots' GPU type where available): the W14b composition re-run once, read by v1 and v2 on the
#             same generated states (4 shards, merged); the arms re-read; the acceptance (G1-G6).
# No phase builds what another phase reads: every input of data/index/train/eval is pinned by SHA-256 here.
# Worker counts are fixed numbers (never nproc: OMP_NUM_THREADS makes it lie).
set -uo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
mark() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], b'done')" "$1"; }
# pinned() NAME VALUE KEY: refuse a placeholder, refuse bytes that differ from the pin (no trust on first use)
pinned() { case "$2" in ""|PIN-*) echo "$1 is not pinned: run the phase that prints it, pin it here, commit"; exit 1;; esac
  [ "$(sha "$3")" = "$2" ] || { echo "$1: $3 differs from its pin"; exit 1; }; }
PHASE=${BT_PHASE:?set BT_PHASE to manifests, data, index, train or eval}
O=acquisition/2026-10-17-v1/youtube-asl-v1; A=acquisition/2026-10-30-v1; OUT=$O/instrument-v2
MIX=$O/round14b-mixed-v1/manifest.json; MSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
TOK=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt; TSHA=021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de
F12=training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt
STEM=$A/asl-stem-wiki/asl-stem-wiki-v1/manifest.json; FLORES=$A/2m-flores-asl/2m-flores-asl-v1/manifest.json
RECON=$OUT/recon-d2-v1
# ---- pins: fill from the manifests and index phases' output, commit, then run the later phases ----
STEM_SHA=PIN-after-manifests
FLORES_SHA=PIN-after-manifests
RECON_INDEX_SHA=PIN-after-index
RECON_SHARDS=4
MONITOR_CLIPS=536                      # amendment 1: every youtube-asl-v1 monitor clip
RUN=back-translator-state-t5-v1; SMOKE=back-translator-state-t5-smoke-v1
H2S=how2sign-two-view-text-v1; YT=youtube-asl-v1
[ "$(sha $MIX)" = "$MSHA" ] && [ "$(sha $R10)" = "$R10SHA" ] && [ "$(sha $TOK)" = "$TSHA" ] || { echo "pin mismatch"; exit 1; }

if [ "$PHASE" = manifests ]; then
  set -e        # write-once: an existing manifest is refused by the builder, never rebuilt or trusted here
  python scripts/build_video_corpus_manifest.py --prefix $A/asl-stem-wiki/extract-v1 \
    --corpus-id asl-stem-wiki-v1 --text-field "sentence text" --holdout-count 4 --manifest-key $STEM \
    --licence "Microsoft Research License (ASL STEM Wiki): non-commercial research, no distribution; instruments only (owner, 2026-10-30 01:12 UTC)"
  python scripts/build_video_corpus_manifest.py --prefix $A/2m-flores-asl/extract-v2 \
    --corpus-id 2m-flores-asl-v1 --text-field sentence --crossdomain --manifest-key $FLORES --licence "CC BY-SA 4.0"
  echo "PIN STEM_SHA=$(sha $STEM)"; echo "PIN FLORES_SHA=$(sha $FLORES)"
  echo STATE-BT-MANIFESTS-DONE; exit 0
fi
pinned STEM_SHA "$STEM_SHA" $STEM

if [ "$PHASE" = data ]; then
  set -e
  python scripts/build_recon_states.py --manifest $MIX@$MSHA --manifest $STEM@$STEM_SHA --corpus $H2S --corpus $YT \
    --corpus asl-stem-wiki-v1 --partition train --prefix $RECON --prior-checkpoint-key $F12 \
    --shards $RECON_SHARDS --shard ${BT_SHARD:?set BT_SHARD} --workers ${BT_WORKERS:-28} --device cpu
  echo STATE-BT-DATA-DONE; exit 0
fi

if [ "$PHASE" = index ]; then
  set -e
  python scripts/build_recon_states.py --prefix $RECON --merge --shards $RECON_SHARDS
  echo "PIN RECON_INDEX_SHA=$(sha $RECON/index.json)"
  echo STATE-BT-INDEX-DONE; exit 0
fi
pinned FLORES_SHA "$FLORES_SHA" $FLORES
pinned RECON_INDEX_SHA "$RECON_INDEX_SHA" $RECON/index.json
CORPORA="--corpus $MIX@$MSHA:4:$H2S --corpus $MIX@$MSHA:1:$YT --corpus $STEM@$STEM_SHA:1 --corpus $FLORES@$FLORES_SHA:1"
TRAIN="$CORPORA --template $R10@$R10SHA:$H2S --recon-prefix $RECON --recon-index-sha256 $RECON_INDEX_SHA --recon-p 0.4 \
  --validity-dropout 0.3 --heads 8 --feedforward 2048 --monitor-clips $MONITOR_CLIPS --workers 10 --batch-size 64"

if [ "$PHASE" = train ]; then
  set -e
  # smoke first: every script of the chain on a 300-step model, so a bug surfaces in minutes
  exists training-runs/$SMOKE/DONE || { python scripts/train_state_to_text_t5.py $TRAIN --run $SMOKE --steps 300 \
    --val-every 300 --select-clips 64 --report-clips 32 --checkpoint-every 300; mark training-runs/$SMOKE/DONE; }
  SK=$(best $SMOKE)
  if ! exists $OUT/smoke-acceptance.json; then
    set +e      # a smoke model fails the gates (exit 4) or cannot complete them (exit 2); only a crash stops the chain
    python scripts/validate_state_back_translator.py --translator-key $SK --translator-sha256 $(sha $SK) --limit 32 \
      --beam 1 --flores-manifest $FLORES@$FLORES_SHA --flores-clips 32 --output-key $OUT/smoke-acceptance.json
    rc=$?; set -e; [ $rc = 0 ] || [ $rc = 4 ] || [ $rc = 2 ] || { echo "smoke acceptance crashed ($rc)"; exit 1; }
  fi
  exists training-runs/$RUN/DONE || { python scripts/train_state_to_text_t5.py $TRAIN --run $RUN --steps 60000 \
    --val-every 2000 --checkpoint-every 4000 --patience 6; mark training-runs/$RUN/DONE; }
  echo "INSTRUMENT-V2 $(best $RUN) $(sha $(best $RUN))"
  echo STATE-BT-TRAIN-DONE; exit 0
fi

if [ "$PHASE" = eval ]; then
  V2=$(best $RUN); V2SHA=$(sha $V2); echo "INSTRUMENT-V2 $V2 $V2SHA"
  for K in $OUT/reread-arms.json $OUT/acceptance.json; do
    exists $K && { echo "$K exists: written once, not overwritten"; exit 1; }
  done
  # the W14b composition (round 15, 1,000 clips): the declared run's command, read by v1 and (--also-translator-run)
  # by v2 on the same generated states, in one run
  COMP="--clips 1000 --shards 4 --selection stage1 --timing sign-fitted --core-margin 0.1 --variants core_crossfade \
    --expect-model-sha256 42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21 \
    --stage1-report-key $O/round15/compositional-f12-step60000.json --device cuda --word-workers 4 \
    --checkpoint-key training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt \
    --generator-mix-key $MIX --generator-mix-sha256 $MSHA --also-translator-run $RUN"
  for S in 0 1 2 3; do
    exists $OUT/composition-shard$S.json || python scripts/round15_compose_eval.py run $COMP --shard $S \
      --output-key $OUT/composition-shard$S.json || exit 1
  done
  exists $OUT/composition.json || python scripts/round15_compose_eval.py merge \
    --shard-keys $OUT/composition-shard0.json,$OUT/composition-shard1.json,$OUT/composition-shard2.json,$OUT/composition-shard3.json \
    --output-key $OUT/composition.json || exit 1
  python scripts/reread_arms.py --translator-key $V2 --translator-sha256 $V2SHA --composition-rerun $OUT/composition.json \
    --instrument-run $RUN --output-key $OUT/reread-arms.json || exit 1
  python scripts/validate_state_back_translator.py --translator-key $V2 --translator-sha256 $V2SHA \
    --reread-key $OUT/reread-arms.json --flores-manifest $FLORES@$FLORES_SHA --output-key $OUT/acceptance.json
  rc=$?
  echo "ACCEPTANCE exit $rc (0 pass, 4 a gate failed, 2 incomplete)"
  case $rc in 0|2|4) ;; *) echo "the acceptance crashed ($rc)"; exit 1;; esac
  echo STATE-BT-EVAL-DONE; exit 0
fi
echo "unknown BT_PHASE $PHASE"; exit 1
