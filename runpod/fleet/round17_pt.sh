#!/usr/bin/env bash
# round17_pt.sh — GPU pod: round 17 stage (a), the plan tokenizer plan-vq-v1 (plan "Round 17 design" 04:39 UTC, declared
# gates 7f06613 + amendments 16d2e00, a560fd0; reviewed and merged 9ad176a): smoke 300 steps, full 20000 steps, then the
# PT gate instruments (fidelity incl. plan round-trip BLEU, recognisability). Commands verbatim from
# reports/round17_plan_dev.md (a). ROUND17-PT-DONE at the end.
set -uo pipefail
source /root/r2env.sh; export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
uv pip install -q 'transformers>=4.40,<5' sentencepiece        # the back-translator (fidelity, diagnostics, pilot)
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
O=acquisition/2026-10-17-v1/youtube-asl-v1
MIX=$O/round14b-mixed-v1/manifest.json; MSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
TOK=training-runs/tokens-rvq-d0b-r3d-d2/checkpoints/step-00005000.pt; TSHA=021ee00a777261f915c2738efb671115c71eaf790f953ffc791931761dda08de
TMIX=$O/round3c-mixed-v1/manifest.json; TMSHA=004bf6596a2330ee5f840bfeef0d233174f63d0602f72465b6c28a21078d00fd
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
WEIGHTS="--corpus-weight asllvd-text-v1=3 --corpus-weight rit-text-r6=3 --corpus-weight wlasl-text-r6=3 --corpus-weight semlex-text-v2=2"
DATA="--fps 24 --conform-to-corpus how2sign-two-view-text-v1 --workers 10 --cache-bytes 107374182400"
DIAG="--mix-key $MIX --mix-sha256 $MSHA --clips 1000 --device cuda"
[ "$(sha $MIX)" = "$MSHA" ] && [ "$(sha $TOK)" = "$TSHA" ] && [ "$(sha $W14B)" = "$W14BSHA" ] || { echo "pin mismatch"; exit 1; }
python scripts/prefetch_cache.py $MIX || exit 1

set -e
python scripts/train_plan_tokenizer.py --run plan-vq-v1-smoke --manifest-key $MIX --manifest-sha256 $MSHA \
  --tokenizer-key $TOK --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA $DATA $WEIGHTS \
  --window 4 --hidden 256 --code-dim 64 --codebook 512 --plan-stages 2 --blocks 2 --pooling mean \
  --latent-weight 1 --motion-weight 1 --steps 300 --batch-size 64 --lr 3e-4 --warmup 500 \
  --val-every 100 --val-batches 32 --checkpoint-every 100 --fidelity-clips 64
python scripts/train_plan_tokenizer.py --run plan-vq-v1 --manifest-key $MIX --manifest-sha256 $MSHA \
  --tokenizer-key $TOK --tokenizer-sha256 $TSHA --tokenizer-manifest-sha256 $TMSHA $DATA $WEIGHTS \
  --window 4 --hidden 256 --code-dim 64 --codebook 512 --plan-stages 2 --blocks 2 --pooling mean \
  --latent-weight 1 --motion-weight 1 --steps 20000 --batch-size 64 --lr 3e-4 --warmup 500 \
  --val-every 1000 --val-batches 32 --checkpoint-every 2000 --fidelity-clips 64
P=training-runs/plan-vq-v1/checkpoints/step-00020000.pt; PSHA=$(sha $P); echo "PLAN-TOKENIZER $P $PSHA"
KP2D=$(best back-translator-kp2d-uwb-ft-v1); KPSHA=ad8a5876c37fb17e80bf0c5e8d45e6bc8cd51f6849c23057e47ad6ae4d12cfd2
python scripts/tokenizer_fidelity_check.py --manifest-key $TMIX --manifest-sha256 $TMSHA --tokenizer-key $TOK \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --translator-key "$KP2D" --translator-manifest-sha256 $KPSHA \
  --limit 1000 --device cuda --speed-joints-key $O/round3d/joint-reliability-how2sign-train.json \
  --output-key $O/round17/plan-fidelity-plan-vq-v1.json
python scripts/tokenizer_recognisability_check.py --manifest-key $TMIX --manifest-sha256 $TMSHA --corpus asllvd-text-v1 \
  --gloss-map-key acquisition/2026-10-17-v1/derived/signbank-v1/asl-citizen-gloss-map.json --tokenizer-key $TOK \
  --plan-tokenizer-key $P --plan-tokenizer-sha256 $PSHA --limit 300 --device cuda \
  --output-key $O/round17/plan-recognisability-plan-vq-v1.json
echo ROUND17-PT-DONE
