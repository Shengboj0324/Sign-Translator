#!/usr/bin/env bash
# One-command stages of the How2Sign two-view pipeline (docs/RUNPOD.md), resumable.
#
#   runpod/pipeline.sh sapiens SHARD SHARDS     # GPU pods: Sapiens keypoints, both views, all splits
#   runpod/pipeline.sh raw SPLIT                # one pod: raw recordings of both cameras into R2 (take-5 recovery)
#   runpod/pipeline.sh recut SHARD SHARDS       # GPU pods: take-5 lateral keypoints re-cut from the raw recordings
#   runpod/pipeline.sh cameras                  # one pod: shared camera intrinsics (written once)
#   runpod/pipeline.sh triangulate SHARD SHARDS # CPU pods: sync, calibrate, gate, two-view states
#   runpod/pipeline.sh manifest                 # one pod: training manifest from the two-view reports
#   runpod/pipeline.sh train tokens|diffusion|regression [STEPS]
#   runpod/pipeline.sh evaluate RUN             # signer-disjoint validation metrics of a run's newest checkpoint
#
# Every stage re-reads its inputs from R2 and pins them by SHA-256; every job skips
# work already written, so re-running a stage after an interruption continues it.
# Overridable: DERIVED (output root), MONO (monocular states), SPLITS, MATCH (clip
# filename regex, for smoke tests), SAPIENS_FLAGS, MODELS, TRAIN_FLAGS, RECUT_MATCH (session
# regex for recut), RECUT_FLAGS, LEDGER (local acquisition ledger of the raw stage).
set -euo pipefail
cd "$(dirname "$0")/.."
for v in R2_ENDPOINT_URL R2_BUCKET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  [[ -n "${!v:-}" ]] || { echo "missing secret $v" >&2; exit 1; }
done
PY=${PY:-python}
H=acquisition/2026-10-17-v1/how2sign-public
DERIVED=${DERIVED:-acquisition/2026-10-17-v1/derived}
MONO=${MONO:-acquisition/2026-10-17-v1/derived/how2sign-video-estimate-v1}
SPLITS=${SPLITS:-train val test}
MODELS=${MODELS:-/workspace/models}
CK=$MODELS/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2
CS=d1077c15e66fb2798893e569f71cb063f0b0778af2a62e3c0233807a351c31f3
FOLD_PLAN="Sign Translator Stage Documentation/evidence/w1-fold-plan-2026-10-17/fold-plan.json"
FOLD_SHA=00ea2bb9305bb78d96b25e4dbbc8eb2d1393ab862cd79857db56d73c11a34778
FRONT_PREFIX=$DERIVED/how2sign-sapiens-front-v1
SIDE_PREFIX=$DERIVED/how2sign-sapiens-side-v1
TWO_VIEW=$DERIVED/how2sign-two-view-v1
RAW=acquisition/2026-10-17-v1/how2sign-raw
RECUT=$DERIVED/how2sign-sapiens-side-recut-v1
CAMERAS=$DERIVED/how2sign-cameras-v1/intrinsics.json
MANIFEST=$DERIVED/how2sign-text-manifest-v1/manifest.json
MATCH_ARGS=()
[[ -n "${MATCH:-}" ]] && MATCH_ARGS=(--match "$MATCH")

zip_sha() {   # verified SHA-256 recorded at acquisition
  $PY -c "import os,json,sys; from signtranslator.data.text_motion_dataset import R2Store; p=json.loads(R2Store(os.environ).get(sys.argv[1]+'.provenance.json')); assert p['verified_by_r2_reread']; print(p['sha256'])" "$1"
}
object_sha() {
  $PY -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"
}
newest() {    # newest checkpoint key of a run
  $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; k=[x for x in R2Store(os.environ).list('training-runs/'+sys.argv[1]+'/checkpoints/') if x.endswith('.pt')]; assert k, 'no checkpoint'; print(k[-1])" "$1"
}

stage=${1:?stage required}; shift
case "$stage" in
  sapiens)
    shard=${1:?shard}; shards=${2:?shards}
    [[ -f $CK ]] || $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; open(sys.argv[1],'wb').write(R2Store(os.environ).get('acquisition/2026-10-17-v1/models-sapiens/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2'))" "$CK"
    for split in $SPLITS; do
      for view in front side; do
        key=$H/${split}_rgb_${view}_clips.zip
        prefix=$([[ $view == front ]] && echo "$FRONT_PREFIX" || echo "$SIDE_PREFIX")
        $PY scripts/sapiens_keypoints_runpod.py --zip-key "$key" --zip-sha256 "$(zip_sha "$key")" --prefix "$prefix" \
          --checkpoint "$CK" --checkpoint-sha256 $CS --shard "$shard" --shards "$shards" ${SAPIENS_FLAGS:-} ${MATCH_ARGS[@]+"${MATCH_ARGS[@]}"}
      done
    done ;;
  raw)
    split=${1:?split}
    $PY scripts/acquisition/acquire_how2sign_raw.py --split "$split" --ledger "${LEDGER:-${WORK:-/workspace}/acquisition-ledger.jsonl}" ;;
  recut)
    shard=${1:?shard}; shards=${2:?shards}
    [[ -f $CK ]] || $PY -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; open(sys.argv[1],'wb').write(R2Store(os.environ).get('acquisition/2026-10-17-v1/models-sapiens/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2'))" "$CK"
    inputs=()
    for split in $SPLITS; do
      key=$H/${split}_rgb_front_clips.zip
      inputs+=(--front-zip "$key=$(zip_sha "$key")" --sentences "$H/how2sign_${split}.csv")
    done
    $PY scripts/recut_how2sign_take5.py --front-keypoints "$FRONT_PREFIX" "${inputs[@]}" --raw-prefix "$RAW" --prefix "$RECUT" \
      --checkpoint "$CK" --checkpoint-sha256 $CS --shard "$shard" --shards "$shards" \
      ${RECUT_MATCH:+--match "$RECUT_MATCH"} ${RECUT_FLAGS:-} ;;
  cameras)
    $PY scripts/calibrate_how2sign_cameras.py --front-keypoints "$FRONT_PREFIX" --side-keypoints "$SIDE_PREFIX" \
      --sessions "${SESSIONS:-40}" --output-key "$CAMERAS" ;;
  triangulate)
    shard=${1:?shard}; shards=${2:?shards}
    $PY scripts/triangulate_how2sign_sessions.py --front-keypoints "$FRONT_PREFIX" --side-keypoints "$SIDE_PREFIX" \
      --recut-side-keypoints "$RECUT" --monocular "$MONO" --prefix "$TWO_VIEW" --intrinsics-key "$CAMERAS" --intrinsics-sha256 "$(object_sha "$CAMERAS")" \
      --shard "$shard" --shards "$shards" ;;
  manifest)
    $PY scripts/build_how2sign_text_manifest.py --reports-prefix "$TWO_VIEW/reports/" --corpus-id how2sign-two-view-text-v1 \
      --sentences $H/how2sign_realigned_train.csv $H/how2sign_realigned_val.csv $H/how2sign_realigned_test.csv \
      --fold-plan "$FOLD_PLAN" --fold-plan-sha256 $FOLD_SHA --manifest-key "$MANIFEST" ;;
  train)
    generator=${1:?tokens|diffusion|regression}; steps=${2:-200000}
    msha=$(object_sha "$MANIFEST")
    case "$generator" in
      tokens)
        $PY scripts/train_motion_tokens.py --stage tokenizer --manifest-key "$MANIFEST" --manifest-sha256 "$msha" \
          --run tokens-vq-v1 --steps "$steps" --hidden 512 --code-dim 256 --codebook 1024 --down-layers 1 ${TRAIN_FLAGS:-}
        tok=$(newest tokens-vq-v1)
        $PY scripts/train_motion_tokens.py --stage prior --manifest-key "$MANIFEST" --manifest-sha256 "$msha" \
          --run tokens-prior-v1 --tokenizer-key "$tok" --tokenizer-sha256 "$(object_sha "$tok")" --steps "$steps" ${TRAIN_FLAGS:-} ;;
      diffusion|regression)
        $PY scripts/train_text_to_motion.py --manifest-key "$MANIFEST" --manifest-sha256 "$msha" --run "t2m-$generator-v1" \
          --objective "$generator" --steps "$steps" ${TRAIN_FLAGS:-} ;;
      *) echo "unknown generator $generator" >&2; exit 2 ;;
    esac ;;
  evaluate)
    run=${1:?run}
    $PY scripts/evaluate_text_to_motion.py --manifest-key "$MANIFEST" --manifest-sha256 "$(object_sha "$MANIFEST")" \
      --checkpoint-key "$(newest "$run")" --partition val --guidance 1,2,3 \
      --output-key "$DERIVED/evaluations/$run.json" ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac
