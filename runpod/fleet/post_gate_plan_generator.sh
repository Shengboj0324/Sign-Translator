#!/usr/bin/env bash
# post_gate_plan_generator.sh — plan item 2 on one GPU pod, started after task No.1's exit gate passes:
# the plan-conditioned generator. The full-corpus RVQ prior (pod A) is warm-started on How2Sign English + ASLLRP plan
# lines (one mixed manifest, each corpus's own signer-disjoint partitions), with the full-corpus tokenizer, all states at
# 24 fps (ASLLRP is 30). Then, on validation partitions only (test untouched):
#   - ASLLRP val (Jonathan): plan generator on the gold plan line vs the full English prior on the same utterances'
#     English (the pre-plan baseline) - retrieval R-precision and back-translation BLEU against the English, NMM
#     grammar against the plan's markers, real and blank references beside them;
#   - ASLLRP val: the citation-stitching baseline on the gold plan (recorded citation forms joined; no coarticulation,
#     agreement or phrasal non-manuals) - what the generator has to beat;
#   - How2Sign val: the plan generator on English, so any cost of the plan data to the English condition shows.
# The retrieval evaluator and back-translator stay the full-corpus ones (frozen, How2Sign-trained); their real/blank
# scores on ASLLRP show how far they transfer. Every stage log goes to R2. PLAN-DONE at the end.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /root/work/Sign-Translator
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
best() { python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get('training-runs/'+sys.argv[1]+'/best.json'))['checkpoint_key'])" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
upload() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" "$1" "$2" || true; }
stage() { local run=$1 name=$2; shift 2; echo "STAGE $name $run $(date)"; "$@" > "/root/stage-$name-$run.log" 2>&1; local s=$?
  tail -3 "/root/stage-$name-$run.log"; upload "training-runs/$run/stage-$name.log" "/root/stage-$name-$run.log"
  [ $s = 0 ] || echo "FAILED $name $run (status $s)"; return $s; }
ACQ=acquisition/2026-10-17-v1/derived
FULL=$ACQ/full-v1/how2sign-text-manifest-v1/manifest.json
PLAN=$ACQ/asllrp-dai2-v1/plan-manifest-v2/manifest.json
MIX=$ACQ/asllrp-dai2-v1/how2sign-asllrp-plan-mixed-v1/manifest.json      # research-only (ASLLRP): private prefix
OUT=$ACQ/asllrp-dai2-v1/plan-generator-v1
TOKRUN=tokens-rvq-d0b-full-v1 PRIOR=tokens-rvq-d0b-prior-loci-full-v1 RET=retrieval-hold-full-v1 BT=back-translator-full-v1
RUN=tokens-rvq-d0b-prior-loci-plan-v1
FSHA=$(sha "$FULL"); PSHA=$(sha "$PLAN")
[ "$PSHA" = b01877253ea45b62aeaec313e2b9ffb075c0363af9d9e33ff5e4fe8c1902044a ] || { echo "plan manifest changed"; # sign level: one-sign plans of the recogniser's reliable glosses (recogniser v2 passed its instrument validation)
python - "$MIX" "$MSHA" "$OUT/plan-glosses.json" <<'PY'
import os, sys, json, tempfile
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.corpus_manifest import load_manifest
s = R2Store(os.environ)
with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "m.json"; p.write_bytes(s.get(sys.argv[1]) + b"\n")
    m = load_manifest(p, expected_sha256=sys.argv[2])
glosses = sorted({g for r in m["records"] if r.get("partition") == "train" and r.get("plan") for g in r["plan"]["glosses"]})
s.put(sys.argv[3], json.dumps(glosses).encode())
PY
stage $RUN sign-level python scripts/score_sign_generation.py --recognizer-run sign-recognizer-v2 \
  --recognizer-validation-key training-runs/sign-recognizer-v2/validation.json --generator-key "$GEN" \
  --generator-manifest-sha256 "$MSHA" --plan-glosses-key "$OUT/plan-glosses.json" \
  --citizen-manifest-key $ACQ/asl-citizen-v1/manifest-v1/manifest.json \
  --citizen-manifest-sha256 708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6 --fps 24 --device cuda \
  --output-key "$OUT/evaluations/sign-level.json"
echo PLAN-DONE; exit 1; }
exists "$MIX" || stage $RUN mixed-manifest python scripts/build_mixed_text_manifest.py --input "$FULL@$FSHA" \
  --input "$PLAN@$PSHA" --corpus-id how2sign-asllrp-plan-mixed-v1 --manifest-key "$MIX" || { # sign level: one-sign plans of the recogniser's reliable glosses (recogniser v2 passed its instrument validation)
python - "$MIX" "$MSHA" "$OUT/plan-glosses.json" <<'PY'
import os, sys, json, tempfile
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.corpus_manifest import load_manifest
s = R2Store(os.environ)
with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "m.json"; p.write_bytes(s.get(sys.argv[1]) + b"\n")
    m = load_manifest(p, expected_sha256=sys.argv[2])
glosses = sorted({g for r in m["records"] if r.get("partition") == "train" and r.get("plan") for g in r["plan"]["glosses"]})
s.put(sys.argv[3], json.dumps(glosses).encode())
PY
stage $RUN sign-level python scripts/score_sign_generation.py --recognizer-run sign-recognizer-v2 \
  --recognizer-validation-key training-runs/sign-recognizer-v2/validation.json --generator-key "$GEN" \
  --generator-manifest-sha256 "$MSHA" --plan-glosses-key "$OUT/plan-glosses.json" \
  --citizen-manifest-key $ACQ/asl-citizen-v1/manifest-v1/manifest.json \
  --citizen-manifest-sha256 708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6 --fps 24 --device cuda \
  --output-key "$OUT/evaluations/sign-level.json"
echo PLAN-DONE; exit 1; }
MSHA=$(sha "$MIX")
TOK=$(best $TOKRUN); TSHA=$(sha "$TOK"); INIT=$(best $PRIOR); ISHA=$(sha "$INIT")
# same architecture as the full prior (so its weights load); fine-tuning rate and schedule
stage $RUN prior python scripts/train_motion_tokens.py --architecture rvq --manifest-key "$MIX" --manifest-sha256 "$MSHA" \
  --workers 8 --val-every 1000 --val-batches 32 --checkpoint-every 2000 --patience 5 --batch-size 64 --stage prior --run $RUN \
  --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" --tokenizer-manifest-sha256 "$FSHA" --init-key "$INIT" --init-sha256 "$ISHA" \
  --fps 24 --lr 1e-4 --warmup 500 --steps 20000 --d-model 384 --feedforward 1536 --token-layers 6 --depth-layers 2 \
  --locus-conditioning --cache-bytes 12884901888 || { # sign level: one-sign plans of the recogniser's reliable glosses (recogniser v2 passed its instrument validation)
python - "$MIX" "$MSHA" "$OUT/plan-glosses.json" <<'PY'
import os, sys, json, tempfile
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.corpus_manifest import load_manifest
s = R2Store(os.environ)
with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "m.json"; p.write_bytes(s.get(sys.argv[1]) + b"\n")
    m = load_manifest(p, expected_sha256=sys.argv[2])
glosses = sorted({g for r in m["records"] if r.get("partition") == "train" and r.get("plan") for g in r["plan"]["glosses"]})
s.put(sys.argv[3], json.dumps(glosses).encode())
PY
stage $RUN sign-level python scripts/score_sign_generation.py --recognizer-run sign-recognizer-v2 \
  --recognizer-validation-key training-runs/sign-recognizer-v2/validation.json --generator-key "$GEN" \
  --generator-manifest-sha256 "$MSHA" --plan-glosses-key "$OUT/plan-glosses.json" \
  --citizen-manifest-key $ACQ/asl-citizen-v1/manifest-v1/manifest.json \
  --citizen-manifest-sha256 708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6 --fps 24 --device cuda \
  --output-key "$OUT/evaluations/sign-level.json"
echo PLAN-DONE; exit 1; }
GEN=$(best $RUN)
EVAL="--manifest-key $MIX --manifest-sha256 $MSHA --partition val --guidance 1,2 --fps 24 --device cuda \
  --part-temperature body=1.2,left_hand=1.2,right_hand=1.2,face=1.0 --retrieval-key $(best $RET) --retrieval-manifest-sha256 $FSHA"
A=asllrp-dai2-plan-v2 H=how2sign-two-view-text-v1
LEX=$ACQ/asllrp-dai2-v1/citation-lexicon-v1/lexicon.json
# the four evaluations are independent and generation leaves the GPU mostly idle: run them in parallel
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
( stage $RUN eval-asllrp-plan python scripts/evaluate_text_to_motion.py $EVAL --checkpoint-key "$GEN" --corpus $A --limit 412 \
  --save-generated "$OUT/generated/plan-asllrp" --output-key "$OUT/evaluations/plan-asllrp.json" ) &
( stage $RUN eval-asllrp-english-baseline python scripts/evaluate_text_to_motion.py $EVAL --checkpoint-key "$INIT" \
  --checkpoint-manifest-sha256 "$FSHA" --condition english --corpus $A --limit 412 \
  --save-generated "$OUT/generated/english-baseline-asllrp" --output-key "$OUT/evaluations/english-baseline-asllrp.json" ) &
( stage $RUN eval-asllrp-stitch python scripts/evaluate_text_to_motion.py $EVAL --lexicon-key "$LEX" --lexicon-sha256 "$(sha "$LEX")" \
  --corpus $A --limit 412 --save-generated "$OUT/generated/stitch-asllrp" --output-key "$OUT/evaluations/stitch-asllrp.json" ) &
( stage $RUN eval-how2sign python scripts/evaluate_text_to_motion.py $EVAL --checkpoint-key "$GEN" --corpus $H --limit 600 \
  --save-generated "$OUT/generated/plan-how2sign" --output-key "$OUT/evaluations/plan-how2sign.json" ) &
wait
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
stage $RUN back-translation python scripts/score_back_translation.py --manifest-key "$MIX" --manifest-sha256 "$MSHA" \
  --translator-key "$(best $BT)" --translator-manifest-sha256 "$FSHA" --fps 24 \
  --generated "$OUT/generated/plan-asllrp" "$OUT/generated/english-baseline-asllrp" "$OUT/generated/stitch-asllrp" \
  "$OUT/generated/plan-how2sign" \
  --output-key "$OUT/evaluations/back-translation.json"
# sign level: one-sign plans of the recogniser's reliable glosses (recogniser v2 passed its instrument validation)
python - "$MIX" "$MSHA" "$OUT/plan-glosses.json" <<'PY'
import os, sys, json, tempfile
from pathlib import Path
from signtranslator.data.text_motion_dataset import R2Store
from signtranslator.data_engineering.corpus_manifest import load_manifest
s = R2Store(os.environ)
with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "m.json"; p.write_bytes(s.get(sys.argv[1]) + b"\n")
    m = load_manifest(p, expected_sha256=sys.argv[2])
glosses = sorted({g for r in m["records"] if r.get("partition") == "train" and r.get("plan") for g in r["plan"]["glosses"]})
s.put(sys.argv[3], json.dumps(glosses).encode())
PY
stage $RUN sign-level python scripts/score_sign_generation.py --recognizer-run sign-recognizer-v2 \
  --recognizer-validation-key training-runs/sign-recognizer-v2/validation.json --generator-key "$GEN" \
  --generator-manifest-sha256 "$MSHA" --plan-glosses-key "$OUT/plan-glosses.json" \
  --citizen-manifest-key $ACQ/asl-citizen-v1/manifest-v1/manifest.json \
  --citizen-manifest-sha256 708af3a6748742c0ce8d147a984ef1de57abcec94da47ff0ce82391b067661a6 --fps 24 --device cuda \
  --output-key "$OUT/evaluations/sign-level.json"
echo PLAN-DONE
