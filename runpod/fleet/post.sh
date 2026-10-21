#!/usr/bin/env bash
# post.sh — after the fleet's Sapiens pass: cameras, triangulation (16 parallel shards), manifest,
# motion-token training (tokenizer then prior) and evaluation, all on the subset prefix.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH
cd /root/work/Sign-Translator
export SPLITS=train SUBSET=runpod/subset-v1.json MODELS=/root/work/models PY=python \
       DERIVED=acquisition/2026-10-17-v1/derived/subset-v1
count() { python - "$1" <<'PY'
import os, sys, boto3
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"], region_name="auto")
n, token = 0, None
while True:
    kw = dict(Bucket=os.environ["R2_BUCKET"], Prefix=sys.argv[1], MaxKeys=1000)
    if token: kw["ContinuationToken"] = token
    r = s3.list_objects_v2(**kw); n += r.get("KeyCount", 0)
    if not r.get("IsTruncated"): break
    token = r["NextContinuationToken"]
print(n)
PY
}
echo "waiting for the fleet $(date)"
last=-1; still=0
while true; do
  f=$(count "$DERIVED/how2sign-sapiens-front-v1/keypoints/"); s=$(count "$DERIVED/how2sign-sapiens-side-v1/keypoints/")
  total=$((f + s)); echo "$(date +%H:%M) front $f side $s"
  if [ "$total" -eq "$last" ]; then still=$((still + 1)); else still=0; fi
  last=$total
  # done when both views are (nearly) complete and nothing new has landed for 10 minutes
  if [ "$f" -ge 6500 ] && [ "$s" -ge 6500 ] && [ "$still" -ge 2 ]; then break; fi
  sleep 300
done
echo "STAGE cameras $(date)"
bash runpod/pipeline.sh cameras || { echo "FAILED cameras"; exit 1; }
echo "STAGE triangulate $(date)"
for i in $(seq 0 15); do bash runpod/pipeline.sh triangulate $i 16 > /root/tri-$i.log 2>&1 & done; wait
grep -h '"sessions"' /root/tri-*.log
echo "STAGE manifest $(date)"
bash runpod/pipeline.sh manifest || { echo "FAILED manifest"; exit 1; }
echo "STAGE train $(date)"
TRAIN_FLAGS="--batch-size 64 --workers 8" bash runpod/pipeline.sh train tokens "${STEPS:-30000}" || { echo "FAILED train"; exit 1; }
echo "STAGE evaluate $(date)"
bash runpod/pipeline.sh evaluate tokens-prior-v1 || echo "FAILED evaluate"
echo "STAGE prior-loci $(date)"
M=$DERIVED/how2sign-text-manifest-v1/manifest.json
MSHA=$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$M")
TOK=$(python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; k=[x for x in R2Store(os.environ).list('training-runs/tokens-vq-v1/checkpoints/') if x.endswith('.pt')]; print(k[-1])")
TSHA=$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$TOK")
python scripts/train_motion_tokens.py --stage prior --manifest-key "$M" --manifest-sha256 "$MSHA" --run tokens-prior-loci-v1   --tokenizer-key "$TOK" --tokenizer-sha256 "$TSHA" --steps "${STEPS:-30000}" --batch-size 64 --workers 8 --locus-conditioning   && bash runpod/pipeline.sh evaluate tokens-prior-loci-v1 || echo "FAILED prior-loci"
echo "STAGE grammar-tracks $(date)"
M=$DERIVED/how2sign-text-manifest-v1/manifest.json
python scripts/annotate_grammar_tracks.py --manifest-key "$M" --manifest-sha256 "$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$M")"   --prefix "$DERIVED/grammar-tracks-v2" --workers 16 || echo "FAILED grammar-tracks"
echo "STAGE grammar-measure-two-view $(date)"
python scripts/measure_grammar_on_how2sign.py --states "$DERIVED/how2sign-two-view-v1" --clips 3000 --enrich-questions 1500   --workers 16 --output /root/grammar-two-view.json --output-key "$DERIVED/grammar-motion-v1/two-view-measurements.json"   || echo "FAILED grammar-measure"
echo POST-DONE
