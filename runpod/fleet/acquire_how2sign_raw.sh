#!/usr/bin/env bash
# acquire_how2sign_raw.sh I N SPLITS... — How2Sign raw 30 fps recordings (both cameras) into R2 for the take-5 recovery
# (audit B80): mirror shards I mod N of each split, each hashed against the mirror's LFS SHA-256 while streaming and
# re-read from R2; val/test members also checked against the publisher zips by CRC-32 and size. CC BY-NC 4.0.
set -uo pipefail
I=$1 N=$2; shift 2
source /root/r2env.sh
cd /root/work/Sign-Translator
python3 -m pip -q install boto3 > /dev/null 2>&1 || pip -q install boto3
for split in "$@"; do
  echo "STAGE raw $split shard $I/$N $(date)"
  for attempt in 1 2 3; do
    python3 scripts/acquisition/acquire_how2sign_raw.py --split "$split" --shard "$I" --shards "$N" \
      --ledger "/root/raw-ledger-$split-$I-of-$N.jsonl" && break
    echo "retrying $split after failure (attempt $attempt) $(date)"; sleep 60
    [ $attempt = 3 ] && echo "FAILED raw $split $I"
  done
done
python3 - "$I" "$N" <<'PY'
import glob, hashlib, os, sys, boto3
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"], region_name="auto",
                  aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
for local in glob.glob(f"/root/raw-ledger-*-{sys.argv[1]}-of-{sys.argv[2]}.jsonl"):
    data = open(local, "rb").read()
    s3.put_object(Bucket=os.environ["R2_BUCKET"], Key="acquisition/2026-10-17-v1/how2sign-raw/ledgers/" + os.path.basename(local),
                  Body=data, Metadata={"sha256": hashlib.sha256(data).hexdigest()})
    print("stored ledger", local, len(data))
PY
echo RAW-DONE
