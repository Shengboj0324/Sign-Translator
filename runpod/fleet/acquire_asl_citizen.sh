#!/usr/bin/env bash
# ASL Citizen into private R2 (acquisition/2026-10-17-v1/asl-citizen-v1/): both Download Center files streamed with exact
# byte counts and end-to-end SHA-256 (acquire_resumable.py: HTTP -> memory parts -> R2 multipart, re-read from R2), the
# licence page archived beside them. Microsoft Research Licence Terms: non-commercial research only, no redistribution.
set -uo pipefail
source /root/r2env.sh
cd /root/work/Sign-Translator
python3 -m pip -q install boto3 >/dev/null 2>&1 || pip -q install boto3
BASE=https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4
PAGE=https://www.microsoft.com/en-us/download/details.aspx?id=105253
LIC="Microsoft Research License Terms (ASL Citizen): non-commercial, non-revenue-generating research only; no redistribution"
for spec in Demographics.zip:5077 ASL_Citizen.zip:45924134223; do
  f=${spec%%:*} n=${spec##*:}
  echo "STAGE acquire $f $(date)"
  python3 scripts/acquisition/acquire_resumable.py /root/work/Sign-Translator "$BASE/$f" "asl-citizen-v1/$f" \
    --expected-size "$n" --license "$LIC" --source-page "$PAGE" --ledger /root/asl-citizen-ledger.jsonl \
    || echo "FAILED acquire $f"
done
curl -sL https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/ -o /root/asl-citizen-license.html
python3 - <<'PY'
import os, boto3, hashlib
s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"], region_name="auto",
                  aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
for local, key in (("/root/asl-citizen-license.html", "acquisition/2026-10-17-v1/asl-citizen-v1/LICENSE-page-2026-10-21.html"),
                   ("/root/asl-citizen-ledger.jsonl", "acquisition/2026-10-17-v1/asl-citizen-v1/acquisition-ledger.jsonl")):
    if os.path.exists(local):
        data = open(local, "rb").read()
        s3.put_object(Bucket=os.environ["R2_BUCKET"], Key=key, Body=data, Metadata={"sha256": hashlib.sha256(data).hexdigest()})
        print("stored", key, len(data))
PY
echo ACQUIRE-DONE
