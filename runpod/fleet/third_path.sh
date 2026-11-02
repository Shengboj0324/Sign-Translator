#!/bin/bash
# third_path.sh — CPU is enough: the SignAvatars format check on 5 c02 train clips by exact member (scripts/third_path.py). Report only: no D20 member, no conversion and no instrument. The check stopped the conversion on 2026-11-02; this job only reproduces it.
# Ends with THIRD-PATH-DONE <status> (0 the format is as claimed, 1 failed, 4 it is not: nothing further may run). The exit status is the same.
[ -f "${R2ENV:-/root/r2env.sh}" ] && source "${R2ENV:-/root/r2env.sh}"
export PATH=${VENV_BIN:-/root/venv/bin}:$PATH
cd "${ST_REPO:-/root/work/Sign-Translator}" || { echo "THIRD-PATH-DONE 1"; exit 1; }
set -uo pipefail
D=acquisition/2026-10-17-v1/youtube-asl-v1/round20/third-path
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=${WORK:-/root/third-path}
OUT=$W/format-$TS.json
mkdir -p "$W" || { echo "THIRD-PATH-DONE 1"; exit 1; }
python scripts/third_path.py format --out-file "$OUT" 2>&1 | tee -a "$W/third_path.log"
codes=("${PIPESTATUS[@]}")
status=${codes[0]}
[ "${codes[1]}" = 0 ] || { echo "ALERT the log could not be written (tee ${codes[1]})"; [ "$status" != 0 ] || status=1; }
if [ -f "$OUT" ]; then python scripts/third_path.py put --file "$OUT" --key "$D/format-$TS.json" || { echo "ALERT the format record was not stored"; [ "$status" != 0 ] || status=1; }; fi
echo "THIRD-PATH-DONE $status"
exit "$status"
