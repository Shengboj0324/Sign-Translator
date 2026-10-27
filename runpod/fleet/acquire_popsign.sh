#!/usr/bin/env bash
# acquire_popsign.sh — PopSign ASL v1.0 (CC BY 4.0) into R2: every game-category training tar (raw, with provenance)
# and a per-sign selection of up to PER_SIGN clips (distinct participants first), extracted to canonical states as the
# selections land (extract_video_states.py, default hand crops: phone framing). Resumes. Research builds; attribution
# kept in every provenance file.
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1
cd /root/work/Sign-Translator
command -v apt-get > /dev/null && { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgles2 libgl1 curl > /dev/null; }
P=acquisition/2026-10-17-v1/popsign-v1
SIGNS=data/popsign_v1_game_signs.txt
echo "STAGE acquire $(wc -l < $SIGNS) signs $(date)"
python scripts/acquire_popsign.py --signs $SIGNS --per-sign ${PER_SIGN:-60} --parallel ${PAR:-3} > /root/acquire.jsonl 2>&1 &
ACQ=$!
done_signs=""
extract_one() {
  local s=$1 K=$P/selection/game/train/$1.zip
  local SHA=$(python -c "import os,sys,json; from signtranslator.data.text_motion_dataset import R2Store; print(json.loads(R2Store(os.environ).get(sys.argv[1]+'.provenance.json'))['sha256'])" $K)
  for pass in 1 2 3; do
    python scripts/extract_video_states.py --zip-key $K --zip-sha256 $SHA --prefix acquisition/2026-10-17-v1/derived/popsign-v1 \
      --source-id popsign-v1 --models-dir /root/work/models --workers ${WORKERS:-12} 2>&1 | grep -v -E "^W0000|inference_feedback" | tail -1
    [ ${PIPESTATUS[0]} -eq 0 ] && return 0
    sleep 20
  done
  echo "FAILED extract $s"
}
while true; do
  alive=0; kill -0 $ACQ 2>/dev/null && alive=1
  for s in $(grep -o '"sign": "[^"]*", "status": "\(converted\|already_present\)"' /root/acquire.jsonl | sed 's/"sign": "\([^"]*\)".*/\1/'); do
    case " $done_signs " in *" $s "*) continue;; esac
    extract_one $s; done_signs="$done_signs $s"
  done
  [ $alive = 0 ] && break
  sleep 30
done
wait $ACQ; echo "acquire exit $?"
echo "signs acquired: $(grep -c '"status": "converted"\|"status": "already_present"' /root/acquire.jsonl); errors: $(grep -c '"status": "error"' /root/acquire.jsonl)"
python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; R2Store(os.environ).put(sys.argv[1], open(sys.argv[2],'rb').read())" $P/logs/acquire.jsonl /root/acquire.jsonl || true
echo POPSIGN-DONE
