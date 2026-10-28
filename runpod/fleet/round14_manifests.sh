#!/usr/bin/env bash
# round14_manifests.sh — round 14's data (one CPU pod, ~32 vCPU; docs/NEXT_PHASE_PLAN.md "Round 14"):
#   1. semlex-text-v1: Sem-Lex val + train one-word records from our own video extraction (states-video-v1, every
#      train/val shard's latest full-pass report per clip, each state hashed against its report), text from ASL-LEX
#      lemmas and plain-English free text, fps / hand / trim filters (scripts/build_semlex_manifest.py --source video).
#      The Sem-Lex test split is not in it (a separate held-out set). Written once to
#      sem-lex-mirror-v1/semlex-text-v1/{manifest,summary}.json, with R5 coverage against round10-mixed-v1.
#   2. round14-mixed-v1 = round10-mixed-v1 (F12's training mix, every record byte-identical) + semlex-text-v1
#      (scripts/build_mixed_text_manifest.py --base), written once to youtube-asl-v1/round14-mixed-v1/manifest.json.
# Stops without writing when: a train/val shard has no full-pass extraction report (train shard 2 is still running
# on 2026-10-28), a clip's best extraction result is error/failed/exists_different (rerun that shard), a state's bytes
# differ from its report, or the mix would move the trainer's first 1536/2048 validation records.
# Prints both SHA-256s and the summary. The free-text word list is Debian's wamerican (/usr/share/dict/american-english,
# its SHA-256 recorded in the manifest). Corpus weights are training flags: --corpus-weight semlex-text-v1=W.
# ROUND14-DONE <status> at the end. Research use only (Sem-Lex: CC BY-NC-SA 4.0).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
ACQ=acquisition/2026-10-17-v1; SL=$ACQ/sem-lex-mirror-v1
R10=$ACQ/youtube-asl-v1/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
SKEY=$SL/semlex-text-v1/manifest.json; SSUM=$SL/semlex-text-v1/summary.json
MKEY=$ACQ/youtube-asl-v1/round14-mixed-v1/manifest.json
OUT=/root/round14-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT   # a fresh local directory per run (local outputs are write-once)
done_() { echo "ROUND14-DONE $1"; exit $1; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
upload_once() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; s=R2Store(os.environ); k=sys.argv[1]; sys.exit('exists: '+k) if s.exists(k) else s.put(k, open(sys.argv[2],'rb').read())" "$1" "$2"; }
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
W=${WORKERS:-$(( C > 4 ? C - 4 : 1 ))}
WORDS=/usr/share/dict/american-english
[ -r $WORDS ] || { apt-get update -qq > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wamerican > /dev/null; }
[ -r $WORDS ] || { echo "ALERT no word list at $WORDS"; done_ 1; }
echo "word list $WORDS sha256 $(sha256sum $WORDS | cut -d' ' -f1) ($(wc -l < $WORDS) lines); workers $W"

echo "STAGE semlex-text-v1 $(date -u)"
if exists $SKEY; then
  echo "stored: $SKEY (not rebuilt)"
  exists $SSUM || { echo "ALERT $SKEY is stored but $SSUM is not (an interrupted upload): investigate"; done_ 3; }
  python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; s=R2Store(os.environ); open('$OUT/semlex.summary.json','wb').write(s.get('$SSUM'))" || done_ 1
else
  # UNMIRRORED=1: the re-solved states of runpod/fleet/unmirror_semlex_states.sh (only if Sem-Lex is judged mirrored)
  python scripts/build_semlex_manifest.py --source video --splits val,train --workers $W --wordlist $WORDS \
    ${UNMIRRORED:+--states-prefix $SL/states-video-v1-unmirrored} \
    --coverage-mix $R10@$R10SHA --output $OUT/semlex-text-v1.json --summary $OUT/semlex.summary.json \
    --output-key $SKEY --summary-key $SSUM > $OUT/stage-semlex.log 2>&1
  s=$?; tail -5 $OUT/stage-semlex.log
  upload_once $SL/semlex-text-v1/logs/stage-semlex-$(date -u +%Y%m%dT%H%M%SZ).log $OUT/stage-semlex.log || true
  [ $s = 0 ] || { echo "FAILED semlex-text-v1 (status $s): see $OUT/stage-semlex.log"; done_ $s; }
fi
SSHA=$(python -c "import json; print(json.load(open('$OUT/semlex.summary.json'))['manifest_sha256'])")
[ "$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" $SKEY)" = "$SSHA" ] \
  || { echo "ALERT $SKEY does not hash to the summary's $SSHA"; done_ 3; }

echo "STAGE round14-mixed-v1 $(date -u)"
if exists $MKEY; then
  echo "stored: $MKEY (not rebuilt)"
else
  python scripts/build_mixed_text_manifest.py --base $R10@$R10SHA --input $SKEY@$SSHA --corpus-id round14-mixed-v1 \
    --manifest-key $MKEY --check-val-prefix 1536 --check-val-prefix 2048 > $OUT/stage-mix.log 2>&1
  s=$?; tail -c 3000 $OUT/stage-mix.log; echo
  upload_once $SL/semlex-text-v1/logs/stage-round14-mix-$(date -u +%Y%m%dT%H%M%SZ).log $OUT/stage-mix.log || true
  [ $s = 0 ] || { echo "FAILED round14-mixed-v1 (status $s)"; done_ $s; }
fi
MSHA=$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" $MKEY)

python - "$OUT/semlex.summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print("FILTERS", json.dumps(s["counts"], sort_keys=True))
print("SPLIT", json.dumps(s["split"]["counts"]), "signer_overlap", json.dumps(s["split"]["signer_overlap"]))
for part in ("train", "val", "all"):
    d = s["describe"][part]
    print(f"{part.upper()}: records {d['records']}; trimmed s {d['trimmed_seconds']}; untrimmed s {d['untrimmed_seconds']}")
    print(f"  signers {d['signers']} (clips per signer {d['clips_per_signer']}); texts {d['texts']}, gloss labels "
          f"{d['gloss_labels']}, ASL-LEX codes {d['asl_lex_codes']}, label types {d['label_types']}")
    print(f"  signers left by forearm tracking {d.get('signers_left_by_forearm_tracking')} "
          f"(their clips {d.get('clips_of_left_signers_by_forearm_tracking')}); trim side {d['trim_side']}")
c = s["coverage"]
print(f"R5 COVERAGE of the {c['uncovered']} uncovered words: >=1 train clip {c['now_with_semlex_train_clips']}, "
      f">=5 {c['now_with_5_semlex_train_clips']}; still uncovered {c['still_uncovered']}")
print(f"  ceiling {c['ceiling_before']} -> {c['projected_ceiling']} (>=1 clip); at >=5 clips {c['ceiling_before_5']} -> "
      f"{c['projected_ceiling_5']}")
print("  clips per word", json.dumps(c["clips_per_word"], sort_keys=True))
print("REPORTS used", [(u["report"].rsplit("/", 1)[-1], u["shard"]) for u in s["reports"]["used"]],
      "ignored", [(i["report"].rsplit("/", 1)[-1], i["reason"]) for i in s["reports"]["ignored"]])
EOF
echo "SEMLEX-TEXT-V1 $SKEY sha256 $SSHA"
echo "ROUND14-MIXED-V1 $MKEY sha256 $MSHA"
done_ 0
