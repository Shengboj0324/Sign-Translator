#!/usr/bin/env bash
# round14_manifests.sh — round 14's data (one CPU pod, ~32 vCPU; docs/NEXT_PHASE_PLAN.md "Round 14"):
#   1. semlex-text-v1: Sem-Lex val + train one-word records from our own video extraction (states-video-v1, every
#      train/val shard's latest full-pass report per clip, each state hashed against its report), text from ASL-LEX
#      lemmas and plain-English free text, fps / hand / trim filters (scripts/build_semlex_manifest.py --source video).
#      The Sem-Lex test split is not in it (a separate held-out set). Written once to
#      sem-lex-mirror-v1/semlex-text-v1/{manifest,summary}.json, with R5 coverage against round10-mixed-v1.
#   2. round14-mixed-v1 = round10-mixed-v1 (F12's training mix, every record byte-identical) + semlex-text-v1
#      (scripts/build_mixed_text_manifest.py --base), written once to youtube-asl-v1/round14-mixed-v1/manifest.json.
# Environment (the defaults reproduce round 14 exactly):
#   CORPUS_ID (default semlex-text-v1): the Sem-Lex manifest's id; it is written to sem-lex-mirror-v1/$CORPUS_ID/.
#   MIX_ID (default round14-mixed-v1) and MKEY (default youtube-asl-v1/$MIX_ID/manifest.json): the mix.
#   EXCLUDE_SIGNERS (default none) with EXCLUDE_REASON (a default is given): Sem-Lex signers whose clips are dropped.
#   UNMIRRORED=1 (default unset): the re-solved states of runpod/fleet/unmirror_semlex_states.sh (withdrawn, 2026-10-29).
#   Round 14b (docs/NEXT_PHASE_PLAN.md "Sem-Lex per-signer sides", the original states-video-v1, image-right signers
#   excluded): CORPUS_ID=semlex-text-v2 MIX_ID=round14b-mixed-v1 EXCLUDE_SIGNERS=11,23,46,76
#   -> sem-lex-mirror-v1/semlex-text-v2/{manifest,summary}.json and youtube-asl-v1/round14b-mixed-v1/manifest.json.
# Stops without writing when: a train/val shard has no full-pass extraction report, a clip's best extraction result is
# error/failed/exists_different (rerun that shard), a state's bytes differ from its report, the mix would move the
# trainer's first 1536/2048 validation records, or a stored manifest or mix is not what these settings would build
# (another exclusion, or a mix without this corpus).
# Prints both SHA-256s and the summary. The free-text word list is Debian's wamerican (/usr/share/dict/american-english,
# its SHA-256 recorded in the manifest). Corpus weights are training flags: --corpus-weight $CORPUS_ID=W.
# ROUND14-DONE <status> at the end. Research use only (Sem-Lex: CC BY-NC-SA 4.0).
set -uo pipefail
source /root/r2env.sh
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/work/Sign-Translator
ACQ=acquisition/2026-10-17-v1; SL=$ACQ/sem-lex-mirror-v1
R10=$ACQ/youtube-asl-v1/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
# ---- parameters (env) ----
CORPUS_ID=${CORPUS_ID:-semlex-text-v1}
MIX_ID=${MIX_ID:-round14-mixed-v1}
SKEY=$SL/$CORPUS_ID/manifest.json; SSUM=$SL/$CORPUS_ID/summary.json
MKEY=${MKEY:-$ACQ/youtube-asl-v1/$MIX_ID/manifest.json}
EXCLUDE_SIGNERS=${EXCLUDE_SIGNERS:-}
EXCLUDE_REASON=${EXCLUDE_REASON:-"image-right Sem-Lex signers (per-signer active side about 0.22 image-left, against 30 of 34 signers leaning image-left like natural video): excluded from round 14b (docs/NEXT_PHASE_PLAN.md, Sem-Lex per-signer sides, 2026-10-29)"}
PARAM_ERROR=
[[ $CORPUS_ID =~ ^semlex-text-v[0-9]+$ ]] || PARAM_ERROR="CORPUS_ID $CORPUS_ID is not semlex-text-v<N>"
[[ $MIX_ID =~ ^[a-z0-9][a-z0-9.-]*$ ]] || PARAM_ERROR="MIX_ID $MIX_ID is not a corpus id"
[ "$CORPUS_ID" != semlex-text-v1 ] && [ "$MIX_ID" = round14-mixed-v1 ] && \
  PARAM_ERROR="CORPUS_ID $CORPUS_ID needs its own MIX_ID (round14-mixed-v1 holds semlex-text-v1)"
[ -n "$EXCLUDE_SIGNERS" ] && [ "$CORPUS_ID" = semlex-text-v1 ] && \
  PARAM_ERROR="EXCLUDE_SIGNERS needs a new CORPUS_ID (semlex-text-v1 is round 14's, built without exclusion)"
SEMLEX_EXTRA=()
[ -n "${UNMIRRORED:-}" ] && SEMLEX_EXTRA+=(--states-prefix $SL/states-video-v1-unmirrored)
[ -n "$EXCLUDE_SIGNERS" ] && SEMLEX_EXTRA+=(--exclude-signers "$EXCLUDE_SIGNERS" --exclude-reason "$EXCLUDE_REASON")
MIXLOG=stage-${MIX_ID%-mixed-v1}-mix                    # round14-mixed-v1 -> stage-round14-mix (round 14's log name)
STAG=$(printf %s "$CORPUS_ID" | tr a-z A-Z); MTAG=$(printf %s "$MIX_ID" | tr a-z A-Z)
# ---- end parameters ----
OUT=/root/round14-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $OUT   # a fresh local directory per run (local outputs are write-once)
done_() { echo "ROUND14-DONE $1"; exit $1; }
[ -z "$PARAM_ERROR" ] || { echo "ALERT $PARAM_ERROR"; done_ 2; }
echo "corpus $CORPUS_ID ($SKEY), mix $MIX_ID ($MKEY), excluded signers ${EXCLUDE_SIGNERS:-none}, unmirrored ${UNMIRRORED:-no}"
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

echo "STAGE $CORPUS_ID $(date -u)"
if exists $SKEY; then
  echo "stored: $SKEY (not rebuilt)"
  exists $SSUM || { echo "ALERT $SKEY is stored but $SSUM is not (an interrupted upload): investigate"; done_ 3; }
  python -c "import os; from signtranslator.data.text_motion_dataset import R2Store; s=R2Store(os.environ); open('$OUT/semlex.summary.json','wb').write(s.get('$SSUM'))" || done_ 1
  # the stored manifest must be the one these settings build: the same excluded signers (none by default)
  python -c "import json,sys; s=json.load(open(sys.argv[1])); got=(s.get('excluded_signers') or {}).get('signers') or []; want=sorted([x for x in sys.argv[2].split(',') if x], key=int); sys.exit(0 if got == want else f'stored summary excludes {got}, these settings {want}')" \
    "$OUT/semlex.summary.json" "$EXCLUDE_SIGNERS" || { echo "ALERT $SKEY was built with other excluded signers"; done_ 3; }
else
  python scripts/build_semlex_manifest.py --source video --splits val,train --workers $W --wordlist $WORDS \
    --corpus-id $CORPUS_ID ${SEMLEX_EXTRA[@]+"${SEMLEX_EXTRA[@]}"} \
    --coverage-mix $R10@$R10SHA --output $OUT/$CORPUS_ID.json --summary $OUT/semlex.summary.json \
    --output-key $SKEY --summary-key $SSUM > $OUT/stage-semlex.log 2>&1
  s=$?; tail -5 $OUT/stage-semlex.log
  upload_once $SL/$CORPUS_ID/logs/stage-semlex-$(date -u +%Y%m%dT%H%M%SZ).log $OUT/stage-semlex.log || true
  [ $s = 0 ] || { echo "FAILED $CORPUS_ID (status $s): see $OUT/stage-semlex.log"; done_ $s; }
fi
SSHA=$(python -c "import json; print(json.load(open('$OUT/semlex.summary.json'))['manifest_sha256'])")
[ "$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" $SKEY)" = "$SSHA" ] \
  || { echo "ALERT $SKEY does not hash to the summary's $SSHA"; done_ 3; }

echo "STAGE $MIX_ID $(date -u)"
if exists $MKEY; then
  echo "stored: $MKEY (not rebuilt)"
else
  python scripts/build_mixed_text_manifest.py --base $R10@$R10SHA --input $SKEY@$SSHA --corpus-id $MIX_ID \
    --manifest-key $MKEY --check-val-prefix 1536 --check-val-prefix 2048 > $OUT/stage-mix.log 2>&1
  s=$?; tail -c 3000 $OUT/stage-mix.log; echo
  upload_once $SL/$CORPUS_ID/logs/$MIXLOG-$(date -u +%Y%m%dT%H%M%SZ).log $OUT/stage-mix.log || true
  [ $s = 0 ] || { echo "FAILED $MIX_ID (status $s)"; done_ $s; }
fi
MSHA=$(python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" $MKEY)
# the mix (built now or stored) must be round10-mixed-v1 extended by exactly this corpus manifest
python -c "import json,os,sys; from signtranslator.data.text_motion_dataset import R2Store; m=json.loads(R2Store(os.environ).get(sys.argv[1])); i=m.get('inputs') or {}; want={'round10-mixed-v1': sys.argv[2], sys.argv[3]: sys.argv[4]}; sys.exit(0 if m.get('corpus_id') == sys.argv[5] and i == want else f'mix {m.get(\"corpus_id\")} inputs {i}, expected {want}')" \
  $MKEY $R10SHA $CORPUS_ID $SSHA $MIX_ID || { echo "ALERT $MKEY is not round10-mixed-v1 + $CORPUS_ID@$SSHA"; done_ 3; }

python - "$OUT/semlex.summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print("FILTERS", json.dumps(s["counts"], sort_keys=True))
if s.get("excluded_signers"):
    e = s["excluded_signers"]
    print("EXCLUDED signers", e["signers"], "clips", e["metadata_clips"], json.dumps(e["metadata_clips_by_signer"]),
          "reason:", e["reason"])
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
echo "$STAG $SKEY sha256 $SSHA"
echo "$MTAG $MKEY sha256 $MSHA"
done_ 0
