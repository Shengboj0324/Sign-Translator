#!/usr/bin/env bash
# round20_phase_b.sh — Round 20 Phase B on the 1,000 (docs/NEXT_PHASE_PLAN.md "Round 20: composition v2", Phase B and M1,
# amendments 1-3). Evaluation only: NOTHING TRAINS, nothing is tuned. One stacked arm of every lever Phase A adopted, against
# the stored W14b composition (2.306). PREPARED, NOT LAUNCHED: which levers are adopted is not known yet.
#
#   R20B_PHASE=e2e|cache|run|verdict PRIOR=w14b|r16 NO_GAINS=0|1 BEST_OF=0|4|8|16 SET=val1000|c20 [LOWPASS=0|1] [SMOKE=1] \
#     bash runpod/fleet/round20_phase_b.sh
#
# The arm: PRIOR's words (A5/a8), without decoder gains if NO_GAINS=1 (A7), from a best-of-BEST_OF cache if BEST_OF is not 0
# (A4); BEST_OF=0 generates fresh words. PRIOR=w14b NO_GAINS=0 BEST_OF=0 is the comparator itself and is refused.
# Tempo (A3), vocabulary (A1) and continuous words (A2) were not adopted: their options stay off.
# LOWPASS=1 (amendment 4, rule 4: A6 enters only if the stack's @lowpass has both point estimates above 0 against the stack):
# --lowpass-hz 6 on the main form and its controls, applied after rendering, so seeds, codes, plans and the cache are the
# unfiltered arm's; real and blank stay unfiltered. Its only evidence is a9's read of the W14b no-gains stack, so it is refused
# on anything else: gained words (A6 was rejected there), fresh words (position seeds, not a9's entry seeds), R16 (other codes).
#
# Stored D20 reads gate run and verdict (scripts/round20_phase_b_verdict.py stack-check; cache and e2e read none, so they can be
# built ahead). Amendment 3 and rule 5: nothing runs until a8 is stored; if a8 meets amendment 3, W14b arms wait for the stack
# question repeated with R16's words (a4-r16-nN-r10, a7-nogains-r16-r10, a9-nogains-r16-nN-r10; Phase A has no such phases yet,
# so R16 arms are refused today); if it fails, W14b proceeds. Rule 1: N is computed from the a4 reads. Rule 3: the stack, or
# only if its stored read failed, the better single lever. Rule 4: LOWPASS=1 needs the stack's @lowpass above the stack.
#
# One arm on a set ("one stacked arm"): run refuses if any other arm's report or shard is stored on that set, and the verdict and
# C20's gate refuse the same way. R20B_DECLARED_REPLACEMENT=<plan commit> lifts that for run only when an amendment declares a
# replacement: it is logged and recorded in the report and the verdict. Pods have no git history, so the commit's format is
# checked but its contents are not: whoever launches must have read that amendment.
#
# Order and pods (one L40S as in Phase A, wall-clock = GPU-hours; the 1,000 hold 894 entries under r10, against D20's 705):
#   e2e        (GPU, only if NO_GAINS=1, any time before run; about 50 min, as the R16 pilot job)  the prior's own end-to-end
#              without gains: its stored pilot's command (seeds, clips, words) plus --no-decoder-gains, write-once, one per prior.
#   1. cache   (GPU, only if BEST_OF is not 0)  894 x BEST_OF samples, CACHE_PROCS (4) shard processes on one GPU, at Phase
#              A's 705 x 16 in 3.9 h = 2,892 samples/h: n4 1.2 h, n8 2.5 h, n16 4.9 h, the same with or without gains (the gain
#              is one multiplication after the decoder). Samples = BEST_OF: seeds are prefixes, so the choice is what a
#              best-of-16 run would choose at N.
#   2. run     (GPU, as the comparator: reads differ slightly by device; a CPU run is refused)  the comparator's 4 shards, in
#              sequence on one pod (shards change no clip's seeds or translation batch): from a cache about 1 h; fresh words
#              (BEST_OF=0, 4 word workers) about 6 h, from a0's 21 s per clip and the comparator's 20 s.
#   3. verdict (CPU, or the same pod; seconds)  scripts/round20_phase_b_verdict.py: stored reports only, write-once.
#   SET=c20 repeats 1-3, and is refused unless the val1000 verdict of this same arm passed with a point estimate >= 2.55.
#
# The comparator is $CMP: --clips 1000 --shards 4 --selection stage1 --timing sign-fitted --core-margin 0.1 --stage1-report-key
# $S1 --generator-mix-key $R14B (the "declared run's command", runpod/fleet/state_back_translator.sh), settings confirmed from
# the report. Pairing: `merge --paired-with $CMP` refuses other references or clip ids, and gives minus_paired_core_crossfade;
# recogniser v2's recall is paired from both reports' shards in the verdict. The comparator has NO likelihood at any level, so
# the arm's likelihood minus blank is its own read, unpaired with W14b (the rule does not ask for it).
#
# Its own end-to-end: W14b $E2E_W14B (the comparator's own), R16 $E2E_R16 (step 10000), both gained. A NO_GAINS=1 arm is compared
# with the e2e phase's pilot without gains; the verdict refuses a gained pilot for it, and a pilot without gains otherwise.
#
# SMOKE=1: plumbing only, under $OUT/smoke/<UTC stamp>/: D20's first 5 sentences (--clips 5 of the 1,000's list would draw
# other clips, some of them possibly C20's), 4 samples per entry, 1 shard, no comparator (other clips cannot pair). Smoke the
# run with CACHE_KEY=<the smoke cache's index.json>. The e2e smoke is the pilot's first 5 clips of the 1,000 (--first-clips: the
# pilot has no sentence set), real and generated hypotheses of those clips only, into the smoke prefix. No smoke of the verdict:
# its tests read real merges of the synthetic store.
# Thread counts come from the cgroup CPU quota or sched_getaffinity, never nproc.
set -uo pipefail
source /root/r2env.sh
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(/root/venv/bin/python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
T=${THREADS:-$C}
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
die() { echo "ALERT $*"; echo "R20-PHASE-B-DONE 1"; exit 1; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }

O=acquisition/2026-10-17-v1/youtube-asl-v1
OUT=$O/round20/phase-b
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
R14B=$O/round14b-mixed-v1/manifest.json; R14BSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
C20=$O/round20/c20.json; C20SHA=24ac16f220139365af2bc7a62e7555ab88d13e9adb9b0a5f078c8b4107f72956
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
R16=training-runs/tokens-rvq-d0b-prior-r16-rollout/checkpoints/step-00010000.pt
R16SHA=abaf1b4b24a0eee25772e6d0ea481e4b84b9172893cd2997384099b6fce2a3bc
# the stored W14b composition (BLEU 2.306), its shards (per-clip spotting), the stage-1 report it checked its words against
CMP=$O/round15/eval1000/w14b-step30000.json; CMPSHA=6a763d503d21ec873abde69427d81bb92e1b38e8539f3e7d5f86b29eaf75042a
S1=$O/round15/compositional-f12-step60000.json; S1SHA=15f1aee2a7b7ca15432f5df3991f3a88630d6b79f1ef7d9a5732a5e80eee32d9
CMPS0=641d7a899c4d5a9fb44403d55bacdb49d35597f506b885a65218945fcf016a75
CMPS1=5b5e324fb7fa56d37cb66035758cb09fff089be7b2225604e21d0f6fef74f769
CMPS2=f4f72c57c3305e6d9b709ed366656f0f102acf1a9891cc72ef35d77fe7f72151
CMPS3=fe342ce0943f648940a8a1a8dbb54990455affe33859924043622b218ead2d3c
SPLIT=$O/round14/r5-word-split.json; SPLITSHA=676f56cbbaa9bac08f0bc305027807827f982cd4c242de03f7891323e723102a
A=$O/round20/phase-a      # the stored D20 reads
# each prior's own end-to-end on the same 1,000 (references and real/blank hypotheses checked identical to the comparator's)
E2E_W14B=$O/round14b/pilot-w14b-step30000.json; E2E_W14BSHA=090f68b1f26e33ff9a8c26f1d90e8a89d33f215687dbd6c40d21cc7232eddb68
E2E_R16=$O/round16/pilot-r16.json; E2E_R16SHA=8f24b7126326a818d2f1f7a389e05ad13f189e1bb4c995fc4e34b8ed233c8f63
# the instruments (best.json and the checkpoint it names), as Phase A
BT=training-runs/back-translator-kp2d-uwb-ft-v1
BTSHA=3db49bb084ed499bceffbbd93b83d6f62059da0d6727369c9472b029668b0dba
BTCK=$BT/checkpoints/step-00009000.pt; BTCKSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
V2=training-runs/sign-recognizer-v2; V2SHA=8989a03b8af3137659f375f1f7c64507a7341061e77b19a5f677cfa230661497
V2CK=$V2/checkpoints/step-00027000.pt; V2CKSHA=a126bf42c0fa4ce8a00e168efbe785668a9ee98600d1146e0d0503c087ba82fb
V3D=training-runs/sign-recognizer-v3d; V3DSHA=3b50a8168e9365984a1f5eb42dd1525466e700bae4036c9bb485a23704cd7ba0
V3DCK=$V3D/checkpoints/step-00012000.pt; V3DCKSHA=0c6c10e232037e869f3983ed6df16df44b3e0a092cbfa34d40ca744cbf00a4e2

# ---- the inputs, refused before anything is read ------------------------------------------------------------------------
PHASE=${R20B_PHASE:-}; PRIOR=${PRIOR:-}; NO_GAINS=${NO_GAINS:-}; BEST_OF=${BEST_OF:-}; SET=${SET:-}
case $PHASE in e2e|cache|run|verdict) ;; *) die "R20B_PHASE must be e2e, cache, run or verdict" ;; esac
case $PRIOR in
  w14b) CK=$W14B; CKSHA=$W14BSHA; STORED_E2E=$E2E_W14B; E2ESHA=$E2E_W14BSHA
        WORDS_E2E="--word-split-key $SPLIT --word-split-sha256 $SPLITSHA --recognizer-run sign-recognizer-v2" ;;
  r16)  CK=$R16;  CKSHA=$R16SHA;  STORED_E2E=$E2E_R16;  E2ESHA=$E2E_R16SHA
        WORDS_E2E="--word-split-key $SPLIT --word-split-sha256 $SPLITSHA --word-list-key $SPLIT --word-list-sha256 $SPLITSHA --recognizer-run sign-recognizer-v3d" ;;
  *) die "PRIOR must be w14b or r16" ;;
esac
case $NO_GAINS in 0) GAINS=gains; NG="" ;; 1) GAINS=nogains; NG=--no-decoder-gains ;; *) die "NO_GAINS must be 0 or 1" ;; esac
case $BEST_OF in 0|4|8|16) ;; *) die "BEST_OF must be 0, 4, 8 or 16" ;; esac
case $SET in val1000|c20) ;; *) die "SET must be val1000 or c20" ;; esac
[ "$PRIOR" = w14b ] && [ "$NO_GAINS" = 0 ] && [ "$BEST_OF" = 0 ] && die "no lever adopted: this arm is the stored W14b composition itself"
[ "$PHASE" = cache ] && [ "$BEST_OF" = 0 ] && die "cache: BEST_OF=0 uses no cache"
[ "$PHASE" = e2e ] && [ "$NO_GAINS" = 0 ] && die "e2e: only for NO_GAINS=1 (the gained pilots are stored)"
[ "$PHASE" = e2e ] && [ "$SET" != val1000 ] && die "e2e: SET=val1000 only (no pilot or comparator reads C20)"
DECLARED=${R20B_DECLARED_REPLACEMENT:-}
if [ -n "$DECLARED" ]; then
  [ "$PHASE" = run ] && [ "${SMOKE:-0}" != 1 ] || die "R20B_DECLARED_REPLACEMENT belongs to a full run"
  case $DECLARED in *[!0-9a-f]*) die "R20B_DECLARED_REPLACEMENT must be a plan commit (hex)" ;; esac
  [ ${#DECLARED} -ge 7 ] && [ ${#DECLARED} -le 40 ] || die "R20B_DECLARED_REPLACEMENT must be a plan commit (7-40 hex)"
fi
[ "$PHASE" = verdict ] && [ "${SMOKE:-0}" = 1 ] && die "verdict: no smoke (a smoke run has no comparator)"
LOWPASS=${LOWPASS:-0}
case $LOWPASS in
  0) LP=""; LPTAG="" ;;
  1) { [ "$PRIOR" = w14b ] && [ "$NO_GAINS" = 1 ] && [ "$BEST_OF" != 0 ]; } ||
       die "LOWPASS=1: only on the stack a9 read (PRIOR=w14b NO_GAINS=1, BEST_OF 4, 8 or 16)"
     LP="--lowpass-hz 6"; LPTAG=-lp6 ;;
  *) die "LOWPASS must be 0 or 1" ;;
esac
uv pip install -q 'transformers>=4.40,<5' sentencepiece

ARM=$PRIOR-$GAINS-n$BEST_OF$LPTAG      # the cache's prefix has no LOWPASS: the filter comes after rendering
REPORT=$ARM-$SET
VAL_VERDICT=$OUT/verdict-$ARM-val1000.json
# C20 stays untouched until an M1 claim: no key of it is read, not even to pin it, before this arm's val1000 verdict opens it
if [ "$SET" = c20 ]; then
  exists $VAL_VERDICT || die "c20: no stored val1000 verdict of this arm ($VAL_VERDICT)"
  python scripts/round20_phase_b_verdict.py c20-gate --verdict-key $VAL_VERDICT --out $OUT --report $ARM-val1000 --prior $PRIOR \
    --prior-checkpoint-key $CK --no-gains $NO_GAINS --best-of $BEST_OF --lowpass $LOWPASS || die "c20: the val1000 verdict does not open it"
  pin $C20 $C20SHA
fi
for p in "$R10 $R10SHA" "$R14B $R14BSHA" "$D20 $D20SHA" "$CMP $CMPSHA" "$S1 $S1SHA" \
         "${CMP%.json}-shard0.json $CMPS0" "${CMP%.json}-shard1.json $CMPS1" "${CMP%.json}-shard2.json $CMPS2" \
         "${CMP%.json}-shard3.json $CMPS3" "$STORED_E2E $E2ESHA" "$SPLIT $SPLITSHA" "$BT/best.json $BTSHA" "$BTCK $BTCKSHA" \
         "$V2/best.json $V2SHA" "$V2CK $V2CKSHA" "$V3D/best.json $V3DSHA" "$V3DCK $V3DCKSHA"; do
  set -- $p; pin "$1" "$2"
done
# the stored D20 reads decide whether this arm may read the 1,000; caches and e2e read none of them, so they can be built ahead
if { [ "$PHASE" = run ] || [ "$PHASE" = verdict ]; } && [ "${SMOKE:-0}" != 1 ]; then
  python scripts/round20_phase_b_verdict.py stack-check --phase-a-out $A --prior $PRIOR --w14b-checkpoint-key $W14B \
    --r16-checkpoint-key $R16 --no-gains $NO_GAINS --best-of $BEST_OF --lowpass $LOWPASS --sentences-sha256 $D20SHA \
    || die "the stored D20 reads do not allow this arm on the 1,000"
fi

# ---- the sentences: the 1,000 are validation_sentences(round10, 1000), so no --sentences-key: the comparator's own list ----
CLIPS=1000; SHARDS=4; SAMPLES=$BEST_OF; CHOOSE=$BEST_OF; NCH=$BEST_OF
if [ "$SET" = val1000 ]; then
  SENTENCES=""; STAGE1="--stage1-report-key $S1"
  if [ "$NO_GAINS" = 1 ]; then E2E=$OUT/pilot-$PRIOR-nogains-val1000.json; else E2E=$STORED_E2E; fi
  MERGE="--end-to-end-report $E2E --paired-with $CMP --rule bleu:end_to_end,bleu:mismatched_words,spotting"
else
  SENTENCES="--sentences-key $C20 --sentences-sha256 $C20SHA"; STAGE1=""
  MERGE="--rule bleu:mismatched_words,spotting"     # no pilot or comparator read C20
fi
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=$OUT/smoke/$(date -u +%Y%m%dT%H%M%SZ); CLIPS=5; SHARDS=1; SAMPLES=4; CHOOSE=4; NCH=4
  SENTENCES="--sentences-key $D20 --sentences-sha256 $D20SHA"; STAGE1=""
  MERGE="--rule bleu:mismatched_words,spotting"
fi
PREFIX=$OUT/cache-$PRIOR-$GAINS-$SET-n$BEST_OF
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
if [ "$PHASE" != verdict ] && [ "${SMOKE:-0}" != 1 ]; then
  [ "$DEV" = cuda ] || die "$PHASE: a full run needs a GPU as the comparator had (DEV=$DEV)"
  echo "R20B $REPORT $PHASE GPU: $(nvidia-smi -L 2>&1)"
fi

# Phase A's shard processes: CACHE_PROCS on one GPU, each with its share of the threads; a failure stops the others at once
cache_shards() {
  local N=${CACHE_PROCS:-4} pids=() s q failed=0 ended=" " running
  for s in $(seq 0 $(( N - 1 ))); do
    exists $PREFIX/shard-$s-of-$N.json && continue
    OMP_NUM_THREADS=$(( T / N > 0 ? T / N : 1 )) python scripts/round20_word_cache.py run --prefix $PREFIX \
      --mix-key $R10 --mix-sha256 $R10SHA $SENTENCES \
      --checkpoint-key $CK --generator-mix-sha256 $R14BSHA --samples $SAMPLES --choose $CHOOSE \
      --recognizer-run sign-recognizer-v3d --device $DEV --shards $N --shard $s --clips $CLIPS $NG \
      > /root/r20b-cache-$REPORT-$s.log 2>&1 & pids+=($!)
  done
  while [ ${#pids[@]} -gt 0 ]; do
    running=0
    for q in "${pids[@]}"; do
      case "$ended" in *" $q "*) continue ;; esac
      if kill -0 $q 2>/dev/null; then running=1; continue; fi
      wait $q || failed=1; ended="$ended$q "
    done
    if [ $failed = 1 ]; then for q in "${pids[@]}"; do kill $q 2>/dev/null; done; break; fi
    [ $running = 0 ] && break
    sleep 10
  done
  wait
  [ $failed = 0 ] || die "a cache shard failed; the others were stopped (logs /root/r20b-cache-$REPORT-*.log)"
  python scripts/round20_word_cache.py merge --prefix $PREFIX --shards $N || die "cache merge"
}

case $PHASE in
  e2e)
    PILOT=$OUT/pilot-$PRIOR-nogains-val1000.json; FIRST=""
    [ "${SMOKE:-0}" = 1 ] && FIRST="--first-clips 5"
    if exists $PILOT; then echo "stored: $PILOT (not rerun)"
    else
      pin $CK $CKSHA
      # the stored pilot's command (its report records these settings); the mix is the prior's own, and its first 1,000 are the
      # comparator's clips (checked 2026-10-31)
      python scripts/pilot_content_check.py --checkpoint-key $CK --mix-key $R14B --mix-sha256 $R14BSHA --vocab-mix-key $R10 \
        --vocab-mix-sha256 $R10SHA --clips 1000 --part-temperature body=1.1,left_hand=1.1,right_hand=1.1,face=1.0 --guidance 1.5 \
        --translator-run $(basename $BT) $WORDS_E2E --device $DEV --no-decoder-gains $FIRST --output-key $PILOT \
        || die "e2e $PILOT"
      echo "R20B e2e $PILOT $(sha $PILOT)"
    fi ;;
  cache)
    if exists $PREFIX/index.json; then echo "stored: $PREFIX/index.json"
    else
      pin $CK $CKSHA
      cache_shards
      echo "R20B cache $PREFIX/index.json $(sha $PREFIX/index.json)"
    fi ;;
  run)
    # the comparator's settings (read from its report), every one spelled out; apart from the words only --likelihood is new
    BASE="--mix-key $R10 --mix-sha256 $R10SHA $SENTENCES $STAGE1 --clips $CLIPS --block 32 --seed 0 --variants core_crossfade \
      --selection stage1 --timing sign-fitted --core-margin 0.1 --total-fit pad --guidance 1.5 \
      --part-temperature body=1.1,left_hand=1.1,right_hand=1.1,face=1.0 --expect-model-sha256 $MODEL \
      --generator-mix-key $R14B --generator-mix-sha256 $R14BSHA --translator-run $(basename $BT) \
      --recognizer-run $(basename $V2) --likelihood --device $DEV $LP"
    if [ "$SET" = val1000 ] && [ "$NO_GAINS" = 1 ] && [ "${SMOKE:-0}" != 1 ]; then
      exists $E2E || die "run: the own end-to-end without gains $E2E is not stored: run the e2e phase first"
    fi
    if [ "$BEST_OF" = 0 ]; then
      pin $CK $CKSHA
      WORDS="--checkpoint-key $CK --word-workers ${WORD_WORKERS:-4} $NG"
    else
      IDX=$PREFIX/index.json
      if [ "${SMOKE:-0}" = 1 ]; then
        [ -n "${CACHE_KEY:-}" ] || die "run smoke: give CACHE_KEY (a smoke cache's index.json)"; IDX=$CACHE_KEY
      fi
      exists $IDX || die "run: the cache $IDX is not stored: run the cache phase first"
      WORDS="--word-cache-key $IDX --word-cache-sha256 $(sha $IDX) --word-cache-n $NCH"     # its gains are the cache's
    fi
    MERGED=$OUT/$REPORT.json
    if exists $MERGED; then echo "stored: $MERGED (not rerun)"
    else
      if [ "${SMOKE:-0}" != 1 ]; then
        python scripts/round20_phase_b_verdict.py one-arm --out $OUT --set $SET --report $REPORT \
          ${DECLARED:+--declared-replacement $DECLARED} || die "run $REPORT: this set already holds another arm"
      fi
      if [ -n "$DECLARED" ]; then
        echo "R20B DECLARED REPLACEMENT $DECLARED: $REPORT on a set that holds another arm"
        MERGE="$MERGE --declared-replacement $DECLARED"
      fi
      KEYS=""
      for s in $(seq 0 $(( SHARDS - 1 ))); do
        part=$OUT/$REPORT-shard$s.json; KEYS=$KEYS${KEYS:+,}$part
        exists $part || python scripts/round15_compose_eval.py run $BASE $WORDS --shards $SHARDS --shard $s \
          --output-key $part || die "run $REPORT shard $s"
      done
      python scripts/round15_compose_eval.py merge --shard-keys $KEYS $MERGE --output-key $MERGED || die "merge $REPORT"
      echo "R20B $REPORT $MERGED $(sha $MERGED)"
    fi ;;
  verdict)
    MERGED=$OUT/$REPORT.json; V=$OUT/verdict-$REPORT.json
    if exists $V; then echo "stored: $V"
    else
      exists $MERGED || die "verdict: $MERGED is not stored: run the run phase first"
      if [ "$SET" = val1000 ]; then
        if [ "$NO_GAINS" = 1 ]; then
          exists $E2E || die "verdict: the own end-to-end without gains $E2E is not stored"
          FOR_SET="--end-to-end-key $E2E --end-to-end-sha256 $(sha $E2E) --stored-pilot-key $STORED_E2E --stored-pilot-sha256 $E2ESHA"
        else FOR_SET="--end-to-end-key $E2E --end-to-end-sha256 $E2ESHA"; fi
      else FOR_SET="--val1000-verdict-key $VAL_VERDICT --sentences-key $C20 --sentences-sha256 $C20SHA"; fi
      python scripts/round20_phase_b_verdict.py verdict --arm-key $MERGED --comparator-key $CMP --comparator-sha256 $CMPSHA \
        $FOR_SET --out $OUT --report $REPORT --lowpass $LOWPASS --prior $PRIOR --prior-checkpoint-key $CK --no-gains $NO_GAINS --best-of $BEST_OF --set $SET \
        --output-key $V || die "verdict $REPORT, nothing stored (exit 2: a rule item is unavailable; 1: refused)"
      echo "R20B verdict $V $(sha $V)"
    fi ;;
esac
echo "R20-PHASE-B-DONE 0"
