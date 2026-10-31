#!/usr/bin/env bash
# round20_phase_a.sh — Round 20 Phase A on D20 (docs/NEXT_PHASE_PLAN.md "Round 20: composition v2", declared
# 2026-10-30 19:09 UTC; tooling, decisions and the launch sequence in reports/round20_dev.md). Evaluation only:
# NOTHING TRAINS. PREPARED, NOT LAUNCHED until the review fixes are approved.
#
#   R20_PHASE=<phase> VOCAB=<vocabulary> [ARM=<arm>] [SMOKE=1] bash runpod/fleet/round20_phase_a.sh
#
# VOCAB (every phase; its name is in every output key and cache prefix):
#   r10            round10's vocabulary (today's)
#   r14b-fraction  round14b's vocabulary, Sem-Lex-only cores at ASLLVD's pooled median fraction (attribution arm)
#   r14b-activity  round14b's vocabulary, Sem-Lex-only cores by the activity rule (A1's declared switch)
# Every oracle run draws Sem-Lex-only entries' real clips among clips whose right arm is tracked on >= 0.3 of the frames
# (--semlex-oracle-tracked; none: the best-tracked clip), under every VOCAB.
# Every variant is its own invocation with write-once keys under $OUT (SMOKE=1: the first 5 D20 sentences, a best-of-{1,4}
# cache over those sentences' entries, keys under $OUT/smoke/<UTC stamp>/). Smoke each phase first.
# Phases (pod type):
#   a1    (CPU)  real-word oracle; also renders oracle_stage2@lowpass (6 Hz, same material: A6 on real words)
#   a0    (GPU)  W14b fresh composition (4 word workers); also core_crossfade@lowpass (6 Hz, same material: A6).
#                Run under EVERY VOCAB: the A1 winner is named from a1 and a0 together (reports/round20_dev.md)
#   a7    (GPU, VOCAB=r10 only: core_window cuts on gained motion under r14b-activity)  a0 with the tokenizer's accepted
#                decoder gains off: a0's arguments, jobs and seeds, so the same codes decode without the gain. Logits differ
#                by device and a flipped draw changes the rest of a word, so a full run needs a GPU, logs it, and first
#                reruns a0's smoke on this pod: its per-clip core_crossfade motion must equal the stored a0 shard's
#                exactly (first 5 clips) or nothing further runs. Reports store no codes: this is the only check.
#   a2    (CPU)  continuous-form oracle (ASLLRP segments where available, else isolated) on VOCAB (the A1 winner)
#   a3    (CPU)  continuous retiming, velocity-bounded bridges, ARM=tempo-0.8 | tempo-1.0 | tempo-1.25 | tempo-1.5
#                (tempo-1.0 also renders the report-only, deployable @predicted form)
#   cache (GPU)  best-of-16 word cache, ARM=w14b | r16, on VOCAB's D20 entries (choices 4, 8, 16); CACHE_PROCS shard
#                processes on one GPU (a failed shard stops the others), then the merge. NO_GAINS=1 (ARM=w14b, VOCAB=r10
#                only; amendment 4): the words decode without gains, under the prefix cache-w14b-nogains-r10
#   a4    (CPU or GPU)  composition from the W14b cache, ARM=n4 | n8 | n16 (v2 spotting: the independent guard)
#   a5    (CPU)  A5: R16 step 10000 against W14b, word-level v3d reads entry by entry (both caches first)
#   a8    (GPU)  A5's sentence read: a0 with R16 step 10000's words instead of W14b's, against a0 (amendment 3)
#   a9    (CPU or GPU, VOCAB=r10 only)  amendment 4's stack read: composition from the no-gains cache, ARM=n4 | n8 | n16, with
#                @lowpass; against both a7-nogains-r10 and a4-w14b-nN-r10. It passes no --no-decoder-gains: the gains are the
#                cache's, and the report records them (settings.round20.word_cache.no_decoder_gains)
# Comparisons (CPU, seconds): scripts/round20_compare.py forms --a KEY:FORM --b KEY:FORM (paired BLEU + likelihood).
# Thread counts come from the cgroup CPU quota or sched_getaffinity, never nproc.
set -uo pipefail
source /root/r2env.sh
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(/root/venv/bin/python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
T=${THREADS:-$C}
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
uv pip install -q 'transformers>=4.40,<5' sentencepiece
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
die() { echo "ALERT $*"; echo "R20-PHASE-A-DONE 1"; exit 1; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }

O=acquisition/2026-10-17-v1/youtube-asl-v1
OUT=$O/round20/phase-a
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
R14B=$O/round14b-mixed-v1/manifest.json; R14BSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
R16=training-runs/tokens-rvq-d0b-prior-r16-rollout/checkpoints/step-00010000.pt
R16SHA=abaf1b4b24a0eee25772e6d0ea481e4b84b9172893cd2997384099b6fce2a3bc
GLOSS=acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json
GLOSSSHA=f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944
# the instruments (best.json and the checkpoint it names), read from R2 on 2026-10-30
BT=training-runs/back-translator-kp2d-uwb-ft-v1
BTSHA=3db49bb084ed499bceffbbd93b83d6f62059da0d6727369c9472b029668b0dba
BTCK=$BT/checkpoints/step-00009000.pt; BTCKSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
V2=training-runs/sign-recognizer-v2; V2SHA=8989a03b8af3137659f375f1f7c64507a7341061e77b19a5f677cfa230661497
V2CK=$V2/checkpoints/step-00027000.pt; V2CKSHA=a126bf42c0fa4ce8a00e168efbe785668a9ee98600d1146e0d0503c087ba82fb
V3D=training-runs/sign-recognizer-v3d; V3DSHA=3b50a8168e9365984a1f5eb42dd1525466e700bae4036c9bb485a23704cd7ba0
V3DCK=$V3D/checkpoints/step-00012000.pt; V3DCKSHA=0c6c10e232037e869f3983ed6df16df44b3e0a092cbfa34d40ca744cbf00a4e2
for p in "$R10 $R10SHA" "$R14B $R14BSHA" "$D20 $D20SHA" "$GLOSS $GLOSSSHA" "$BT/best.json $BTSHA" "$BTCK $BTCKSHA" \
         "$V2/best.json $V2SHA" "$V2CK $V2CKSHA" "$V3D/best.json $V3DSHA" "$V3DCK $V3DCKSHA"; do
  set -- $p; pin "$1" "$2"
done
PHASE=${R20_PHASE:?set R20_PHASE: a1 a0 a7 a2 a3 cache a4 a5 a8 a9}
ARM=${ARM:-}
VOCAB=${VOCAB:?set VOCAB: r10, r14b-fraction or r14b-activity}
case $VOCAB in
  r10)           VMIX="";              VCACHE="" ;;
  r14b-fraction) VMIX="--vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA"; VCACHE=$VMIX ;;
  r14b-activity) VMIX="--vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA --semlex-core activity"
                 VCACHE="--vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA" ;;
  *) die "VOCAB must be r10, r14b-fraction or r14b-activity" ;;
esac
case ${NO_GAINS:-0} in 0) ;; 1) [ "$PHASE" = cache ] || die "NO_GAINS=1 belongs to the cache phase (a7 and a9 have their own)" ;;
  *) die "NO_GAINS must be 0 or 1" ;; esac
CLIPS=500; SAMPLES=16; CHOOSE=4,8,16
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=$OUT/smoke/$(date -u +%Y%m%dT%H%M%SZ); CLIPS=5; SAMPLES=4; CHOOSE=1,4
fi
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
# the declared round-15 composition settings (amendment 5) on D20, every form read with the likelihood metric
BASE="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS \
  --selection stage1 --timing sign-fitted --core-margin 0.1 --expect-model-sha256 $MODEL --likelihood --device $DEV"
ORACLE="--variants oracle_stage2 $VMIX --semlex-oracle-tracked"
GEN="--variants core_crossfade $VMIX --checkpoint-key $W14B --generator-mix-key $R14B --generator-mix-sha256 $R14BSHA \
  --word-workers ${WORD_WORKERS:-4}"

# one write-once run (single shard) and its merge: compose NAME ARGS...
compose() {
  local name=$1; shift
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json
  exists $merged && { echo "stored: $merged (not rerun)"; return 0; }
  exists $part || python scripts/round15_compose_eval.py run $BASE "$@" --output-key $part || die "run $name"
  python scripts/round15_compose_eval.py merge --shard-keys $part --output-key $merged || die "merge $name"
  echo "R20 $name $merged $(sha $merged)"
}

# A7's guard: a0's smoke on this pod against the stored a0 shard (scripts/round20_compare.py same-codes)
a7_same_codes() {
  local gpu=$1 full=$OUT stored=$OUT/a0-w14b-$VOCAB-shard0.json
  exists $stored || die "a7: the stored a0 shard $stored is missing; GPU: $gpu"
  OUT=$full/smoke/$(date -u +%Y%m%dT%H%M%SZ)
  compose a0-w14b-$VOCAB $GEN --lowpass-report-hz 6 --clips 5
  python scripts/round20_compare.py same-codes --a $OUT/a0-w14b-$VOCAB-shard0.json --b $stored \
    --form core_crossfade --first 5 || die "a7: this pod does not reproduce a0's codes on the first 5 clips, nothing further runs; GPU: $gpu"
  OUT=$full
}

case $PHASE in
  a1) compose a1-oracle-$VOCAB $ORACLE --lowpass-report-hz 6 ;;
  a0) pin $W14B $W14BSHA
      compose a0-w14b-$VOCAB $GEN --lowpass-report-hz 6 ;;
  a8) pin $R16 $R16SHA        # A5's sentence read (amendment 3): as a0, with R16's words
      compose a8-r16-$VOCAB --variants core_crossfade $VMIX --checkpoint-key $R16 --generator-mix-key $R14B \
        --generator-mix-sha256 $R14BSHA --word-workers ${WORD_WORKERS:-4} --lowpass-report-hz 6 ;;
  a7) [ "$VOCAB" = r10 ] || die "a7: VOCAB must be r10 (core_window cuts on gained motion under r14b-activity)"
      pin $W14B $W14BSHA
      if [ "${SMOKE:-0}" != 1 ]; then
        [ "$DEV" = cuda ] || die "a7: a full run needs a GPU as a0 did (DEV=$DEV): on another device a flipped draw changes the rest of a word"
        GPU=$(nvidia-smi -L 2>&1)
        echo "R20 a7 GPU: $GPU"
        exists $OUT/a7-nogains-$VOCAB.json || a7_same_codes "$GPU"
      fi
      compose a7-nogains-$VOCAB $GEN --lowpass-report-hz 6 --no-decoder-gains ;;
  a2) compose a2-continuous-$VOCAB $ORACLE --oracle-source continuous \
        --gloss-manifest-key $GLOSS --gloss-manifest-sha256 $GLOSSSHA ;;
  a3)
    case $ARM in
      tempo-1.0) compose a3-$ARM-$VOCAB $ORACLE --retime continuous --tempo 1.0 --predicted-duration ;;
      tempo-0.8|tempo-1.25|tempo-1.5) compose a3-$ARM-$VOCAB $ORACLE --retime continuous --tempo ${ARM#tempo-} ;;
      *) die "a3: ARM tempo-0.8, tempo-1.0, tempo-1.25 or tempo-1.5" ;;
    esac ;;
  cache)
    case $ARM in
      w14b) CK=$W14B; CKSHA=$W14BSHA ;; r16) CK=$R16; CKSHA=$R16SHA ;; *) die "cache: ARM w14b or r16" ;;
    esac
    [ "$DEV" = cuda ] || [ "${SMOKE:-0}" = 1 ] || die "cache: a full cache needs a GPU (DEV=$DEV): other devices sample other bits"
    pin $CK $CKSHA
    TAG=$ARM; NG=""
    if [ "${NO_GAINS:-0}" = 1 ]; then
      [ "$ARM" = w14b ] || die "cache: NO_GAINS=1 is for ARM=w14b only"
      [ "$VOCAB" = r10 ] || die "cache: NO_GAINS=1 is for VOCAB=r10 only"
      TAG=$ARM-nogains; NG=--no-decoder-gains
    fi
    PREFIX=$OUT/cache-$TAG-$VOCAB
    exists $PREFIX/index.json && { echo "stored: $PREFIX/index.json"; echo "R20-PHASE-A-DONE 0"; exit 0; }
    N=${CACHE_PROCS:-4}; pids=()
    for s in $(seq 0 $(( N - 1 ))); do
      exists $PREFIX/shard-$s-of-$N.json && continue
      OMP_NUM_THREADS=$(( T / N > 0 ? T / N : 1 )) python scripts/round20_word_cache.py run --prefix $PREFIX \
        --mix-key $R10 --mix-sha256 $R10SHA $VCACHE --sentences-key $D20 --sentences-sha256 $D20SHA \
        --checkpoint-key $CK --generator-mix-sha256 $R14BSHA --samples $SAMPLES --choose $CHOOSE \
        --recognizer-run sign-recognizer-v3d --device $DEV --shards $N --shard $s --clips $CLIPS $NG \
        > /root/r20-cache-$TAG-$s.log 2>&1 & pids+=($!)
    done
    failed=0; ended=" "   # poll every shard: an early failure stops the others at once (re-review; any bash, no wait -n)
    while :; do
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
    [ $failed = 0 ] || die "a cache shard failed; the others were stopped (logs /root/r20-cache-$TAG-*.log)"
    python scripts/round20_word_cache.py merge --prefix $PREFIX --shards $N || die "cache merge"
    echo "R20 cache-$TAG-$VOCAB $PREFIX/index.json $(sha $PREFIX/index.json)" ;;
  a4)
    case $ARM in n4|n8|n16) ;; *) die "a4: ARM n4, n8 or n16" ;; esac
    if [ "${SMOKE:-0}" = 1 ]; then      # a smoke cache holds best-of-1 and best-of-4 only
      [ -n "${CACHE_KEY:-}" ] || die "a4 smoke: give CACHE_KEY (a smoke cache's index.json)"; IDX=$CACHE_KEY; NCH=4
    else IDX=$OUT/cache-w14b-$VOCAB/index.json; NCH=${ARM#n}; fi
    compose a4-w14b-$ARM-$VOCAB --variants core_crossfade $VMIX --word-cache-key $IDX \
      --word-cache-sha256 $(sha $IDX) --word-cache-n $NCH --recognizer-run sign-recognizer-v2 ;;
  a9)
    case $ARM in n4|n8|n16) ;; *) die "a9: ARM n4, n8 or n16" ;; esac
    [ "$VOCAB" = r10 ] || die "a9: VOCAB must be r10"
    if [ "${SMOKE:-0}" = 1 ]; then      # a smoke cache holds best-of-1 and best-of-4 only
      [ -n "${CACHE_KEY:-}" ] || die "a9 smoke: give CACHE_KEY (a no-gains smoke cache's index.json)"; IDX=$CACHE_KEY; NCH=4
    else IDX=$OUT/cache-w14b-nogains-$VOCAB/index.json; NCH=${ARM#n}; fi
    exists $IDX || die "a9: the no-gains cache $IDX is not stored"
    compose a9-nogains-$ARM-$VOCAB --variants core_crossfade $VMIX --word-cache-key $IDX \
      --word-cache-sha256 $(sha $IDX) --word-cache-n $NCH --recognizer-run sign-recognizer-v2 --lowpass-report-hz 6 ;;
  a5)
    A=${A5_A:-$OUT/cache-r16-$VOCAB/index.json}; B=${A5_B:-$OUT/cache-w14b-$VOCAB/index.json}
    K=$OUT/a5-r16-vs-w14b-$VOCAB.json
    exists $K && { echo "stored: $K"; echo "R20-PHASE-A-DONE 0"; exit 0; }
    python scripts/round20_compare.py words --a $A --b $B --output-key $K || die "a5 compare"
    echo "R20 a5 $K $(sha $K)" ;;
  *) die "unknown R20_PHASE $PHASE" ;;
esac
echo "R20-PHASE-A-DONE 0"
