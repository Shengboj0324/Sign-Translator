#!/usr/bin/env bash
# round20_phase_a.sh — Round 20 Phase A on D20 (docs/NEXT_PHASE_PLAN.md "Round 20: composition v2", declared
# 2026-10-30 19:09 UTC; tooling and commands in reports/round20_dev.md). Evaluation only: NOTHING TRAINS.
# PREPARED, NOT LAUNCHED (independent review first).
#
#   R20_PHASE=<phase> [ARM=<arm>] [VOCAB=r10|r14b] [SMOKE=1] bash runpod/fleet/round20_phase_a.sh
#
# Every variant is its own invocation with write-once output keys under $OUT (SMOKE=1: the first 5 D20 sentences, a
# small cache, keys under $OUT/smoke/<UTC stamp>/, so a smoke never blocks the real key). Run each phase's smoke first.
# Phases (pod type):
#   a1    (CPU)  vocabulary oracle: ARM=r10 | r14b-fraction | r14b-activity (A1 declared: r10 vs r14b-activity;
#                r14b-fraction is the attribution arm for the Sem-Lex activity core)
#   a2    (CPU)  continuous-form oracle (ASLLRP segments where available, else isolated) on VOCAB (the A1 winner)
#   a3    (CPU)  continuous retiming + velocity-bounded bridges, ARM=tempo-0.8 | tempo-1.0 | tempo-1.25 | tempo-1.5
#                (tempo-1.0 also renders the report-only @predicted form) on VOCAB's oracle
#   a6o   (CPU)  Butterworth 6 Hz on VOCAB's oracle (A6, real words)
#   a0    (GPU)  W14b fresh composition on D20 (the generated baseline of A4 and A6), VOCAB
#   a6    (GPU)  the same W14b composition low-passed at 6 Hz (A6; identical seeds, so identical words)
#   cache (GPU)  best-of-16 word cache for ARM=w14b | r16 on VOCAB's D20 entries (choices 4, 8, 16), CACHE_PROCS
#                shard processes on one GPU, then the merge; the R16 cache doubles as A5's word-level read
#   a4    (CPU or GPU)  composition from the W14b cache, ARM=n4 | n8 | n16 (v2 spotting: the independent guard)
#   a5    (CPU)  A5: R16 step 10000 against W14b, word-level v3d reads entry by entry (both caches merged first)
# Comparisons (CPU, seconds): scripts/round20_compare.py forms --a KEY:FORM --b KEY:FORM (paired BLEU + likelihood).
# Worker and thread counts come from the cgroup CPU quota or sched_getaffinity, never nproc.
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

O=acquisition/2026-10-17-v1/youtube-asl-v1
OUT=$O/round20/phase-a
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
R14B=$O/round14b-mixed-v1/manifest.json; R14BSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
R16=training-runs/tokens-rvq-d0b-prior-r16-rollout/checkpoints/step-00010000.pt
GLOSS=acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json
GLOSSSHA=f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944
for pin in "$R10 $R10SHA" "$R14B $R14BSHA" "$D20 $D20SHA" "$GLOSS $GLOSSSHA"; do
  set -- $pin; [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"
done
PHASE=${R20_PHASE:?set R20_PHASE: a1 a2 a3 a6o a0 a6 cache a4 a5}
ARM=${ARM:-}
VOCAB=${VOCAB:-r14b}
CLIPS=500; SAMPLES=16; CHOOSE=4,8,16
if [ "${SMOKE:-0}" = 1 ]; then
  OUT=$OUT/smoke/$(date -u +%Y%m%dT%H%M%SZ); CLIPS=5; SAMPLES=4; CHOOSE=1,4
fi
case $VOCAB in
  r10)  VMIX="" ;;
  r14b) VMIX="--vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA --semlex-core activity" ;;
  *) die "VOCAB must be r10 or r14b" ;;
esac
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
# the declared round-15 composition settings (amendment 5) on D20, every form read with the likelihood metric
BASE="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS \
  --selection stage1 --timing sign-fitted --core-margin 0.1 --expect-model-sha256 $MODEL --likelihood --device $DEV"
GEN="--checkpoint-key $W14B --generator-mix-key $R14B --generator-mix-sha256 $R14BSHA --word-workers ${WORD_WORKERS:-4}"

# one write-once run (single shard) and its merge: compose NAME ARGS...
compose() {
  local name=$1; shift
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json
  exists $merged && { echo "stored: $merged (not rerun)"; return 0; }
  exists $part || python scripts/round15_compose_eval.py run $BASE "$@" --output-key $part || die "run $name"
  python scripts/round15_compose_eval.py merge --shard-keys $part --output-key $merged || die "merge $name"
  echo "R20 $name $merged $(sha $merged)"
}

case $PHASE in
  a1)
    case $ARM in
      r10)           compose a1-r10 --variants oracle_stage2 ;;
      r14b-fraction) compose a1-r14b-fraction --variants oracle_stage2 \
                       --vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA ;;
      r14b-activity) compose a1-r14b-activity --variants oracle_stage2 \
                       --vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA --semlex-core activity ;;
      *) die "a1: ARM r10, r14b-fraction or r14b-activity" ;;
    esac ;;
  a2)
    compose a2-continuous-$VOCAB --variants oracle_stage2 $VMIX --oracle-source continuous \
      --gloss-manifest-key $GLOSS --gloss-manifest-sha256 $GLOSSSHA ;;
  a3)
    case $ARM in
      tempo-1.0) compose a3-$ARM-$VOCAB --variants oracle_stage2 $VMIX --retime continuous --tempo 1.0 \
                   --predicted-duration ;;
      tempo-0.8|tempo-1.25|tempo-1.5) compose a3-$ARM-$VOCAB --variants oracle_stage2 $VMIX --retime continuous \
                   --tempo ${ARM#tempo-} ;;
      *) die "a3: ARM tempo-0.8, tempo-1.0, tempo-1.25 or tempo-1.5" ;;
    esac ;;
  a6o) compose a6-oracle-lp6-$VOCAB --variants oracle_stage2 $VMIX --lowpass-hz 6 ;;
  a0)  compose a0-w14b-$VOCAB --variants core_crossfade $VMIX $GEN ;;
  a6)  compose a6-w14b-lp6-$VOCAB --variants core_crossfade $VMIX $GEN --lowpass-hz 6 ;;
  cache)
    case $ARM in
      w14b) CK=$W14B ;; r16) CK=$R16 ;; *) die "cache: ARM w14b or r16" ;;
    esac
    PREFIX=$OUT/cache-$ARM-$VOCAB
    VCACHE=""; [ $VOCAB = r14b ] && VCACHE="--vocabulary-mix-key $R14B --vocabulary-mix-sha256 $R14BSHA"
    exists $PREFIX/index.json && { echo "stored: $PREFIX/index.json"; echo "R20-PHASE-A-DONE 0"; exit 0; }
    N=${CACHE_PROCS:-4}; pids=()
    for s in $(seq 0 $(( N - 1 ))); do
      exists $PREFIX/shard-$s-of-$N.json && continue
      OMP_NUM_THREADS=$(( T / N > 0 ? T / N : 1 )) python scripts/round20_word_cache.py run --prefix $PREFIX \
        --mix-key $R10 --mix-sha256 $R10SHA $VCACHE --sentences-key $D20 --sentences-sha256 $D20SHA \
        --checkpoint-key $CK --generator-mix-sha256 $R14BSHA --samples $SAMPLES --choose $CHOOSE \
        --recognizer-run sign-recognizer-v3d --device $DEV --shards $N --shard $s --clips $CLIPS \
        > /root/r20-cache-$ARM-$s.log 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || die "a cache shard failed (logs /root/r20-cache-$ARM-*.log)"; done
    python scripts/round20_word_cache.py merge --prefix $PREFIX --shards $N || die "cache merge"
    echo "R20 cache-$ARM $PREFIX/index.json $(sha $PREFIX/index.json)" ;;
  a4)
    case $ARM in n4|n8|n16) ;; *) die "a4: ARM n4, n8 or n16" ;; esac
    if [ "${SMOKE:-0}" = 1 ]; then      # a smoke cache holds best-of-1 and best-of-4 only
      [ -n "${CACHE_KEY:-}" ] || die "a4 smoke: give CACHE_KEY (a smoke cache's index.json)"; IDX=$CACHE_KEY; NCH=4
    else IDX=$OUT/cache-w14b-$VOCAB/index.json; NCH=${ARM#n}; fi
    compose a4-w14b-$ARM-$VOCAB --variants core_crossfade $VMIX --word-cache-key $IDX \
      --word-cache-sha256 $(sha $IDX) --word-cache-n $NCH --recognizer-run sign-recognizer-v2 ;;
  a5)
    A=$OUT/cache-r16-$VOCAB/index.json; B=$OUT/cache-w14b-$VOCAB/index.json
    [ -n "${A5_A:-}" ] && A=$A5_A; [ -n "${A5_B:-}" ] && B=$A5_B
    K=$OUT/a5-r16-vs-w14b-$VOCAB.json
    exists $K && { echo "stored: $K"; echo "R20-PHASE-A-DONE 0"; exit 0; }
    python scripts/round20_compare.py words --a $A --b $B --output-key $K || die "a5 compare"
    echo "R20 a5 $K $(sha $K)" ;;
  *) die "unknown R20_PHASE $PHASE" ;;
esac
echo "R20-PHASE-A-DONE 0"
