#!/usr/bin/env bash
# comp_a.sh — post-S2 workstream A on D20 (reports/post_s2_plan.md section 3). Evaluation only: NOTHING TRAINS. PREPARED, NOT LAUNCHED.
# NOTHING IS DECLARED BY THIS FILE: a full read stops unless COMP_A_DECLARED is a real ancestor commit whose plan holds that test's rule text
# (scripts/comp_a_declared.py).
#
#   COMP_PHASE=<phase> [ARM=<arm>] [SMOKE=1] [COMP_A_DECLARED=<plan commit>] [the comparator hashes below] bash runpod/fleet/comp_a.sh
#
# Every run reads D20, pinned, and nothing else: there is no option for another set, SET other than d20 stops the job, and the Python
# options refuse any other key before it is read (comp_a_lib.require_d20). C20, the 1,000, test and c00 (How2Sign signer 9) are refused.
# Every report is --report-only and write-once under $OUT; each read adds loop share, word echo and gap share (scripts/comp_a_read.py).
# SMOKE=1 reads at most 5 real D20 clips under $OUT/smoke/<UTC stamp>/, with no comparator. Nothing decides on a smoke.
# One primary arm per phase is pre-registered (comp_a_read.PRIMARY); only it has a rule. A comparator this job wrote itself is pinned by the hash its
# own COMP-A line printed: MARGIN0_SHA, BASE_SHA, SAMPLE0_SHA, TOP10_READ_SHA. Those must also be this code and environment (comp_a_read.check_provenance).
# Phases (GPU for a full run; a smoke runs anywhere):
#   a1         ARM=margin0 (comparator) | hold (PRIMARY) | left0.5, both0.7 (report-only sensitivities, each with its real control). a9's stored
#              words (no gains, best of 16), margin 0, at predicted timing. Read first against a1-margin0, second against a10's a9@predicted.
#   a3-oracle  ARM=base (comparator) | top10 (PRIMARY) | top30 (sensitivity). Real words at predicted timing, margin 0: r10, or r10 plus the first ranks
#              of reports/comp_a3_tokens.json. Comparators: the base at the same timing, then a1.
#   a3-cache   The new entries of the top 30 only, no gains, 16 samples, chosen by v3d (PMI where a word has no label). Stops unless the top-10 oracle's
#              read, pinned by TOP10_READ_SHA, passes its gate: the real-word oracle comes first.
#   a3-gen     ARM=top10 (PRIMARY) | top30 (sensitivity). a9's stored words plus that cache, margin 0, predicted timing. Comparator: a1's margin0.
#   a4-index   The clip-choice index: 16 candidates per D20 r10 entry, v3d's choice, sample0 and four fixed draws (scripts/comp_a_clip_choice.py).
#   a4         ARM=sample0 (comparator) | n16 (PRIMARY) | draw1 ... draw4 (sensitivities). Real-word oracle at a1's options. Not a ceiling: a non-rise is
#              not informative. Comparators: sample0 (not for itself), a1, then a4's generated best of 16 for reference.
#   a4-summary chosen minus sample0, sample0 minus a1, the draws' variance, with a bootstrap over sentences and entries.
# Ends with one line: COMP-A-DONE <0|1> phase=<phase> arm=<arm> smoke=<0|1>. Any other way out prints ALERT and COMP-A-DONE 1.
set -uo pipefail
PHASE=${COMP_PHASE:-none}; ARM=${ARM:-}; ENDED=""; SMOKING=0
done_line() { ENDED=1; echo "COMP-A-DONE $1 phase=$PHASE arm=${ARM:-none} smoke=$SMOKING"; }
die() { echo "ALERT $*"; done_line 1; exit 1; }
unexpected() { code=$?; if [ -z "$ENDED" ]; then echo "ALERT comp_a stopped unexpectedly, exit $code"; done_line 1; exit 1; fi; }
trap unexpected EXIT
source /root/r2env.sh
cd /root/work/Sign-Translator
if [ -r /sys/fs/cgroup/cpu.max ]; then read Q P < /sys/fs/cgroup/cpu.max; [ "$Q" = max ] && Q=-1
else Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1) P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); fi
if [ "$Q" -gt 0 ]; then C=$(( Q / P )); else C=$(/root/venv/bin/python -c "import os; print(len(os.sched_getaffinity(0)))"); fi
T=${THREADS:-$C}
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
uv pip install -q 'transformers==4.57.6' sentencepiece || die "uv pip install failed"     # the version the stored a10 reads used
sha() { python -c "import os,sys,hashlib; from signtranslator.data.text_motion_dataset import R2Store; print(hashlib.sha256(R2Store(os.environ).get(sys.argv[1])).hexdigest())" "$1"; }
exists() { python -c "import os,sys; from signtranslator.data.text_motion_dataset import R2Store; sys.exit(0 if R2Store(os.environ).exists(sys.argv[1]) else 1)" "$1"; }
pin() { [ "$(sha "$1")" = "$2" ] || die "$1 differs from its pin"; }

[ -z "${SET:-}" ] || [ "$SET" = d20 ] || die "SET: these tests read D20 only; C20, the 1,000 and test are refused"
case ${SMOKE:-0} in 0) ;; 1) SMOKING=1 ;; *) die "SMOKE must be 0 or 1" ;; esac
case $PHASE in a1) TEST=a1 ;; a3-oracle) TEST=a3-oracle ;; a3-cache|a3-gen) TEST=a3-gen ;; a4-index|a4|a4-summary) TEST=a4 ;; *) die "unknown COMP_PHASE $PHASE: a1 a3-oracle a3-cache a3-gen a4-index a4 a4-summary" ;; esac
if [ $SMOKING = 0 ]; then
  [ -n "${COMP_A_DECLARED:-}" ] || die "a full read needs COMP_A_DECLARED=<plan commit>: nothing is declared yet"
  python scripts/comp_a_declared.py "$COMP_A_DECLARED" $TEST || die "COMP_A_DECLARED $COMP_A_DECLARED does not declare $TEST"
fi
O=acquisition/2026-10-17-v1/youtube-asl-v1
A=$O/round20/phase-a
OUT=$O/round20/comp-a
R10=$O/round10-mixed-v1/manifest.json; R10SHA=ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a
R14B=$O/round14b-mixed-v1/manifest.json; R14BSHA=710a3941dce4abf099a336705203a719f441c8cac5847b0e59ed7679d515eb7d
D20=$O/round20/d20.json; D20SHA=d311703b748821822f65f8e65c5e6bd81422d5e16231171f0ab1ea54e185dabe
MODEL=42389ca48d48ac9923843557d66cd3e98c0faf0d5df54a3d5752ddb41f4e6f21      # the declared stage-2 model
W14B=training-runs/tokens-rvq-d0b-prior-r14b-w14b/checkpoints/step-00030000.pt
W14BSHA=0a689471fc6ea90277ba29fd96650119e404da6ccc63907978b9e3edaa71e563
GLOSS=acquisition/2026-10-17-v1/derived/asllrp-dai2-v1/gloss-manifest-v1/manifest.json
GLOSSSHA=f8c40aead29ad1cf97c4a3986d326d475f07bf539deec87219ba69cd47e72944
BT=training-runs/back-translator-kp2d-uwb-ft-v1; BTSHA=3db49bb084ed499bceffbbd93b83d6f62059da0d6727369c9472b029668b0dba
BTCK=$BT/checkpoints/step-00009000.pt; BTCKSHA=2e3ded3fa5524371bdf0986c515ebdfffadb8040cb6cf94a67881c4a4326f5d8
V2=training-runs/sign-recognizer-v2; V2SHA=8989a03b8af3137659f375f1f7c64507a7341061e77b19a5f677cfa230661497
V2CK=$V2/checkpoints/step-00027000.pt; V2CKSHA=a126bf42c0fa4ce8a00e168efbe785668a9ee98600d1146e0d0503c087ba82fb
V3D=training-runs/sign-recognizer-v3d; V3DSHA=3b50a8168e9365984a1f5eb42dd1525466e700bae4036c9bb485a23704cd7ba0
V3DCK=$V3D/checkpoints/step-00012000.pt; V3DCKSHA=0c6c10e232037e869f3983ed6df16df44b3e0a092cbfa34d40ca744cbf00a4e2
# the stored words and reads; hashes read from R2 on 2026-11-01 (a10's and a1's by their bytes only, none of their numbers were read here)
CACHE9=$A/cache-w14b-nogains-r10/index.json; CACHE9SHA=507ee9cc8f436107863d203ec844e46114d55d7a053883f030d901b07c7f137b
A10=$A/a10-predicted-nogains-r10; A10SHA=1e912c9a3cac0f8c581e1542c53de41c8152196d371ee61e450e691836af537d
A10SHARD=ded8847085fa6aded3dfd6e40a1c2279b13b87d798dff9f9c809a53a4636646e
A1R=$A/a1-oracle-r10; A1RSHA=adf3a11010e32a66e9909d07705155aa3b0601e25d9f5b4b0de4240829060835
A1RSHARD=4a40096bd817b105809d5f1d971725ffc7e2104bf623ecd43e40f67a2dff2e14
A4R=$A/a4-w14b-n16-r10; A4RSHA=25bff85153d8a3dd8ad2a7205a586d526621bb17a124ea420e086012c9217e4f
A4RSHARD=57055254ca0dd67b8ad12d216a29397077cd911e20a7f3519f714352d9d3970e
TOKENS=reports/comp_a3_tokens.json; TOKENSSHA=c1105410ef8d502cfcf145768329fe1f72886106afdf83cd0474bfdd654cb776
HOLDSTATS=reports/comp_a_hold_stats.json; HOLDSHA=1b7c24a81aa75c58c523096595b40df537fcd4a49094b88b1eb19686c17440d7
A2FILE=reports/comp_a2_left_arm.json; A2SHA=1dab85c88ba9380cdba89fe4b830d0987cfbeb29d0dec638636af5034e84644a
DURATION=-0.083,0.3691,18289
for p in "$R10 $R10SHA" "$R14B $R14BSHA" "$D20 $D20SHA" "$GLOSS $GLOSSSHA" "$BT/best.json $BTSHA" "$BTCK $BTCKSHA" \
         "$V2/best.json $V2SHA" "$V2CK $V2CKSHA" "$V3D/best.json $V3DSHA" "$V3DCK $V3DCKSHA" "$CACHE9 $CACHE9SHA"; do
  set -- $p; pin "$1" "$2"
done
DEV=cpu; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && DEV=cuda
CLIPS=500; READSMOKE=""
if [ $SMOKING = 1 ]; then
  OUT=$OUT/smoke/$(date -u +%Y%m%dT%H%M%SZ); CLIPS=5; READSMOKE=--smoke
else
  [ "$DEV" = cuda ] || die "a full read needs a GPU, as the stored reads' GPU pods (DEV=$DEV)"
fi
DECL=""; [ -z "${COMP_A_DECLARED:-}" ] || DECL="--declared-commit $COMP_A_DECLARED"
BASE="--mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS \
  --selection stage1 --timing sign-fitted --expect-model-sha256 $MODEL --likelihood --device $DEV"
PREDICT="--core-margin 0 --predicted-duration --expect-duration-model=$DURATION"
WORDS9="--variants core_crossfade --word-cache-key $CACHE9 --word-cache-sha256 $CACHE9SHA --word-cache-n 16 --recognizer-run sign-recognizer-v2"
ORACLE="--variants oracle_stage2 --semlex-oracle-tracked"
EXTRA="--vocabulary-extra-file $TOKENS --vocabulary-extra-sha256 $TOKENSSHA"

# a hash for a success line, or stop: no success prints with an empty hash
said() { local hash; hash=$(sha "$3"); [ -n "$hash" ] || die "$2: no hash of $3"; echo "$1 $2 $3 $hash"; }
# one write-once run, its report-only merge and its read: arm NAME TEST ARMNAME "COMPARATORS" ARGS...
arm() {
  local name=$1 test=$2 arm_name=$3 comparators=$4; shift 4
  local part=$OUT/$name-shard0.json merged=$OUT/$name.json read=$OUT/$name-read.json
  if exists $read; then echo "stored: $read (not rerun)"; return 0; fi
  if ! exists $merged; then
    exists $part || python scripts/round15_compose_eval.py run $BASE "$@" --output-key $part || die "run $name"
    python scripts/round15_compose_eval.py merge --report-only --shard-keys $part --output-key $merged || die "merge $name"
  fi
  local c=""
  if [ $SMOKING = 0 ]; then for x in $comparators; do c="$c --comparator $x"; done; fi
  python scripts/comp_a_read.py read --test $test --arm-name $arm_name --arm $merged $c $READEXTRA $READSMOKE $DECL --output-key $read || die "read $name"
  said COMP-A $name $merged
  said COMP-A-READ $name $read
}
stored_reads() {        # the stored comparators' bytes, before a full read relies on them
  [ $SMOKING = 1 ] && return 0
  pin $A10.json $A10SHA; pin $A10-shard0.json $A10SHARD
  pin $A1R.json $A1RSHA; pin $A1R-shard0.json $A1RSHARD
  pin $A4R.json $A4RSHA; pin $A4R-shard0.json $A4RSHARD
}
own_comparator() {      # a comparator this job wrote: stored, and the bytes its COMP-A line printed
  [ $SMOKING = 1 ] && return 0
  exists $1 || die "$PHASE: its comparator $1 comes first"
  [ -n "$2" ] || die "$PHASE: give the hash of its comparator $1 (the COMP-A line of its run)"
  pin $1 $2
}
A9P=a9_predicted=$A10.json:core_crossfade@predicted
A1P=a1=$A1R.json:oracle_stage2
READEXTRA=""

case $PHASE in
  a1)
    stored_reads
    M0=$OUT/a1-margin0.json
    READEXTRA="--a9-sides-file $A2FILE --a9-sides-sha256 $A2SHA"
    case $ARM in
      margin0) arm a1-margin0 a1 margin0 "$A9P:core_margin" $WORDS9 $PREDICT --comp-a-read ;;
      hold)
        own_comparator $M0 "${MARGIN0_SHA:-}"
        arm a1-hold a1 hold "margin0=$M0:core_crossfade@predicted:hold_left $A9P:core_margin+hold_left" $WORDS9 $PREDICT \
          --hold-left-file $HOLDSTATS --hold-left-sha256 $HOLDSHA ;;
      left0.5|both0.7)
        own_comparator $M0 "${MARGIN0_SHA:-}"
        if [ $ARM = left0.5 ]; then SCALE="left=0.5,right=1"; else SCALE="left=0.7,right=0.7"; fi
        arm a1-$ARM a1 $ARM "margin0=$M0:core_crossfade@predicted:arm_scale $A9P:core_margin+arm_scale" $WORDS9 $PREDICT --arm-scale $SCALE ;;
      *) die "a1: ARM margin0, hold, left0.5 or both0.7" ;;
    esac ;;
  a3-oracle)
    stored_reads
    READEXTRA="--a9-sides-file $A2FILE --a9-sides-sha256 $A2SHA"
    BASEREPORT=$OUT/a3-oracle-base.json
    case $ARM in
      base)  arm a3-oracle-base a3-oracle base "$A1P:core_margin" $ORACLE $PREDICT --comp-a-read ;;
      top10|top30)
        own_comparator $BASEREPORT "${BASE_SHA:-}"
        arm a3-oracle-$ARM a3-oracle $ARM "base=$BASEREPORT:oracle_stage2@predicted:vocabulary_extra $A1P:core_margin+vocabulary_extra" $ORACLE $PREDICT $EXTRA --vocabulary-extra-top ${ARM#top} ;;
      *) die "a3-oracle: ARM base, top10 or top30" ;;
    esac ;;
  a3-cache)
    PREFIX=$OUT/cache-a3-top30-nogains
    if exists $PREFIX/index.json; then echo "stored: $PREFIX/index.json"
    else
      if [ $SMOKING = 0 ]; then
        own_comparator $OUT/a3-oracle-top10-read.json "${TOP10_READ_SHA:-}"
        python scripts/comp_a_read.py gate --read $OUT/a3-oracle-top10-read.json || die "a3-cache: the top-10 oracle did not pass its gate; no generation"
      fi
      pin $W14B $W14BSHA
      exists $PREFIX/shard-0-of-1.json || python scripts/round20_word_cache.py run --prefix $PREFIX --mix-key $R10 --mix-sha256 $R10SHA \
        --sentences-key $D20 --sentences-sha256 $D20SHA --clips $CLIPS --checkpoint-key $W14B --generator-mix-sha256 $R14BSHA \
        --samples 16 --choose 16 --recognizer-run sign-recognizer-v3d --device $DEV --no-decoder-gains \
        $EXTRA --vocabulary-extra-top 30 || die "a3-cache run"
      python scripts/round20_word_cache.py merge --prefix $PREFIX || die "a3-cache merge"
      said COMP-A cache-a3-top30-nogains $PREFIX/index.json
    fi ;;
  a3-gen)
    case $ARM in top10|top30) ;; *) die "a3-gen: ARM top10 or top30" ;; esac
    stored_reads
    if [ $SMOKING = 1 ]; then
      [ -n "${CACHE_KEY:-}" ] || die "a3-gen smoke: give CACHE_KEY (a smoke's a3-cache index.json)"; IDX=$CACHE_KEY
    else
      IDX=$OUT/cache-a3-top30-nogains/index.json
      exists $IDX || die "a3-gen: the new entries' cache $IDX is not stored"
    fi
    READEXTRA="--a9-sides-file $A2FILE --a9-sides-sha256 $A2SHA"
    M0=$OUT/a1-margin0.json
    own_comparator $M0 "${MARGIN0_SHA:-}"
    IDXSHA=$(sha $IDX); [ -n "$IDXSHA" ] || die "a3-gen: no hash of $IDX"
    arm a3-gen-$ARM a3-gen $ARM "margin0=$M0:core_crossfade@predicted:vocabulary_extra+word_cache_extra" $WORDS9 $PREDICT $EXTRA --vocabulary-extra-top ${ARM#top} \
      --word-cache-extra-key $IDX --word-cache-extra-sha256 $IDXSHA ;;
  a4-index)
    KEY=$OUT/a4-clip-choice.json
    if exists $KEY; then echo "stored: $KEY"
    else
      python scripts/comp_a_clip_choice.py --mix-key $R10 --mix-sha256 $R10SHA --sentences-key $D20 --sentences-sha256 $D20SHA \
        --clips $CLIPS --samples 16 --choose 4,8,16 --recognizer-run sign-recognizer-v3d --device $DEV --output-key $KEY || die "a4-index"
      said COMP-A a4-clip-choice $KEY
    fi ;;
  a4)
    stored_reads
    if [ $SMOKING = 1 ]; then
      [ -n "${CHOICE_KEY:-}" ] || die "a4 smoke: give CHOICE_KEY (a smoke's a4-clip-choice.json)"; IDX=$CHOICE_KEY
    else
      IDX=$OUT/a4-clip-choice.json
      exists $IDX || die "a4: the choice $IDX is not stored"
    fi
    IDXSHA=$(sha $IDX); [ -n "$IDXSHA" ] || die "a4: no hash of $IDX"
    S0=$OUT/a4-clipchoice-sample0.json
    COMMON="$A1P:oracle_choice a4_generated=$A4R.json:core_crossfade:word_cache+oracle_choice"
    case $ARM in
      sample0) PICK=sample0; FIRST="" ;;
      n16) PICK=16; own_comparator $S0 "${SAMPLE0_SHA:-}"; FIRST="sample0=$S0:oracle_stage2:oracle_choice " ;;
      draw1|draw2|draw3|draw4) PICK=$ARM; own_comparator $S0 "${SAMPLE0_SHA:-}"; FIRST="sample0=$S0:oracle_stage2:oracle_choice " ;;
      *) die "a4: ARM sample0, n16 or draw1 to draw4" ;;
    esac
    arm a4-clipchoice-$ARM a4 $ARM "$FIRST$COMMON" $ORACLE --core-margin 0.1 --recognizer-run sign-recognizer-v2 \
      --oracle-choice-key $IDX --oracle-choice-sha256 $IDXSHA --oracle-choice-pick $PICK ;;
  a4-summary)
    [ $SMOKING = 0 ] || die "a4-summary reads all of D20: no smoke"
    stored_reads
    KEY=$OUT/a4-clipchoice-decomposition.json
    if exists $KEY; then echo "stored: $KEY"
    else
      S0=$OUT/a4-clipchoice-sample0.json
      own_comparator $S0 "${SAMPLE0_SHA:-}"
      DRAWS=""; for k in 1 2 3 4; do exists $OUT/a4-clipchoice-draw$k.json || die "a4-summary: draw$k is not stored"; DRAWS="$DRAWS --draw $OUT/a4-clipchoice-draw$k.json"; done
      exists $OUT/a4-clipchoice-n16.json || die "a4-summary: n16 is not stored"
      python scripts/comp_a_read.py decompose --chosen $OUT/a4-clipchoice-n16.json --sample0 $S0 $DRAWS --a1 $A1R.json $DECL --output-key $KEY || die "a4-summary"
      said COMP-A a4-clipchoice-decomposition $KEY
    fi ;;
esac
done_line 0
