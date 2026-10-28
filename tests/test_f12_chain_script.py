"""runpod/fleet/f12_chain.sh in dry-run mode (F12_DRY=1: every stage's command printed, nothing run): the declared
round-3e settings on F12's step-60000 checkpoint, and no test-reading stage unless the test-set guard allows it."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "runpod" / "fleet" / "f12_chain.sh"
GEN = "training-runs/tokens-rvq-d0b-prior-r12-f12/checkpoints/step-00060000.pt"
D = "acquisition/2026-10-17-v1/youtube-asl-v1/round12/chain-f12-step60000"
TEST_READS = ("--test-limit", "--r6-from-validation-key", "review-generated", "r6_guard.py verdict --gate-key")


def run(*args, **env):
    out = subprocess.run(["bash", str(CHAIN), *args], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "F12_DRY": "1", "PY": sys.executable, **env})
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def stages(lines):
    return [(m.group(1), m.group(2)) for m in (re.match(r"STAGE (\S+): (.*)", line) for line in lines) if m]


def test_sweep_is_round_3e_on_f12_step_60000():
    seen = []
    for part in "abc":
        evals = [cmd for name, cmd in stages(run("sweep", part)) if name.startswith("eval-")]
        evals = list(dict.fromkeys(evals))          # the dry run's "rerun once" pass repeats each command
        assert len(evals) == 6
        seen += evals
    assert len(set(seen)) == 18
    temps = set()
    for cmd in seen:
        assert f"--checkpoint-key {GEN}" in cmd and "--partition val " in cmd and "--limit 300 " in cmd
        assert "--guidance 1,1.5,2 " in cmd and "--manifest-key acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1" in cmd
        assert f"--output-key {D}/evaluations/" in cmd
        t = re.search(r"--part-temperature body=([0-9.]+),left_hand=([0-9.]+),right_hand=([0-9.]+),face=([0-9.]+)", cmd)
        g = re.search(r"--signer-component (c0\d)", cmd).group(1)
        assert t.group(1) == t.group(2) == t.group(3)
        temps.add((t.group(1), t.group(4), g))
    assert temps == {(h, f, g) for h in ("1.0", "1.05", "1.1") for f in ("0.9", "1.0") for g in ("c01", "c03", "c05")}


def test_grid_is_round_3e():
    grids = [cmd for name, cmd in stages(run("grid", F12_DRY_STORED="1")) if name.startswith("grid-")]
    assert len(grids) == 6
    for cmd in grids:
        assert "--round3c" in cmd and "--head-smoothing 0.1,0.2,0.3" in cmd and "--head-damping 1,0.8,0.6" in cmd
        assert "--groups c01,c03,c05 --pool-group pool" in cmd


def test_blocked_guard_reads_no_test_data():
    lines = run("review", F12_DRY_GUARD="block")
    names = [n for n, _ in stages(lines)]
    assert names == ["cross-signer-baseline-validation", "gate", "strict-review-validation", "r7-validation", "guard",
                     "verdict"]
    for name, cmd in stages(lines):
        assert not any(t in cmd for t in TEST_READS[:3]), (name, cmd)
    assert "R6 not run: validation rules failed (test partition left untouched)" in lines
    verdict = dict(stages(lines))["verdict"]
    assert "strict-review-validation.json" in verdict and "--r7-test-key" not in verdict


def test_validation_stages_never_touch_the_test_partition():
    for guard in ("block", "pass"):
        cmds = dict(stages(run("review", F12_DRY_GUARD=guard)))
        assert "--test-limit" not in cmds["cross-signer-baseline-validation"]
        review = cmds["strict-review-validation"]
        assert "--validation-only" in review and "--envelope-key" not in review and "--heldout-save-generated" not in review
        assert "--r5-vocab-mix-key acquisition/2026-10-17-v1/youtube-asl-v1/round10-mixed-v1/manifest.json" in review
        assert "--r5-expect-words-key acquisition/2026-10-17-v1/youtube-asl-v1/round12/pilot-f12-step10000.json" in review
        assert "--word-seconds 0 " in review and "--translation-smoothing 0.1 " in review and f"--checkpoint-key {GEN}" in review
        assert "--envelope-key " + D + "/evaluations/cross-signer-baseline-validation.json" in cmds["gate"]


def test_passed_guard_runs_r6_after_it():
    lines = run("review", F12_DRY_GUARD="pass")
    names = [n for n, _ in stages(lines)]
    assert names == ["cross-signer-baseline-validation", "gate", "strict-review-validation", "r7-validation", "guard",
                     "cross-signer-baseline", "strict-review-r6", "r7-test", "verdict"]
    cmds = dict(stages(lines))
    assert "--test-limit 92" in cmds["cross-signer-baseline"]
    r6 = cmds["strict-review-r6"]
    assert "--r6-from-validation-key " + D + "/evaluations/tokens-rvq-d0b-prior-r12-f12-strict-review-validation.json" in r6
    assert "--envelope-key " + D + "/evaluations/cross-signer-baseline.json" in r6 and "--validation-only" not in r6
    # the R6 review repeats every argument of the validation review (it re-checks they match)
    validation = cmds["strict-review-validation"].split(" --validation-only")[0]
    assert r6.startswith(validation)
    assert "--r7-test-key" in cmds["verdict"]


@pytest.mark.parametrize("guard", ["block", "pass"])
def test_smoke_is_quick_and_separate(guard):
    cmds = dict(stages(run("review", F12_SMOKE="1", F12_DRY_GUARD=guard)))
    assert "--quick" in cmds["strict-review-validation"] and f"{D}-smoke/" in cmds["strict-review-validation"]
