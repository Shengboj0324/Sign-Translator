"""The round-5 prior pod script, run with a stub python: the trainer's learning rate, warmup and steps come from PLR,
PWARMUP and PSTEPS_FULL (round 14b), and the defaults are every earlier run's exact command."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runpod/fleet/round5_prior.sh"
STUB = """#!/bin/sh
# python stub: '-c' snippets (the SHA reads and uploads) print the expected mix SHA; the trainer's argv is recorded
case "$1" in
  -c) echo "$MSHA" ;;
  scripts/train_motion_tokens.py) shift; printf '%s\\n' "$@" > "$STUB_OUT"; echo trained ;;
  *) : ;;
esac
"""


def trainer_argv(tmp_path: Path, env: dict) -> list[str]:
    """The script with /root replaced by ``tmp_path`` and python by the stub; the trainer's arguments."""
    root = tmp_path / "root"
    (root / "work/Sign-Translator").mkdir(parents=True, exist_ok=True)
    (root / "venv/bin").mkdir(parents=True, exist_ok=True)
    stub = root / "venv/bin/python"
    stub.write_text(STUB)
    stub.chmod(0o755)
    text = SCRIPT.read_text().replace("source /root/r2env.sh", ":").replace("/root/", f"{root}/")
    out = tmp_path / "argv.txt"
    if out.exists():
        out.unlink()
    base = dict(PATH="/usr/bin:/bin", MIX="mix.json", MSHA="m" * 64, TOKKEY="tok.pt", PRUN="run", STUB_OUT=str(out))
    subprocess.run(["bash", "-c", text], env=dict(base, **env), check=True, capture_output=True, text=True)
    return out.read_text().splitlines()


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_defaults_are_the_earlier_runs_command(tmp_path):
    argv = trainer_argv(tmp_path, {})
    assert (flag(argv, "--lr"), flag(argv, "--warmup"), flag(argv, "--steps")) == ("3e-4", "1000", "60000")
    assert flag(argv, "--run") == "run" and flag(argv, "--patience") == "5" and flag(argv, "--val-every") == "1000"
    smoke = trainer_argv(tmp_path, dict(R3D_SMOKE="1", PSTEPS_FULL="30000"))
    assert flag(smoke, "--steps") == "300" and flag(smoke, "--run") == "run-smoke"          # the smoke keeps 300


def test_round14b_overrides(tmp_path):
    argv = trainer_argv(tmp_path, dict(PLR="1e-4", PWARMUP="200", PSTEPS_FULL="30000", PATIENCE="0",
                                       EXTRA_WEIGHTS="--corpus-weight semlex-text-v2=2 --part-valid-loss"))
    assert (flag(argv, "--lr"), flag(argv, "--warmup"), flag(argv, "--steps")) == ("1e-4", "200", "30000")
    assert flag(argv, "--patience") == "0" and "--part-valid-loss" in argv
    assert flag(argv, "--corpus-weight") == "asllvd-text-v1=3" and "semlex-text-v2=2" in argv
    source = (ROOT / "scripts/train_motion_tokens.py").read_text()
    missing = sorted(f for f in {a for a in argv if a.startswith("--")} if f'"{f}"' not in source)
    assert not missing                                                  # every flag the trainer gets, it defines


def test_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert re.search(r"--lr \$\{PLR:-3e-4\} --warmup \$\{PWARMUP:-1000\}", SCRIPT.read_text())
