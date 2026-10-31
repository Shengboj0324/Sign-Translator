import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_suite_sees_no_credentials():
    assert not [k for k in os.environ if k.startswith(("R2_", "AWS_", "RUNPOD_"))]


def test_the_guard_removes_a_planted_credential(tmp_path):
    probe = tmp_path / "test_probe.py"
    probe.write_text("import os\n\ndef test_probe():\n    assert 'R2_SECRET_ACCESS_KEY' not in os.environ\n")
    (tmp_path / "conftest.py").write_text((ROOT / "tests" / "conftest.py").read_text())
    env = {**os.environ, "R2_SECRET_ACCESS_KEY": "planted"}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
        env=env, capture_output=True, text=True, timeout=120, cwd=tmp_path,
    )
    assert run.returncode == 0, run.stdout + run.stderr
