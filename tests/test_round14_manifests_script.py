"""The round-14 pod script: valid bash, the pinned base mix, and only flags its two builders define."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runpod/fleet/round14_manifests.sh"


def invocations(text: str) -> dict[str, set[str]]:
    """Per ``scripts/<name>.py`` invoked, the long flags passed to it (across continuation lines)."""
    out, lines = {}, text.splitlines()
    for k, line in enumerate(lines):
        m = re.search(r"python (scripts/\w+\.py)", line)
        if not m:
            continue
        command = line
        while command.endswith("\\") and k + 1 < len(lines):
            k += 1
            command = command[:-1] + lines[k]
        out.setdefault(m.group(1), set()).update(re.findall(r"(--[a-z][a-z0-9-]*)", command.split(m.group(1), 1)[1]))
    return out


def test_round14_script_is_valid_and_its_flags_exist():
    text = SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert "ccfd10b8961e95523ac7938c84261c89f22c8ae786af961908fbeba21fa5647a" in text      # round10-mixed-v1
    calls = invocations(text)
    assert set(calls) == {"scripts/build_semlex_manifest.py", "scripts/build_mixed_text_manifest.py"}
    for script, flags in calls.items():
        source = (ROOT / script).read_text()
        missing = sorted(f for f in flags if f'"{f}"' not in source)
        assert not missing, (script, missing)
    assert "--source video" in text and "--splits val,train" in text and "--base $R10@$R10SHA" in text
