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
        own = command.split(m.group(1), 1)[1].split("|")[0]          # up to a pipe: later commands' flags are theirs
        out.setdefault(m.group(1), set()).update(re.findall(r"(--[a-z][a-z0-9-]*)", own))
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
    source = (ROOT / "scripts/build_semlex_manifest.py").read_text()
    extra = re.findall(r"SEMLEX_EXTRA\+=\(([^)]*)\)", text)
    flags = {f for e in extra for f in re.findall(r"(--[a-z][a-z0-9-]*)", e)}
    assert flags == {"--states-prefix", "--exclude-signers", "--exclude-reason"}
    assert all(f'"{f}"' in source for f in flags)


def parameters(env: dict) -> dict:
    """The script's parameter block (from ACQ= to the end marker) evaluated by bash under ``env``: its variables, and
    SEMLEX_EXTRA as the list of words the builder receives."""
    lines = SCRIPT.read_text().splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith("ACQ="))
    end = next(k for k, line in enumerate(lines) if line.startswith("# ---- end parameters"))
    names = ("CORPUS_ID", "MIX_ID", "SKEY", "SSUM", "MKEY", "MIXLOG", "STAG", "MTAG", "PARAM_ERROR")
    probe = "\n".join(lines[start:end]) + "\n" + "".join(f'printf "%s\\0" "${n}"\n' for n in names) + \
        'printf "%s\\0" "${#SEMLEX_EXTRA[@]}" ${SEMLEX_EXTRA[@]+"${SEMLEX_EXTRA[@]}"}\n'
    out = subprocess.run(["bash", "-c", "set -uo pipefail\n" + probe], env=dict(PATH="/usr/bin:/bin", **env),
                         check=True, capture_output=True, text=True).stdout.split("\0")[:-1]
    values = dict(zip(names, out))
    count = int(out[len(names)])
    values["SEMLEX_EXTRA"] = out[len(names) + 1:]
    assert len(values["SEMLEX_EXTRA"]) == count
    return values


SL = "acquisition/2026-10-17-v1/sem-lex-mirror-v1"
YT = "acquisition/2026-10-17-v1/youtube-asl-v1"


def test_round14_defaults_are_round_14s_keys_and_flags():
    """No env: exactly round 14's keys, log name, tags and builder flags (the script before parameterisation)."""
    v = parameters({})
    assert v["SKEY"] == f"{SL}/semlex-text-v1/manifest.json" and v["SSUM"] == f"{SL}/semlex-text-v1/summary.json"
    assert v["MKEY"] == f"{YT}/round14-mixed-v1/manifest.json" and v["MIX_ID"] == "round14-mixed-v1"
    assert v["MIXLOG"] == "stage-round14-mix" and v["STAG"] == "SEMLEX-TEXT-V1" and v["MTAG"] == "ROUND14-MIXED-V1"
    assert v["SEMLEX_EXTRA"] == [] and v["PARAM_ERROR"] == ""
    assert parameters({"UNMIRRORED": "1"})["SEMLEX_EXTRA"] == ["--states-prefix", f"{SL}/states-video-v1-unmirrored"]


def test_round14b_env_builds_semlex_text_v2_and_its_mix():
    v = parameters(dict(CORPUS_ID="semlex-text-v2", MIX_ID="round14b-mixed-v1", EXCLUDE_SIGNERS="11,23,46,76"))
    assert v["SKEY"] == f"{SL}/semlex-text-v2/manifest.json" and v["SSUM"] == f"{SL}/semlex-text-v2/summary.json"
    assert v["MKEY"] == f"{YT}/round14b-mixed-v1/manifest.json" and v["PARAM_ERROR"] == ""
    assert v["MIXLOG"] == "stage-round14b-mix" and v["STAG"] == "SEMLEX-TEXT-V2" and v["MTAG"] == "ROUND14B-MIXED-V1"
    extra = v["SEMLEX_EXTRA"]
    assert extra[:3] == ["--exclude-signers", "11,23,46,76", "--exclude-reason"] and len(extra) == 4
    assert "image-right" in extra[3] and "per-signer sides" in extra[3]          # one word, spaces kept
    v = parameters(dict(CORPUS_ID="semlex-text-v2", MIX_ID="round14b-mixed-v1", EXCLUDE_SIGNERS="11",
                        EXCLUDE_REASON="a reason, with spaces", MKEY="x/m.json"))
    assert v["SEMLEX_EXTRA"] == ["--exclude-signers", "11", "--exclude-reason", "a reason, with spaces"]
    assert v["MKEY"] == "x/m.json"


def test_round14_parameter_refusals():
    assert "needs its own MIX_ID" in parameters(dict(CORPUS_ID="semlex-text-v2"))["PARAM_ERROR"]
    assert "needs a new CORPUS_ID" in parameters(dict(EXCLUDE_SIGNERS="11"))["PARAM_ERROR"]
    assert "not semlex-text-v<N>" in parameters(dict(CORPUS_ID="semlex-v2", MIX_ID="m-v1"))["PARAM_ERROR"]
    assert "not a corpus id" in parameters(dict(CORPUS_ID="semlex-text-v2", MIX_ID="a b"))["PARAM_ERROR"]
    text = SCRIPT.read_text()
    assert text.index('[ -z "$PARAM_ERROR" ]') < text.index("exists $SKEY")   # refused before anything is read


def test_unmirror_script_is_valid_and_its_flags_exist():
    script = ROOT / "runpod/fleet/unmirror_semlex_states.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    calls = invocations(script.read_text())
    assert set(calls) == {"scripts/unmirror_semlex_states.py", "scripts/handedness_check.py",
                          "scripts/tokenizer_domain_read.py"}
    for name, flags in calls.items():
        source = (ROOT / name).read_text()
        missing = sorted(f for f in flags if f'"{f}"' not in source)
        assert not missing, (name, missing)
