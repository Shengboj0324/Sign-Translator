"""The SignAvatars format check: what the unpickler refuses, members by exact name only, the verdict that stops the conversion, and the job."""
import io
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s2_lib as L  # noqa: E402
import third_path as X  # noqa: E402
from signtranslator.data.text_motion_dataset import DirectoryStore  # noqa: E402

JOB = ROOT / "runpod/fleet/third_path.sh"
SIGN = "abc_1-2-rgb_front"


class Payload:
    """Runs a command if ever unpickled."""

    def __init__(self, flag):
        self.flag = flag

    def __reduce__(self):
        return (os.system, (f"touch {self.flag}",))


def named_clip(frames=48, hand=45):
    return pickle.dumps(dict(left_hand_pose=np.zeros((frames, hand), np.float32), right_hand_pose=np.zeros((frames, hand), np.float32), length=frames, other=[frames, frames, frames, frames]))


def test_numpy_and_plain_types_load():
    out = X.safe_load(pickle.dumps(dict(a=np.arange(3.), b=[1, "x", (2., None)], c=np.float32(1.))))
    assert np.array_equal(out["a"], np.arange(3.)) and out["b"] == [1, "x", (2., None)]


def test_a_payload_never_runs_and_any_other_global_refuses(tmp_path):
    flag = tmp_path / "ran"
    with pytest.raises(X.Refused, match="system"):
        X.safe_load(pickle.dumps(Payload(flag)))
    assert not flag.exists()
    for module, name in (("torch._utils", "_rebuild_tensor_v2"), ("torch.storage", "_load_from_bytes"), ("builtins", "eval"), ("collections", "OrderedDict"), ("numpy", "load"), ("numpy.testing._private.utils", "runstring")):
        with pytest.raises(X.Refused):
            X.RestrictedUnpickler(io.BytesIO(b"")).find_class(module, name)


def test_a_persistent_id_refuses():
    with pytest.raises(X.Refused, match="persistent"):
        X.RestrictedUnpickler(io.BytesIO(b"")).persistent_load("x")


def test_the_allowed_list_is_numpy_reconstruction_alone():
    assert all(m.split(".")[0] == "numpy" for m, _ in X.ALLOWED) and {n for _, n in X.ALLOWED} == {"_reconstruct", "ndarray", "dtype", "scalar"}


@pytest.mark.parametrize("sign", ["", "a/b-rgb_front", "..-rgb_front", "a-rgb_side", "a", "a\\b-rgb_front"])
def test_a_name_that_is_not_a_front_clip_refuses(sign):
    with pytest.raises(SystemExit, match="not a How2Sign front"):
        X.member_of(sign)


class NoNames(zipfile.ZipFile):
    def namelist(self):
        raise AssertionError("listed")

    def infolist(self):
        raise AssertionError("listed")


def zip_of(tmp_path, members):
    path = tmp_path / "z.zip"
    with zipfile.ZipFile(path, "w") as z:
        for sign, raw in members.items():
            z.writestr(X.member_of(sign), raw)
    return NoNames(path)


def test_a_member_is_read_by_its_exact_name_and_never_by_listing(tmp_path):
    archive = zip_of(tmp_path, {SIGN: b"x", "other-rgb_front": b"y"})
    assert X.read_member(archive, SIGN) == b"x" and X.read_member(archive, "missing-rgb_front") is None


def test_a_member_over_the_ceiling_refuses_unread(tmp_path, monkeypatch):
    archive = zip_of(tmp_path, {SIGN: b"xxxx"})
    monkeypatch.setattr(X, "MEMBER_CEILING", 3)
    with pytest.raises(SystemExit, match="over 3"):
        X.read_member(archive, SIGN)


def test_another_sized_zip_refuses(tmp_path):
    store = DirectoryStore(tmp_path)
    store.put(X.ZIP_KEY, b"short")
    with pytest.raises(SystemExit, match="its index says"):
        X.open_zip(store)


def test_the_range_file_reads_what_a_local_file_would():
    data = bytes(range(256)) * 10

    class S3:
        def head_object(self, Bucket, Key):
            return dict(ContentLength=len(data))

        def get_object(self, Bucket, Key, Range):
            a, b = Range.removeprefix("bytes=").split("-")
            return dict(Body=io.BytesIO(data[int(a):int(b) + 1]))
    store = type("S", (), dict(s3=S3(), bucket="b"))()
    f = io.BufferedReader(X.RangeFile(store, "k"), buffer_size=64)
    assert f.read(5) == data[:5]
    f.seek(-5, 2)
    assert f.read() == data[-5:] and f.read(1) == b""
    f.seek(100)
    assert f.read(300) == data[100:400]


def test_disassembly_names_fields_and_globals_and_runs_nothing(tmp_path):
    flag = tmp_path / "ran"
    info = X.disassemble(pickle.dumps(dict(smplx=Payload(flag), width=1280, height=720, blob=b"abc")))
    assert not flag.exists() and info["fields"][:1] == ["smplx"] and any("system" in g for g in info["globals"]) and "width" in info["fields"] and info["byte_blobs"] == [3] and info["shape_integers"] == {1280: 1, 720: 1}


def row(**change):
    return dict(dict(member=True, restricted_load="loaded", hands=dict(left_hand_pose=[48, 45], right_hand_pose=[48, 45]), fps=24.), **change)


def test_the_format_as_claimed_passes():
    assert X.verdict([row() for _ in range(5)])["stop"] is False


@pytest.mark.parametrize("change,word", [(dict(restricted_load="refused: torch._utils._rebuild_tensor_v2"), "refuses the pickles: torch._utils"), (dict(hands=dict(left_hand_pose=[48, 15], right_hand_pose=[48, 45])), "45 values"), (dict(hands={}), "45 values"), (dict(fps=30.), "24.0 fps"), (dict(fps=None), "24.0 fps"), (dict(member=False), "no member")])
def test_a_format_other_than_claimed_stops(change, word):
    rows = [row() for _ in range(4)] + [row(**change)]
    out = X.verdict(rows)
    assert out["stop"] is True and any(word in m for m in out["misses"])


def test_too_few_clips_stop():
    assert X.verdict([row()])["stop"] is True


def test_a_clip_row_loads_what_is_allowed_and_describes_what_is_refused(tmp_path):
    good = X.clip_row(SIGN, named_clip(), 2.)
    assert good["restricted_load"] == "loaded" and good["hands"] == dict(left_hand_pose=[48, 45], right_hand_pose=[48, 45]) and good["frames"] == 48 and good["fps"] == 24.
    flag = tmp_path / "ran"
    bad = X.clip_row(SIGN, pickle.dumps(dict(smplx=Payload(flag), a=48, b=48)), 2.)
    assert bad["restricted_load"].startswith("refused:") and not flag.exists() and "hands" not in bad and bad["fps"] == 24.
    assert X.clip_row(SIGN, None, 2.) == dict(sign=SIGN, member=False)


def world(monkeypatch, tmp_path, raw):
    store = DirectoryStore(tmp_path / "s")
    records = [dict(sign=f"s{i}-rgb_front", partition="train", signer_component="c02", continuous=dict(seconds=2.)) for i in range(5)]
    monkeypatch.setattr(L, "load_manifest_at", lambda s, k, sha: dict(records=records))
    monkeypatch.setattr(L, "smoke_records", lambda m, n: records[:n])
    path = Path(store.root) / X.ZIP_KEY
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as z:
        for r in records:
            z.writestr(X.member_of(r["sign"]), raw)
    monkeypatch.setattr(X, "ZIP_SIZE", path.stat().st_size)
    monkeypatch.setattr(zipfile, "ZipFile", NoNames)
    return store, records


def test_the_format_command_stops_with_4_on_a_refused_pickle_and_writes_once(monkeypatch, tmp_path, capsys):
    flag = tmp_path / "ran"
    store, _ = world(monkeypatch, tmp_path, pickle.dumps(dict(smplx=Payload(flag), a=48)))
    out = tmp_path / "format.json"
    assert X.main(["format", "--store", f"dir:{store.root}", "--out-file", str(out)]) == X.EXIT_FORMAT
    report = json.loads(out.read_text())
    assert report["verdict"]["stop"] is True and report["report_only"] is True and len(report["clips"]) == 5 and not flag.exists() and "ALERT" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="write-once"):
        X.main(["format", "--store", f"dir:{store.root}", "--out-file", str(out)])


def test_the_format_command_passes_a_named_format(monkeypatch, tmp_path):
    store, _ = world(monkeypatch, tmp_path, named_clip())
    assert X.main(["format", "--store", f"dir:{store.root}", "--out-file", str(tmp_path / "f.json")]) == 0


def test_a_non_train_clip_stops_the_check_before_the_zip_opens(monkeypatch, tmp_path):
    store, records = world(monkeypatch, tmp_path, named_clip())
    records[0]["signer_component"] = "c00"
    monkeypatch.setattr(X, "open_zip", lambda s: (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(SystemExit, match="not a train clip"):
        X.main(["format", "--store", f"dir:{store.root}", "--out-file", str(tmp_path / "f.json")])


def test_this_commit_opens_no_d20_member_and_loads_no_instrument():
    text = (ROOT / "scripts/third_path.py").read_text().split('"""', 2)[2]
    for word in ("D20_KEY", "load_read_list", "open_read_list", "make_reader", "load_reader", "load_unfitted", "namelist", "infolist", "torch"):
        if word == "torch":
            assert "import torch" not in text
        else:
            assert word not in text, word
    assert "stopped the conversion" in X.__doc__ and "How a full read would be read" in X.__doc__ and "45 values per hand and the order are unverified" in X.__doc__


def run_job(tmp_path, code=0, tee_fails=False):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "python").write_text(f'#!/bin/bash\necho "$*" >> {log}\nif [ "$2" = format ]; then echo "{{}}" > "$4"; exit {code}; fi\nexit 0\n')
    names = ["python"]
    if tee_fails:
        (bin_dir / "tee").write_text("#!/bin/bash\n/bin/cat > /dev/null\nexit 9\n")
        names.append("tee")
    for name in names:
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)
    done = subprocess.run(["bash", str(JOB)], env=dict(PATH=f"{bin_dir}:/usr/bin:/bin", HOME=str(tmp_path), VENV_BIN=str(bin_dir), R2ENV=str(tmp_path / "none"), ST_REPO=str(tmp_path), WORK=str(tmp_path / "work")), capture_output=True, text=True, timeout=60)
    return done, log.read_text()


@pytest.mark.parametrize("code", [0, 4, 7])
def test_the_job_passes_the_status_through_the_pipe_and_stores_the_record(tmp_path, code):
    done, calls = run_job(tmp_path, code)
    assert done.returncode == code and done.stdout.strip().endswith(f"THIRD-PATH-DONE {code}") and "put --file" in calls


def test_a_log_that_cannot_be_written_fails_the_job(tmp_path):
    done, _ = run_job(tmp_path, tee_fails=True)
    assert done.returncode == 1 and "log could not be written" in done.stdout


def test_the_job_is_valid_bash_and_uses_pipestatus():
    text = JOB.read_text()
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0 and "PIPESTATUS" in text and "set -uo pipefail" in text and "create-pod" not in text and "c20" not in text.lower()
