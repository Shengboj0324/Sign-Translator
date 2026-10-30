"""run_resilient: a worker that dies (os._exit, as an OOM kill does) costs only the jobs that keep breaking the pool."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("extract_video_corpus", ROOT / "scripts" / "extract_video_corpus.py")
evc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evc)


def job_fn(job, args):
    if job == "killer":
        os._exit(9)                 # the worker dies abruptly: the pool breaks
    if job == "raises":
        raise ValueError("bad video")
    return dict(status="converted")


def run(jobs, workers=2, attempts=2):
    seen = []
    evc.run_resilient(job_fn, jobs, None, workers, lambda n, r: seen.append((n, r["status"])), max_attempts=attempts)
    return seen


def test_all_succeed():
    seen = run([f"v{i}" for i in range(9)])
    assert [n for n, _ in seen] == list(range(1, 10))
    assert all(s == "converted" for _, s in seen)


def test_one_killer_fails_only_itself():
    jobs = [f"v{i}" for i in range(20)] + ["killer"] + [f"w{i}" for i in range(20)]
    seen = run(jobs)
    statuses = [s for _, s in seen]
    assert len(seen) == len(jobs) and [n for n, _ in seen] == list(range(1, len(jobs) + 1))
    assert statuses.count("failed") == 1 and statuses.count("converted") == len(jobs) - 1


def test_ordinary_exception_fails_one_job():
    statuses = [s for _, s in run(["a", "raises", "b"])]
    assert sorted(statuses) == ["converted", "converted", "failed"]


def test_auto_workers_budget():
    assert evc.auto_workers() >= 1


def test_two_killers_and_many_innocents():
    jobs = [f"v{i}" for i in range(30)] + ["killer"] + [f"w{i}" for i in range(30)] + ["killer"] + ["x"]
    statuses = [s for _, s in run(jobs, workers=3)]
    assert len(statuses) == len(jobs)
    assert statuses.count("failed") == 2 and statuses.count("converted") == len(jobs) - 2
