"""Round 19 amendment 1: arms R19 / R19w. The smoke-recipe check allows R19w exactly one extra argument (the YouTube
corpus weight), and that weight keeps every non-YouTube record's per-step sampling probability at R16C's."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]
import round19_recipe_check as rc  # noqa: E402

R16C_WEIGHTS = ["asllvd-text-v1=3", "rit-text-r6=3", "wlasl-text-r6=3", "asllvd-text-v1=3", "semlex-text-v2=2"]
# trainer-visible training records (partition train, text, usable continuous, <= 20 s), read from R2 on 2026-10-30
COUNTS_14B = {"asllrp-dai2-plan-v2": 1651, "asllvd-text-v1": 5155, "how2sign-two-view-text-v1": 17727,
              "rit-text-r6": 5295, "semlex-text-v2": 26698, "wlasl-text-r6": 1045, "youtube-asl-v1": 99887}
COUNTS_19 = dict(COUNTS_14B, **{"youtube-asl-v1": 173202})


def r16c_smoke():
    return dict(kind="motion_rvq_prior", config_sha256="6a92" + "0" * 60, train_records=157458,
                manifest_key="acquisition/2026-10-17-v1/youtube-asl-v1/round14b-mixed-v1/manifest.json",
                args=dict(run="tokens-rvq-d0b-prior-r16c-control-smoke", corpus_weight=list(R16C_WEIGHTS), lr=1e-4,
                          steps=300, manifest_key="…/round14b-mixed-v1/manifest.json", manifest_sha256="710a",
                          history_dropout=.5, init_sha256="0a68"))


def candidate(arm, weights=()):
    c = copy.deepcopy(r16c_smoke())
    run = {"r19": "tokens-rvq-d0b-prior-r19-refresh", "r19w": "tokens-rvq-d0b-prior-r19w-refresh-weighted"}[arm]
    c["manifest_key"] = "acquisition/2026-10-17-v1/youtube-asl-v1/round19-mixed-v1/manifest.json"
    c["args"].update(run=f"{run}-smoke", manifest_key=c["manifest_key"], manifest_sha256="9950", plan_dropout=0.,
                     plan_memory=True)
    c["args"]["corpus_weight"] += list(weights)
    return c


def test_r19_matches_r16c_and_r19w_may_add_exactly_the_youtube_weight():
    assert rc.differences(r16c_smoke(), candidate("r19"), "r19") == []
    assert rc.differences(r16c_smoke(), candidate("r19w", [rc.YOUTUBE_WEIGHT]), "r19w") == []
    assert rc.YOUTUBE_WEIGHT == "youtube-asl-v1=0.5767"
    # the weight belongs to r19w only, exactly once, with exactly that value
    assert rc.differences(r16c_smoke(), candidate("r19", [rc.YOUTUBE_WEIGHT]), "r19")
    assert rc.differences(r16c_smoke(), candidate("r19w"), "r19w")
    assert rc.differences(r16c_smoke(), candidate("r19w", ["youtube-asl-v1=0.6"]), "r19w")
    assert rc.differences(r16c_smoke(), candidate("r19w", [rc.YOUTUBE_WEIGHT] * 2), "r19w")
    assert rc.differences(r16c_smoke(), candidate("r19w", ["how2sign-two-view-text-v1=2", rc.YOUTUBE_WEIGHT]), "r19w")


@pytest.mark.parametrize("change", [
    lambda c: c["args"].update(lr=3e-4),
    lambda c: c["args"].pop("history_dropout"),
    lambda c: c["args"].update(plan_dropout=.2),
    lambda c: c["args"].update(rollout_blocks=.5),
    lambda c: c.update(config_sha256="1" * 64),
    lambda c: c.update(manifest_key="acquisition/2026-10-17-v1/youtube-asl-v1/round18-mixed-v1/manifest.json"),
])
def test_anything_else_is_refused_for_both_arms(change):
    for arm, weights in (("r19", []), ("r19w", [rc.YOUTUBE_WEIGHT])):
        c = candidate(arm, weights)
        change(c)
        assert rc.differences(r16c_smoke(), c, arm), arm


def test_main_reads_files_and_exits_1_on_a_difference(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(r16c_smoke()))
    (tmp_path / "b.json").write_text(json.dumps(candidate("r19w", [rc.YOUTUBE_WEIGHT])))
    args = ["--control", str(tmp_path / "a.json"), "--candidate", str(tmp_path / "b.json")]
    assert rc.main(["--arm", "r19w", *args]) == 0
    assert rc.main(["--arm", "r19", *args]) == 1
    assert "corpus_weight" in capsys.readouterr().out


def per_step_probability(counts: dict, weights: list[str]) -> dict:
    """{corpus: one record's probability per draw} with the trainer's own weighting (corpus_sample_weights; the
    WeightedRandomSampler draws in proportion to the weights)."""
    from train_motion_tokens import corpus_sample_weights
    records = [dict(corpus=c) for c, n in counts.items() for _ in range(n)]
    w = corpus_sample_weights(records, {k: float(v) for k, v in (x.split("=", 1) for x in weights)})
    total = sum(w)
    first = {}
    for r, x in zip(records, w):
        first.setdefault(r["corpus"], x / total)
    return first


def test_r19w_keeps_every_non_youtube_record_at_r16cs_per_step_probability():
    r16c = per_step_probability(COUNTS_14B, R16C_WEIGHTS)
    r19w = per_step_probability(COUNTS_19, R16C_WEIGHTS + [rc.YOUTUBE_WEIGHT])
    r19 = per_step_probability(COUNTS_19, R16C_WEIGHTS)
    for corpus in COUNTS_14B:
        if corpus != "youtube-asl-v1":
            assert abs(r19w[corpus] / r16c[corpus] - 1) < 1e-3, corpus
            assert r19[corpus] / r16c[corpus] < .8, corpus            # the dilution R19w removes
    # YouTube's total mass matches R16C's too (the diversity, not the mass, changes)
    mass = lambda p, n: p["youtube-asl-v1"] * n["youtube-asl-v1"]  # noqa: E731
    assert abs(mass(r19w, COUNTS_19) / mass(r16c, COUNTS_14B) - 1) < 1e-3


def test_the_jobs_wire_each_arm_to_its_run_weight_and_keys():
    train = (ROOT / "runpod/fleet/round19_train.sh").read_text()
    assert "r19w) PRUN=tokens-rvq-d0b-prior-r19w-refresh-weighted; YTW=' --corpus-weight youtube-asl-v1=0.5767' ;;" in train
    assert "r19)  PRUN=tokens-rvq-d0b-prior-r19-refresh; YTW= ;;" in train
    assert train.count("'\"$YTW\" bash runpod/fleet/round5_prior.sh") == 2
    assert "round19_recipe_check.py --arm $ARM" in train
    pilot = (ROOT / "runpod/fleet/round19_pilot.sh").read_text()
    for key in ("OUTKEY=$O/round19/pilot-$ARM.json", "compare-$ARM-vs-r16c.json", "verdict-$ARM.json", "--arm $VARM"):
        assert key in pilot
    for script in ("round19_train.sh", "round19_pilot.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / "runpod/fleet" / script)]).returncode == 0
