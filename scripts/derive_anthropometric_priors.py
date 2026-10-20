"""Derive the population priors of ``body_space.CUES`` from the ANSUR II public data.

ANSUR II (US Army anthropometric survey, 4,082 men and 1,986 women, public release
2017) is archived in R2 under ``acquisition/2026-10-17-v1/anthropometry-ansur2/`` and
pinned here by SHA-256. A signer's sex is not known to the pipeline, so the prior is
the equal-weight mixture of the two sexes, summarised in log space: mean
``μ = (μ_M + μ_F)/2`` and covariance ``(Σ_M + Σ_F)/2 + Σ_s (μ_s − μ)(μ_s − μ)ᵀ / 2``
(within-sex plus between-sex). Prints the medians, log SDs and correlations, and
with ``--check`` fails unless ``body_space`` embeds the same values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PREFIX = "acquisition/2026-10-17-v1/anthropometry-ansur2"
FILES = {"male": ("ANSUR_II_MALE_Public.csv", "0547aea0170e5293de519389981135e75803f02e6d40c77ace740decc0caac64"),
         "female": ("ANSUR_II_FEMALE_Public.csv", "ed7e800aa97a4d42f8286b988be2fe1883a2e789588169db959864b707d1757a")}
# body_space cue -> (ANSUR variable, millimetres per recorded unit)
VARIABLES = {"interpupillary": ("interpupillarybreadth", .1), "upper_arm": ("acromionradialelength", 1.),
             "forearm": ("radialestylionlength", 1.), "hand_length": ("handlength", 1.)}


def read(source: str, name: str, sha256: str) -> bytes:
    if source == "r2":
        from train_text_to_motion import make_store
        data = make_store("r2").get(f"{PREFIX}/{name}")
    else:
        data = (Path(source) / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{name}: SHA-256 differs from the archived file")
    return data


def priors(tables: dict) -> dict:
    logs = {}
    for sex, data in tables.items():
        rows = list(csv.DictReader(io.StringIO(data.decode("latin-1"))))
        logs[sex] = np.log([[float(r[v]) * unit / 1000 for v, unit in VARIABLES.values()] for r in rows])
    mus = {s: x.mean(0) for s, x in logs.items()}
    mu = sum(mus.values()) / len(mus)
    cov = sum(np.cov(x.T) for x in logs.values()) / len(logs) + sum(np.outer(m - mu, m - mu) for m in mus.values()) / len(mus)
    sd = np.sqrt(np.diag(cov))
    names = list(VARIABLES)
    return dict(subjects={s: len(x) for s, x in logs.items()},
                median_m={n: round(float(np.exp(m)), 5) for n, m in zip(names, mu)},
                log_sd={n: round(float(s), 4) for n, s in zip(names, sd)},
                correlation={f"{a}|{b}": round(float(cov[i, j] / sd[i] / sd[j]), 2)
                             for i, a in enumerate(names) for j, b in enumerate(names) if i < j})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="r2", help="'r2' or a directory holding the two CSVs")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = priors({s: read(args.source, n, h) for s, (n, h) in FILES.items()})
    print(json.dumps(result, indent=2))
    if args.check:
        from signtranslator.data_engineering import body_space
        for name, (_, sd, _, _) in body_space.POPULATION.items():
            if abs(sd - result["log_sd"][name]) > 5e-4:
                raise SystemExit(f"{name}: embedded log SD {sd} differs from ANSUR II {result['log_sd'][name]}")
        for (a, b), r in body_space.CORRELATION.items():
            if abs(r - result["correlation"].get(f"{a}|{b}", result["correlation"].get(f"{b}|{a}"))) > .005:
                raise SystemExit(f"{a}/{b}: embedded correlation differs from ANSUR II")
        for name, median in body_space.ANSUR_MEDIAN.items():
            if median is not None and abs(median - result["median_m"][name]) > 5e-5:
                raise SystemExit(f"{name}: embedded median differs from ANSUR II")
        print("body_space priors match ANSUR II")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
