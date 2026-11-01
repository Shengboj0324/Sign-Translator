"""Post-S2 A5: does A3 generated top 10 hold across word draws? top10 against margin0, paired within each of five pre-registered fresh word draws, on D20. Report
only: nothing adopts from it, and nothing here reads a form until the job runs it.

**Why.** A3's stored read (+0.343 BLEU, its bootstrap over sentences only) rests on one draw of generated words, and five draws of real clips spanned 1.82-2.41. A draw is
a whole fresh cache at the same recipe (best of 16 by v3d, no gains; ``round20_word_cache.py --draw``) plus the seven new entries at the same draw. The stored read chose
this arm, so it is in no mean.

**Summary, per form (predicted and true length) and metric (BLEU, likelihood minus its own-length blank, v2 spotting):** each draw's paired top10 minus margin0, the mean,
the between-draw SD, how many draws are above 0, a 95% interval from the two-way bootstrap (sentences resampled, draws resampled, every draw on the same sentences), and a
t interval on the five draw differences alone. Each arm's own BLEU and gap share are given too.

``decide`` applies ``RULE`` at predicted timing; true length decides nothing. Fewer than five finished draws is ``incomplete``, never a smaller test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import comp_a_lib as lib  # noqa: E402
import comp_a_read as rd  # noqa: E402

DRAWS = (1, 2, 3, 4, 5)
JOB_FILE = "runpod/fleet/comp_a5.sh"
FORMS = dict(predicted="core_crossfade@predicted", true_length="core_crossfade")
DECLARED_BETWEEN = ("vocabulary_extra", "word_cache_extra")
MIN_POSITIVE = 4
RULE = ("Primary comparison: A3 generated top10 against margin0, paired within each of exactly five pre-registered fresh word draws (draw 1 to 5: new seeds for every entry and "
        "the same recipe, best of 16 by v3d, no gains), on D20 at predicted timing, report only. Confirmed only if all hold: the mean paired BLEU's 95% lower bound above 0 on both "
        "the two-way bootstrap over sentences and draws and the t interval on the five draws; the mean paired likelihood minus blank@predicted's 95% lower bound above 0 on both; "
        "v2 spotting's two-way lower bound above -0.02; BLEU and likelihood each above 0 in at least 4 of the 5 draws; mean wrist no further from 1.0x real than a9@predicted's; mean "
        "hands at least 0.9x margin0's. Mean BLEU and likelihood above 0 with at least 4 of 5 draws positive on each, but a bound not met, is direction only, which is not confirmed. "
        "Anything else is not confirmed. What does not confirm: the stored read of A3 counts in no mean; true-length forms are reported and decide nothing; rank-2 or rank-3 samples, "
        "best-of-4 or best-of-8 choices and any subset of entries are not draws; fewer than five finished draws is incomplete, not a smaller test; no draw is replaced, added or dropped "
        "after any result; a job failure reruns that draw's same seeds and nothing else; no second A5 with more draws on D20 follows a not-confirmed. Confirmed means only that top10 "
        "beats margin0 across word draws on D20's own sentences: D20 chose this arm, so its sentences are not fresh. It is no M1 claim, no adoption and no release bar.")
KNOWN_AT_WRITING = ("A3 generated top10's stored read on D20 was known: BLEU +0.343 [-0.018, +0.718], likelihood +0.0128, spotting +0.0149, wrist 1.774x, and that it passed. A4's real-clip "
                    "draws were known (SD 0.237). No read of any fresh draw existed, and no fresh draw had been generated beyond a two-entry smoke with no reader.")


def report_settings(report: dict) -> dict:
    return (report.get("settings") or {}).get("round20") or {}


def check_draw(k: int, margin0: dict, top10: dict) -> None:
    """One draw's two reports: D20, report-only, all of it, what their names say, the same fresh words, and this draw's."""
    from round20_compare import check_pairable
    for report in (margin0, top10):
        rd.check_arm(report, smoke=False)
    rd.check_arm_name("a1", "margin0", margin0.get("settings") or {})
    rd.check_arm_name("a3-gen", "top10", top10.get("settings") or {})
    check_pairable(top10, margin0, DECLARED_BETWEEN)
    for name, report in (("margin0", margin0), ("top10", top10)):
        cache = report_settings(report).get("word_cache") or {}
        if cache.get("draw") != k:
            raise SystemExit(f"draw {k} {name}: its word cache is draw {cache.get('draw')}; the stored read's words count in no mean")
    extra = report_settings(top10).get("word_cache_extra")
    if not extra:
        raise SystemExit(f"draw {k} top10: no new entries' cache")
    if report_settings(top10).get("vocabulary_extra", {}).get("top") != 10 or (top10.get("round20") or {}).get("vocabulary_extra", {}).get("fingerspelled_single_clips"):
        raise SystemExit(f"draw {k} top10: not the top-10 arm")
    a, b = report_settings(margin0).get("provenance"), report_settings(top10).get("provenance")
    if not a or a != b:
        raise SystemExit(f"draw {k}: its two arms came from other code or another environment: {a} against {b}")


def check_draws(pairs: dict) -> None:
    """Exactly the pre-registered draws, each paired, all on one digest and packages, none sharing words. GPU may differ between draws, never within one."""
    if tuple(sorted(pairs)) != DRAWS:
        raise SystemExit(f"incomplete: draws {sorted(pairs)}, the rule names {list(DRAWS)}; fewer than five finished draws is no smaller test")
    for k, (m, t) in pairs.items():
        check_draw(k, m, t)
    first = report_settings(pairs[DRAWS[0]][0])
    prov = first["provenance"]
    packages = {x: prov["environment"].get(x) for x in ("python", "torch", "transformers", "tokenizers")}
    for k, (m, _) in pairs.items():
        other = report_settings(m)["provenance"]
        if other["code_digest"] != prov["code_digest"] or {x: other["environment"].get(x) for x in packages} != packages:
            raise SystemExit(f"draw {k} came from other code or packages")
    caches = [report_settings(m)["word_cache"]["sha256"] for m, _ in pairs.values()]
    extras = [report_settings(t)["word_cache_extra"]["sha256"] for _, t in pairs.values()]
    if len(set(caches)) != len(DRAWS) or len(set(extras)) != len(DRAWS):
        raise SystemExit("two draws share a word cache: they are one draw")
    refs = pairs[DRAWS[0]][0]["references"]
    if any(m["references"] != refs or m["signs"] != pairs[DRAWS[0]][0]["signs"] for m, _ in pairs.values()):
        raise SystemExit("the draws read other sentences")


def t_interval(values) -> list[float]:
    """95% on the mean of the draw values alone."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [round(float(v.mean() - half), 4), round(float(v.mean() + half), 4)]


def summary_row(per_draw, boot_means) -> dict:
    v = np.asarray(per_draw, dtype=float)
    return dict(per_draw=[round(float(x), 4) for x in v], mean=round(float(v.mean()), 4), between_draw_sd=round(float(v.std(ddof=1)), 4), positive_draws=int((v > 0).sum()),
                two_way_ci95=[round(float(np.percentile(boot_means, 2.5)), 4), round(float(np.percentile(boot_means, 97.5)), 4)], t_ci95=t_interval(v))


def _resamples(n: int, k: int, boot: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n).astype(float), rng.integers(0, k, k)


def bleu_row(stats_a, stats_b, *, boot: int, seed: int = 0) -> dict:
    """Corpus BLEU a - b per draw (arrays draws x sentences x stats). Each resample takes the same sentences in every draw, and draws with replacement."""
    from signtranslator.eval.s0_metrics import bleu_from_statistics
    a, b = np.asarray(stats_a), np.asarray(stats_b)
    per_draw = bleu_from_statistics(a.sum(1)) - bleu_from_statistics(b.sum(1))
    means = np.empty(boot)
    for j, (w, picks) in enumerate(_resamples(a.shape[1], a.shape[0], boot, seed)):
        d = bleu_from_statistics((a * w[None, :, None]).sum(1)) - bleu_from_statistics((b * w[None, :, None]).sum(1))
        means[j] = d[picks].mean()
    return summary_row(per_draw, means)


def mean_row(values, *, boot: int, seed: int = 0) -> dict:
    """A per-sentence paired value per draw (draws x sentences; NaN where absent), as ``bleu_row``."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    filled = np.where(ok, v, 0.)
    per_draw = filled.sum(1) / ok.sum(1)
    means = np.empty(boot)
    for j, (w, picks) in enumerate(_resamples(v.shape[1], v.shape[0], boot, seed)):
        d = (filled * w).sum(1) / np.clip((ok * w).sum(1), 1e-12, None)
        means[j] = d[picks].mean()
    return summary_row(per_draw, means)


def _numbers(values) -> list[float]:
    return [np.nan if x is None else float(x) for x in values]


def spotting_rows(report: dict, shards, form: str) -> list[float]:
    from round20_compare import shard_recalls
    recalls = shard_recalls(report, shards, form)
    if recalls is None:
        raise SystemExit(f"{form}: no per-clip spotting in the shards")
    return _numbers(recalls)


def form_summary(pairs: dict, shards: dict, form: str, *, boot: int) -> dict:
    """Every number of one form across the draws (module docstring)."""
    from signtranslator.eval.s0_metrics import bleu_statistics
    order = sorted(pairs)
    refs = pairs[order[0]][0]["references"]
    blank = rd.blank_of(form, pairs[order[0]][0]["forms"])
    stat = lambda r, f: bleu_statistics(r["forms"][f]["hypotheses"], refs)              # noqa: E731
    m_stats, t_stats = [stat(pairs[k][0], form) for k in order], [stat(pairs[k][1], form) for k in order]
    ll = lambda r, f: np.array(_numbers(r["likelihood"]["per_form"][f]))                 # noqa: E731
    likelihood = [(ll(pairs[k][1], form) - ll(pairs[k][1], blank)) - (ll(pairs[k][0], form) - ll(pairs[k][0], blank)) for k in order]
    spotting = [np.array(spotting_rows(pairs[k][1], shards[k][1], form)) - np.array(spotting_rows(pairs[k][0], shards[k][0], form)) for k in order]
    share = {}
    for name, index in (("margin0", 0), ("top10", 1)):
        share[name] = [rd.form_read(pairs[k][index], form, boot=50)["gap_share"]["share"] for k in order]
    motion = {name: [(pairs[k][index].get("motion_ratio_to_real") or {}).get(form) or {} for k in order] for name, index in (("margin0", 0), ("top10", 1))}
    mean_of = lambda rows, key: round(float(np.mean([r[key] for r in rows if r.get(key) is not None])), 4) if any(r.get(key) is not None for r in rows) else None    # noqa: E731
    return dict(form=form, blank=blank, bleu=bleu_row(t_stats, m_stats, boot=boot), likelihood_minus_blank=mean_row(likelihood, boot=boot), spotting=mean_row(spotting, boot=boot),
                margin0_bleu=[pairs[k][0]["forms"][form]["bleu"] for k in order], top10_bleu=[pairs[k][1]["forms"][form]["bleu"] for k in order], gap_share=share,
                motion=dict(top10_wrist=mean_of(motion["top10"], "wrist_speed"), margin0_wrist=mean_of(motion["margin0"], "wrist_speed"),
                            top10_hands=mean_of(motion["top10"], "hands_tracked"), margin0_hands=mean_of(motion["margin0"], "hands_tracked")))


def decide(predicted: dict, a9_wrist: float | None) -> dict:
    """The rule on the predicted-timing summary. An unknown number fails its check."""
    bleu, ll, spot, motion = predicted["bleu"], predicted["likelihood_minus_blank"], predicted["spotting"], predicted["motion"]
    hands, base = motion.get("top10_hands"), motion.get("margin0_hands")
    checks = dict(bleu_two_way_lower_bound_above_0=bleu["two_way_ci95"][0] > 0, bleu_t_lower_bound_above_0=bleu["t_ci95"][0] > 0,
                  likelihood_two_way_lower_bound_above_0=ll["two_way_ci95"][0] > 0, likelihood_t_lower_bound_above_0=ll["t_ci95"][0] > 0,
                  spotting_two_way_lower_bound_above_minus_0_02=spot["two_way_ci95"][0] > -.02, bleu_positive_in_4_of_5=bleu["positive_draws"] >= MIN_POSITIVE,
                  likelihood_positive_in_4_of_5=ll["positive_draws"] >= MIN_POSITIVE, wrist_no_further_than_a9=rd.no_further(motion.get("top10_wrist"), a9_wrist),
                  hands_at_least_0_9_of_margin0=hands is not None and base is not None and hands >= .9 * base)
    direction = bleu["mean"] > 0 and ll["mean"] > 0 and bleu["positive_draws"] >= MIN_POSITIVE and ll["positive_draws"] >= MIN_POSITIVE
    outcome = "confirmed" if all(checks.values()) else "direction_only_not_confirmed" if direction else "not_confirmed"
    return dict(checks=checks, outcome=outcome)


def read(pairs: dict, shards: dict, *, a9_sides: dict, boot: int = 1000, declared_commit: str | None = None, job_file: str = JOB_FILE) -> dict:
    check_draws(pairs)
    forms = {name: form_summary(pairs, shards, form, boot=boot) for name, form in FORMS.items()}
    return dict(kind="comp-a5-read", report_only=True, declared=bool(declared_commit), declared_commit=declared_commit, draws=list(DRAWS), set=dict(key=lib.D20_KEY, sha256=lib.D20_SHA),
                forms=forms, rule=dict(text=RULE, known_when_written=KNOWN_AT_WRITING, form="predicted", **decide(forms["predicted"], rd.a9_reference(a9_sides)["wrist"]),
                                       note="evaluated for reading only; this field adopts nothing" if declared_commit else "DRAFT, NOT DECLARED: adopts nothing"),
                stored_read_excluded="A3's stored read of top10 counts in no mean", true_length="reported; decides nothing",
                sentences_note="D20 chose this arm: draws are fresh, sentences are not", provenance=lib.provenance_row(job_file))


def main(argv=None) -> int:
    from round20_compare import report_shards
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draw", action="append", default=[], help="K=MARGIN0_KEY,TOP10_KEY; every draw of the rule")
    p.add_argument("--a9-sides-file", required=True)
    p.add_argument("--a9-sides-sha256", required=True)
    p.add_argument("--declared-commit")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--store", default="r2")
    p.add_argument("--output-key", required=True)
    args = p.parse_args(argv)
    from train_text_to_motion import make_store
    store = make_store(args.store)
    if store.exists(args.output_key):
        raise SystemExit(f"{args.output_key} exists: refusing to overwrite")
    raw = Path(args.a9_sides_file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.a9_sides_sha256:
        raise SystemExit(f"{args.a9_sides_file} does not match --a9-sides-sha256")
    pairs, shards = {}, {}
    for spec in args.draw:
        k, _, keys = spec.partition("=")
        m_key, _, t_key = keys.partition(",")
        if not (k.isdigit() and m_key and t_key) or int(k) in pairs:
            p.error(f"--draw {spec!r}: K=MARGIN0_KEY,TOP10_KEY, once each")
        m, t = json.loads(store.get(m_key)), json.loads(store.get(t_key))
        pairs[int(k)] = (m, t)
        shards[int(k)] = (report_shards(store, m_key, m, None), report_shards(store, t_key, t, None))
    out = read(pairs, shards, a9_sides=json.loads(raw), boot=args.bootstrap, declared_commit=args.declared_commit)
    store.put(args.output_key, json.dumps(out, indent=1, default=float).encode())
    print(f"A5 outcome: {out['rule']['outcome']} {json.dumps(out['rule']['checks'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
